"""
Font input pipeline.

输入文件 input.xlsx 中全部为「字体」，与软件包(npm/pypi/maven...)的处理逻辑不同：
- 字体来源站点高度分散（Google Fonts、猫啃网、Fontshare、微信公众号、站酷……），
  大部分没有标准化 API，需要「针对性爬虫 + 小片段 LLM 抽取」来获取授权(license)与版权(copyright)信息。
- 能复用的能力尽量复用：来源是 github.com 的字体直接走既有 GitHub 流程；
  Google Fonts 复用 GitHub API client 读 google/fonts 仓库。

核心约束：**绝不把整页 HTML 丢给 LLM**。每个站点适配器先把页面裁剪到「授权/版权/作者」
相关的小片段（默认上限 _SNIPPET_CHAR_LIMIT 字符），再交给 LLM 归一化；能用规则直接判定
的（github / google_fonts / fontshare）完全不调 LLM。

每个 handler 返回与软件包流程一致的结果字段（见 RESULT_COLUMNS_ORDER），
以便复用 main.py 中既有的输出/风险评级/列重排逻辑。
"""

import re
import json
import asyncio
import logging
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse, unquote

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger("font")

# ---------------------------------------------------------------------------
# 通用配置
# ---------------------------------------------------------------------------
_UA = {"User-Agent": "Mozilla/5.0 (compatible; FontLicenseBot/1.0)"}
_HTTP_TIMEOUT = 20.0
# 交给 LLM 的片段字符上限（控制 token 成本，绝不传整页）
_SNIPPET_CHAR_LIMIT = 1800
# 抽取片段时围绕关键词的上下文窗口（字符）
_KEYWORD_WINDOW = 220

# 授权/版权相关关键词（中英文），用于在正文中定位相关片段
_LICENSE_KEYWORDS = [
    "授权", "版权", "著作权", "商用", "商业", "免费", "可商用", "可免费", "禁止",
    "字体来源", "字体作者", "设计师", "版权说明", "使用许可", "协议",
    "license", "licence", "copyright", "©", "commercial", "free for",
    "open font", "ofl", "apache", "all rights reserved",
]

# ---------------------------------------------------------------------------
# 站点分类（用于路由 + 统计）
# ---------------------------------------------------------------------------
# category 说明：
#   github        -> 复用既有 GitHub 流程（process_github_repository）
#   google_fonts  -> 复用 GitHub API client 读 google/fonts 仓库（按真实 LICENSE 判定）
#   fontshare     -> Indian Type Foundry，统一 ITF Free Font License（规则判定，无需 LLM）
#   maoken        -> 猫啃网聚合站，结构化页面，定向片段抽取
#   crawl_llm     -> 长尾分散站点，通用爬虫 + 小片段 LLM 抽取
GITHUB_DOMAINS = {"github.com"}
GOOGLE_FONTS_DOMAINS = {"fonts.google.com"}
FONTSHARE_DOMAINS = {"www.fontshare.com", "fontshare.com"}
MAOKEN_DOMAINS = {"www.maoken.com", "maoken.com"}


def classify_font_source(url: str) -> str:
    """根据 URL 域名返回处理类别。"""
    domain = urlparse(str(url)).netloc.lower()
    if domain in GITHUB_DOMAINS:
        return "github"
    if domain in GOOGLE_FONTS_DOMAINS:
        return "google_fonts"
    if domain in FONTSHARE_DOMAINS:
        return "fontshare"
    if domain in MAOKEN_DOMAINS:
        return "maoken"
    return "crawl_llm"


# ---------------------------------------------------------------------------
# 结果构造
# ---------------------------------------------------------------------------
def _make_result(
    *,
    name, version, url,
    repo_url=None,
    license_type=None,
    license_file_license=None,
    copyright_notice=None,
    license_files=None,
    reason=None,
    source=None,
    status="success",
    error=None,
) -> Dict[str, Any]:
    """构造与软件包流程一致的结果字段。"""
    return {
        "input_url": url,
        "input_name": name,
        "input_version": version,
        "component_name": name,
        "repo_url": repo_url,
        "license_type": license_type,
        "license_file_license": license_file_license,
        "readme_license": None,
        "copyright_notice": copyright_notice,
        "license_files": license_files,
        "license_analysis": {
            "license_determination_reason": reason,
            "license_source": source,
        },
        "has_license_conflict": False,
        "status": status,
        "error": error,
    }


def _pending_result(name, version, url, category) -> Dict[str, Any]:
    return _make_result(
        name=name, version=version, url=url,
        status="pending_font_adapter",
        error=f"字体来源 [{category}] 的适配器尚未实现（域名: {urlparse(str(url)).netloc}）",
    )


# ---------------------------------------------------------------------------
# 通用爬取 / 片段抽取工具
# ---------------------------------------------------------------------------
async def _fetch_html(url: str) -> Optional[str]:
    """抓取页面 HTML，失败返回 None。"""
    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT, follow_redirects=True, headers=_UA) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.text
    except Exception as e:
        logger.warning(f"抓取失败 {url}: {type(e).__name__} {e}")
        return None


def _html_to_text(html: str) -> str:
    """提取正文纯文本（去脚本/样式）。"""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    text = soup.get_text("\n")
    # 压缩空行
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def extract_license_snippet(text: str, char_limit: int = _SNIPPET_CHAR_LIMIT) -> str:
    """在正文中围绕授权/版权关键词截取片段，控制送入 LLM 的体量。

    若未命中关键词，则回退取正文开头若干字符（仍受 char_limit 限制）。
    """
    if not text:
        return ""
    lowered = text.lower()
    spans: List[tuple] = []
    for kw in _LICENSE_KEYWORDS:
        start = 0
        kw_l = kw.lower()
        while True:
            idx = lowered.find(kw_l, start)
            if idx == -1:
                break
            spans.append((max(0, idx - _KEYWORD_WINDOW), min(len(text), idx + len(kw) + _KEYWORD_WINDOW)))
            start = idx + len(kw)
            if len(spans) > 60:
                break

    if not spans:
        return text[:char_limit]

    # 合并重叠区间
    spans.sort()
    merged = [spans[0]]
    for s, e in spans[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    out: List[str] = []
    total = 0
    for s, e in merged:
        chunk = text[s:e]
        if total + len(chunk) > char_limit:
            chunk = chunk[: char_limit - total]
        out.append(chunk)
        total += len(chunk)
        if total >= char_limit:
            break
    return "\n...\n".join(out)


_LLM_PROMPT = """你是字体授权信息抽取助手。下面是从某个字体页面截取的片段（不是完整页面）。
请只依据片段内容，判断该字体的授权情况，并输出**严格的 JSON**，不要任何额外文字：

{{
  "license_type": "授权类型，如 OFL-1.1 / Apache-2.0 / SIL OFL / 商用免费 / 个人非商用 / 未知",
  "copyright_notice": "版权/作者信息，没有则填 null",
  "commercial_use": "allowed / not_allowed / unknown",
  "reason": "判断依据，一句话中文"
}}

字体名称：{name}
来源 URL：{url}
页面片段：
\"\"\"
{snippet}
\"\"\"
"""


async def _llm_extract_license(name, url, snippet: str) -> Dict[str, Any]:
    """用小片段调用 LLM 抽取授权信息，返回 dict。"""
    from core.llm_provider import generate_text_async
    prompt = _LLM_PROMPT.format(name=name, url=url, snippet=snippet)
    raw = await generate_text_async(prompt)
    # 去掉可能的 ```json 包裹
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        # 容错：尝试截取第一个 {...}
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        logger.warning(f"LLM 返回无法解析为 JSON: {raw[:200]}")
        return {"license_type": "未知", "copyright_notice": None, "commercial_use": "unknown", "reason": "LLM 输出解析失败"}


# ---------------------------------------------------------------------------
# Google Fonts 适配器（复用 GitHub API client 读 google/fonts 仓库）
# ---------------------------------------------------------------------------
# google/fonts 顶层目录即授权类型；不假定 OFL，按真实命中目录 + LICENSE 文件判定
_GF_LICENSE_DIRS = ["ofl", "apache", "ufl", "cc-by-sa"]
_GF_DIR_TO_LICENSE = {
    "ofl": "OFL-1.1",
    "apache": "Apache-2.0",
    "ufl": "Ubuntu Font License 1.0",
    "cc-by-sa": "CC-BY-SA",
}
_GF_METADATA_LICENSE_MAP = {
    "OFL": "OFL-1.1",
    "APACHE2": "Apache-2.0",
    "UFL": "Ubuntu Font License 1.0",
}


def _gf_family_slug(url: str) -> str:
    """从 fonts.google.com/specimen/<Family> 解析 family slug（小写去非字母数字）。"""
    path = urlparse(url).path
    seg = path.rstrip("/").split("/")[-1]
    family = unquote(seg).replace("+", " ")
    return re.sub(r"[^a-z0-9]", "", family.lower())


def _parse_copyright_from_license(text: str) -> Optional[str]:
    """从 LICENSE/OFL 文本首个 Copyright 行提取版权声明。"""
    if not text:
        return None
    for line in text.splitlines():
        if "copyright" in line.lower():
            return line.strip()
    return None


async def process_google_font(api, name, version, url) -> Dict[str, Any]:
    """在 google/fonts 的 4 个授权目录中定位 family，按真实 LICENSE 判定授权。"""
    slug = _gf_family_slug(url)
    if not slug:
        return _make_result(name=name, version=version, url=url,
                            status="error", error="无法从 URL 解析 Google Fonts family")

    found_dir = None
    files: List[str] = []
    for d in _GF_LICENSE_DIRS:
        try:
            listing = await api._make_request(f"/repos/google/fonts/contents/{d}/{slug}")
            if isinstance(listing, list) and listing:
                found_dir = d
                files = [item.get("name", "") for item in listing]
                break
        except Exception:
            continue  # 该目录下无此 family，尝试下一个

    if not found_dir:
        return _make_result(name=name, version=version, url=url,
                            status="not_found",
                            error=f"google/fonts 4 个授权目录均未找到 family: {slug}")

    base = f"{found_dir}/{slug}"
    repo_url = f"https://github.com/google/fonts/tree/main/{base}"
    raw_base = f"https://raw.githubusercontent.com/google/fonts/main/{base}"

    # 读取真实 LICENSE 文件确定授权 + 版权
    # 注意：直接抓 raw 文件，规避 github_utils.get_file_content 的 proxies kwarg 兼容问题
    license_file_name = next(
        (f for f in files if f.upper() in ("OFL.TXT", "LICENSE.TXT", "LICENSE", "UFL.TXT")),
        None,
    )
    copyright_notice = None
    if license_file_name:
        license_text = await _fetch_html(f"{raw_base}/{license_file_name}")
        copyright_notice = _parse_copyright_from_license(license_text or "")

    # 读取 METADATA.pb 校验 license 字段 + 兜底 copyright
    license_type = _GF_DIR_TO_LICENSE.get(found_dir)
    metadata = await _fetch_html(f"{raw_base}/METADATA.pb")
    if metadata:
        m = re.search(r'license:\s*"([^"]+)"', metadata)
        if m:
            license_type = _GF_METADATA_LICENSE_MAP.get(m.group(1).strip(), license_type)
        if not copyright_notice:
            mc = re.search(r'copyright:\s*"([^"]+)"', metadata)
            if mc:
                copyright_notice = mc.group(1).strip()

    license_file_url = (
        f"https://github.com/google/fonts/blob/main/{base}/{license_file_name}"
        if license_file_name else None
    )
    return _make_result(
        name=name, version=version, url=url,
        repo_url=repo_url,
        license_type=license_type,
        license_file_license=license_type,
        copyright_notice=copyright_notice,
        license_files=license_file_url,
        reason=f"命中 google/fonts/{found_dir} 目录，依据真实 {license_file_name or 'METADATA.pb'} 判定",
        source="google_fonts_repo",
    )


# ---------------------------------------------------------------------------
# Fontshare 适配器（每个字体的授权按官方 API 的 license_type 字段判定）
# ---------------------------------------------------------------------------
# Fontshare 同时提供两类免费字体：
#   itf_ffl  -> Closed Source，ITF Free Font License（ITF 自有）
#   sil_ofl  -> Open Source，SIL Open Font License（归各设计师/发布者所有）
# 因此不能一刀切，必须按官方目录里每个字体的 license_type 判定。
_FONTSHARE_API = "https://api.fontshare.com/v2/fonts"
_FONTSHARE_LICENSE_MAP = {
    "itf_ffl": "ITF Free Font License",
    "sil_ofl": "OFL-1.1",
}
_FONTSHARE_LICENSE_URL = {
    "itf_ffl": "https://www.fontshare.com/licenses/itf-ffl",
    "sil_ofl": "https://www.fontshare.com/licenses/ofl",
}
_fontshare_catalog: Optional[Dict[str, dict]] = None
_fontshare_lock = asyncio.Lock()


async def _load_fontshare_catalog() -> Dict[str, dict]:
    """加载并缓存 Fontshare 全量字体目录（slug -> font 元数据）。"""
    global _fontshare_catalog
    async with _fontshare_lock:
        if _fontshare_catalog:  # 仅在已成功缓存（非空）时复用
            return _fontshare_catalog
        catalog: Dict[str, dict] = {}
        try:
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT, follow_redirects=True, headers=_UA) as client:
                offset = 0
                while True:
                    fonts = None
                    # 单页重试，规避代理/SSL 偶发抖动
                    for attempt in range(3):
                        try:
                            resp = await client.get(_FONTSHARE_API, params={"limit": 100, "offset": offset})
                            resp.raise_for_status()
                            fonts = resp.json().get("fonts", [])
                            break
                        except Exception as e:
                            if attempt == 2:
                                raise
                            await asyncio.sleep(1.5 * (attempt + 1))
                    for f in fonts:
                        if f.get("slug"):
                            catalog[f["slug"].lower()] = f
                    if len(fonts) < 100:
                        break
                    offset += 100
                    if offset > 2000:  # 安全上限
                        break
        except Exception as e:
            logger.warning(f"加载 Fontshare 目录失败: {type(e).__name__} {e}")
            # 不缓存失败结果，留待下次重试
            return catalog

        if catalog:
            _fontshare_catalog = catalog  # 仅缓存成功结果
        return catalog


async def process_fontshare_font(name, version, url) -> Dict[str, Any]:
    """按 Fontshare API 的 license_type 判定单个字体授权（itf_ffl / sil_ofl）。"""
    slug = urlparse(url).path.rstrip("/").split("/")[-1].lower()
    catalog = await _load_fontshare_catalog()
    font = catalog.get(slug)
    if not font:
        return _make_result(name=name, version=version, url=url, repo_url=url,
                            status="not_found",
                            error=f"Fontshare 目录中未找到 slug: {slug}", source="fontshare_api")

    lt = font.get("license_type")
    license_type = _FONTSHARE_LICENSE_MAP.get(lt, lt or "未知")
    designers = [d.get("name") for d in (font.get("designers") or []) if d.get("name")]
    publisher = (font.get("publisher") or {}).get("name")
    copyright_notice = ", ".join(designers) if designers else publisher

    license_terms = _FONTSHARE_LICENSE_URL.get(lt, "https://www.fontshare.com/licenses")
    kind = "开源 SIL OFL" if lt == "sil_ofl" else ("闭源 ITF FFL" if lt == "itf_ffl" else lt)
    return _make_result(
        name=name, version=version, url=url,
        repo_url=url,
        license_type=license_type,
        license_file_license=license_type,
        copyright_notice=copyright_notice,
        # 证据URL指向该字体页面；许可证条款文本见 reason 中的链接
        license_files=url,
        reason=f"Fontshare API license_type={lt}（{kind}），条款见 {license_terms}",
        source="fontshare_api",
    )


# ---------------------------------------------------------------------------
# 猫啃网适配器（结构化聚合站，定向片段抽取 + 小片段 LLM）
# ---------------------------------------------------------------------------
async def process_maoken_font(name, version, url) -> Dict[str, Any]:
    """猫啃网：截取授权说明片段，必要时用小片段 LLM 归一化。"""
    html = await _fetch_html(url)
    if not html:
        return _make_result(name=name, version=version, url=url,
                            status="error", error="页面抓取失败", source="maoken")
    text = _html_to_text(html)
    snippet = extract_license_snippet(text)
    extracted = await _llm_extract_license(name, url, snippet)
    return _make_result(
        name=name, version=version, url=url,
        repo_url=url,
        license_type=extracted.get("license_type"),
        license_file_license=extracted.get("license_type"),
        copyright_notice=extracted.get("copyright_notice"),
        license_files=url,  # 证据URL：实际抽取片段所在的猫啃网页面
        reason=extracted.get("reason"),
        source="maoken_snippet_llm",
    )


# ---------------------------------------------------------------------------
# 长尾通用适配器（爬虫 + 小片段 LLM）
# ---------------------------------------------------------------------------
async def process_generic_crawl_llm(name, version, url) -> Dict[str, Any]:
    """通用：抓正文 -> 关键词窗口截取 -> 小片段 LLM 抽取。"""
    html = await _fetch_html(url)
    if not html:
        return _make_result(name=name, version=version, url=url,
                            status="error", error="页面抓取失败", source="generic_crawl_llm")
    text = _html_to_text(html)
    snippet = extract_license_snippet(text)
    if not snippet.strip():
        return _make_result(name=name, version=version, url=url,
                            status="error", error="正文为空，无法抽取", source="generic_crawl_llm")
    extracted = await _llm_extract_license(name, url, snippet)
    return _make_result(
        name=name, version=version, url=url,
        repo_url=url,
        license_type=extracted.get("license_type"),
        license_file_license=extracted.get("license_type"),
        copyright_notice=extracted.get("copyright_notice"),
        license_files=url,  # 证据URL：实际抽取片段所在的来源页面
        reason=extracted.get("reason"),
        source="generic_crawl_llm",
    )


# ---------------------------------------------------------------------------
# 分发器
# ---------------------------------------------------------------------------
async def process_font_entry(
    api,
    name: Optional[str],
    version: Optional[str],
    url: str,
) -> Dict[str, Any]:
    """处理单个字体条目，按来源站点路由到对应 handler。"""
    category = classify_font_source(url)
    logger.info(f"字体条目路由: name={name} category={category} url={url}")

    try:
        if category == "github":
            from core.github_utils import process_github_repository, normalize_github_url
            result = await process_github_repository(
                api, normalize_github_url(url), version, name=name
            )
            result["input_name"] = name
            result["input_url"] = url
            return result

        if category == "google_fonts":
            return await process_google_font(api, name, version, url)

        if category == "fontshare":
            return await process_fontshare_font(name, version, url)

        if category == "maoken":
            return await process_maoken_font(name, version, url)

        return await process_generic_crawl_llm(name, version, url)

    except Exception as e:
        logger.error(f"字体处理失败 [{category}] {url}: {e}", exc_info=True)
        return _make_result(name=name, version=version, url=url,
                            status="error", error=f"{type(e).__name__}: {e}", source=category)
