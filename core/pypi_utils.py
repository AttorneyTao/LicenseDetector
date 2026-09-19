import os
import re
import json
import asyncio
import time
import logging
import requests
import yaml
from .llm_provider import get_llm_provider
from datetime import datetime, timezone
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib3.exceptions import InsecureRequestWarning
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
from .utils import extract_copyright_info, extract_copyright_info_async  # 添加这行导入
from .config import LLM_CONFIG, SCORE_THRESHOLD  # 修改导入

# 日志设置
logger = logging.getLogger('main')
llm_logger = logging.getLogger('llm_interaction')

def _parse_package_name(url: str) -> str:
    """从 PyPI URL 中提取包名"""
    try:
        parsed = urlparse(url)
        parts = [p for p in parsed.path.split('/') if p]
        if len(parts) >= 2 and parts[0] == "project":
            return parts[1].strip('/')  # 移除尾部斜杠
        return parts[0].strip('/') if parts else ""
    except Exception as e:
        logger.error(f"Failed to parse package name from URL {url}: {str(e)}")
        return ""

# ---------------------------------------------------------------------------
# GitHub 仓库 URL 提取
#
# PyPI 元数据的 project_urls 是一个混杂字典：既有 Source/Repository/Homepage 这类
# 真正的仓库地址，也有 Funding（github.com/sponsors/<user>）、Issues、Wiki、
# Documentation 等非仓库页面。「取第一个值里含 github.com 的 URL」会把赞助页当
# 仓库，GitHub 流程解析失败后静默回落到 PyPI 元数据，丢失 LICENSE 全文与版权声明。
# 因此改为按 key 优先级挑选 + 规范化校验，只接受 github.com/owner/repo 形态。
# ---------------------------------------------------------------------------

# GitHub 站点上不是仓库的一级路径（/sponsors/xxx、/topics/xxx ...）
_GITHUB_RESERVED_TOP_SEGMENTS = {
    "sponsors", "topics", "features", "orgs", "settings", "notifications",
    "marketplace", "explore", "pricing", "login", "join", "about", "security",
    "enterprise", "team", "readme", "collections", "trending", "events",
    "search", "apps", "account", "sessions", "users", "site", "blog",
    "customer-stories", "solutions", "resources", "contact", "sponsors-explore",
}

# 仓库下的非仓库尾段：/owner/repo/issues、/wiki、/blob ... 一律收敛到仓库根
_GITHUB_NON_REPO_SUFFIXES = {
    "issues", "issue", "pulls", "pull", "wiki", "discussions", "discussion",
    "releases", "tags", "actions", "graphs", "network", "compare", "commits",
    "blob", "raw", "archive", "pulse", "contributors", "deployments",
    "environments", "projects", "insights", "labels", "milestones", "stars",
    "forks", "watchers", "activity", "branches", "stargazers", "subscribers",
}

# project_urls 里优先采信的 key（命中即排在最前）
_REPO_URL_KEY_PRIORITY = (
    "source", "sourcecode", "source code", "repository", "repo", "code",
    "github", "homepage", "home", "project home", "project",
    "源码", "源代码", "仓库", "项目主页", "主页",
)

# project_urls 里需要降权的 key：通常不是仓库，但部分包（如 sqlalchemy）的
# "Issue Tracker" 恰恰指向仓库根，因此不丢弃、只降到最低优先级——URL 形态校验
# （是否为 github.com/owner/repo）比 key 名更可靠。
_DEPRECATED_URL_KEY_HINTS = (
    "funding", "sponsor", "sponsors", "donate", "donation", "issue", "issues",
    "bug", "bugs", "tracker", "wiki", "discussion", "documentation", "docs",
    "changelog", "change log", "release notes", "download", "downloads",
)

_GITHUB_URL_IN_TEXT_RE = re.compile(
    r"https?://(?:www\.)?github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+"
    r"(?:/[^\s\"'<>)\]]*)?",
    re.IGNORECASE,
)


def _normalize_github_repo_url(raw: str) -> Optional[str]:
    """把任意形态的 GitHub 链接规范化为 https://github.com/owner/repo[/tree/...]。

    非仓库地址（赞助页、issues/wiki 页、docs.github.com、单段用户页等）返回 None。
    返回 None 表示"这不是一个可用的仓库地址"，调用方应继续尝试下一个候选。
    """
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip()

    if text.startswith("git+"):
        text = text[4:]
    if text.startswith("git@github.com:"):
        text = "https://github.com/" + text[len("git@github.com:"):]
    elif text.startswith("git://github.com/"):
        text = "https://github.com/" + text[len("git://github.com/"):]
    if not re.match(r"^https?://", text):
        text = "https://" + text.lstrip("/")

    parsed = urlparse(text)
    host = (parsed.hostname or "").lower()
    if host in ("raw.githubusercontent.com", "raw.github.com", "gist.github.com"):
        return None
    if host not in ("github.com", "www.github.com"):
        return None  # docs.github.com 等子域不是仓库

    segments = [s for s in parsed.path.split("/") if s]
    if len(segments) < 2:
        return None  # /owner 单段是用户页，不是仓库
    owner, repo = segments[0], segments[1]
    if owner.lower() in _GITHUB_RESERVED_TOP_SEGMENTS:
        return None  # /sponsors/xxx、/topics/xxx
    if repo.endswith(".git"):
        repo = repo[:-4]
    if not owner or not repo:
        return None

    rest = segments[2:]
    if rest and rest[0].lower() == "tree":
        if len(rest) > 1:
            # 保留子目录路径（monorepo 子包），供 GitHub 流程定位 sub_path
            return f"https://github.com/{owner}/{repo}/tree/" + "/".join(rest[1:])
        return f"https://github.com/{owner}/{repo}"
    if rest and rest[0].lower() in _GITHUB_NON_REPO_SUFFIXES:
        return f"https://github.com/{owner}/{repo}"
    return f"https://github.com/{owner}/{repo}"


def _extract_github_repo_url(info: Dict[str, Any]) -> Optional[str]:
    """从 PyPI 元数据中提取可用的 GitHub 仓库 URL。

    候选来源按可靠性排序：project_urls 优先 key > project_urls 其他 key >
    home_page > download_url > description 正文里的链接。
    """
    candidates: List[tuple] = []

    project_urls = info.get("project_urls") or {}
    if isinstance(project_urls, dict):
        for key, value in project_urls.items():
            if not isinstance(value, str):
                continue
            lower_key = str(key).strip().lower()
            if any(hint in lower_key for hint in _DEPRECATED_URL_KEY_HINTS):
                candidates.append((4, value))  # 降权：仅在别处找不到仓库时使用
            elif any(lower_key == p or p in lower_key for p in _REPO_URL_KEY_PRIORITY):
                candidates.append((0, value))
            else:
                candidates.append((1, value))

    for field, rank in (("home_page", 2), ("homepage", 2), ("download_url", 3)):
        value = info.get(field)
        if isinstance(value, str) and value:
            candidates.append((rank, value))

    description = info.get("description") or ""
    if isinstance(description, str) and description:
        for match in _GITHUB_URL_IN_TEXT_RE.finditer(description):
            candidates.append((5, match.group(0)))

    candidates.sort(key=lambda item: item[0])
    for _, raw in candidates:
        normalized = _normalize_github_repo_url(raw)
        if normalized:
            return normalized
    return None


def _pypi_project_page(package_name: str, version: Optional[str]) -> str:
    """PyPI 项目页链接（Description 标签页）。

    此前用的是 ``#files`` 锚点，落到只罗列构件的 Files 标签页；带许可证分类器
    与 README 正文的是 Description 页，即去掉锚点的裸项目页 URL。
    """
    version_part = f"/{version}" if version else ""
    return f"https://pypi.org/project/{package_name}{version_part}/"


class PyPIAPIError(Exception):
    """PyPI API 调用异常"""
    pass

def _create_session(retries: int = 3, backoff_factor: float = 0.5) -> requests.Session:
    """创建带重试机制的 Session"""
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[500, 502, 503, 504, 404],
        allowed_methods=frozenset(['GET', 'POST'])
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    
    # 设置代理（如果环境变量中有配置）
    http_proxy = os.getenv('HTTP_PROXY')
    https_proxy = os.getenv('HTTPS_PROXY')
    if http_proxy or https_proxy:
        session.proxies = {
            'http': http_proxy,
            'https': https_proxy or http_proxy
        }
    
    return session

def _fetch_pypi_metadata(package_name: str, max_retries: int = 3) -> Dict[str, Any]:
    """获取 PyPI 包的元数据，带重试机制"""
    url = f"https://pypi.org/pypi/{package_name}/json"
    session = _create_session(retries=max_retries)
    timeout = (5, 15)  # (连接超时, 读取超时)
    
    for attempt in range(max_retries):
        try:
            logger.debug(f"Attempting to fetch PyPI metadata for {package_name} (attempt {attempt + 1}/{max_retries})")
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.SSLError as e:
            logger.warning(f"SSL verification failed (attempt {attempt + 1}), retrying without verification")
            try:
                response = session.get(url, verify=False, timeout=timeout)
                response.raise_for_status()
                return response.json()
            except Exception as ssl_e:
                if attempt == max_retries - 1:
                    raise PyPIAPIError(f"SSL Error: {str(ssl_e)}")
        except requests.exceptions.Timeout:
            if attempt == max_retries - 1:
                raise PyPIAPIError(f"Timeout fetching metadata for {package_name}")
            logger.warning(f"Timeout fetching metadata (attempt {attempt + 1}), retrying...")
            time.sleep(attempt * 2)  # 指数退避
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise PyPIAPIError(f"Failed to fetch PyPI metadata: {str(e)}")
            logger.warning(f"Request failed (attempt {attempt + 1}): {str(e)}, retrying...")
            time.sleep(attempt * 2)

async def _standardize_license(license_info: Dict[str, Any]) -> str:
    """
    使用多级判断逻辑标准化 license 信息
    Args:
        license_info: 包含 license 相关信息的字典
    Returns:
        标准化的 SPDX 标识符
    """
    try:
        # 1. 首先检查 license_expression
        if license_info.get("license_expression"):
            return license_info["license_expression"]
            
        # 2. 检查 classifiers 中的 license 信息
        classifiers = license_info.get("classifiers", [])
        for classifier in classifiers:
            if classifier.startswith("License :: OSI Approved :: "):
                # 移除前缀并进行简单转换
                license_name = classifier.replace("License :: OSI Approved :: ", "")
                if "MIT" in license_name:
                    return "MIT"
                elif "Apache" in license_name:
                    return "Apache-2.0"
                elif "BSD" in license_name:
                    if "3" in license_name:
                        return "BSD-3-Clause"
                    return "BSD-2-Clause"
                    
        # 3. 检查基本的 license 字段
        raw_license = license_info.get("license")
        if raw_license:
            # 如果存在原始 license 文本，使用 LLM 进行分析
            try:
                # 读取提示词
                with open("prompts.yaml", 'r', encoding='utf-8') as f:
                    prompts = yaml.safe_load(f)
                
                # 初始化 LLM Provider
                provider = get_llm_provider()
                
                # 准备提示词
                prompt = prompts["license_standardize"].format(
                    license_string=raw_license
                )
                
                # 调用模型
                response = provider.generate(prompt)
                logger.debug(f"LLM raw response: {response}")
                
                # 清理并解析响应
                cleaned_response = response.strip()
                if cleaned_response.startswith("```"):
                    cleaned_response = "\n".join(cleaned_response.split("\n")[1:-1])
                    
                result = json.loads(cleaned_response)
                
                # 检查置信度
                confidence_threshold = SCORE_THRESHOLD / 100.0
                if result.get("confidence", 0) >= confidence_threshold:
                    return result["spdx_identifier"]
                    
            except Exception as e:
                logger.error(f"LLM analysis failed: {str(e)}")
                
        # 4. 如果所有方法都失败，返回 UNKNOWN
        return "UNKNOWN"
        
    except Exception as e:
        logger.error(f"Failed to standardize license info: {str(e)}")
        return "UNKNOWN"

async def process_pypi_repository(url: str, version: Optional[str] = None) -> Dict[str, Any]:
    """处理 PyPI 仓库信息"""
    logger.info(f"Starting PyPI repository processing: {url}")
    
    try:
        # 1. 解析包名
        package_name = _parse_package_name(url)
        logger.info(f"Parsed package name: {package_name}")
        
        if not package_name:
            logger.error(f"Could not parse package name from URL: {url}")
            return {
                "status": "error", 
                "error": "Invalid PyPI URL", 
                "input_url": url
            }
        
        # 2. 获取元数据
        try:
            metadata = _fetch_pypi_metadata(package_name)
            logger.info(f"Successfully fetched metadata for {package_name}")
            logger.debug(f"Raw PyPI metadata: {json.dumps(metadata, indent=2)}")  # 添加这行
        except PyPIAPIError as e:
            logger.error(f"PyPI API error for {package_name}: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "input_url": url,
                "component_name": package_name
            }
            
        # 3. 版本处理
        # PyPI releases 的 key 是精确 release 串（"1.0.0"），输入常缺省 patch 位（"1.0"/"1.25"）。
        # 先精确命中；不中则按「补零后的数值元组相等」匹配（"1.0" == "1.0.0"），避免静默回退到最新版。
        releases = metadata["releases"]
        resolved_version = None
        if version:
            if version in releases:
                resolved_version = version
            else:
                def _ver_tuple(s):
                    parts = []
                    for p in str(s).strip().lower().lstrip("v").split("."):
                        if p.isdigit():
                            parts.append(int(p))
                        else:
                            return None  # 含非纯数字段（rc/dev 等）→ 不参与数值等价匹配
                    return tuple(parts) if parts else None

                req = _ver_tuple(version)
                if req:
                    for key in releases:
                        kt = _ver_tuple(key)
                        if kt is None or not releases[key]:  # 跳过非纯数字 key 和被 yank 的空 release
                            continue
                        n = max(len(req), len(kt))
                        if req + (0,) * (n - len(req)) == kt + (0,) * (n - len(kt)):
                            resolved_version = key
                            logger.info(f"Normalized version match: input {version} -> PyPI release {key}")
                            break
        if resolved_version is None:
            resolved_version = metadata["info"]["version"]  # 回退到最新版本
            if version:
                logger.warning(
                    f"No PyPI release matched requested version {version!r}, falling back to latest: {resolved_version}"
                )
        
        # 4. 获取版本特定信息
        version_info = next((r for r in metadata["releases"][resolved_version] 
                           if r["packagetype"] == "sdist"), 
                          metadata["releases"][resolved_version][0])
        
        # 5. 基本信息提取
        info = metadata["info"]
        license_type = await _standardize_license(info)  # 传入完整的 info 字典
        readme_content = info.get("description", "")
        
        # 6. 源码仓库 URL 处理
        # 按 key 优先级挑选并规范化：Funding/Issues/Wiki 等非仓库页会被过滤掉，
        # 只有真正形如 github.com/owner/repo 的地址才会进入 GitHub 分析流程。
        repo_url = _extract_github_repo_url(info)
        if repo_url:
            logger.info(f"Resolved GitHub repository from PyPI metadata: {repo_url}")
        else:
            logger.debug(f"No usable GitHub repository URL in PyPI metadata for {package_name}")

        # 7. 调用 GitHub API 获取完整信息（如果有 GitHub 仓库）
        github_result = None
        use_github_result = False

        # 在调用 GitHub API 时也添加重试逻辑
        if repo_url:
            logger.info(f"Found GitHub repository: {repo_url}, using GitHub analysis as primary source")
            from core.github_utils import process_github_repository, GitHubAPI
            for attempt in range(3):  # GitHub API 重试3次
                try:
                    api = GitHubAPI()
                    github_result = await process_github_repository(
                        api,
                        repo_url,
                        resolved_version,
                        name=package_name,  # 供 monorepo 子包按组件名定位子目录
                    )
                    if github_result and github_result.get("status") == "success":
                        use_github_result = True
                        logger.info("Successfully obtained GitHub analysis results, will use as primary source")
                    else:
                        status = github_result.get("status") if github_result else None
                        logger.warning(
                            f"GitHub analysis attempt {attempt + 1} returned status={status}, "
                            f"falling back to PyPI metadata"
                        )
                    # 拿到结果即退出：非 success 多为确定性失败（仓库 404 等），
                    # 重试只会浪费 GitHub 配额；只有异常才需要重试。
                    break
                except Exception as e:
                    logger.warning(f"GitHub API attempt {attempt + 1} failed: {str(e)}")
                    if attempt == 2:  # 最后一次尝试失败
                        logger.error(f"Failed to process GitHub repository after 3 attempts: {str(e)}")
                    else:
                        await asyncio.sleep(2 ** attempt)  # 异步等待，避免阻塞事件循环
                        
        # 8. 处理版权信息和许可证信息
        if use_github_result and github_result:
            # 如果有GitHub结果，优先使用GitHub的信息
            logger.info("Using GitHub analysis results as primary source")
            
            # 基础信息保持PyPI的
            final_license_type = github_result.get("license_type", license_type)
            fallback_page = _pypi_project_page(package_name, resolved_version)
            final_license_files = github_result.get("license_files") or fallback_page

            # GitHub 仓库无对应版本 tag（回退到默认分支 blob 链接）时，改用 PyPI
            # 带版本号的页面链接，保证链接与版本对应；其余字段仍以 GitHub 分析为准。
            if github_result.get("used_default_branch"):
                versioned_url = fallback_page
                from core.utils import is_url_reachable
                if await is_url_reachable(versioned_url):
                    logger.info(
                        "GitHub has no matching version tag, using versioned PyPI link: %s",
                        versioned_url,
                    )
                    final_license_files = versioned_url
            final_license_analysis = github_result.get("license_analysis")
            final_has_license_conflict = github_result.get("has_license_conflict")
            final_readme_license = github_result.get("readme_license")
            final_license_file_license = github_result.get("license_file_license")
            final_copyright_notice = github_result.get("copyright_notice")
            
            # 如果GitHub没有找到版权信息，使用PyPI的author信息构建
            if not final_copyright_notice:
                author = info.get("author", "")
                if not author:
                    author = f"{package_name} original author and authors"
                current_year = datetime.now(timezone.utc).year
                final_copyright_notice = f"Copyright (c) {current_year} {author}"
                
            license_determination_reason = "Analyzed via GitHub repository (primary source)"
            
        else:
            # 没有GitHub结果或GitHub分析失败，使用PyPI信息
            logger.info("Using PyPI analysis results as primary source")
            
            final_license_type = license_type
            final_license_files = _pypi_project_page(package_name, resolved_version)
            final_license_analysis = None
            final_has_license_conflict = None
            final_readme_license = None
            final_license_file_license = None
            
            # 分析 README 中的许可证信息
            if readme_content:
                try:
                    from core.utils import analyze_license_content_async
                    readme_license_analysis = await analyze_license_content_async(readme_content)
                    if readme_license_analysis and readme_license_analysis.get("licenses"):
                        final_readme_license = readme_license_analysis.get("spdx_expression") if readme_license_analysis else None
                        final_license_analysis = readme_license_analysis
                except Exception as e:
                    logger.warning(f"Failed to analyze README license content: {str(e)}")
            
            # 处理版权信息
            author = info.get("author", "")
            if not author:
                author = f"{package_name} original author and authors"
            
            try:
                copyright_notice = await extract_copyright_info_async(readme_content)
            except Exception:
                copyright_notice = extract_copyright_info(readme_content)
            if not copyright_notice:
                current_year = datetime.now(timezone.utc).year
                copyright_notice = f"Copyright (c) {current_year} {author}"
            final_copyright_notice = copyright_notice
            
            license_determination_reason = "Fetched from PyPI registry"
        
        # 9. 返回结果
        result = {
            "input_url": url,
            "repo_url": repo_url,
            "input_version": version,
            "resolved_version": resolved_version,
            "used_default_branch": version is None,
            "component_name": package_name,
            "license_files": final_license_files,
            "license_analysis": final_license_analysis,
            "license_type": final_license_type,
            "has_license_conflict": final_has_license_conflict,
            "readme_license": final_readme_license,
            "license_file_license": final_license_file_license,
            "copyright_notice": final_copyright_notice,
            "status": "success",
            "license_determination_reason": license_determination_reason,
            "readme": readme_content[:5000] if readme_content else None
        }
        
        logger.info(f"Processing completed for PyPI package: {package_name}@{resolved_version}")
        if use_github_result:
            logger.info(f"Final result based on GitHub analysis from: {repo_url}")
        else:
            logger.info("Final result based on PyPI metadata")
        return result
        
    except Exception as e:
        logger.error(f"Error processing PyPI repository: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "input_url": url
        }