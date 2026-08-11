"""Debian / Ubuntu (deb) 包的许可证分析。

deb 是少数拥有强制性机器可读许可证元数据的生态：每个源码包都必须携带
``debian/copyright``，现代包使用 DEP-5（copyright-format 1.0）格式，其中
逐段声明了 ``Files`` / ``Copyright`` / ``License``。因此这里以**确定性解析**
为主，只有在 copyright 文件不是 DEP-5、或解析不出任何许可证时，才退回
LLM 分析——此时原文已经在手，LLM 只是兜底而非主路径。

整体流程：
    1. 判定发行版（namespace / distro qualifier / 版本号特征，均不确定时两边都试）
    2. 解析源码包名、版本与组件（二进制包名会映射回源码包）
    3. 下载 debian/copyright 原文（每个发行版都有主源与备用源）
    4. 解析 DEP-5 得到 SPDX 表达式与版权声明，失败则交给 LLM

已验证的数据源：
    Ubuntu  api.launchpad.net          版本 / 组件 / 二进制→源码包
            changelogs.ubuntu.com      copyright 原文（版本需剥掉 epoch）
    Debian  sources.debian.org/api     版本 / area
            sources.debian.org/data    copyright 原文（版本需保留 epoch）
            metadata.ftp-master        copyright 备用源（版本需剥掉 epoch）
            api.ftp-master madison     二进制→源码包 + component
            snapshot.debian.org        二进制→源码包备用源
"""

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, quote

import aiohttp

from core.utils import (
    analyze_license_content_async,
    infer_deb_vendor,
    prepare_license_text,
)

logger = logging.getLogger("main")
substep_logger = logging.getLogger("substep")

# ---------------------------------------------------------------------------
# 数据源
# ---------------------------------------------------------------------------

LAUNCHPAD_API = "https://api.launchpad.net/devel"
UBUNTU_CHANGELOGS = "https://changelogs.ubuntu.com/changelogs/pool"
DEBIAN_SOURCES_API = "https://sources.debian.org/api"
DEBIAN_SOURCES_DATA = "https://sources.debian.org/data"
DEBIAN_SOURCES_WEB = "https://sources.debian.org/src"
DEBIAN_METADATA = "https://metadata.ftp-master.debian.org/changelogs"
DEBIAN_MADISON = "https://api.ftp-master.debian.org/madison"
DEBIAN_SNAPSHOT_BINARY = "https://snapshot.debian.org/mr/binary"
LAUNCHPAD_WEB = "https://launchpad.net/ubuntu/+source"

HTTP_TIMEOUT_SECONDS = 30
HTTP_RETRIES = 2
USER_AGENT = "LicenseDetector/1.0 (license compliance scanner)"

VENDOR_UBUNTU = "ubuntu"
VENDOR_DEBIAN = "debian"


# ---------------------------------------------------------------------------
# URL 识别与解析
# ---------------------------------------------------------------------------

def is_deb_package_url(url: Any) -> bool:
    """判断 URL 是否指向 Debian / Ubuntu 的包页面。"""
    if not isinstance(url, str) or not url.strip():
        return False
    parsed = urlparse(url.strip())
    host = parsed.netloc.lower()
    path = parsed.path.strip("/")
    if host in {"launchpad.net", "www.launchpad.net"} and path.startswith("ubuntu/+source/"):
        return True
    if host == "sources.debian.org" and path.startswith("src/"):
        return True
    return False


def parse_deb_url(url: str) -> Tuple[str, str, Optional[str]]:
    """把包页面 URL 解析为 (vendor, package_name, version)。

    Raises:
        ValueError: URL 不是可识别的 deb 包地址。
    """
    parsed = urlparse((url or "").strip())
    host = parsed.netloc.lower()
    parts = [p for p in parsed.path.split("/") if p]

    if host in {"launchpad.net", "www.launchpad.net"} and len(parts) >= 3:
        # /ubuntu/+source/<name>[/<version>]
        name = parts[2]
        version = parts[3] if len(parts) > 3 else None
        return VENDOR_UBUNTU, name, version

    if host == "sources.debian.org" and len(parts) >= 2:
        # /src/<name>[/<version>]
        name = parts[1]
        version = parts[2] if len(parts) > 2 else None
        return VENDOR_DEBIAN, name, version

    raise ValueError(f"Not a deb package URL: {url}")


# ---------------------------------------------------------------------------
# 版本与路径规则
# ---------------------------------------------------------------------------

def strip_epoch(version: Optional[str]) -> Optional[str]:
    """去掉 deb 版本号的 epoch 前缀（``2:9.1.0-1`` -> ``9.1.0-1``）。

    Ubuntu 的 pool 路径与 Debian 的 metadata 路径都不含 epoch，
    而 sources.debian.org 的 data 路径必须保留 epoch。
    """
    if not version:
        return version
    return version.split(":", 1)[1] if ":" in version else version


def pool_prefix(source_name: str) -> str:
    """deb 归档池的目录前缀：``lib`` 开头的包取前 4 个字符，其余取首字母。"""
    return source_name[:4] if source_name.startswith("lib") else source_name[:1]


def _versions_equal(a: Optional[str], b: Optional[str]) -> bool:
    """比较两个 deb 版本号，epoch 有无都视为相同。"""
    if not a or not b:
        return False
    a, b = a.strip(), b.strip()
    return a == b or strip_epoch(a) == strip_epoch(b)


# ---------------------------------------------------------------------------
# HTTP 帮助函数
# ---------------------------------------------------------------------------

async def _http_get(url: str, as_json: bool = False) -> Optional[Any]:
    """GET 请求，失败返回 None（不抛异常，由调用方决定回退路径）。"""
    last_error: Optional[Exception] = None
    for attempt in range(HTTP_RETRIES + 1):
        if attempt:
            await asyncio.sleep(1.5 * attempt)
        try:
            timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT_SECONDS)
            headers = {"User-Agent": USER_AGENT}
            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                async with session.get(url, allow_redirects=True) as resp:
                    if resp.status == 404:
                        substep_logger.info(f"[DEB] 404: {url}")
                        return None
                    if resp.status >= 400:
                        last_error = RuntimeError(f"HTTP {resp.status}")
                        continue
                    text = await resp.text()
                    if not as_json:
                        return text
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError as e:
                        substep_logger.warning(f"[DEB] 响应不是合法 JSON: {url} ({e})")
                        return None
        except Exception as e:  # 网络异常、超时等
            last_error = e
    substep_logger.warning(f"[DEB] 请求失败 {url}: {last_error}")
    return None


# ---------------------------------------------------------------------------
# Ubuntu：源码包解析
# ---------------------------------------------------------------------------

async def _resolve_ubuntu_source(name: str, version: Optional[str]) -> Optional[Dict[str, Any]]:
    """通过 Launchpad 解析 Ubuntu 源码包名、版本与组件。

    先按源码包名查；查不到再按二进制包名查，经 build 资源映射回源码包。
    """
    url = (
        f"{LAUNCHPAD_API}/ubuntu/+archive/primary"
        f"?ws.op=getPublishedSources&source_name={quote(name)}&exact_match=true"
    )
    data = await _http_get(url, as_json=True)
    entries = (data or {}).get("entries") or []

    if entries:
        entry = _pick_launchpad_entry(entries, version, "source_package_version")
        return {
            "vendor": VENDOR_UBUNTU,
            "source_name": entry.get("source_package_name") or name,
            "source_version": entry.get("source_package_version"),
            "component": entry.get("component_name") or "main",
            "matched_requested": _versions_equal(entry.get("source_package_version"), version),
        }

    substep_logger.info(f"[DEB] Ubuntu 源码包未命中，改按二进制包查询: {name}")
    return await _resolve_ubuntu_binary(name, version)


async def _resolve_ubuntu_binary(name: str, version: Optional[str]) -> Optional[Dict[str, Any]]:
    """Ubuntu 二进制包名 -> 源码包（getPublishedBinaries -> build -> source）。"""
    url = (
        f"{LAUNCHPAD_API}/ubuntu/+archive/primary"
        f"?ws.op=getPublishedBinaries&binary_name={quote(name)}&exact_match=true&ordered=false"
    )
    data = await _http_get(url, as_json=True)
    entries = (data or {}).get("entries") or []
    if not entries:
        return None

    entry = _pick_launchpad_entry(entries, version, "binary_package_version")
    build_link = entry.get("build_link")
    if not build_link:
        return None

    build = await _http_get(build_link, as_json=True)
    source_name = (build or {}).get("source_package_name")
    if not source_name:
        return None

    source_version = (build or {}).get("source_package_version") or entry.get("binary_package_version")
    substep_logger.info(f"[DEB] Ubuntu 二进制包 {name} -> 源码包 {source_name}")
    return {
        "vendor": VENDOR_UBUNTU,
        "source_name": source_name,
        "source_version": source_version,
        "component": entry.get("component_name") or "main",
        "matched_requested": _versions_equal(entry.get("binary_package_version"), version),
        "binary_name": name,
    }


def _pick_launchpad_entry(
    entries: List[Dict[str, Any]], version: Optional[str], version_field: str
) -> Dict[str, Any]:
    """优先取版本完全匹配的发布记录，否则取最新的一条（Launchpad 按时间倒序返回）。"""
    if version:
        for entry in entries:
            if _versions_equal(entry.get(version_field), version):
                return entry
        substep_logger.info(f"[DEB] Launchpad 未找到版本 {version}，改用最新发布记录")
    return entries[0]


# ---------------------------------------------------------------------------
# Debian：源码包解析
# ---------------------------------------------------------------------------

async def _resolve_debian_source(name: str, version: Optional[str]) -> Optional[Dict[str, Any]]:
    """通过 sources.debian.org 解析 Debian 源码包版本与 area。"""
    data = await _http_get(f"{DEBIAN_SOURCES_API}/src/{quote(name)}/", as_json=True)
    versions = (data or {}).get("versions") or []

    if not versions:
        substep_logger.info(f"[DEB] Debian 源码包未命中，改按二进制包查询: {name}")
        return await _resolve_debian_binary(name, version)

    picked = None
    if version:
        picked = next((v for v in versions if _versions_equal(v.get("version"), version)), None)
        if picked is None:
            substep_logger.info(f"[DEB] Debian 未找到版本 {version}，改用最新版本")
    picked = picked or versions[0]

    return {
        "vendor": VENDOR_DEBIAN,
        "source_name": name,
        "source_version": picked.get("version"),
        "component": picked.get("area") or "main",
        "matched_requested": _versions_equal(picked.get("version"), version),
    }


async def _resolve_debian_binary(name: str, version: Optional[str]) -> Optional[Dict[str, Any]]:
    """Debian 二进制包名 -> 源码包。madison 为主源，snapshot 为备用源。"""
    source_name, source_version, component = await _debian_madison_lookup(name, version)

    if not source_name:
        source_name, source_version = await _debian_snapshot_lookup(name, version)
        component = None

    if not source_name:
        return None

    substep_logger.info(f"[DEB] Debian 二进制包 {name} -> 源码包 {source_name}")

    # area 以 sources.debian.org 为准（madison 的 component 可能与其命名不同）
    resolved = await _resolve_debian_source(source_name, source_version)
    if resolved:
        resolved["binary_name"] = name
        resolved["matched_requested"] = _versions_equal(source_version, version) or resolved["matched_requested"]
        return resolved

    if not source_version:
        return None
    return {
        "vendor": VENDOR_DEBIAN,
        "source_name": source_name,
        "source_version": source_version,
        "component": component or "main",
        "matched_requested": _versions_equal(source_version, version),
        "binary_name": name,
    }


async def _debian_madison_lookup(
    name: str, version: Optional[str]
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """查询 api.ftp-master.debian.org 的 madison 接口。

    返回结构形如::

        [{"libssl3": {"oldstable": {"3.0.17-1~deb12u2": {
            "component": "main", "source": "openssl", "source_version": "..."}}}}]
    """
    data = await _http_get(f"{DEBIAN_MADISON}?package={quote(name)}&f=json", as_json=True)
    if not isinstance(data, list):
        return None, None, None

    candidates: List[Dict[str, Any]] = []
    for block in data:
        if not isinstance(block, dict):
            continue
        for _pkg, suites in block.items():
            if not isinstance(suites, dict):
                continue
            for _suite, versions in suites.items():
                if not isinstance(versions, dict):
                    continue
                for binary_version, info in versions.items():
                    if not isinstance(info, dict):
                        continue
                    candidates.append({
                        "binary_version": binary_version,
                        "source": info.get("source") or name,
                        "source_version": info.get("source_version") or binary_version,
                        "component": info.get("component"),
                    })

    if not candidates:
        return None, None, None

    picked = None
    if version:
        picked = next(
            (c for c in candidates
             if _versions_equal(c["binary_version"], version)
             or _versions_equal(c["source_version"], version)),
            None,
        )
    picked = picked or candidates[0]
    return picked["source"], picked["source_version"], picked["component"]


async def _debian_snapshot_lookup(
    name: str, version: Optional[str]
) -> Tuple[Optional[str], Optional[str]]:
    """snapshot.debian.org 的二进制包索引，作为 madison 不可用时的备用源。"""
    data = await _http_get(f"{DEBIAN_SNAPSHOT_BINARY}/{quote(name)}/", as_json=True)
    results = (data or {}).get("result") or []
    if not results:
        return None, None

    picked = None
    if version:
        picked = next((r for r in results if _versions_equal(r.get("binary_version"), version)), None)
    picked = picked or results[0]
    return picked.get("source"), picked.get("version")


# ---------------------------------------------------------------------------
# copyright 原文获取
# ---------------------------------------------------------------------------

def build_copyright_urls(info: Dict[str, Any]) -> List[str]:
    """按发行版拼出 copyright 文件地址（主源在前，备用源在后）。

    注意两边的 epoch 规则相反：Ubuntu pool 路径与 Debian metadata 路径
    都不含 epoch，而 sources.debian.org 的 data 路径必须保留 epoch。
    """
    source = info["source_name"]
    version = info.get("source_version")
    component = info.get("component") or "main"
    prefix = pool_prefix(source)
    if not version:
        return []

    if info["vendor"] == VENDOR_UBUNTU:
        no_epoch = strip_epoch(version)
        return [
            f"{UBUNTU_CHANGELOGS}/{component}/{prefix}/{source}/{source}_{no_epoch}/copyright"
        ]

    return [
        f"{DEBIAN_SOURCES_DATA}/{component}/{prefix}/{source}/{version}/debian/copyright",
        f"{DEBIAN_METADATA}//{component}/{prefix}/{source}/{source}_{strip_epoch(version)}_copyright",
    ]


async def _fetch_copyright(info: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """依次尝试各个 copyright 地址，返回 (原文, 命中的 URL)。"""
    for url in build_copyright_urls(info):
        text = await _http_get(url)
        if text and text.strip():
            substep_logger.info(f"[DEB] 已获取 copyright: {url}")
            return text, url
    return None, None


# ---------------------------------------------------------------------------
# DEP-5 解析
# ---------------------------------------------------------------------------

_DEP5_FORMAT_MARKER = "copyright-format/1.0"

# DEP-5 使用 Debian 自己的许可证短名，与 SPDX 标识符并不一致，这里做映射。
# 未收录的短名保持原样输出，风险等级会落到 unknown，便于后续补充。
_DEP5_LICENSE_MAP = {
    "expat": "MIT",
    "mit": "MIT",
    "gpl-1": "GPL-1.0-only",
    "gpl-1+": "GPL-1.0-or-later",
    "gpl-2": "GPL-2.0-only",
    "gpl-2+": "GPL-2.0-or-later",
    "gpl-3": "GPL-3.0-only",
    "gpl-3+": "GPL-3.0-or-later",
    "lgpl-2": "LGPL-2.0-only",
    "lgpl-2+": "LGPL-2.0-or-later",
    "lgpl-2.1": "LGPL-2.1-only",
    "lgpl-2.1+": "LGPL-2.1-or-later",
    "lgpl-3": "LGPL-3.0-only",
    "lgpl-3+": "LGPL-3.0-or-later",
    "agpl-3": "AGPL-3.0-only",
    "agpl-3+": "AGPL-3.0-or-later",
    "apache-2.0": "Apache-2.0",
    "apache-2": "Apache-2.0",
    "artistic": "Artistic-1.0",
    "artistic-1": "Artistic-1.0",
    "artistic-1.0": "Artistic-1.0",
    "artistic-2.0": "Artistic-2.0",
    "apache-1.1": "Apache-1.1",
    "bsd-2-clause": "BSD-2-Clause",
    "bsd-3-clause": "BSD-3-Clause",
    "bsd-4-clause": "BSD-4-Clause",
    "boost": "BSL-1.0",
    "bsl-1.0": "BSL-1.0",
    "cc0-1.0": "CC0-1.0",
    "cc-by-3.0": "CC-BY-3.0",
    "cc-by-4.0": "CC-BY-4.0",
    "cc-by-sa-3.0": "CC-BY-SA-3.0",
    "cc-by-sa-4.0": "CC-BY-SA-4.0",
    "cddl-1.0": "CDDL-1.0",
    "epl-1.0": "EPL-1.0",
    "epl-2.0": "EPL-2.0",
    "gfdl-1.2": "GFDL-1.2-only",
    "gfdl-1.2+": "GFDL-1.2-or-later",
    "gfdl-1.3": "GFDL-1.3-only",
    "gfdl-1.3+": "GFDL-1.3-or-later",
    "isc": "ISC",
    "mpl-1.1": "MPL-1.1",
    "mpl-2.0": "MPL-2.0",
    "ofl-1.1": "OFL-1.1",
    "openssl": "OpenSSL",
    "psf-2": "PSF-2.0",
    "python": "Python-2.0",
    "tcl": "TCL",
    "unicode": "Unicode-DFS-2016",
    "wtfpl": "WTFPL",
    "x11": "X11",
    "zlib": "Zlib",
    "zpl-2.1": "ZPL-2.1",
}


def _parse_dep5_stanzas(text: str) -> List[Dict[str, str]]:
    """把 DEP-5 文本切成 stanza，每个 stanza 解析为字段名 -> 首行值。

    DEP-5 的多行字段用行首空白续行，字段内的空行写作 " ."，因此真正的
    空行就是 stanza 分隔符。短名类字段（如 License）只取首行。
    """
    stanzas: List[Dict[str, str]] = []
    for block in re.split(r"\n(?:[ \t]*\n)+", text):
        if not block.strip():
            continue
        fields: Dict[str, str] = {}
        current: Optional[str] = None
        continuation: List[str] = []
        for line in block.splitlines():
            match = re.match(r"^([A-Za-z0-9][A-Za-z0-9-]*):\s*(.*)$", line)
            if match:
                if current:
                    fields[f"{current}__full"] = "\n".join(continuation).strip()
                current = match.group(1).strip()
                fields[current] = match.group(2).strip()
                continuation = [match.group(2).strip()]
            elif current and line.startswith((" ", "\t")):
                stripped = line.strip()
                continuation.append("" if stripped == "." else stripped)
        if current:
            fields[f"{current}__full"] = "\n".join(continuation).strip()
        if fields:
            stanzas.append(fields)
    return stanzas


def dep5_license_to_spdx(short_name: Optional[str]) -> Optional[str]:
    """把 DEP-5 的许可证短名转成 SPDX 表达式。

    支持 DEP-5 允许的 ``A or B`` / ``A and B`` 组合，以及
    ``GPL-2+ with OpenSSL exception`` 这类例外说明（例外部分丢弃，
    因为 Debian 的措辞不是 SPDX 的 exception 标识符）。
    """
    if not short_name or not short_name.strip():
        return None

    value = short_name.strip()
    # 例外说明不参与 SPDX 表达式
    value = re.split(r"\s+with\s+", value, maxsplit=1, flags=re.IGNORECASE)[0].strip()

    parts = re.split(r"\s+(or|and)\s+", value, flags=re.IGNORECASE)
    if len(parts) == 1:
        key = parts[0].strip().lower()
        if not key:
            return None
        return _DEP5_LICENSE_MAP.get(key, parts[0].strip())

    rendered: List[str] = []
    for index, part in enumerate(parts):
        if index % 2 == 1:  # 运算符
            rendered.append(part.upper())
            continue
        atom = dep5_license_to_spdx(part)
        if not atom:
            return None
        rendered.append(atom)
    return " ".join(rendered)


def parse_dep5_copyright(text: str) -> Dict[str, Any]:
    """确定性解析 DEP-5 copyright 文件。

    Returns:
        dict: 含 ``is_dep5`` / ``primary_license`` / ``bundled_licenses`` /
        ``spdx_expression`` / ``copyright_notice`` / ``upstream_source``。
        非 DEP-5 时 ``is_dep5`` 为 False 且其余字段为空。
    """
    empty = {
        "is_dep5": False,
        "primary_license": None,
        "bundled_licenses": [],
        "spdx_expression": None,
        "copyright_notice": None,
        "upstream_source": None,
        "upstream_name": None,
    }
    if not text or not text.strip():
        return empty

    stanzas = _parse_dep5_stanzas(text)
    if not stanzas:
        return empty

    header = stanzas[0]
    if _DEP5_FORMAT_MARKER not in (header.get("Format") or ""):
        return empty

    files_stanzas = [s for s in stanzas if "Files" in s]
    if not files_stanzas:
        return {**empty, "is_dep5": True}

    # Files: * 声明的是整个包的主许可证；找不到时退而用第一个 Files 段
    primary_stanza = next(
        (s for s in files_stanzas if s.get("Files", "").strip() in {"*", "*.*"}),
        files_stanzas[0],
    )
    primary_license = dep5_license_to_spdx(primary_stanza.get("License"))

    bundled: List[str] = []
    for stanza in files_stanzas:
        if stanza is primary_stanza:
            continue
        spdx = dep5_license_to_spdx(stanza.get("License"))
        if spdx and spdx != primary_license and spdx not in bundled:
            bundled.append(spdx)

    expression = _compose_spdx_expression(primary_license, bundled)

    copyright_notice = primary_stanza.get("Copyright__full") or primary_stanza.get("Copyright")
    if copyright_notice:
        copyright_notice = " ".join(
            line.strip() for line in copyright_notice.splitlines() if line.strip()
        ).strip()

    return {
        "is_dep5": True,
        "primary_license": primary_license,
        "bundled_licenses": sorted(bundled),
        "spdx_expression": expression,
        "copyright_notice": copyright_notice or None,
        "upstream_source": header.get("Source") or header.get("Upstream-Source"),
        "upstream_name": header.get("Upstream-Name"),
    }


def _compose_spdx_expression(primary: Optional[str], bundled: List[str]) -> Optional[str]:
    """把主许可证与内嵌第三方许可证合成一个 SPDX 表达式。

    内嵌许可证是包内确实存在、必须同时遵守的成分，因此用 AND 连接；
    含运算符的子表达式加括号，避免与外层 AND 的优先级混淆。
    """
    if not primary and not bundled:
        return None

    def _wrap(expr: str) -> str:
        return f"({expr})" if re.search(r"\s+(AND|OR)\s+", expr) else expr

    parts = [_wrap(primary)] if primary else []
    parts.extend(_wrap(b) for b in sorted(bundled))
    return " AND ".join(parts) if parts else None


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

async def resolve_deb_package(
    vendor: str, name: str, version: Optional[str]
) -> Optional[Dict[str, Any]]:
    """按猜测的发行版解析包信息，失败时自动尝试另一个发行版。"""
    order = [vendor, VENDOR_DEBIAN if vendor == VENDOR_UBUNTU else VENDOR_UBUNTU]
    for candidate in order:
        substep_logger.info(f"[DEB] 按 {candidate} 解析包: {name} (version={version})")
        if candidate == VENDOR_UBUNTU:
            info = await _resolve_ubuntu_source(name, version)
        else:
            info = await _resolve_debian_source(name, version)
        if info and info.get("source_version"):
            return info
        substep_logger.info(f"[DEB] {candidate} 未找到 {name}，尝试另一个发行版")
    return None


async def process_deb_package(
    url: str,
    version: Optional[str] = None,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """分析一个 Debian / Ubuntu 包的许可证。

    Args:
        url: deb 包页面 URL（由 purl 翻译而来，或用户直接输入）。
        version: 请求的版本，未提供时使用 URL 中的版本或最新版本。
        name: 组件名，仅用于结果展示。

    Returns:
        Dict[str, Any]: 与其他生态 handler 一致的结果字典。
    """
    try:
        vendor, pkg_name, url_version = parse_deb_url(url)
    except ValueError:
        return _error_result(url, version, name, "无法解析的 deb 包地址")

    requested_version = version or url_version
    logger.info(f"[DEB] 开始分析 {vendor} 包 {pkg_name} (version={requested_version})")

    info = await resolve_deb_package(vendor, pkg_name, requested_version)
    if not info:
        return _no_license_result(
            url, requested_version, name or pkg_name,
            f"未能在 Debian / Ubuntu 归档中找到包 {pkg_name}",
        )

    copyright_text, copyright_url = await _fetch_copyright(info)
    if not copyright_text:
        return _no_license_result(
            url, requested_version, name or info["source_name"],
            f"已定位到 {info['source_name']} {info['source_version']}，但未取到 debian/copyright",
            info=info,
        )

    parsed = parse_dep5_copyright(copyright_text)
    license_expression = parsed.get("spdx_expression")
    license_source = "dep5"
    license_analysis: Dict[str, Any] = {
        "license_source": license_source,
        "primary_license": parsed.get("primary_license"),
        "bundled_licenses": parsed.get("bundled_licenses") or [],
        "spdx_expression": license_expression,
    }

    if not license_expression:
        # 老包的 copyright 是自由文本，没有可解析的结构，此时才动用 LLM
        reason = "非 DEP-5 格式" if not parsed.get("is_dep5") else "DEP-5 中未解析出许可证"
        substep_logger.info(f"[DEB] {reason}，改用 LLM 分析 copyright 原文: {copyright_url}")
        llm_analysis = await analyze_license_content_async(copyright_text, copyright_url)
        license_source = "llm"
        if llm_analysis:
            license_expression = llm_analysis.get("spdx_expression")
            license_analysis = {**llm_analysis, "license_source": license_source}

    copyright_notice = parsed.get("copyright_notice")
    upstream_source = parsed.get("upstream_source")
    repo_url = upstream_source if _looks_like_url(upstream_source) else None

    determination = (
        f"Debian/Ubuntu copyright ({license_source}): {copyright_url}"
        if license_expression
        else f"未能从 copyright 文件判定许可证: {copyright_url}"
    )
    license_analysis["license_determination_reason"] = determination

    return {
        "input_url": url,
        "repo_url": repo_url,
        "input_version": requested_version,
        "resolved_version": info.get("source_version"),
        "used_default_branch": not info.get("matched_requested", False),
        "component_name": name or info.get("binary_name") or info["source_name"],
        "license_files": copyright_url,
        "license_analysis": license_analysis,
        "license_type": license_expression,
        "has_license_conflict": False,
        "readme_license": None,
        "license_file_license": license_expression,
        "copyright_notice": copyright_notice,
        "license_text": prepare_license_text(copyright_text),
        "status": "success" if license_expression else "no_license_found",
        "license_determination_reason": determination,
    }


def _looks_like_url(value: Optional[str]) -> bool:
    if not value or not isinstance(value, str):
        return False
    return value.strip().lower().startswith(("http://", "https://"))


def _base_result(url: str, version: Optional[str], name: Optional[str]) -> Dict[str, Any]:
    return {
        "input_url": url,
        "repo_url": None,
        "input_version": version,
        "resolved_version": None,
        "used_default_branch": False,
        "component_name": name,
        "license_files": "",
        "license_analysis": None,
        "license_type": None,
        "has_license_conflict": False,
        "readme_license": None,
        "license_file_license": None,
        "copyright_notice": None,
        "license_text": None,
    }


def _no_license_result(
    url: str,
    version: Optional[str],
    name: Optional[str],
    reason: str,
    info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    logger.warning(f"[DEB] {reason}")
    result = _base_result(url, version, name)
    if info:
        result["resolved_version"] = info.get("source_version")
        result["used_default_branch"] = not info.get("matched_requested", False)
    result["status"] = "no_license_found"
    result["license_determination_reason"] = reason
    return result


def _error_result(url: str, version: Optional[str], name: Optional[str], reason: str) -> Dict[str, Any]:
    logger.warning(f"[DEB] {reason}: {url}")
    result = _base_result(url, version, name)
    result["status"] = "error"
    result["error"] = reason
    result["license_determination_reason"] = reason
    return result


__all__ = [
    "is_deb_package_url",
    "parse_deb_url",
    "process_deb_package",
    "resolve_deb_package",
    "parse_dep5_copyright",
    "dep5_license_to_spdx",
    "build_copyright_urls",
    "strip_epoch",
    "pool_prefix",
]
