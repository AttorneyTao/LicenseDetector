import os
import re
import json
import logging
import requests
from urllib.parse import urlparse
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from .config import LLM_CONFIG
from .llm_provider import get_llm_provider
from .utils import (
    analyze_license_content,
    extract_copyright_info,
    extract_copyright_info_async,
    analyze_license_content_async,
    find_top_level_thirdparty_dirs_local,
    construct_copyright_notice_async,
)
from bs4 import BeautifulSoup
import tempfile
import shutil
import tarfile
import aiofiles
import aiohttp
import yaml

load_dotenv()

# ---------------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------------
# We expect an application-level npm_logger singleton named `npm` to exist.
# If it is absent, create a minimal fallback logger with the same name so that
# logging calls never fail.
# ---------------------------------------------------------------------------
try:
    npm_logger = npm  # type: ignore  # noqa: F821
except NameError:  # pragma: no cover – fallback for standalone usage
    npm_logger = logging.getLogger("npm")
    llm_logger = logging.getLogger("llm_interaction")
    version_resolve_logger = logging.getLogger("version_resolve_interaction")
else:
    llm_logger = logging.getLogger("llm_interaction")
    version_resolve_logger = logging.getLogger("version_resolve_interaction")

# ---------------------------------------------------------------------------
# Constants & Exceptions
# ---------------------------------------------------------------------------

NPM_REGISTRY_BASE = "https://registry.npmjs.org"
NPMMIRROR_REGISTRY_BASE = "https://registry.npmmirror.com"
TENCENT_MIRROR_BASE = "https://mirrors.tencent.com/npm"

USE_LLM = os.getenv("USE_LLM", "true").lower() == "true"

try:
    with open("prompts.yaml", "r", encoding="utf-8") as f:
        PROMPTS = yaml.safe_load(f)
except Exception:
    PROMPTS = {}


class NpmAPIError(Exception):
    """Raised when the npm registry returns an unexpected response."""


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _parse_package_name(url_or_name: str) -> str:
    """Extract npm package name (supports @scope) from full URL or raw name."""
    npm_logger.info("Parsing package name from input: %s", url_or_name)
    if not url_or_name.startswith("http"):
        return url_or_name

    parsed = urlparse(url_or_name)
    path = parsed.path.strip("/")

    # 1. 处理 tgz 或其他格式的直接下载链接
    if "/-/" in path:
        # 例如: aegis-web-sdk/-/aegis-web-sdk-1.39.3.tgz
        # 或者: @types/node/-/node-14.14.31.tgz
        parts = path.split("/-/", 1)[0]  # 取 /-/ 之前的部分
        if parts.startswith("npm/"):  # 处理腾讯镜像的路径前缀
            parts = parts[4:]
        return parts

    # 2. 处理包主页格式
    path = re.sub(r"^(?:npm/|package/)", "", path)
    path = re.sub(r"/v/[^/]+$", "", path)  # 移除版本号路径
    path = path.rstrip("/")

    # 3. 处理作用域包
    parts = path.split("/")
    if parts and parts[0].startswith("@"):
        # 确保作用域包包含两部分
        return "/".join(parts[:2]) if len(parts) > 1 else parts[0]

    # 4. 返回第一段作为包名
    return parts[0] if parts else ""


def is_npm_package_url(url_or_name: str) -> bool:
    """Return True when input points to a supported npm package source."""
    if not isinstance(url_or_name, str):
        return False

    value = url_or_name.strip()
    if not value:
        return False

    if value.startswith("@") and "/" in value:
        return True

    parsed = urlparse(value)
    host = parsed.netloc.lower()
    path = parsed.path.strip("/")

    if host in {"www.npmjs.com", "npmjs.com"} and path.startswith("package/"):
        return True
    if host in {"www.npmjs.org", "npmjs.org"} and path:
        return True
    if host in {"registry.npmjs.org", "registry.npmmirror.com"} and path:
        return True
    if host == "www.npmmirror.com" and path.startswith("package/"):
        return True
    if host == "mirrors.tencent.com" and path.startswith("npm/"):
        return True

    return False


def _normalize_requested_npm_version(version: Optional[str]) -> Optional[str]:
    """
    Normalize incoming npm version string without using LLM.

    Examples:
    - None -> None
    - "" -> None
    - "latest" -> "latest"
    - "v1.2.3" -> "1.2.3"
    - "2.1.8_|_dev" -> "2.1.8"
    """
    if version is None:
        return None

    version_str = str(version).strip()
    if not version_str:
        return None

    if version_str.lower() == "latest":
        return "latest"

    # 提取最前面的 semver 片段，兼容 "2.1.8_|_dev"、"v1.2.3-beta" 等
    semver_prefix_match = re.match(
        r"^[vV]?(\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?)",
        version_str
    )
    if semver_prefix_match:
        return semver_prefix_match.group(1)

    # 普通 v 前缀兼容
    if re.match(r"^[vV]\d", version_str):
        return version_str[1:]

    return version_str


def _fetch_packument(pkg_name: str, version: Optional[str] = None) -> dict:
    """从各个 npm registry 获取包信息。"""
    urls = [
        f"{NPM_REGISTRY_BASE}/{pkg_name}",
        f"{NPMMIRROR_REGISTRY_BASE}/{pkg_name}",
        f"{TENCENT_MIRROR_BASE}/{pkg_name}",
    ]
    if version:
        urls = [
            f"{NPM_REGISTRY_BASE}/{pkg_name}/{version}",
            f"{NPMMIRROR_REGISTRY_BASE}/{pkg_name}/{version}",
            f"{TENCENT_MIRROR_BASE}/{pkg_name}/{version}",
        ]

    last_error = None
    all_errors = []  # 收集所有错误信息

    for url in urls:
        try:
            npm_logger.debug("Trying to fetch packument from: %s", url)
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            status_code = getattr(e.response, "status_code", "unknown")
            reason = getattr(e.response, "reason", str(e))
            error_msg = f"HTTP Error for {url}: {status_code} - {reason}"
            npm_logger.error(error_msg)
            all_errors.append(error_msg)
            last_error = e
        except requests.exceptions.ConnectionError as e:
            error_msg = f"Connection Error for {url}: {str(e)}"
            npm_logger.error(error_msg)
            all_errors.append(error_msg)
            last_error = e
        except requests.exceptions.Timeout as e:
            error_msg = f"Timeout Error for {url}: {str(e)}"
            npm_logger.error(error_msg)
            all_errors.append(error_msg)
            last_error = e
        except requests.exceptions.RequestException as e:
            error_msg = f"Request Error for {url}: {str(e)}"
            npm_logger.error(error_msg)
            all_errors.append(error_msg)
            last_error = e
        except json.JSONDecodeError as e:
            error_msg = f"JSON Decode Error for {url}: {str(e)}"
            npm_logger.error(error_msg)
            all_errors.append(error_msg)
            last_error = e
        except Exception as e:
            error_msg = f"Unexpected Error for {url}: {str(e)}"
            npm_logger.error(error_msg, exc_info=True)
            all_errors.append(error_msg)
            last_error = e

    # 所有尝试都失败了,记录详细错误信息
    error_summary = "\n".join(all_errors)
    npm_logger.error(
        "All attempts to fetch packument failed for %s@%s:\n%s",
        pkg_name,
        version,
        error_summary,
    )
    raise NpmAPIError(f"Failed to fetch packument from all registries: {str(last_error)}")


def _paginate_versions(first_page: Dict[str, Any]):
    """Yield version lists; placeholder for future pagination support."""
    versions = list(first_page.get("versions", {}).keys())
    if versions:
        yield versions
    next_url = first_page.get("next")
    while next_url:
        npm_logger.info("Following pagination to %s", next_url)
        resp = requests.get(next_url, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        yield list(data.get("versions", {}).keys())
        next_url = data.get("next")


def _list_all_versions(packument: Dict[str, Any]) -> List[str]:
    """Return all available versions sorted newest→oldest using semver when possible."""
    versions: List[str] = []
    for page in _paginate_versions(packument):
        versions.extend(page)

    # 去重，防止异常重复
    versions = list(dict.fromkeys(versions))

    try:
        from packaging.version import Version
        versions.sort(key=lambda v: Version(v), reverse=True)
    except Exception:  # pragma: no cover – packaging missing or invalid semver
        versions.sort(reverse=True)
    npm_logger.info("Total versions discovered: %d", len(versions))
    return versions


# ---------------------------------------------------------------------------
# LLM version resolution (npm)
# ---------------------------------------------------------------------------

def _build_npm_version_resolve_prompt(
    candidate_versions: List[str],
    version: str,
    default_version: str,
) -> str:
    if PROMPTS and "version_resolve" in PROMPTS:
        return PROMPTS["version_resolve"].format(
            candidate_versions=candidate_versions,
            version=version,
            default_branch=default_version,
        )

    return f"""
You are an npm package version resolver.
Here is the list of all available published versions (choose only from these):
{candidate_versions}

The user requested version string: {version}

Please determine the most appropriate published version the user probably means.
Only return one value, and it must be strictly from the above list.
Do not return explanations or anything else.
If you cannot determine or there is no suitable match, return "{default_version}".

Return in the following JSON format:
{{
    "resolved_version": "xxx",
    "used_default_branch": true
}}
""".strip()


def _llm_choose_npm_version(
    pkg_name: str,
    candidate_versions: List[str],
    version: str,
    default_version: str,
) -> Tuple[str, bool]:
    """
    Use LLM to choose the most likely npm version from candidate_versions.
    Returns (resolved_version, used_default_branch).
    """
    version_resolve_logger.info(
        "NPM version LLM fallback for %s, requested version: %s",
        pkg_name,
        version,
    )

    if not USE_LLM:
        version_resolve_logger.info(
            "USE_LLM is disabled, fallback to default version: %s",
            default_version,
        )
        return default_version, True

    try:
        prompt = _build_npm_version_resolve_prompt(
            candidate_versions=candidate_versions,
            version=version,
            default_version=default_version,
        )

        llm_logger.info("NPM Version Resolve Request:")
        llm_logger.info("Prompt: %s", prompt)
        version_resolve_logger.info("NPM Version Resolve LLM Request:")

        provider = get_llm_provider()
        response = provider.generate(prompt)

        llm_logger.info("NPM Version Resolve Response:")
        llm_logger.info("Response: %s", response)
        version_resolve_logger.info("NPM Version Resolve LLM Response:")
        version_resolve_logger.info("Response: %s", response)

        if response:
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                resolved_version = result.get("resolved_version", default_version)
                used_default_branch = result.get(
                    "used_default_branch",
                    resolved_version == default_version,
                )

                # 安全校验：LLM 返回值必须在候选列表中
                if resolved_version not in candidate_versions:
                    version_resolve_logger.warning(
                        "LLM returned version not in candidate list: %s, fallback to default version: %s",
                        resolved_version,
                        default_version,
                    )
                    return default_version, True

                version_resolve_logger.info(
                    "LLM resolved npm version: %s, used_default_branch: %s",
                    resolved_version,
                    used_default_branch,
                )
                return resolved_version, used_default_branch
            else:
                version_resolve_logger.warning(
                    "No JSON found in npm version resolve response"
                )
    except Exception as e:
        if "This event loop is already running" in str(e):
            version_resolve_logger.warning(
                "Event loop conflict detected during npm version resolution - falling back to default version"
            )
        else:
            version_resolve_logger.error(
                "Failed to resolve npm version via LLM: %s",
                str(e),
                exc_info=True,
            )

    version_resolve_logger.info(
        "LLM fallback failed, using default version: %s",
        default_version,
    )
    return default_version, True


async def resolve_npm_version(
    pkg_name: str,
    versions: List[str],
    version: Optional[str],
    default_version: str,
) -> Tuple[str, bool]:
    """
    First try text matching, fallback to LLM if no match.
    Supports:
    - exact match
    - leading 'v' prefix
    - case-insensitive matching
    - range-like forms such as 0.x / 1.2.x
    - partial match
    """
    version_resolve_logger.info(
        "Resolving npm version for %s, requested version: %s",
        pkg_name,
        version,
    )

    if not versions:
        version_resolve_logger.warning(
            "No candidate versions for %s, using default version: %s",
            pkg_name,
            default_version,
        )
        return default_version, True

    if not version:
        version_resolve_logger.info("No version specified, using default version")
        return default_version, True

    version_str = str(version).strip()
    version_str_lower = version_str.lower().lstrip("v")

    # 1. Exact match ignoring "v" prefix and case
    for candidate in versions:
        cand_lower = candidate.lower().lstrip("v")
        if version_str_lower == cand_lower:
            version_resolve_logger.info(
                "Found exact npm version match (ignore v/case): %s",
                candidate,
            )
            return candidate, False

    # 2. Range match like "0.x" / "1.2.x"
    if version_str_lower.endswith(".x"):
        base = version_str_lower[:-2]
        for candidate in versions:
            cand_lower = candidate.lower().lstrip("v")
            if cand_lower.startswith(base + "."):
                version_resolve_logger.info(
                    "Found npm version range match: %s for %s",
                    candidate,
                    version_str,
                )
                return candidate, False

    # 3. Partial match (e.g. "1.2" matches "1.2.3")
    for candidate in versions:
        cand_lower = candidate.lower().lstrip("v")
        if version_str_lower in cand_lower:
            version_resolve_logger.info(
                "Found partial npm version match: %s",
                candidate,
            )
            return candidate, False

    # 4. LLM fallback
    return _llm_choose_npm_version(
        pkg_name=pkg_name,
        candidate_versions=versions,
        version=version_str,
        default_version=default_version,
    )


def fetch_npm_readme_simple(pkg_name: str, version: str) -> Optional[str]:
    url = f"https://www.npmjs.com/package/{pkg_name}/v/{version}?activeTab=readme"
    try:
        resp = requests.get(url, timeout=10)
        # 检查响应状态码
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            # 直接提取全部文本
            text = soup.get_text(separator="\n")
            # 可选：去掉前后空行
            return text.strip()
        else:
            npm_logger.warning(
                "Failed to fetch README from %s, status code: %s",
                url,
                resp.status_code,
            )
            return None
    except requests.exceptions.RequestException as e:
        npm_logger.warning("Error fetching README from %s: %s", url, str(e))
        return None
    except Exception as e:
        npm_logger.warning(
            "Unexpected error fetching README from %s: %s",
            url,
            str(e),
        )
        return None


# ---------------------------------------------------------------------------
# Main processor (public API)
# ---------------------------------------------------------------------------

async def process_npm_repository(url: str, version: Optional[str] = None) -> Dict[str, Any]:
    logger = logging.getLogger("main")

    logger.info("Starting processing for %s (requested version=%s)", url, version)

    pkg_name = _parse_package_name(url)
    logger.debug("Parsed package name: %s", pkg_name)

    normalized_version = _normalize_requested_npm_version(version)
    logger.info(
        "Normalized requested npm version: raw=%s, normalized=%s",
        version,
        normalized_version,
    )

    # 先拉取整包 packument，再在本地解析版本，避免非法 version 直接打挂 registry 请求
    try:
        packument = _fetch_packument(pkg_name)
        logger.debug("Fetched full packument keys: %s", list(packument.keys()))
    except Exception as e:
        logger.warning(
            "Error fetching full packument for %s: %s",
            pkg_name,
            e,
            exc_info=True,
        )
        return {"status": "error"}

    # 1. 判断是 packument 还是单版本对象
    if "versions" in packument:
        # 多版本 packument
        default_version = packument.get("dist-tags", {}).get("latest")
        versions = _list_all_versions(packument)
        logger.debug("Available versions: %s", versions)

        resolved_version, used_default_branch = await resolve_npm_version(
            pkg_name=pkg_name,
            versions=versions,
            version=normalized_version,
            default_version=default_version,
        )
        logger.info("Resolved version for %s: %s", pkg_name, resolved_version)

        version_obj = packument.get("versions", {}).get(resolved_version)
        if not version_obj:
            logger.warning(
                "Cannot find version object for %s, fallback to latest version: %s",
                resolved_version,
                default_version,
            )
            resolved_version = default_version
            used_default_branch = True
            version_obj = packument.get("versions", {}).get(resolved_version)
            if not version_obj:
                logger.error(
                    "Cannot find version object for latest version %s",
                    resolved_version,
                )
                return {"status": "error", "error": f"Version {resolved_version} not found"}
    else:
        # 单版本对象
        version_obj = packument
        resolved_version = version_obj.get("version")
        used_default_branch = normalized_version in (None, "", "latest")
        logger.info("Single version object detected, version: %s", resolved_version)

    logger.debug("Version object keys: %s", list(version_obj.keys()))

    # repository 字段兼容
    repo_field = version_obj.get("repository")
    repo_url = None
    if isinstance(repo_field, dict):
        repo_url = repo_field.get("url")
    elif isinstance(repo_field, str):
        repo_url = repo_field
    if repo_url and repo_url.startswith("git+"):
        repo_url = repo_url[4:]
    if repo_url and repo_url.endswith(".git"):
        repo_url = repo_url[:-4]
    if repo_url and repo_url.startswith("git@github.com:"):
        repo_url = repo_url.replace("git@github.com:", "https://github.com/")
    logger.debug("Parsed repo_url: %s", repo_url)

    license_type = version_obj.get("license")
    logger.debug("License type: %s", license_type)

    readme_content = version_obj.get("readme")
    if readme_content:
        logger.info(
            "Readme content found for %s@%s, length=%d",
            pkg_name,
            resolved_version,
            len(readme_content),
        )
    else:
        logger.info(
            "No readme found for %s@%s, trying to fetch from npm registry page...",
            pkg_name,
            resolved_version,
        )
        readme_content = fetch_npm_readme_simple(pkg_name, resolved_version)

    license_analysis = None
    readme_license = None
    if not license_type:
        logger.info("No license info in npm metadata, analyzing readme for license...")
        # 构建 npm 包的 URL 作为 source_url
        npm_source_url = f"https://www.npmjs.com/package/{pkg_name}/v/{resolved_version}"
        license_analysis = await analyze_license_content_async(readme_content or "", npm_source_url)
        if license_analysis and license_analysis.get("licenses"):
            license_type = license_analysis["licenses"][0]
            readme_license = license_analysis["licenses"][0]
            logger.info("Extracted license from readme: %s", readme_license)
        else:
            license_type = None
            readme_license = None

    # npm packument 的 time 字典在根层级（以版本号为键），版本子对象本身不含 time 字段
    if "versions" in packument:
        last_modified_iso = packument.get("time", {}).get(resolved_version)
    else:
        last_modified_iso = None  # 单版本端点不包含发布时间
    logger.debug("Last modified ISO: %s", last_modified_iso)
    if last_modified_iso:
        try:
            dt = datetime.fromisoformat(last_modified_iso.replace("Z", "+00:00"))
        except Exception:
            logger.warning(
                "Failed to parse last_modified_iso: %s, using now()",
                last_modified_iso,
            )
            dt = datetime.now(timezone.utc)
    else:
        dt = datetime.now(timezone.utc)

    author_obj = version_obj.get("author") or {}
    if isinstance(author_obj, dict):
        author = author_obj.get("name") or ""
    elif isinstance(author_obj, str):
        author = author_obj
    else:
        author = ""

    license_files = f"https://www.npmjs.com/package/{pkg_name}/v/{resolved_version}?activeTab=code"
    logger.debug("License files URL: %s", license_files)
    final_license_file = license_files
    final_used_default_branch = used_default_branch

    # 如果 repo_url 为 github 地址，调用 process_github_repository 补充元数据
    github_fields = {
        "license_files": license_files,
        "license_analysis": None,
        "has_license_conflict": None,
        "readme_license": None,
        "license_file_license": None,
        "used_default_branch": None,
    }
    github_copyright_notice = None
    github_scan_success = False

    if repo_url and "github.com" in repo_url:
        logger.info(
            "repo_url is a GitHub URL, calling process_github_repository: %s %s",
            repo_url,
            resolved_version,
        )
        try:
            from core.github_utils import process_github_repository, GitHubAPI

            parsed = urlparse(repo_url)
            path_parts = parsed.path.strip("/").split("/")
            if len(path_parts) >= 2:
                github_url = f"https://github.com/{path_parts[0]}/{path_parts[1]}"
                api = GitHubAPI()
                github_result = await process_github_repository(
                    api,
                    github_url,
                    resolved_version,
                )
                github_scan_success = github_result.get("status") == "success"
                if not github_scan_success:
                    logger.warning(
                        "GitHub scan did not succeed for %s: %s",
                        github_url,
                        github_result.get("license_determination_reason") or github_result.get("error"),
                    )
                    github_result = {}

                gh_used_default_branch = github_result.get("used_default_branch")
                github_fields["used_default_branch"] = gh_used_default_branch
                if gh_used_default_branch is not None:
                    final_used_default_branch = gh_used_default_branch

                # license_files 字段替换逻辑
                if gh_used_default_branch is False and github_result.get("license_files"):
                    final_license_file = github_result.get("license_files", license_files)
                    logger.info(
                        "Replaced license_files with GitHub result because used_default_branch is False"
                    )
                else:
                    final_license_file = license_files
                    logger.info(
                        "Kept original license_files because used_default_branch is True or missing"
                    )

                # 其他字段直接赋值
                for key in [
                    "license_analysis",
                    "has_license_conflict",
                    "readme_license",
                    "license_file_license",
                ]:
                    if github_result.get(key) is not None:
                        github_fields[key] = github_result.get(key)

                # copyright_notice 特殊逻辑
                github_copyright_notice = github_result.get("copyright_notice")
                logger.info("GitHub copyright_notice: %s", github_copyright_notice)
            else:
                logger.warning("repo_url path not valid for github: %s", repo_url)
        except Exception as e:
            github_scan_success = False
            logger.error("Failed to call process_github_repository: %s", e, exc_info=True)

    # 没有 github 仓库，或 github 扫描失败时，分析 npm 包 tarball 中的 thirdparty 目录和 License 文件
    thirdparty_dirs: List[str] = []
    npm_license_content: Optional[str] = None
    should_fallback_to_npm_tarball = (not repo_url or "github.com" not in repo_url) or (
        repo_url and "github.com" in repo_url and not github_scan_success
    )

    if should_fallback_to_npm_tarball:
        tarball_url = version_obj.get("dist", {}).get("tarball")
        if tarball_url:
            try:
                thirdparty_dirs, npm_license_content = await async_analyze_npm_tarball(tarball_url)
                logger.info("Found thirdparty dirs in npm tarball: %s", thirdparty_dirs)
                logger.info(
                    "Extracted npm license content (length=%d)",
                    len(npm_license_content) if npm_license_content else 0,
                )

                if license_analysis is None:
                    license_analysis = {}

                if isinstance(license_analysis, dict):
                    existing_dirs = license_analysis.get("thirdparty_dirs")
                    if not existing_dirs:
                        license_analysis["thirdparty_dirs"] = thirdparty_dirs
                else:
                    license_analysis = {"thirdparty_dirs": thirdparty_dirs}
            except Exception as e:
                logger.warning("Failed to analyze npm tarball: %s", e)
        else:
            logger.info("No tarball URL found for npm fallback analysis")
    else:
        logger.info("GitHub scan succeeded; npm tarball fallback not needed")

    # 处理 copyright_notice 逻辑
    if github_scan_success and github_copyright_notice and "original author and authors" not in github_copyright_notice:
        final_copyright_notice = github_copyright_notice
        logger.info("Using GitHub copyright_notice: %s", final_copyright_notice)
    else:
        # 使用 LLM 从 README 和 npm tarball 中的 License 文件提取版权声明
        final_copyright_notice = await construct_copyright_notice_async(
            year=str(dt.year),
            owner="",
            repo=pkg_name,
            ref=resolved_version,
            component_name=pkg_name,
            readme_content=readme_content,
            license_content=npm_license_content,
        )
        logger.debug("Copyright notice computed from npm data: %s", final_copyright_notice)

    final_license_analysis = (
        github_fields["license_analysis"]
        if github_scan_success and github_fields["license_analysis"] is not None
        else license_analysis
    )
    final_has_license_conflict = (
        github_fields["has_license_conflict"] if github_scan_success else None
    )
    final_readme_license = (
        github_fields["readme_license"]
        if github_scan_success and github_fields["readme_license"] is not None
        else readme_license
    )
    final_license_file_license = (
        github_fields["license_file_license"] if github_scan_success else None
    )

    result = {
        "input_url": url,
        "repo_url": repo_url,
        "input_version": version,
        "normalized_input_version": normalized_version,
        "resolved_version": resolved_version,
        "used_default_branch": final_used_default_branch,
        "component_name": pkg_name,
        "license_files": final_license_file,
        "license_analysis": final_license_analysis,
        "license_type": license_type,
        "has_license_conflict": final_has_license_conflict,
        "readme_license": final_readme_license,
        "license_file_license": final_license_file_license,
        "copyright_notice": final_copyright_notice,
        "status": "success",
        "license_determination_reason": (
            "Fetched from GitHub repository"
            if github_scan_success
            else "Fetched from npm registry"
        ),
        "readme": readme_content[:5000] if readme_content else None,
    }

    logger.info("Processing completed for %s@%s", pkg_name, resolved_version)
    logger.debug("Result dict: %s", json.dumps(result, ensure_ascii=False, indent=2))
    return result


# ---------------------------------------------------------------------------
# This module exposes only process_npm_repository for programmatic use.
# ---------------------------------------------------------------------------

async def async_download_and_extract_npm_tarball(tarball_url: str) -> str:
    """
    异步下载 npm 包 tarball 并解压到临时目录，返回解压后的根目录路径。
    """
    tmp_dir = tempfile.mkdtemp()
    tarball_path = os.path.join(tmp_dir, "package.tgz")

    # 异步下载
    async with aiohttp.ClientSession() as session:
        async with session.get(tarball_url) as resp:
            resp.raise_for_status()
            async with aiofiles.open(tarball_path, "wb") as f:
                async for chunk in resp.content.iter_chunked(8192):
                    await f.write(chunk)

    # 解包（解包用同步方式，tarfile 不支持异步）
    with tarfile.open(tarball_path, "r:gz") as tar:
        tar.extractall(path=tmp_dir)

    return tmp_dir


async def async_analyze_npm_tarball(tarball_url: str) -> Tuple[List[str], Optional[str]]:
    """
    异步下载并分析 npm tarball，返回 (thirdparty_dirs, license_content)。
    license_content 为找到的第一个 License 文件的文本内容，未找到时为 None。
    """
    tmp_dir = None
    try:
        tmp_dir = await async_download_and_extract_npm_tarball(tarball_url)
        package_root = os.path.join(tmp_dir, "package")
        if not os.path.isdir(package_root):
            package_root = tmp_dir

        thirdparty_dirs = find_top_level_thirdparty_dirs_local(package_root)

        license_content: Optional[str] = None
        license_keywords = ("license", "licence", "copying")
        try:
            for filename in os.listdir(package_root):
                if any(kw in filename.lower() for kw in license_keywords):
                    filepath = os.path.join(package_root, filename)
                    if os.path.isfile(filepath):
                        async with aiofiles.open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                            license_content = await f.read()
                        if license_content:
                            npm_logger.debug("Found license file in tarball: %s", filename)
                            break
        except Exception as e:
            npm_logger.warning("Failed to extract license content from tarball: %s", e)

        return thirdparty_dirs, license_content
    finally:
        if tmp_dir and os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)


async def async_analyze_npm_tarball_thirdparty_dirs(tarball_url: str) -> List[str]:
    """向后兼容包装，只返回 thirdparty_dirs。"""
    thirdparty_dirs, _ = await async_analyze_npm_tarball(tarball_url)
    return thirdparty_dirs
