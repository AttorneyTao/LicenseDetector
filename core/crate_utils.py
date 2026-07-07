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
    prepare_license_text,
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
try:
    crate_logger = crate  # type: ignore  # noqa: F821
except NameError:  # pragma: no cover – fallback for standalone usage
    crate_logger = logging.getLogger("crate")
    llm_logger = logging.getLogger("llm_interaction")
    version_resolve_logger = logging.getLogger("version_resolve_interaction")
else:
    llm_logger = logging.getLogger("llm_interaction")
    version_resolve_logger = logging.getLogger("version_resolve_interaction")

# ---------------------------------------------------------------------------
# Constants & Exceptions
# ---------------------------------------------------------------------------

CRATES_IO_API_BASE = "https://crates.io/api/v1"
CRATES_IO_BASE = "https://crates.io"

# crates.io 的爬虫政策要求 User-Agent 能标识调用方，默认的
# python-requests UA 会被直接 403 / 掐断连接
# （可用环境变量 CRATES_IO_USER_AGENT 覆盖，建议带上联系方式）
CRATES_IO_HEADERS = {
    "User-Agent": os.getenv(
        "CRATES_IO_USER_AGENT",
        "Github-Repo-Analyser/1.0 (license compliance scanner; +https://github.com/AttorneyTao/LicenseDetector)",
    )
}

USE_LLM = os.getenv("USE_LLM", "true").lower() == "true"

try:
    with open("prompts.yaml", "r", encoding="utf-8") as f:
        PROMPTS = yaml.safe_load(f)
except Exception:
    PROMPTS = {}


class CrateAPIError(Exception):
    """Raised when the crates.io API returns an unexpected response."""


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _parse_crate_name(url_or_name: str) -> str:
    """Extract crate name from full URL or raw name."""
    crate_logger.info("Parsing crate name from input: %s", url_or_name)
    if not url_or_name.startswith("http"):
        return url_or_name

    parsed = urlparse(url_or_name)
    path = parsed.path.strip("/")

    # 处理 crates.io URL 格式
    # https://crates.io/crates/serde
    # https://crates.io/crates/serde/1.0.0
    if path.startswith("crates/"):
        parts = path.split("/")
        if len(parts) >= 2:
            return parts[1]  # crate name is the second part
    
    # 如果无法解析，返回原路径
    return path if path else ""


def _normalize_requested_crate_version(version: Optional[str]) -> Optional[str]:
    """
    Normalize incoming crate version string.

    Examples:
    - None -> None
    - "" -> None
    - "1.0.0" -> "1.0.0"
    - "v1.0.0" -> "1.0.0"
    """
    if version is None:
        return None

    version_str = str(version).strip()
    if not version_str:
        return None

    # 移除 'v' 或 'V' 前缀
    if re.match(r"^[vV]\d", version_str):
        return version_str[1:]

    return version_str


def _fetch_crate_info(crate_name: str) -> dict:
    """从 crates.io API 获取 crate 信息。"""
    url = f"{CRATES_IO_API_BASE}/crates/{crate_name}"
    
    try:
        crate_logger.debug("Fetching crate info from: %s", url)
        resp = requests.get(url, headers=CRATES_IO_HEADERS, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as e:
        status_code = getattr(e.response, "status_code", "unknown")
        reason = getattr(e.response, "reason", str(e))
        error_msg = f"HTTP Error for {url}: {status_code} - {reason}"
        crate_logger.error(error_msg)
        raise CrateAPIError(error_msg)
    except requests.exceptions.RequestException as e:
        error_msg = f"Request Error for {url}: {str(e)}"
        crate_logger.error(error_msg)
        raise CrateAPIError(error_msg)
    except json.JSONDecodeError as e:
        error_msg = f"JSON Decode Error for {url}: {str(e)}"
        crate_logger.error(error_msg)
        raise CrateAPIError(error_msg)
    except Exception as e:
        error_msg = f"Unexpected Error for {url}: {str(e)}"
        crate_logger.error(error_msg, exc_info=True)
        raise CrateAPIError(error_msg)


def _fetch_version_info(crate_name: str, version: str) -> dict:
    """从 crates.io API 获取特定版本信息。"""
    url = f"{CRATES_IO_API_BASE}/crates/{crate_name}/{version}"
    
    try:
        crate_logger.debug("Fetching version info from: %s", url)
        resp = requests.get(url, headers=CRATES_IO_HEADERS, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as e:
        status_code = getattr(e.response, "status_code", "unknown")
        reason = getattr(e.response, "reason", str(e))
        error_msg = f"HTTP Error for {url}: {status_code} - {reason}"
        crate_logger.error(error_msg)
        raise CrateAPIError(error_msg)
    except requests.exceptions.RequestException as e:
        error_msg = f"Request Error for {url}: {str(e)}"
        crate_logger.error(error_msg)
        raise CrateAPIError(error_msg)
    except json.JSONDecodeError as e:
        error_msg = f"JSON Decode Error for {url}: {str(e)}"
        crate_logger.error(error_msg)
        raise CrateAPIError(error_msg)
    except Exception as e:
        error_msg = f"Unexpected Error for {url}: {str(e)}"
        crate_logger.error(error_msg, exc_info=True)
        raise CrateAPIError(error_msg)


def _list_all_versions(crate_data: Dict[str, Any]) -> List[str]:
    """Return all available versions sorted newest→oldest."""
    versions_data = crate_data.get("versions", [])
    versions = [v.get("num") for v in versions_data if v.get("num")]
    
    # 去重
    versions = list(dict.fromkeys(versions))
    
    try:
        from packaging.version import Version
        versions.sort(key=lambda v: Version(v), reverse=True)
    except Exception:  # pragma: no cover – packaging missing or invalid semver
        versions.sort(reverse=True)
    
    crate_logger.info("Total versions discovered: %d", len(versions))
    return versions


# ---------------------------------------------------------------------------
# LLM version resolution (crate)
# ---------------------------------------------------------------------------

def _build_crate_version_resolve_prompt(
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
You are a Rust crate version resolver.
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


def _llm_choose_crate_version(
    crate_name: str,
    candidate_versions: List[str],
    version: str,
    default_version: str,
) -> Tuple[str, bool]:
    """
    Use LLM to choose the most likely crate version from candidate_versions.
    Returns (resolved_version, used_default_branch).
    """
    version_resolve_logger.info(
        "Crate version LLM fallback for %s, requested version: %s",
        crate_name,
        version,
    )

    if not USE_LLM:
        version_resolve_logger.info(
            "USE_LLM is disabled, fallback to default version: %s",
            default_version,
        )
        return default_version, True

    try:
        prompt = _build_crate_version_resolve_prompt(
            candidate_versions=candidate_versions,
            version=version,
            default_version=default_version,
        )

        llm_logger.info("Crate Version Resolve Request:")
        llm_logger.info("Prompt: %s", prompt)
        version_resolve_logger.info("Crate Version Resolve LLM Request:")

        provider = get_llm_provider()
        response = provider.generate(prompt)

        llm_logger.info("Crate Version Resolve Response:")
        llm_logger.info("Response: %s", response)
        version_resolve_logger.info("Crate Version Resolve LLM Response:")
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
                    "LLM resolved crate version: %s, used_default_branch: %s",
                    resolved_version,
                    used_default_branch,
                )
                return resolved_version, used_default_branch
            else:
                version_resolve_logger.warning(
                    "No JSON found in crate version resolve response"
                )
    except Exception as e:
        if "This event loop is already running" in str(e):
            version_resolve_logger.warning(
                "Event loop conflict detected during crate version resolution - falling back to default version"
            )
        else:
            version_resolve_logger.error(
                "Failed to resolve crate version via LLM: %s",
                str(e),
                exc_info=True,
            )

    version_resolve_logger.info(
        "LLM fallback failed, using default version: %s",
        default_version,
    )
    return default_version, True


async def resolve_crate_version(
    crate_name: str,
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
    - range-like forms such as 1.x / 1.2.x
    - partial match
    """
    version_resolve_logger.info(
        "Resolving crate version for %s, requested version: %s",
        crate_name,
        version,
    )

    if not versions:
        version_resolve_logger.warning(
            "No candidate versions for %s, using default version: %s",
            crate_name,
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
                "Found exact crate version match (ignore v/case): %s",
                candidate,
            )
            return candidate, False

    # 2. Range match like "1.x" / "1.2.x"
    if version_str_lower.endswith(".x"):
        base = version_str_lower[:-2]
        for candidate in versions:
            cand_lower = candidate.lower().lstrip("v")
            if cand_lower.startswith(base + "."):
                version_resolve_logger.info(
                    "Found crate version range match: %s for %s",
                    candidate,
                    version_str,
                )
                return candidate, False

    # 3. Partial match (e.g. "1.2" matches "1.2.3")
    for candidate in versions:
        cand_lower = candidate.lower().lstrip("v")
        if version_str_lower in cand_lower:
            version_resolve_logger.info(
                "Found partial crate version match: %s",
                candidate,
            )
            return candidate, False

    # 4. LLM fallback
    return _llm_choose_crate_version(
        crate_name=crate_name,
        candidate_versions=versions,
        version=version_str,
        default_version=default_version,
    )


# ---------------------------------------------------------------------------
# Main processor (public API)
# ---------------------------------------------------------------------------

async def process_crate_repository(url: str, version: Optional[str] = None) -> Dict[str, Any]:
    """
    Process a crates.io repository URL or crate name.
    
    Args:
        url: crates.io URL or crate name
        version: Optional version string
        
    Returns:
        Dictionary containing analysis results
    """
    logger = logging.getLogger("main")

    logger.info("Starting processing for %s (requested version=%s)", url, version)

    crate_name = _parse_crate_name(url)
    logger.debug("Parsed crate name: %s", crate_name)

    normalized_version = _normalize_requested_crate_version(version)
    logger.info(
        "Normalized requested crate version: raw=%s, normalized=%s",
        version,
        normalized_version,
    )

    # 获取 crate 信息
    try:
        crate_data = _fetch_crate_info(crate_name)
        logger.debug("Fetched crate data keys: %s", list(crate_data.keys()))
    except Exception as e:
        logger.warning(
            "Error fetching crate info for %s: %s",
            crate_name,
            e,
            exc_info=True,
        )
        return {"status": "error"}

    # 提取基本信息
    crate_info = crate_data.get("crate", {})
    versions_data = crate_data.get("versions", [])
    
    # 默认使用最新稳定版本
    default_version = crate_info.get("max_stable_version") or crate_info.get("max_version")
    all_versions = _list_all_versions(crate_data)
    logger.debug("Available versions: %s", all_versions)

    # 解析版本
    resolved_version, used_default_branch = await resolve_crate_version(
        crate_name=crate_name,
        versions=all_versions,
        version=normalized_version,
        default_version=default_version,
    )
    logger.info("Resolved version for %s: %s", crate_name, resolved_version)

    # 获取具体版本的信息
    try:
        version_data = _fetch_version_info(crate_name, resolved_version)
        version_obj = version_data.get("version", {})
    except Exception as e:
        logger.warning(
            "Error fetching version info for %s@%s: %s",
            crate_name,
            resolved_version,
            e,
        )
        # 降级使用 crate_info 中的信息
        version_obj = {}

    logger.debug("Version object keys: %s", list(version_obj.keys()))

    # 提取仓库 URL（repository 字段）
    repo_url = crate_info.get("repository")
    homepage = crate_info.get("homepage")
    documentation = crate_info.get("documentation")
    
    logger.debug("Repository URL: %s", repo_url)
    logger.debug("Homepage: %s", homepage)
    logger.debug("Documentation: %s", documentation)

    # 许可证信息
    license_type = version_obj.get("license") or crate_info.get("license")
    logger.debug("License type: %s", license_type)

    # 获取 README 内容
    readme_content = None
    if version_obj.get("readme_path"):
        readme_path = version_obj.get("readme_path")
        # 尝试从 GitHub 获取 README
        if repo_url and "github.com" in repo_url:
            readme_content = _fetch_github_readme(repo_url, resolved_version, readme_path)
    
    if not readme_content:
        # 尝试从 crates.io 页面获取
        readme_content = _fetch_crate_readme(crate_name, resolved_version)

    # 如果没有许可证信息，分析 README
    license_analysis = None
    readme_license = None
    if not license_type:
        logger.info("No license info in crate metadata, analyzing readme for license...")
        crate_source_url = f"{CRATES_IO_BASE}/crates/{crate_name}/{resolved_version}"
        license_analysis = await analyze_license_content_async(readme_content or "", crate_source_url)
        if license_analysis and license_analysis.get("licenses"):
            license_type = license_analysis["licenses"][0]
            readme_license = license_analysis["licenses"][0]
            logger.info("Extracted license from readme: %s", readme_license)
        else:
            license_type = None
            readme_license = None

    # 时间信息
    created_at = crate_info.get("created_at")
    updated_at = crate_info.get("updated_at")
    
    dt = datetime.now(timezone.utc)
    if updated_at:
        try:
            dt = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
        except Exception:
            logger.warning("Failed to parse updated_at: %s, using now()", updated_at)
    elif created_at:
        try:
            dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        except Exception:
            logger.warning("Failed to parse created_at: %s, using now()", created_at)

    # 作者信息
    owners = []
    try:
        owners_data = _fetch_crate_owners(crate_name)
        owners = [user.get("login", "") for user in owners_data.get("users", [])]
    except Exception:
        pass
    
    author = ", ".join(owners) if owners else f"{crate_name} original author and authors"
    logger.debug("Author: %s", author)

    # 版权信息
    try:
        copyright_notice = await extract_copyright_info_async(readme_content or "")
    except Exception:
        copyright_notice = extract_copyright_info(readme_content or "")
    
    logger.debug("Copyright notice: %s", copyright_notice)

    if not copyright_notice:
        copyright_notice = f"Copyright(c) {dt.year} {author}".strip()
        logger.debug("Fallback copyright notice: %s", copyright_notice)

    # 构建许可证文件 URL
    license_files = f"{CRATES_IO_BASE}/crates/{crate_name}/{resolved_version}"
    logger.debug("License files URL: %s", license_files)
    final_license_file = license_files
    final_used_default_branch = used_default_branch

    # 如果 repo_url 为 GitHub 地址，调用 process_github_repository 补充元数据
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
                github_scan_success = True

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
                    "license_text",
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

    # 处理 copyright_notice 逻辑
    final_copyright_notice = copyright_notice
    if github_copyright_notice:
        if "original author and authors" in github_copyright_notice:
            logger.info(
                "GitHub copyright_notice contains 'original author and authors', keep crate copyright_notice"
            )
        else:
            final_copyright_notice = github_copyright_notice
            logger.info("Replaced copyright_notice with GitHub result")

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
        "component_name": crate_name,
        "license_files": final_license_file,
        "license_analysis": final_license_analysis,
        "license_type": license_type,
        "has_license_conflict": final_has_license_conflict,
        "readme_license": final_readme_license,
        "license_file_license": final_license_file_license,
        "copyright_notice": final_copyright_notice,
        "license_text": prepare_license_text(
            github_fields.get("license_text") if github_scan_success else None
        ),
        "status": "success",
        "license_determination_reason": (
            "Fetched from GitHub repository"
            if github_scan_success
            else "Fetched from crates.io registry"
        ),
        "readme": readme_content[:5000] if readme_content else None,
        "homepage": homepage,
        "documentation": documentation,
    }

    logger.info("Processing completed for %s@%s", crate_name, resolved_version)
    logger.debug("Result dict: %s", json.dumps(result, ensure_ascii=False, indent=2))
    return result


# ---------------------------------------------------------------------------
# Helper functions for fetching data
# ---------------------------------------------------------------------------

def _fetch_github_readme(repo_url: str, version: str, readme_path: str = "README.md") -> Optional[str]:
    """从 GitHub 获取 README 内容。"""
    try:
        parsed = urlparse(repo_url)
        path_parts = parsed.path.strip("/").split("/")
        if len(path_parts) < 2:
            return None
        
        owner, repo = path_parts[0], path_parts[1]
        
        # 尝试使用 GitHub Contents API
        readme_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{version}/{readme_path}"
        resp = requests.get(readme_url, headers=CRATES_IO_HEADERS, timeout=10)
        
        if resp.status_code == 200:
            crate_logger.info("Successfully fetched README from GitHub")
            return resp.text
        else:
            crate_logger.warning("Failed to fetch README from GitHub, status: %s", resp.status_code)
            return None
    except Exception as e:
        crate_logger.warning("Error fetching README from GitHub: %s", e)
        return None


def _fetch_crate_readme(crate_name: str, version: str) -> Optional[str]:
    """从 crates.io 页面获取 README 内容（如果有）。"""
    try:
        url = f"{CRATES_IO_BASE}/crates/{crate_name}/{version}"
        resp = requests.get(url, headers=CRATES_IO_HEADERS, timeout=10)
        
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            # 尝试提取 README 区域
            readme_section = soup.find("div", class_="readme")
            if readme_section:
                crate_logger.info("Successfully extracted README from crates.io page")
                return readme_section.get_text(separator="\n").strip()
        
        return None
    except Exception as e:
        crate_logger.warning("Error fetching README from crates.io: %s", e)
        return None


def _fetch_crate_owners(crate_name: str) -> dict:
    """获取 crate 的维护者信息。"""
    url = f"{CRATES_IO_API_BASE}/crates/{crate_name}/owners"
    
    try:
        resp = requests.get(url, headers=CRATES_IO_HEADERS, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        crate_logger.debug("Failed to fetch crate owners: %s", e)
        return {"users": []}


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------
__all__ = ["process_crate_repository", "resolve_crate_version", "CrateAPIError"]
