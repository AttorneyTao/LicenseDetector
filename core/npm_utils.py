import os
import re
import json
import logging
import requests
from urllib.parse import urlparse
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from .config import LLM_CONFIG
from .llm_provider import get_llm_provider
from .utils import (
    analyze_license_content,
    extract_copyright_info,
    extract_copyright_info_async,
    analyze_license_content_async,
    find_top_level_thirdparty_dirs_local,
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
# We expect an application-level logger singleton named `npm` to exist.
# If it is absent, create minimal fallback loggers so logging calls never fail.
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

    # 1. Handle direct tarball download links
    if "/-/" in path:
        # e.g. aegis-web-sdk/-/aegis-web-sdk-1.39.3.tgz
        # or   @types/node/-/node-14.14.31.tgz
        parts = path.split("/-/", 1)[0]
        if parts.startswith("npm/"):  # handle Tencent mirror prefix
            parts = parts[4:]
        return parts

    # 2. Handle package page URLs
    path = re.sub(r"^(?:npm/|package/)", "", path)
    path = re.sub(r"/v/[^/]+$", "", path)
    path = path.rstrip("/")

    # 3. Handle scoped packages
    parts = path.split("/")
    if parts and parts[0].startswith("@"):
        return "/".join(parts[:2]) if len(parts) > 1 else parts[0]

    # 4. Return first segment as package name
    return parts[0] if parts else ""


def _fetch_packument(pkg_name: str, version: Optional[str] = None) -> dict:
    """Fetch package metadata from npm registries."""
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
    all_errors = []

    for url in urls:
        try:
            npm_logger.debug("Trying to fetch packument from: %s", url)
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP Error for {url}: {e.response.status_code} - {e.response.reason}"
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
        data = resp.json()
        yield list(data.get("versions", {}).keys())
        next_url = data.get("next")


def _list_all_versions(packument: Dict[str, Any]) -> List[str]:
    """Return all available versions sorted newest→oldest using semver when possible."""
    versions: List[str] = []
    for page in _paginate_versions(packument):
        versions.extend(page)
    try:
        from packaging.version import Version
        versions.sort(key=lambda v: Version(v), reverse=True)
    except Exception:  # pragma: no cover – packaging missing or invalid semver
        versions.sort(reverse=True)
    npm_logger.info("Total versions discovered: %d", len(versions))
    return versions


# ---------------------------------------------------------------------------
# Gemini integration (optional)
# ---------------------------------------------------------------------------

def _gemini_choose_version(user_input: Optional[str], versions: List[str], default: str) -> str:
    """Resolve fuzzily-specified version to a concrete one using LLM if available."""
    USE_LLM = os.getenv("USE_LLM", "true").lower() == "true"

    if not user_input or user_input.lower() in {"", "latest"}:
        npm_logger.info("User version empty or 'latest'; using default %s", default)
        return default

    if user_input in versions:
        npm_logger.info("User version %s found exactly in version list", user_input)
        return user_input

    if not USE_LLM:
        npm_logger.info("USE_LLM=false, fallback to default version %s", default)
        return default

    try:
        with open("prompts.yaml", "r", encoding="utf-8") as f:
            prompts = yaml.safe_load(f)

        prompt = prompts["version_resolve"].format(
            candidate_versions=versions,
            version=user_input,
            default_branch=default,
        )

        llm_logger.info("Version Resolve Request:")
        llm_logger.info("Prompt: %s", prompt)
        version_resolve_logger.info("Version Resolve LLM Request:")

        provider = get_llm_provider()
        response = provider.generate(prompt)

        llm_logger.info("Version Resolve Response:")
        llm_logger.info("Response: %s", response)
        version_resolve_logger.info("Version Resolve LLM Response:")
        version_resolve_logger.info("Response: %s", response)

        if response:
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                resolved_version = result.get("resolved_version", default)
                used_default_branch = result.get(
                    "used_default_branch",
                    resolved_version == default,
                )
                npm_logger.info(
                    "LLM resolved version: %s, used_default_branch: %s",
                    resolved_version,
                    used_default_branch,
                )
                version_resolve_logger.info(
                    "LLM resolved version: %s, used_default_branch: %s",
                    resolved_version,
                    used_default_branch,
                )
                return resolved_version
            else:
                version_resolve_logger.warning(
                    "No JSON found in version resolve response"
                )
    except Exception as e:
        version_resolve_logger.error(
            "Failed to resolve version via LLM: %s",
            str(e),
            exc_info=True,
        )
        npm_logger.error(
            "Failed to resolve version via LLM: %s",
            str(e),
            exc_info=True,
        )

    return default


def fetch_npm_readme_simple(pkg_name: str, version: str) -> Optional[str]:
    url = f"https://www.npmjs.com/package/{pkg_name}/v/{version}?activeTab=readme"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            text = soup.get_text(separator="\n")
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

    try:
        packument = _fetch_packument(pkg_name, version)
        logger.debug("Fetched packument keys: %s", list(packument.keys()))
    except Exception as e:
        logger.warning(
            "Error fetching packument for %s@%s: %s, switching to alternative logic.",
            pkg_name,
            version,
            e,
            exc_info=True,
        )
        return {"status": "error"}

    # 1. Determine whether this is a full packument or a single-version object
    if "versions" in packument:
        default_version = packument.get("dist-tags", {}).get("latest")
        versions = _list_all_versions(packument)
        logger.debug("Available versions: %s", versions)

        resolved_version = _gemini_choose_version(
            version,
            versions,
            default=default_version,
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
            version_obj = packument.get("versions", {}).get(resolved_version)
            if not version_obj:
                logger.error(
                    "Cannot find version object for latest version %s",
                    resolved_version,
                )
                return {
                    "status": "error",
                    "error": f"Version {resolved_version} not found",
                }
    else:
        version_obj = packument
        resolved_version = version_obj.get("version")
        logger.info("Single version object detected, version: %s", resolved_version)

    logger.debug("Version object keys: %s", list(version_obj.keys()))

    # repository field compatibility handling
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
            "No readme found for %s@%s, trying to fetch from npm page...",
            pkg_name,
            resolved_version,
        )
        readme_content = fetch_npm_readme_simple(pkg_name, resolved_version)

    # npm-side initial license analysis
    license_analysis = None
    readme_license = None
    if not license_type:
        logger.info("No license info in npm metadata, analyzing readme for license...")
        npm_source_url = f"https://www.npmjs.com/package/{pkg_name}/v/{resolved_version}"
        license_analysis = await analyze_license_content_async(
            readme_content or "",
            npm_source_url,
        )
        if license_analysis and license_analysis.get("licenses"):
            license_type = license_analysis["licenses"][0]
            readme_license = license_analysis["licenses"][0]
            logger.info("Extracted license from readme: %s", readme_license)
        else:
            license_type = None
            readme_license = None

    last_modified_iso = version_obj.get("time") or version_obj.get("date")
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

    try:
        copyright_notice = await extract_copyright_info_async(readme_content or "")
    except Exception:
        copyright_notice = extract_copyright_info(readme_content or "")

    logger.debug("Copyright notice: %s", copyright_notice)

    if not copyright_notice:
        if not author:
            author = f"{pkg_name} original author and authors"
        logger.debug("Author: %s", author)
        copyright_notice = f"Copyright(c) {dt.year} {author}".strip()
        logger.debug("Fallback copyright notice: %s", copyright_notice)

    license_files = f"https://www.npmjs.com/package/{pkg_name}/v/{resolved_version}?activeTab=code"
    logger.debug("License files URL: %s", license_files)

    final_license_file = license_files
    final_used_default_branch = False

    # GitHub-enriched fields
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

    # Try GitHub scan if repo_url points to GitHub
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

                used_default_branch = github_result.get("used_default_branch")
                github_fields["used_default_branch"] = used_default_branch
                if used_default_branch is not None:
                    final_used_default_branch = used_default_branch

                # Replace license_files only when GitHub explicitly resolved a non-default branch/tag
                if used_default_branch is False and github_result.get("license_files"):
                    final_license_file = github_result.get("license_files", license_files)
                    logger.info(
                        "Replaced license_files with GitHub result because used_default_branch is False"
                    )
                else:
                    final_license_file = license_files
                    logger.info(
                        "Kept original npm code URL because used_default_branch is True or missing"
                    )

                for key in [
                    "license_analysis",
                    "has_license_conflict",
                    "readme_license",
                    "license_file_license",
                ]:
                    if github_result.get(key) is not None:
                        github_fields[key] = github_result.get(key)

                github_copyright_notice = github_result.get("copyright_notice")
                logger.info("GitHub copyright_notice: %s", github_copyright_notice)
            else:
                logger.warning("repo_url path not valid for github: %s", repo_url)
        except Exception as e:
            github_scan_success = False
            logger.error("Failed to call process_github_repository: %s", e, exc_info=True)

    # Fallback to npm tarball analysis when:
    # 1. there is no GitHub repo, OR
    # 2. repo is GitHub but GitHub scan failed
    thirdparty_dirs: List[str] = []
    should_fallback_to_npm_tarball = (not repo_url or "github.com" not in repo_url) or (
        repo_url and "github.com" in repo_url and not github_scan_success
    )

    if should_fallback_to_npm_tarball:
        tarball_url = version_obj.get("dist", {}).get("tarball")
        if tarball_url:
            try:
                thirdparty_dirs = await async_analyze_npm_tarball_thirdparty_dirs(tarball_url)
                logger.info("Found thirdparty dirs in npm tarball: %s", thirdparty_dirs)

                if license_analysis is None:
                    license_analysis = {}

                if isinstance(license_analysis, dict):
                    existing_dirs = license_analysis.get("thirdparty_dirs")
                    if not existing_dirs:
                        license_analysis["thirdparty_dirs"] = thirdparty_dirs
                else:
                    # Defensive fallback in case license_analysis is a non-dict shape
                    license_analysis = {"thirdparty_dirs": thirdparty_dirs}
            except Exception as e:
                logger.warning("Failed to analyze npm tarball for thirdparty dirs: %s", e)
        else:
            logger.info("No tarball URL found for npm fallback analysis")
    else:
        logger.info("GitHub scan succeeded; npm tarball fallback not needed")

    # copyright selection logic
    final_copyright_notice = copyright_notice
    if github_copyright_notice:
        if "original author and authors" in github_copyright_notice:
            logger.info(
                "GitHub copyright_notice contains 'original author and authors', keep npm copyright_notice"
            )
        else:
            final_copyright_notice = github_copyright_notice
            logger.info("Replaced copyright_notice with GitHub result")

    # Final field selection:
    # - If GitHub scan succeeded and returned enriched values, prefer GitHub
    # - Otherwise keep npm-side values/fallback values
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
    Asynchronously download npm tarball and extract it to a temp directory.
    Returns the extraction root path.
    """
    tmp_dir = tempfile.mkdtemp()
    tarball_path = os.path.join(tmp_dir, "package.tgz")

    async with aiohttp.ClientSession() as session:
        async with session.get(tarball_url) as resp:
            resp.raise_for_status()
            async with aiofiles.open(tarball_path, "wb") as f:
                async for chunk in resp.content.iter_chunked(8192):
                    await f.write(chunk)

    with tarfile.open(tarball_path, "r:gz") as tar:
        tar.extractall(path=tmp_dir)

    return tmp_dir


async def async_analyze_npm_tarball_thirdparty_dirs(tarball_url: str) -> List[str]:
    """
    Asynchronously download and analyze third-party directories in an npm tarball.
    Cleans up temp files automatically after analysis.
    """
    tmp_dir = None
    try:
        tmp_dir = await async_download_and_extract_npm_tarball(tarball_url)
        package_root = os.path.join(tmp_dir, "package")
        if not os.path.isdir(package_root):
            package_root = tmp_dir
        thirdparty_dirs = find_top_level_thirdparty_dirs_local(package_root)
        return thirdparty_dirs
    finally:
        if tmp_dir and os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)