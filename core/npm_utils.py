import os
import re
import json
import logging
import requests
from urllib.parse import urlparse
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from .config import GEMINI_CONFIG
import google.generativeai as genai

import yaml
load_dotenv()
# ---------------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------------
# We expect an application‑level npm_logger singleton named `npm` to exist.
# If it is absent, create a minimal fallback logger with the same name so that
# logging calls never fail.
# ---------------------------------------------------------------------------
try:
    npm  # type: ignore  # noqa: F401
except NameError:  # pragma: no cover – fallback for standalone usage
    npm_logger = logging.getLogger("npm")
    llm_logger = logging.getLogger("llm_interaction")
    version_resolve_logger = logging.getLogger("version_resolve_interaction")

# ---------------------------------------------------------------------------
# Constants & Exceptions
# ---------------------------------------------------------------------------

NPM_REGISTRY_BASE = "https://registry.npmjs.org"

class NpmAPIError(Exception):
    """Raised when the npm registry returns an unexpected response."""

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _parse_package_name(url_or_name: str) -> str:
    """Extract npm package name (supports @scope) from full URL or raw name."""
    npm_logger.info("Parsing package name from input: %s", url_or_name)
    if url_or_name.startswith("http"):
        path = urlparse(url_or_name).path
        match = re.match(r"/(?:package/)?(?P<name>.+)", path.rstrip("/"))
        if match:
            name = match.group("name")
        else:
            name = path.rstrip("/").split("/")[-1]
    else:
        name = url_or_name
    npm_logger.info("Resolved package name: %s", name)
    return name


def _fetch_packument(pkg_name: str, version: Optional[str] = None) -> dict:
    """
    如果 version 为空，返回 packument（所有版本）。
    如果 version 不为空，返回单个版本对象。
    """
    import requests
    if version:
        url = f"https://registry.npmjs.org/{pkg_name}/{version}"
    else:
        url = f"https://registry.npmjs.org/{pkg_name}"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()


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
    """Resolve fuzzily‑specified version to concrete one using Gemini if API key present."""
    # Fast‑path scenarios ----------------------------------------------------
    USE_LLM = os.getenv("USE_LLM", "true").lower() == "true"
    with open("prompts.yaml", "r", encoding="utf-8") as f:
        PROMPTS = yaml.safe_load(f)
    if not user_input or user_input.lower() in {"", "latest"}:
        npm_logger.info("User version empty or 'latest'; using default %s", default)
        return default
    if user_input in versions:
        npm_logger.info("User version %s found exactly in version list", user_input)
        return user_input

    try:
        prompt = PROMPTS["version_resolve"].format(
                    candidate_versions=versions,
                    version=user_input,
                    default_branch=default
                )
        llm_logger.info("Version Resolve Request:")
        llm_logger.info(f"Prompt: {prompt}")
        version_resolve_logger.info("Version Resolve LLM Request:")

        model = genai.GenerativeModel(GEMINI_CONFIG["model"])
        response = model.generate_content(prompt)
        llm_logger.info("Version Resolve Response:")
        llm_logger.info(f"Response: {response.text}")
        version_resolve_logger.info("Version Resolve LLM Response:")
        version_resolve_logger.info(f"Response: {response.text}")

        if response.text:
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                resolved_version = result.get("resolved_version", default)
                used_default_branch = result.get("used_default_branch", resolved_version == default)
                npm_logger.info(f"LLM resolved version: {resolved_version}, used_default_branch: {used_default_branch}")
                version_resolve_logger.info(f"LLM resolved version: {resolved_version}, used_default_branch: {used_default_branch}")
                return resolved_version
            else:
                version_resolve_logger.warning("No JSON found in version resolve response")
    except Exception as e:
            version_resolve_logger.error(f"Failed to resolve version via LLM: {str(e)}", exc_info=True)
            npm_logger.error(f"Failed to resolve version via LLM: {str(e)}", exc_info=True)

    
    

    return default

# ---------------------------------------------------------------------------
# Main processor (public API)
# ---------------------------------------------------------------------------

def process_npm_repository(url: str, version: Optional[str] = None) -> Dict[str, Any]:
    logger = logging.getLogger("main")

    logger.info("Starting processing for %s (requested version=%s)", url, version)

    pkg_name = _parse_package_name(url)
    logger.debug(f"Parsed package name: {pkg_name}")

    # 1. 获取元数据
    packument = _fetch_packument(pkg_name, version)
    logger.debug(f"Fetched packument keys: {list(packument.keys())}")

    # 2. 判断是packument还是单版本对象
    if "versions" in packument:
        # 多版本packument
        default_version = packument.get("dist-tags", {}).get("latest")
        versions = _list_all_versions(packument)
        logger.debug(f"Available versions: {versions}")
        resolved_version = _gemini_choose_version(version, versions, default=default_version)
        logger.info("Resolved version for %s: %s", pkg_name, resolved_version)
        version_obj = packument.get("versions", {}).get(resolved_version)
        if not version_obj:
            logger.error(f"Cannot find version object for {resolved_version}")
            return {"status": "error", "error": f"Version {resolved_version} not found"}
    else:
        # 单版本对象
        version_obj = packument
        resolved_version = version_obj.get("version")
        logger.info("Single version object detected, version: %s", resolved_version)

    logger.debug(f"Version object keys: {list(version_obj.keys())}")

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
        # 转换为 https://github.com/<owner>/<repo>
        repo_url = repo_url.replace("git@github.com:", "https://github.com/")
    logger.debug(f"Parsed repo_url: {repo_url}")

    license_type = version_obj.get("license")
    logger.debug(f"License type: {license_type}")

    readme_content = version_obj.get("readme")
    if readme_content:
        logger.info(f"Readme content found for {pkg_name}@{resolved_version}, length={len(readme_content)}")
    else:
        logger.info(f"No readme found for {pkg_name}@{resolved_version}")

    last_modified_iso = version_obj.get("time") or version_obj.get("date")
    logger.debug(f"Last modified ISO: {last_modified_iso}")
    if last_modified_iso:
        try:
            dt = datetime.fromisoformat(last_modified_iso.replace("Z", "+00:00"))
        except Exception:
            logger.warning(f"Failed to parse last_modified_iso: {last_modified_iso}, using now()")
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
    logger.debug(f"Author: {author}")

    copyright_notice = f"Copyright(c) {dt.year} {author}".strip()
    logger.debug(f"Copyright notice: {copyright_notice}")

    license_files = f"https://www.npmjs.com/package/{pkg_name}/v/{resolved_version}?activeTab=code"
    logger.debug(f"License files URL: {license_files}")

    # 新增：如果repo_url为github地址，调用process_github_repository补充元数据
    github_fields = {
        "license_files": license_files,
        "license_analysis": None,
        "has_license_conflict": None,
        "readme_license": None,
        "license_file_license": None
    }
    # 新增：用于后续判断
    github_copyright_notice = None

    if repo_url and "github.com" in repo_url:
        logger.info(f"repo_url is a GitHub URL, calling process_github_repository: {repo_url} {resolved_version}")
        try:
            from core.github_utils import process_github_repository, GitHubAPI
            parsed = urlparse(repo_url)
            path_parts = parsed.path.strip("/").split("/")
            if len(path_parts) >= 2:
                github_url = f"https://github.com/{path_parts[0]}/{path_parts[1]}"
                api = GitHubAPI()
                github_result = process_github_repository(
                    None,  # 你的实现可能需要api对象，这里按需传递
                    github_url,
                    resolved_version
                )
                # 1. license_files字段替换逻辑
                if github_result.get("used_default_branch") is False:
                    final_license_file = github_result.get("license_files", license_files)
                    logger.info("Replaced license_files with GitHub result because used_default_branch is False")
                else:
                    final_license_file = license_files
                    logger.info("Kept original license_files because used_default_branch is True")
                # 2. 其他字段直接赋值
                for key in ["license_analysis", "has_license_conflict", "readme_license", "license_file_license"]:
                    if github_result.get(key) is not None:
                        github_fields[key] = github_result.get(key)
                # 3. copyright_notice特殊逻辑
                github_copyright_notice = github_result.get("copyright_notice")
                logger.info(f"GitHub copyright_notice: {github_copyright_notice}")
            else:
                logger.warning(f"repo_url path not valid for github: {repo_url}")
        except Exception as e:
            logger.error(f"Failed to call process_github_repository: {e}", exc_info=True)

    # 处理copyright_notice逻辑
    final_copyright_notice = copyright_notice
    if github_copyright_notice:
        if "original author and authors" in github_copyright_notice:
            logger.info("GitHub copyright_notice contains 'original author and authors', keep npm copyright_notice")
        else:
            final_copyright_notice = github_copyright_notice
            logger.info("Replaced copyright_notice with GitHub result")

    result = {
        "input_url": url,
        "repo_url": repo_url,
        "input_version": version,
        "resolved_version": resolved_version,
        "used_default_branch": False,
        "component_name": pkg_name,
        "license_files": final_license_file,
        "license_analysis": github_fields["license_analysis"],
        "license_type": license_type,
        "has_license_conflict": github_fields["has_license_conflict"],
        "readme_license": github_fields["readme_license"],
        "license_file_license": github_fields["license_file_license"],
        "copyright_notice": final_copyright_notice,
        "status": "success",
        "license_determination_reason": "Fetched from npm registry",
        "readme": readme_content[5000:] if readme_content else None,
    }

    logger.info("Processing completed for %s@%s", pkg_name, resolved_version)
    logger.debug(f"Result dict: {json.dumps(result, ensure_ascii=False, indent=2)}")
# ---------------------------------------------------------------------------
# This module exposes only process_npm_repository for programmatic use.
# ---------------------------------------------------------------------------
