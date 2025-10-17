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
from .utils import analyze_license_content, extract_copyright_info, extract_copyright_info_async, analyze_license_content_async, find_top_level_thirdparty_dirs_local
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
    path = parsed.path.strip('/')
    
    # 1. 处理 tgz 或其他格式的直接下载链接
    if '/-/' in path:
        # 例如: aegis-web-sdk/-/aegis-web-sdk-1.39.3.tgz
        # 或者: @types/node/-/node-14.14.31.tgz
        parts = path.split('/-/', 1)[0]  # 取 /-/ 之前的部分
        if parts.startswith('npm/'):  # 处理腾讯镜像的路径前缀
            parts = parts[4:]
        return parts
    
    # 2. 处理包主页格式
    path = re.sub(r'^(?:npm/|package/)', '', path)
    path = re.sub(r'/v/[^/]+$', '', path)  # 移除版本号路径
    path = path.rstrip('/')
    
    # 3. 处理作用域包
    parts = path.split('/')
    if parts and parts[0].startswith('@'):
        # 确保作用域包包含两部分
        return '/'.join(parts[:2]) if len(parts) > 1 else parts[0]
    
    # 4. 返回第一段作为包名
    return parts[0] if parts else ''


def _fetch_packument(pkg_name: str, version: Optional[str] = None) -> dict:
    """从各个 npm registry 获取包信息。"""
    urls = [
        f"{NPM_REGISTRY_BASE}/{pkg_name}",
        f"{NPMMIRROR_REGISTRY_BASE}/{pkg_name}",
        f"{TENCENT_MIRROR_BASE}/{pkg_name}"
    ]
    if version:
        urls = [
            f"{NPM_REGISTRY_BASE}/{pkg_name}/{version}",
            f"{NPMMIRROR_REGISTRY_BASE}/{pkg_name}/{version}",
            f"{TENCENT_MIRROR_BASE}/{pkg_name}/{version}"
        ]

    last_error = None
    all_errors = []  # 收集所有错误信息
    
    for url in urls:
        try:
            npm_logger.debug(f"Trying to fetch packument from: {url}")
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
    
    # 所有尝试都失败了,记录详细错误信息
    error_summary = "\n".join(all_errors)
    npm_logger.error(f"All attempts to fetch packument failed for {pkg_name}@{version}:\n{error_summary}")
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

        provider = get_llm_provider()
        response = provider.generate(prompt)
        llm_logger.info("Version Resolve Response:")
        llm_logger.info(f"Response: {response}")
        version_resolve_logger.info("Version Resolve LLM Response:")
        version_resolve_logger.info(f"Response: {response}")

        if response:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
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

def fetch_npm_readme_simple(pkg_name, version):
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
            npm_logger.warning(f"Failed to fetch README from {url}, status code: {resp.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        npm_logger.warning(f"Error fetching README from {url}: {str(e)}")
        return None
    except Exception as e:
        npm_logger.warning(f"Unexpected error fetching README from {url}: {str(e)}")
        return None

# ---------------------------------------------------------------------------
# Main processor (public API)
# ---------------------------------------------------------------------------

async def process_npm_repository(url: str, version: Optional[str] = None) -> Dict[str, Any]:
    logger = logging.getLogger("main")

    logger.info("Starting processing for %s (requested version=%s)", url, version)

    pkg_name = _parse_package_name(url)
    logger.debug(f"Parsed package name: {pkg_name}")
    try:
        packument = _fetch_packument(pkg_name, version)
        logger.debug(f"Fetched packument keys: {list(packument.keys())}")
    except Exception as e:
        logger.warning(f"Error fetching packument for {pkg_name}@{version}: {e}, switching to alternative logic.", exc_info=True)
        # 调用你的备用逻辑
        return  {"status": "error"}

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
            logger.warning(f"Cannot find version object for {resolved_version}, fallback to latest version: {default_version}")
            resolved_version = default_version
            version_obj = packument.get("versions", {}).get(resolved_version)
            if not version_obj:
                logger.error(f"Cannot find version object for latest version {resolved_version}")
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
        repo_url = repo_url.replace("git@github.com:", "https://github.com/")
    logger.debug(f"Parsed repo_url: {repo_url}")

    license_type = version_obj.get("license")
    logger.debug(f"License type: {license_type}")

    readme_content = version_obj.get("readme")
    if readme_content:
        logger.info(f"Readme content found for {pkg_name}@{resolved_version}, length={len(readme_content)}")
    else:
        logger.info(f"No readme found for {pkg_name}@{resolved_version}, trying to fetch from npm registry...")
        readme_content = fetch_npm_readme_simple(pkg_name, resolved_version)
    license_analysis = None
    readme_license = None
    if not license_type:
        logger.info("No license info in npm metadata, analyzing readme for license...")
        # 构建npm包的URL作为source_url
        npm_source_url = f"https://www.npmjs.com/package/{pkg_name}/v/{resolved_version}"
        license_analysis = await analyze_license_content_async(readme_content or "", npm_source_url)
        if license_analysis and license_analysis.get("licenses"):
            license_type = license_analysis["licenses"][0]
            readme_license = license_analysis["licenses"][0]
            logger.info(f"Extracted license from readme: {readme_license}")
        else:
            license_type = None
            readme_license = None
    else:
        license_analysis = None
        readme_license = None
    

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
    
    # Prefer async copyright extraction since we're in async context
    try:
        copyright_notice = await extract_copyright_info_async(readme_content or "")
    except Exception:
        # Fallback to sync version if async extraction fails for any reason
        copyright_notice = extract_copyright_info(readme_content or "")
    logger.debug(f"Copyright notice: {copyright_notice}")

    if not copyright_notice:
        
        if not author:
            author = f"{pkg_name} original author and authors"

        logger.debug(f"Author: {author}")

        copyright_notice = f"Copyright(c) {dt.year} {author}".strip()
        logger.debug(f"Copyright notice: {copyright_notice}")

        
    license_files = f"https://www.npmjs.com/package/{pkg_name}/v/{resolved_version}?activeTab=code"
    logger.debug(f"License files URL: {license_files}")
    final_license_file = license_files

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
                github_result = await process_github_repository(
                    api,  # 你的实现可能需要api对象，这里按需传递
                    github_url,
                    resolved_version
                )
                # 1. license_files字段替换逻辑
                if github_result.get("used_default_branch") is False and github_result.get("license_files"):
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

    # 新增逻辑：没有github仓库时，分析npm包tarball中的thirdparty目录
    thirdparty_dirs = []
    if not repo_url or "github.com" not in repo_url:
        tarball_url = version_obj.get("dist", {}).get("tarball")
        if tarball_url:
            try:
                thirdparty_dirs = await async_analyze_npm_tarball_thirdparty_dirs(tarball_url)
                logger.info(f"Found thirdparty dirs in npm tarball: {thirdparty_dirs}")
            except Exception as e:
                logger.warning(f"Failed to analyze npm tarball for thirdparty dirs: {e}")
        # 写入license_analysis
        license_analysis = {"thirdparty_dirs": thirdparty_dirs}
    else:
        license_analysis = None

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
        "license_analysis": github_fields["license_analysis"] if repo_url and "github.com" in repo_url else license_analysis,
        "license_type": license_type,
        "has_license_conflict": github_fields["has_license_conflict"] if repo_url and "github.com" in repo_url else None,
        "readme_license": github_fields["readme_license"] if repo_url and "github.com" in repo_url else readme_license,
        "license_file_license": github_fields["license_file_license"] if repo_url and "github.com" in repo_url else None,
        "copyright_notice": final_copyright_notice,
        "status": "success",
        "license_determination_reason": "Fetched from npm registry",
        "readme": readme_content[5000:] if readme_content else None,
    }

    logger.info("Processing completed for %s@%s", pkg_name, resolved_version)
    logger.debug(f"Result dict: {json.dumps(result, ensure_ascii=False, indent=2)}")
    return result
# ---------------------------------------------------------------------------
# This module exposes only process_npm_repository for programmatic use.
# ---------------------------------------------------------------------------
async def async_download_and_extract_npm_tarball(tarball_url: str) -> str:
    """
    异步下载npm包tarball并解压到临时目录，返回解压后的根目录路径。
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
    # 解包（解包用同步方式，tarfile不支持异步）
    with tarfile.open(tarball_path, "r:gz") as tar:
        tar.extractall(path=tmp_dir)
    return tmp_dir

async def async_analyze_npm_tarball_thirdparty_dirs(tarball_url: str) -> List[str]:
    """
    异步下载并分析npm tarball中的第三方目录，分析后自动清理临时文件。
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