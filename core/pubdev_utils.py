import re
import logging
import os
import shutil
import tarfile
import tempfile
from datetime import datetime
from typing import Optional, Tuple
import aiohttp
import aiofiles
from .utils import find_matching_version

log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("pubdev_utils")
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = logging.FileHandler(os.path.join(log_dir, "pubdev_utils.log"), encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

PUBDEV_API_BASE = "https://pub.dev/api"


def _parse_pubdev_url(url: str) -> tuple[str, Optional[str]]:
    """
    Parse a pub.dev or pub.dartlang.org URL into (package_name, version_or_None).

    Supported formats:
      https://pub.dev/packages/{name}
      https://pub.dev/packages/{name}/versions/{version}
      https://pub.dartlang.org/packages/{name}
      https://pub.dartlang.org/packages/{name}/versions/{version}
    """
    url = url.strip()
    # Normalise old domain
    url = re.sub(r"https?://pub\.dartlang\.org", "https://pub.dev", url)

    m = re.match(
        r"https://pub\.dev/packages/(?P<name>[^/?#]+)"
        r"(?:/versions/(?P<version>[^/?#]+))?",
        url,
    )
    if not m:
        raise ValueError(f"Cannot parse pub.dev URL: {url}")

    name = m.group("name")
    version = m.group("version")
    logger.info(f"Parsed pub.dev URL: name={name}, version={version}")
    return name, version


async def _fetch_pubdev_metadata(name: str, version: Optional[str]) -> dict:
    """
    Fetch package metadata from the pub.dev API.

    With a version → GET /api/packages/{name}/versions/{version}
    Without        → GET /api/packages/{name}  (contains 'latest' and 'versions')
    """
    if version:
        url = f"{PUBDEV_API_BASE}/packages/{name}/versions/{version}"
    else:
        url = f"{PUBDEV_API_BASE}/packages/{name}"

    logger.info(f"Fetching pub.dev metadata: {url}")
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=20)) as resp:
            if resp.status != 200:
                logger.warning(f"pub.dev API returned {resp.status} for {url}")
                return {}
            data = await resp.json()
            logger.debug(f"pub.dev raw response keys: {list(data.keys())}")
            return data


def _resolve_pubdev_version(all_versions: list[str], requested: Optional[str]) -> tuple[str, bool]:
    """
    Resolve a requested version string against the list of available versions.

    Priority:
      1. Exact match (ignoring leading 'v')
      2. Prefix match (e.g. '1.2' matches '1.2.3')
      3. Fallback to latest (first in list, which pub.dev returns newest-first)

    Returns (resolved_version, used_default).
    """
    if not requested:
        # pub.dev API returns versions in ascending order (oldest first)
        latest = all_versions[-1] if all_versions else ""
        logger.info(f"No version requested, using latest: {latest}")
        return latest, True

    # 1+2. 精确匹配 + 补零数值等价匹配（"1.2" == "1.2.0"）
    #      取代原来的 startswith 前缀匹配（会把 "1.2" 误配到 "1.25.0"）
    matched = find_matching_version(requested, all_versions)
    if matched is not None:
        logger.info(f"Version match: {matched} for requested {requested}")
        return matched, False

    # 3. Fallback
    latest = all_versions[0] if all_versions else requested
    logger.warning(f"No version match for {requested}, falling back to latest: {latest}")
    return latest, True


def _extract_github_url(pubspec: dict) -> Optional[str]:
    """
    Extract a GitHub URL from pubspec fields, trying in order:
      repository → homepage → issue_tracker

    Returns the URL string or None.
    """
    for field in ("repository", "homepage", "issue_tracker"):
        val = pubspec.get(field)
        if isinstance(val, str) and "github.com" in val:
            url = val.strip()
            logger.info(f"Found GitHub URL in pubspec[{field}]: {url}")
            return url
    return None


async def get_github_url_from_pubdev(
    url: str,
    version: Optional[str] = None,
    name: Optional[str] = None,
) -> dict:
    """
    Resolve a pub.dev package URL to its GitHub repository URL and basic metadata.

    Returns a dict with:
      github_url        – GitHub repo/subdir URL from pubspec, or None
      resolved_version  – the actual package version used
      used_default      – True if no matching version was found and latest was used
      license_in_pubspec – license field from pubspec if present, else None
      pubspec           – full pubspec dict
      raw_info          – raw API response
    """
    try:
        pkg_name, version_in_url = _parse_pubdev_url(url)
    except ValueError as e:
        logger.error(str(e))
        return {"github_url": None, "resolved_version": None, "used_default": False,
                "license_in_pubspec": None, "pubspec": {}, "raw_info": {}}

    # Version precedence: explicit param > version embedded in URL
    effective_version = version or version_in_url

    raw = await _fetch_pubdev_metadata(pkg_name, effective_version)
    if not raw:
        return {"github_url": None, "resolved_version": effective_version, "used_default": False,
                "license_in_pubspec": None, "pubspec": {}, "raw_info": {}}

    # If we got the full package listing (no specific version requested), resolve version
    used_default = False
    if "versions" in raw:
        # If a specific version was requested, resolve it; otherwise use the 'latest' object
        # directly to avoid an extra API call
        if effective_version:
            all_versions = [v.get("version", "") for v in raw.get("versions", []) if v.get("version")]
            effective_version, used_default = _resolve_pubdev_version(all_versions, effective_version)
            raw = await _fetch_pubdev_metadata(pkg_name, effective_version)
            if not raw:
                return {"github_url": None, "resolved_version": effective_version, "used_default": used_default,
                        "license_in_pubspec": None, "pubspec": {}, "raw_info": {}}
        else:
            # No version requested: use 'latest' embedded in the package listing
            latest_obj = raw.get("latest", {})
            if latest_obj:
                raw = latest_obj
                used_default = True
            else:
                all_versions = [v.get("version", "") for v in raw.get("versions", []) if v.get("version")]
                effective_version, used_default = _resolve_pubdev_version(all_versions, None)
                raw = await _fetch_pubdev_metadata(pkg_name, effective_version)
                if not raw:
                    return {"github_url": None, "resolved_version": effective_version, "used_default": used_default,
                            "license_in_pubspec": None, "pubspec": {}, "raw_info": {}}

    pubspec = raw.get("pubspec", {})
    resolved_version = raw.get("version", effective_version)
    github_url = _extract_github_url(pubspec)
    license_in_pubspec = pubspec.get("license") or pubspec.get("license_type")

    logger.info(
        f"pub.dev resolution: pkg={pkg_name} version={resolved_version} "
        f"github_url={github_url} license={license_in_pubspec}"
    )

    return {
        "github_url": github_url,
        "resolved_version": resolved_version,
        "used_default": used_default,
        "license_in_pubspec": license_in_pubspec,
        "pubspec": pubspec,
        "raw_info": raw,
    }


# ---------------------------------------------------------------------------
# Archive-based fallback (used when GitHub has no matching version tag)
# ---------------------------------------------------------------------------

async def _download_and_extract_pubdev_archive(archive_url: str) -> str:
    """
    Download a pub.dev .tar.gz archive and extract it to a temp directory.
    Returns the path to the temp directory (caller must clean up).
    """
    tmp_dir = tempfile.mkdtemp(prefix="pubdev_")
    archive_path = os.path.join(tmp_dir, "package.tar.gz")

    logger.info(f"Downloading pub.dev archive: {archive_url}")
    async with aiohttp.ClientSession() as session:
        async with session.get(archive_url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
            resp.raise_for_status()
            async with aiofiles.open(archive_path, "wb") as f:
                async for chunk in resp.content.iter_chunked(8192):
                    await f.write(chunk)

    logger.info(f"Extracting archive to {tmp_dir}")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=tmp_dir)

    return tmp_dir


async def _find_license_in_archive(extract_dir: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Search for a LICENSE file in the root of the extracted pub.dev archive.
    pub.dev packages use a flat layout (no 'package/' subfolder like npm).

    Returns (license_content, license_filename) or (None, None).
    """
    license_keywords = ("license", "licence", "copying", "notice")
    try:
        for filename in os.listdir(extract_dir):
            if any(kw in filename.lower() for kw in license_keywords):
                filepath = os.path.join(extract_dir, filename)
                if os.path.isfile(filepath):
                    async with aiofiles.open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                        content = await f.read()
                    if content:
                        logger.info(f"Found license file in archive: {filename}")
                        return content, filename
    except Exception as e:
        logger.warning(f"Error scanning archive for license files: {e}")
    return None, None


async def process_pubdev_package(
    url: str,
    version: Optional[str] = None,
    name: Optional[str] = None,
    pubdev_info: Optional[dict] = None,
) -> dict:
    """
    Fallback processor: download the pub.dev package archive for the resolved
    version, extract the LICENSE file, and use LLM to determine license type
    and copyright notice.

    pubdev_info may be passed in from a prior get_github_url_from_pubdev call
    to avoid re-fetching the API metadata.
    """
    from .utils import analyze_license_content_async, construct_copyright_notice_async, prepare_license_text

    # Re-use already-fetched metadata if available
    if pubdev_info is None:
        pubdev_info = await get_github_url_from_pubdev(url, version, name)

    pkg_name = name or pubdev_info.get("pubspec", {}).get("name", "unknown")
    resolved_version = pubdev_info.get("resolved_version") or version or "unknown"
    raw_info = pubdev_info.get("raw_info", {})

    # Construct archive URL: prefer API-provided one, fall back to canonical pattern
    archive_url = (
        raw_info.get("archive_url")
        or f"https://pub.dev/packages/{pkg_name}/versions/{resolved_version}.tar.gz"
    )
    source_url = f"https://pub.dev/packages/{pkg_name}/versions/{resolved_version}"

    logger.info(f"pub.dev archive fallback for {pkg_name}@{resolved_version}: {archive_url}")

    license_content: Optional[str] = None
    license_filename: Optional[str] = None
    tmp_dir: Optional[str] = None
    try:
        tmp_dir = await _download_and_extract_pubdev_archive(archive_url)
        license_content, license_filename = await _find_license_in_archive(tmp_dir)
    except Exception as e:
        logger.warning(f"Failed to download/extract pub.dev archive: {e}")
    finally:
        if tmp_dir and os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)

    # LLM license analysis
    license_analysis = None
    license_type = pubdev_info.get("license_in_pubspec")

    if license_content:
        license_analysis = await analyze_license_content_async(license_content, source_url)
        if license_analysis and license_analysis.get("licenses"):
            license_type = (
                license_analysis.get("spdx_expression")
                or license_analysis["licenses"][0]
            )
    elif license_type:
        # pubspec declared a license but no file found — trust the declaration
        license_analysis = {"licenses": [license_type], "spdx_expression": license_type,
                            "confidence": 0.7, "source": "pubspec_declaration"}

    # Copyright notice via LLM
    year = str(datetime.now().year)
    copyright_notice = await construct_copyright_notice_async(
        year=year,
        owner="",
        repo=pkg_name,
        ref=resolved_version,
        component_name=pkg_name,
        readme_content=None,
        license_content=license_content,
    )

    logger.info(
        f"pub.dev fallback result: pkg={pkg_name} version={resolved_version} "
        f"license={license_type} copyright={copyright_notice}"
    )

    return {
        "input_url": url,
        "repo_url": pubdev_info.get("github_url") or source_url,
        "input_version": version,
        "resolved_version": resolved_version,
        "used_default_branch": pubdev_info.get("used_default", False),
        "component_name": pkg_name,
        "license_files": source_url,
        "license_analysis": license_analysis,
        "license_type": license_type,
        "has_license_conflict": False,
        "readme_license": None,
        "license_file_license": license_type,
        "copyright_notice": copyright_notice,
        "license_text": prepare_license_text(license_content),
        "status": "success" if license_type else "no_license_found",
        "license_determination_reason": (
            f"pub.dev archive fallback: license extracted from {license_filename}"
            if license_filename
            else (
                "pub.dev archive fallback: license from pubspec declaration"
                if pubdev_info.get("license_in_pubspec")
                else "pub.dev archive fallback: no license found"
            )
        ),
    }
