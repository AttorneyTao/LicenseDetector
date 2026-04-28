import re
import logging
import os
from typing import Optional
import aiohttp

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
        latest = all_versions[0] if all_versions else ""
        logger.info(f"No version requested, using latest: {latest}")
        return latest, True

    req = requested.lstrip("v")

    # 1. Exact match
    for v in all_versions:
        if v.lstrip("v") == req:
            logger.info(f"Exact version match: {v}")
            return v, False

    # 2. Prefix match
    for v in all_versions:
        if v.lstrip("v").startswith(req):
            logger.info(f"Prefix version match: {v} for requested {requested}")
            return v, False

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
        all_versions = [v.get("version", "") for v in raw.get("versions", []) if v.get("version")]
        effective_version, used_default = _resolve_pubdev_version(all_versions, effective_version)
        # Re-fetch the specific version object
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
