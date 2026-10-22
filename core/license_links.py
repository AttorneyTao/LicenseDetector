"""Final, link-only improvement of package-download evidence.

No repository discovery, version resolution, copyright extraction or audit retry
is performed here. Existing page results are left alone. New URLs are candidates
until their identity, version and license content have been checked.
"""
import asyncio
import hashlib
import logging
import re
from dataclasses import dataclass
from urllib.parse import parse_qs, quote, unquote, urlparse

import httpx
from bs4 import BeautifulSoup

from core.purl_fallback import build_fallback_purl, _has_license
from core.utils import analyze_license_content_async, get_concluded_license, parse_purl, purl_to_url

logger = logging.getLogger('main')
LOOKUP_TIMEOUT = 30
MAX_PAGE_BYTES = 1_000_000
EVIDENCE_KEY = '_archive_license_link_evidence'
PACKAGE_SUFFIXES = ('.tar.gz', '.tar.xz', '.tar.bz2', '.tar.zst', '.tgz',
                    '.tbz2', '.txz', '.zip', '.tar', '.crate', '.whl', '.egg',
                    '.nupkg', '.jar', '.aar', '.deb', '.rpm', '.7z', '.gz', '.xz')


def content_digest(text):
    return hashlib.sha256(' '.join(text.split()).encode('utf-8')).hexdigest()


def is_package_download(url, headers=None):
    if not isinstance(url, str) or not url.strip():
        return False
    parsed = urlparse(url)
    path = unquote(parsed.path).lower().rstrip('/')
    if path.endswith(PACKAGE_SUFFIXES) or path.endswith('/download'):
        return True
    if parsed.hostname in {'codeload.github.com'} or '/zipball/' in path or '/tarball/' in path:
        return True
    query = parse_qs(parsed.query)
    if any(str(value).lower().endswith(PACKAGE_SUFFIXES)
           for values in query.values() for value in values):
        return True
    if headers:
        disposition = headers.get('content-disposition', '').lower()
        if any(suffix in disposition for suffix in PACKAGE_SUFFIXES):
            return True
        media_type = headers.get('content-type', '').split(';')[0].lower()
        if media_type in {'application/zip', 'application/gzip', 'application/x-gzip',
                          'application/x-tar', 'application/x-xz', 'application/x-7z-compressed',
                          'application/java-archive', 'application/vnd.debian.binary-package',
                          'application/octet-stream'}:
            return True
    return False


def _known_evidence_url(url):
    """Avoid extra network requests for existing file/registry page results."""
    parsed = urlparse(url)
    path = unquote(parsed.path).lower()
    filename = path.rsplit('/', 1)[-1]
    if filename.startswith(('license', 'licence', 'copying', 'copyright', 'readme', 'notice')) or filename.endswith('.pom'):
        return True
    return any(parsed.hostname == host and path.startswith(prefix) for host, prefix in (
        ('crates.io', '/crates/'), ('www.npmjs.com', '/package/'),
        ('pypi.org', '/project/'), ('pub.dev', '/packages/'),
        ('www.nuget.org', '/packages/'), ('pkg.go.dev', '/'),
        ('mvnrepository.com', '/artifact/'), ('central.sonatype.com', '/artifact/'),
        ('sources.debian.org', '/src/'), ('launchpad.net', '/ubuntu/+source/'),
    ))


def _same_url(left, right):
    a, b = urlparse(left), urlparse(right)
    return (a.scheme == b.scheme and a.hostname == b.hostname and a.port == b.port
            and unquote(a.path).rstrip('/') == unquote(b.path).rstrip('/')
            and parse_qs(a.query) == parse_qs(b.query))


def _same_version(left, right):
    # Link selection never broadens a requested version or chooses a nearby tag.
    return isinstance(left, str) and isinstance(right, str) and left.removeprefix('v') == right.removeprefix('v')


@dataclass(frozen=True)
class Candidate:
    url: str
    digest: str = ''
    name: str = ''
    version: str = ''


def _github_archive_candidates(source_url, evidence, requested):
    """Map an explicit GitHub archive ref and the *selected* local file only."""
    parsed = urlparse(source_url)
    path = unquote(parsed.path)
    match = None
    if parsed.hostname == 'github.com':
        match = re.fullmatch(r'/([^/]+)/([^/]+)/archive/(?:refs/tags/)?(.+?)(?:\.tar\.gz|\.zip)', path)
    elif parsed.hostname == 'codeload.github.com':
        match = re.fullmatch(r'/([^/]+)/([^/]+)/(?:tar\.gz|zip)/(?:refs/tags/)?(.+)', path)
    if not match or not evidence:
        return []
    owner, repo, ref = match.groups()
    if not requested or not (_same_version(requested, ref) or ref in (f'{repo}-{requested}', f'{repo}@{requested}')):
        return []
    selected = evidence.get('path', '')
    if not selected or selected.startswith('/') or any(p in ('.', '..') for p in selected.split('/')):
        return []
    relative = '/'.join(quote(p, safe='') for p in selected.split('/'))
    ref = quote(ref, safe='')
    digest = evidence.get('digest', '')
    if not digest:
        return []
    return [Candidate(f'https://github.com/{owner}/{repo}/blob/{ref}/{relative}', digest),
            Candidate(f'https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{relative}', digest)]


def _registry_candidate(row, result):
    from core.crate_utils import parse_crates_io_reference
    reference = parse_crates_io_reference(result.get('license_files'))
    parsed = parse_purl(row.get('github_url'))
    if not parsed:
        parsed = parse_purl(build_fallback_purl(row))
    if not parsed:
        if reference and reference[1]:
            parsed = parse_purl(f'pkg:cargo/{reference[0]}@{reference[1]}')
    if not parsed:
        return None
    version = next((v for v in (parsed.version, row.get('version'), result.get('resolved_version'))
                    if isinstance(v, str) and v.strip()), None)
    if not isinstance(version, str) or not version.strip():
        return None
    if reference and (parsed.type != 'cargo' or parsed.name != reference[0]
                      or (reference[1] and not _same_version(version, reference[1]))):
        return None
    requested = row.get('version')
    if isinstance(requested, str) and requested.strip() and not _same_version(requested, version):
        return None
    resolved = result.get('resolved_version')
    if isinstance(resolved, str) and resolved.strip() and not _same_version(version, resolved):
        return None
    if isinstance(resolved, str) and resolved.strip():
        version = resolved
    base = purl_to_url(parsed)
    if not base:
        return None
    suffix = quote(version, safe='')
    def candidate(url):
        return Candidate(url, name=parsed.name if parsed.type == 'maven' else parsed.full_name, version=version)
    if parsed.type == 'npm':
        return candidate(f'{base}/v/{suffix}')
    if parsed.type == 'pub':
        return candidate(f'{base}/versions/{suffix}')
    if parsed.type == 'golang':
        return candidate(f'{base}@{suffix}?tab=licenses')
    if parsed.type == 'deb':
        # Supply the version to the existing purl mapping, including distro rules.
        return Candidate(purl_to_url(parsed._replace(version=version)), name=parsed.name, version=version)
    if parsed.type in {'cargo', 'pypi', 'nuget', 'maven'}:
        return candidate(f'{base}/{suffix}')
    return None


def _candidates(row, result, evidence, earlier_result=None):
    requested = row.get('version')
    if not isinstance(requested, str) or not requested.strip():
        requested = result.get('resolved_version')
    # Archive processors record the supplied version. Do not let that field hide
    # a contradictory version explicitly present in a download URL.
    from core.archive_utils import ARCHIVE_SUFFIXES, guess_name_version_from_url
    source_version = None
    if urlparse(result['license_files']).path.lower().endswith(ARCHIVE_SUFFIXES):
        _, source_version = guess_name_version_from_url(result['license_files'])
    if source_version and requested and not _same_version(requested, source_version):
        return []
    candidates = _github_archive_candidates(result['license_files'], evidence, requested)
    # Archive fallback used to discard these references. Reuse only a repository
    # and exact ref already resolved by the existing audit, checking file bytes.
    prior = earlier_result or {}
    repo = prior.get('repo_url')
    ref = prior.get('resolved_version')
    if (evidence and isinstance(repo, str) and re.fullmatch(r'https://github\.com/[^/]+/[^/]+/?', repo)
            and prior.get('used_default_branch') is False and _same_version(requested, ref)):
        archive = f'{repo.rstrip("/")}/archive/refs/tags/{quote(ref, safe="")}.tar.gz'
        candidates.extend(_github_archive_candidates(archive, evidence, requested))
    registry = _registry_candidate(row, result)
    if registry:
        candidates.append(registry)
    # No generic license homepages or guessed repository/default-branch URLs.
    return list(dict.fromkeys(candidates))


async def _fetch_page(client, url):
    async with client.stream('GET', url) as response:
        response.raise_for_status()
        if not _same_url(url, str(response.url)) or is_package_download(str(response.url), response.headers):
            return None
        content_type = response.headers.get('content-type', '').lower()
        if not any(t in content_type for t in ('text/', 'json', 'xml', 'xhtml')):
            return None
        body = bytearray()
        async for chunk in response.aiter_bytes():
            body.extend(chunk)
            if len(body) > MAX_PAGE_BYTES:
                return None
        text = bytes(body).decode(response.encoding or 'utf-8', errors='replace')
        if 'html' in content_type:
            soup = BeautifulSoup(text, 'html.parser')
            for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
                tag.decompose()
            # GitHub's file contents; exclude controls/navigation from digest.
            source = soup.select_one('[data-testid="read-only-cursor-text-area"]')
            text = source.get_text() if source else soup.get_text(' ', strip=True)
        return text.strip() or None


def _expression_key(expression):
    return re.sub(r'\s+', '', expression or '').casefold()


async def _matches_license(text, candidate, original):
    if candidate.digest:
        return content_digest(text) == candidate.digest
    if not all(re.search(r'(?<![\w.-])' + re.escape(value) + r'(?![\w.-])', text, re.I)
               for value in (candidate.name, candidate.version) if value):
        return False
    # Reuse the existing LLM interpretation, without changing prompts or extracting
    # copyright. Its result only validates this link and never replaces the audit.
    analysis = await analyze_license_content_async(text, candidate.url)
    expression = analysis.get('spdx_expression')
    if not expression and len(analysis.get('licenses') or []) == 1:
        expression = analysis['licenses'][0]
    expected = get_concluded_license(original.get('license_type'), original.get('readme_license'),
                                     original.get('license_file_license'), original.get('status'))
    return bool(expression and _expression_key(expression) == _expression_key(expected))


async def prefer_license_webpage(row, original, earlier_result=None):
    """Run once after all audit fallbacks; change only license_files, fail closed."""
    evidence = original.get(EVIDENCE_KEY)
    result = {k: v for k, v in original.items() if k != EVIDENCE_KEY} if evidence is not None else original
    try:
        link = result.get('license_files')
        if not _has_license(result) or not isinstance(link, str) or not link.startswith(('https://', 'http://')):
            return result
        download = is_package_download(link)
        if not download and _known_evidence_url(link):
            return result
        async with asyncio.timeout(LOOKUP_TIMEOUT):
            async with httpx.AsyncClient(timeout=8, follow_redirects=True,
                                        headers={'User-Agent': 'LicenseDetector/license-link-check'}) as client:
                if not download:
                    response = await client.head(link)
                    response.raise_for_status()
                    download = is_package_download(str(response.url), response.headers)
                    if not download:
                        return result
                for candidate in _candidates(row, result, evidence, earlier_result):
                    try:
                        text = await _fetch_page(client, candidate.url)
                        if text and await _matches_license(text, candidate, result):
                            logger.info('License link improved: %s -> %s', link, candidate.url)
                            return {**result, 'license_files': candidate.url}
                    except Exception as exc:
                        logger.info('License page unavailable; keeping other candidates: %s (%s)', candidate.url, type(exc).__name__)
        return result
    except Exception as exc:
        logger.info('License link lookup failed; retaining original link (%s)', type(exc).__name__)
        return result
