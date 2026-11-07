"""CRAN/Bioconductor web evidence. Archive fallback stays in the dispatchers."""
import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import parse_qs, quote, unquote, urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup

from .utils import analyze_license_content_async, construct_copyright_notice_async, prepare_license_text

logger = logging.getLogger('main')
MAX_PAGE_BYTES = 1_000_000
CRAN_HOSTS = {'cran.r-project.org', 'cloud.r-project.org', 'cran.rstudio.com'}
BIOC_SECTIONS = r'(bioc|data/annotation|data/experiment|workflows)'
PACKAGE_NAME = r'[A-Za-z][A-Za-z0-9.]*'


@dataclass(frozen=True)
class RReference:
    registry: str
    name: str
    version: str = ''
    release: str = 'release'
    section: str = 'bioc'

    def page_url(self, release=None):
        name = quote(self.name, safe='')
        if self.registry == 'cran':
            return f'https://cran.r-project.org/web/packages/{name}/index.html'
        return f'https://bioconductor.org/packages/{release or self.release}/{self.section}/html/{name}.html'


def parse_r_reference(url, package_manager=None, name=None):
    """Use explicit registry coordinates only; never infer a repository from prose."""
    if isinstance(url, str):
        try:
            parsed = urlparse(url)
        except ValueError:
            return None
        if parsed.scheme not in {'http', 'https'}:
            return None
        host, path = (parsed.hostname or '').lower(), unquote(parsed.path)
        if host in CRAN_HOSTS:
            archive = re.fullmatch(rf'/src/contrib/(?:Archive/{PACKAGE_NAME}/)?({PACKAGE_NAME})_([^/]+)\.tar\.gz', path)
            if archive:
                return RReference('cran', *archive.groups())
            page = re.fullmatch(rf'/web/packages/({PACKAGE_NAME})(?:/index\.html|/)?', path)
            short = re.fullmatch(rf'/package=({PACKAGE_NAME})/?', path)
            query_name = parse_qs(parsed.query).get('package', [''])[0] if path == '/' else ''
            if page or short:
                return RReference('cran', (page or short).group(1))
            if re.fullmatch(PACKAGE_NAME, query_name):
                return RReference('cran', query_name)
        elif host in {'bioconductor.org', 'www.bioconductor.org'}:
            prefix = rf'/packages/(release|devel|\d+\.\d+)/{BIOC_SECTIONS}'
            archive = re.fullmatch(prefix + rf'/src/contrib/({PACKAGE_NAME})_([^/]+)\.tar\.gz', path)
            page = re.fullmatch(prefix + rf'/html/({PACKAGE_NAME})\.html', path)
            if archive:
                release, section, package, version = archive.groups()
                return RReference('bioconductor', package, version, release, section)
            if page:
                release, section, package = page.groups()
                return RReference('bioconductor', package, '', release, section)
            short = re.fullmatch(rf'/packages/({PACKAGE_NAME})/?', path)
            if short:
                return RReference('bioconductor', short.group(1))
    # An explicit manager can supply coordinates when the input is not a registry URL.
    manager = package_manager.strip().lower() if isinstance(package_manager, str) else ''
    if manager in {'cran', 'bioconductor', 'bioc'} and isinstance(name, str) and re.fullmatch(PACKAGE_NAME, name):
        return RReference('cran' if manager == 'cran' else 'bioconductor', name)
    return None


def _same_version(left, right):
    """R numeric components permit '.' and '-' separators and trailing zeros."""
    def parts(value):
        if not isinstance(value, str) or not re.fullmatch(r'\d+(?:[.-]\d+)*', value):
            return None
        values = [int(v) for v in re.split(r'[.-]', value)]
        while len(values) > 1 and values[-1] == 0:
            values.pop()
        return values
    a, b = parts(left), parts(right)
    return a is not None and b is not None and a == b


def _parse_page(html, ref, page_url):
    soup = BeautifulSoup(html, 'html.parser')
    title = soup.title.get_text(' ', strip=True) if soup.title else ''
    headings = [h.get_text(' ', strip=True).split(':', 1)[0] for h in soup.select('h1,h2')]
    if not (ref.name in headings or re.fullmatch(rf'CRAN\s*:\s*Package\s+{re.escape(ref.name)}', title, re.I)):
        return None
    fields, rows = {}, {}
    for tr in soup.select('tr'):
        cells = tr.find_all(['td', 'th'], recursive=False)
        if len(cells) == 2:
            key = cells[0].get_text(' ', strip=True).rstrip(':')
            fields[key] = cells[1].get_text(' ', strip=True)
            rows[key] = cells[1]
    if not fields.get('Version'):
        return None
    fields.setdefault('License', '')
    full_text = soup.get_text(' ', strip=True)
    release = None
    if ref.registry == 'bioconductor':
        match = re.search(r'Bioconductor version:\s*(?:(?:Release|Development|Devel)\s*\()?([0-9]+\.[0-9]+)', full_text)
        if match:
            release = match.group(1)
        else:
            match = re.match(r'/packages/(\d+\.\d+)/', urlparse(page_url).path)
            release = match.group(1) if match else None
        if not release:
            return None
    # Only package fields go to the LLM; site-wide footer copyright is excluded.
    evidence = [f'Package: {ref.name}', f'Version: {fields["Version"]}', f'License: {fields["License"]}']
    for key in ('Author', 'Authors', 'Maintainer', 'Copyright'):
        if fields.get(key):
            evidence.append(f'{key}: {fields[key]}')
    references = re.findall(r'\bfile\s+([^\s+|]+)', fields['License'], re.I)
    links = {}
    for key, cell in rows.items():
        if key not in {'License', 'Copyright', 'LICENSE', 'LICENCE', 'COPYRIGHTS'}:
            continue
        for a in cell.select('a[href]'):
            target = urljoin(page_url, a['href'])
            filename = unquote(urlparse(target).path).rsplit('/', 1)[-1]
            if filename in references or filename.lower().startswith(('license', 'licence', 'copying', 'copyright')):
                links[filename] = target
    # Bioconductor lists standalone LICENSE files in its Documentation table.
    for a in soup.select('a[href]'):
        target = urljoin(page_url, a['href'])
        filename = unquote(urlparse(target).path).rsplit('/', 1)[-1]
        if filename in references:
            links[filename] = target
    histories = []
    if ref.registry == 'bioconductor':
        pattern = rf'/packages/(\d+\.\d+)/{re.escape(ref.section)}/html/{re.escape(ref.name)}\.html'
        for a in soup.select('a[href]'):
            target = urlparse(urljoin(page_url, a['href']))
            match = re.fullmatch(pattern, target.path)
            if target.hostname == 'bioconductor.org' and match and match.group(1) not in histories:
                histories.append(match.group(1))
    source_url = None
    for a in soup.select('a[href]'):
        target = urljoin(page_url, a['href'])
        source = parse_r_reference(target)
        if (source and source.version and source.registry == ref.registry
                and source.name == ref.name and _same_version(source.version, fields['Version'])
                and (ref.registry == 'cran' or source.release == release)):
            source_url = target
            break
    return dict(version=fields['Version'], license=fields['License'], evidence='\n'.join(evidence),
                release=release, references=references, links=links, histories=histories, source_url=source_url)


async def _fetch_text(session, url):
    async with session.get(url, allow_redirects=True) as response:
        response.raise_for_status()
        final = str(response.url)
        parsed = urlparse(final)
        if parsed.scheme != 'https' or parsed.hostname not in CRAN_HOSTS | {'bioconductor.org', 'www.bioconductor.org'}:
            raise ValueError('R evidence redirected outside official registries')
        chunks, size = [], 0
        async for chunk in response.content.iter_chunked(65536):
            size += len(chunk)
            if size > MAX_PAGE_BYTES:
                raise ValueError('R evidence page exceeds size limit')
            chunks.append(chunk)
        return final, b''.join(chunks).decode(response.charset or 'utf-8', errors='replace')


def _same_package_file(url, ref, release):
    parsed = urlparse(url)
    path = unquote(parsed.path)
    if ref.registry == 'cran':
        prefix = f'/web/packages/{ref.name}/'
        valid_host = parsed.hostname in CRAN_HOSTS
    else:
        prefix = f'/packages/{release}/{ref.section}/licenses/{ref.name}/'
        valid_host = parsed.hostname in {'bioconductor.org', 'www.bioconductor.org'}
    tail = path.removeprefix(prefix)
    return (parsed.scheme == 'https' and valid_host and path.startswith(prefix)
            and bool(tail) and '/' not in tail and tail not in {'.', '..'})


def _description_fields(text):
    fields, key = {}, None
    for line in text.splitlines():
        if line[:1].isspace() and key:
            fields[key] += ' ' + line.strip()
        elif ':' in line:
            key, value = line.split(':', 1)
            fields[key] = value.strip()
        else:
            key = None
    return fields


def _cran_description_page(description, ref):
    fields = _description_fields(description)
    if (fields.get('Package') != ref.name or not fields.get('License')
            or not _same_version(fields.get('Version'), fields.get('Version'))):
        return None
    references = re.findall(r'\bfile\s+([^\s+|]+)', fields['License'], re.I)
    return dict(version=fields['Version'], license=fields['License'],
                evidence=f'DESCRIPTION:\n{description}', release=None, histories=[],
                references=references,
                links={filename:urljoin(ref.page_url(),quote(filename)) for filename in references},
                source_url=f'https://cran.r-project.org/src/contrib/{ref.name}_{fields["Version"]}.tar.gz')


async def process_r_package(url, version=None, name=None, package_manager=None):
    """Return verified web evidence or a normal miss for the shared archive fallback.

    A historical version never borrows the current page's licence. Bioconductor
    history lookup follows only release links published on that package's page.
    No repository discovery, archive download, or recursive audit is performed.
    """
    ref = parse_r_reference(url, package_manager, name)
    requested = str(version).strip() if version is not None else ''
    requested = '' if requested.lower() in {'', 'default', 'latest', 'nan', 'none'} else requested
    requested = requested or (ref.version if ref else '')
    result = dict(input_url=url, input_version=version, component_name=ref.name if ref else name,
                  resolved_version='', used_default_branch=False, status='error', license_type=None,
                  license_file_license=None, readme_license=None, license_files=None,
                  license_determination_reason='R registry web evidence unavailable; shared archive fallback may apply')
    if not ref:
        return result
    if requested and ref.version and not _same_version(requested, ref.version):
        result['license_determination_reason'] = 'R input version conflicts with archive URL version'
        return result
    try:
        async with asyncio.timeout(60), aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as session:
            page = None
            cran_description_verified = False
            if ref.registry == 'cran':
                # This standalone text preserves the package's original >= qualifiers.
                target = urljoin(ref.page_url(), 'DESCRIPTION')
                try:
                    page_url, description = await _fetch_text(session, target)
                except (aiohttp.ClientError, TimeoutError, ValueError):
                    description = None
                if description:
                    if not _same_package_file(page_url, ref, None):
                        return result
                    page = _cran_description_page(description, ref)
                    if not page:
                        return result
                    cran_description_verified = True
            if page is None:
                page_url, html = await _fetch_text(session, ref.page_url())
                page = _parse_page(html, ref, page_url)
            if not page:
                return result
            if requested and not _same_version(requested, page['version']):
                if ref.registry != 'bioconductor':
                    return result
                # A numeric release URL is an explicit boundary; aliases may use published history.
                histories = page['histories'] if ref.release in {'release', 'devel'} else []
                for release in histories:
                    try:
                        candidate_url, candidate_html = await _fetch_text(session, ref.page_url(release))
                        candidate = _parse_page(candidate_html, ref, candidate_url)
                        if candidate and _same_version(requested, candidate['version']):
                            page_url, page = candidate_url, candidate
                            break
                    except (aiohttp.ClientError, TimeoutError, ValueError):
                        continue
                else:
                    return result
            if ref.registry == 'bioconductor':
                # Pin release/devel aliases and recheck before combining linked licence evidence.
                pinned = ref.page_url(page['release'])
                if page_url != pinned:
                    _, pinned_html = await _fetch_text(session, pinned)
                    pinned_page = _parse_page(pinned_html, ref, pinned)
                    if not pinned_page or pinned_page['version'] != page['version'] or pinned_page['license'] != page['license']:
                        return result
                    page_url, page = pinned, pinned_page
            # Page inputs use the same shared downloader if web analysis later fails.
            # Source coordinates come from the verified page or CRAN DESCRIPTION.
            if page['source_url']:
                result['_r_archive_fallback_url'] = page['source_url']
            # The CRAN index expands licence ranges into today's known versions.
            # Without original DESCRIPTION, use its source coordinates only.
            if ref.registry == 'cran' and not cran_description_verified:
                return result
            if not page['license']:
                return result
            parts = [page['evidence']]
            copyright_parts = []
            for filename in dict.fromkeys([*page['references'], *page['links']]):
                target = page['links'].get(filename)
                if not target or not _same_package_file(target, ref, page['release']):
                    if filename in page['references']:
                        return result
                    continue
                try:
                    final, content = await _fetch_text(session, target)
                    if not _same_package_file(final, ref, page['release']) or not content.strip() or '<html' in content[:200].lower():
                        raise ValueError('Invalid R licence file response')
                    if filename in page['references'] or filename.lower().startswith(('license', 'licence', 'copying')):
                        parts.append(f'{filename}:\n{content}')
                    else:
                        copyright_parts.append(f'{filename}:\n{content}')
                except (aiohttp.ClientError, TimeoutError, ValueError):
                    if filename in page['references']:
                        return result
            evidence = '\n\n'.join(parts)
        analysis = await analyze_license_content_async(evidence, page_url)
        if not analysis or not analysis.get('licenses'):
            return result
        expression = analysis.get('spdx_expression') or analysis['licenses'][0]
        notice = await construct_copyright_notice_async(
            str(datetime.now().year), ref.name, ref.name, page['version'], ref.name,
            license_content='\n\n'.join([evidence, *copyright_parts]))
        result.update(status='success', repo_url=page_url, resolved_version=page['version'], license_files=page_url,
                      license_type=expression, license_file_license=expression,
                      license_analysis=analysis, has_license_conflict=False,
                      copyright_notice=notice, license_text=prepare_license_text(evidence),
                      license_determination_reason='Verified R registry package/version and license evidence on official web pages')
        return result
    except Exception as exc:
        logger.info('[R] Official web evidence unavailable (%s); returning to shared fallback', type(exc).__name__)
        return result
