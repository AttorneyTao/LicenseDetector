"""Revision-pinned Hugging Face evidence, without repository discovery or weights."""
import asyncio
from dataclasses import dataclass
import ipaddress
import json
import logging
import os
from pathlib import PurePosixPath
import re
from urllib.parse import quote, unquote, urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup
import yaml

from .utils import analyze_license_content_async, extract_copyright_info_async, is_blank_value, prepare_license_text

logger = logging.getLogger('main')
BASE = 'https://huggingface.co'
MAX_BYTES = 2_000_000
MAX_METADATA_BYTES = 12_000_000
MAX_EVIDENCE_FILES = 8
TEXT_MEDIA_TYPES = {'text/plain', 'text/markdown', 'text/html', 'text/x-rst',
                    'text/yaml', 'text/x-yaml', 'application/json',
                    'application/ld+json', 'application/yaml', 'application/x-yaml'}
RESERVED = {'spaces', 'models', 'datasets', 'docs', 'collections', 'organizations',
            'api', 'settings', 'join', 'login', 'blog', 'papers', 'tasks'}


@dataclass(frozen=True)
class HFReference:
    kind: str
    repo: str
    revision: str = ''
    path: str = ''
    file: bool = False

    @property
    def page(self):
        return BASE + ('/datasets/' if self.kind == 'dataset' else '/') + self.repo

    def blob(self, sha, path):
        return f'{self.page}/blob/{sha}/{quote(path, safe="/")}'

    def raw(self, sha, path):
        return f'{self.page}/resolve/{sha}/{quote(path, safe="/")}'


def is_huggingface_url(url):
    try:
        return isinstance(url, str) and urlparse(url).hostname == 'huggingface.co'
    except ValueError:
        return False


def parse_huggingface_reference(url):
    if not isinstance(url, str):
        return None
    try:
        parsed = urlparse(url)
        if parsed.scheme != 'https' or parsed.netloc != 'huggingface.co':
            return None
        parts = parsed.path.strip('/').split('/')
        kind = 'dataset' if parts[0] == 'datasets' else 'model'
        if kind == 'dataset':
            parts = parts[1:]
        if len(parts) < 2 or parts[0] in RESERVED:
            return None
        if not all(re.fullmatch(r'[\w.-]+', p, flags=re.ASCII) and p not in {'.', '..'} for p in parts[:2]):
            return None
        repo = '/'.join(parts[:2])
        if len(parts) == 2:
            return HFReference(kind, repo)
        if len(parts) < 4 or parts[2] not in {'tree', 'blob', 'resolve', 'raw'}:
            return None
        rev = unquote(parts[3])
        path = unquote('/'.join(parts[4:]))
        if not rev or any(p in {'.', '..', ''} for p in rev.split('/')):
            return None
        if path and any(p in {'.', '..', ''} for p in path.split('/')):
            return None
        file = parts[2] != 'tree'
        if file and not path:
            return None
        return HFReference(kind, repo, rev, path, file)
    except (TypeError, ValueError, IndexError):
        return None


async def _fetch(session, url, *, metadata=False, external=False):
    """Bounded reads; never forward HF credentials to an external license host."""
    for _ in range(4):
        parsed = urlparse(url)
        if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError('Invalid evidence URL')
        if parsed.hostname != 'huggingface.co' and not external:
            raise ValueError('Unexpected HF redirect')
        if unquote(parsed.path).lower().endswith(('.safetensors', '.bin', '.gguf', '.pt', '.pth', '.ckpt', '.onnx', '.parquet', '.zip', '.gz', '.xz', '.bz2', '.tar', '.conda')):
            raise ValueError('Weight/data/archive downloads are outside the HF evidence scope')
        if parsed.hostname == 'localhost' or parsed.hostname.endswith(('.local', '.internal')):
            raise ValueError('Nonpublic evidence host')
        try:
            address = ipaddress.ip_address(parsed.hostname)
        except ValueError:
            address = None
        if address is not None and not address.is_global:
            raise ValueError('Nonpublic evidence address')
        headers = {'User-Agent': 'LicenseDetector/HuggingFace-license-audit'}
        if parsed.hostname == 'huggingface.co' and os.getenv('HF_TOKEN'):
            headers['Authorization'] = 'Bearer ' + os.environ['HF_TOKEN']
        async with session.get(url, headers=headers, allow_redirects=False) as response:
            if response.status in {301, 302, 303, 307, 308}:
                url = urljoin(url, response.headers.get('Location', ''))
                continue
            response.raise_for_status()
            media_type = response.headers.get('Content-Type', '').split(';')[0].strip().lower()
            if media_type not in ({'application/json'} if metadata else TEXT_MEDIA_TYPES):
                raise ValueError('Non-text HF evidence response rejected before reading body')
            limit = MAX_METADATA_BYTES if metadata else MAX_BYTES
            length = response.headers.get('Content-Length')
            if length is not None and int(length) > limit:
                raise ValueError('HF evidence Content-Length exceeds read limit')
            chunks, size = [], 0
            async for chunk in response.content.iter_chunked(65536):
                size += len(chunk)
                if size > limit:
                    raise ValueError('HF evidence exceeds read limit')
                chunks.append(chunk)
            text = b''.join(chunks).decode('utf-8')
            logger.info('[HF] Read %s evidence: %s (%s bytes)', 'metadata' if metadata else 'text', parsed.path, size)
            if metadata:
                data = json.loads(text)
                data['_hf_response_url'] = url
                return data
            return text
    raise ValueError('Too many evidence redirects')


def _card(text):
    match = re.match(r'\A---\s*\n(.*?)\n---(?:\s*\n|$)', text, re.S)
    data = yaml.safe_load(match.group(1)) if match else {}
    return data if isinstance(data, dict) else {}


def _evidence_files(ref, files):
    parent = PurePosixPath(ref.path).parent if ref.file else PurePosixPath(ref.path)
    directories = [str(p) for p in [parent, *parent.parents]]
    directories = list(dict.fromkeys('' if p == '.' else p for p in directories))
    candidates = []
    for path in files:
        p = PurePosixPath(path)
        directory = '' if str(p.parent) == '.' else str(p.parent)
        if directory in directories and p.name.lower().startswith(('license', 'licence', 'copying', 'notice', 'copyright', 'readme')):
            candidates.append(path)
    return sorted(candidates, key=lambda p: (
        directories.index('' if str(PurePosixPath(p).parent) == '.' else str(PurePosixPath(p).parent)),
        PurePosixPath(p).name.lower().startswith('readme'), p,
    ))[:MAX_EVIDENCE_FILES]


async def process_huggingface_repository(url, version=None, name=None):
    ref = parse_huggingface_reference(url)
    result = dict(input_url=url, input_version=version, component_name=ref.repo if ref else name,
                  status='error', license_type=None, license_file_license=None, readme_license=None,
                  license_files=None, used_default_branch=False, resolved_version='',
                  license_determination_reason='Hugging Face evidence unavailable; no cross-ecosystem fallback')
    if not ref:
        result['error'] = 'Unsupported Hugging Face model/dataset URL'
        return result
    requested = '' if is_blank_value(version) or str(version).lower() in {'latest', 'default'} else str(version).strip()
    revision = ref.revision or requested or 'main'
    reasons, failures = [], []
    if ref.revision and requested and ref.revision != requested:
        reasons.append('URL revision takes precedence over version column')
    api = f'{BASE}/api/{ref.kind}s/{ref.repo}/revision/'
    try:
        async with asyncio.timeout(120), aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            try:
                info = await _fetch(session, api + quote(revision, safe=''), metadata=True)
            except aiohttp.ClientResponseError as exc:
                if exc.status != 404 or revision == 'main':
                    raise
                reasons.append(f'Requested revision {revision} unavailable; same repository main used')
                result['used_default_branch'] = True
                info = await _fetch(session, api + 'main', metadata=True)
            canonical = info.get('id', '')
            if str(canonical).casefold() != ref.repo.casefold():
                final = urlparse(info.get('_hf_response_url', ''))
                if (final.scheme != 'https' or final.netloc != 'huggingface.co'
                        or not final.path.startswith(f'/api/{ref.kind}s/{canonical}/revision/')
                        or not re.fullmatch(r'[\w.-]+/[\w.-]+', canonical, flags=re.ASCII)):
                    raise ValueError('HF repository identity mismatch')
                reasons.append(f'Official HF redirect: {ref.repo} -> {canonical}')
                ref = HFReference(ref.kind, canonical, ref.revision, ref.path, ref.file)
                result['component_name'] = canonical
            sha = info.get('sha', '')
            if not re.fullmatch(r'[0-9a-f]{40}', sha):
                raise ValueError('HF did not return a full commit SHA')
            result.update(repo_url=ref.page, resolved_version=sha)
            files = [x['rfilename'] for x in info.get('siblings', []) if isinstance(x.get('rfilename'), str)]
            if ref.file and ref.path not in files:
                raise ValueError(f'Input file not found at selected commit: {ref.path}')
            if ref.path and not ref.file and not any(p.startswith(ref.path.rstrip('/') + '/') for p in files):
                raise ValueError('Input directory not found at selected commit')
            evidence = []
            for path in _evidence_files(ref, files):
                try:
                    text = await _fetch(session, ref.raw(sha, path))
                    if not text.strip() or '<html' in text[:200].lower():
                        raise ValueError('Invalid HF text evidence')
                    evidence.append((path, text, ref.blob(sha, path)))
                except (aiohttp.ClientError, TimeoutError, ValueError) as exc:
                    failures.append(f'{path}: {type(exc).__name__}')
            root_card = next((text for path, text, _ in evidence if path.lower() == 'readme.md'), '')
            try:
                card = _card(root_card)
            except yaml.YAMLError:
                card = {}
                failures.append('README metadata: invalid YAML; text evidence retained')
            declaration = card.get('license')
            result['metadata_licenses'] = json.dumps(declaration, ensure_ascii=False) if declaration else ''
            link = card.get('license_link')
            if isinstance(link, str) and link.strip():
                linked = urljoin(ref.page + '/', link.strip())
                same = parse_huggingface_reference(linked)
                if same and same.kind == ref.kind and same.repo.casefold() == ref.repo.casefold() and same.file:
                    linked = ref.raw(sha, same.path)
                    display = ref.blob(sha, same.path)
                elif urlparse(linked).hostname == 'huggingface.co':
                    # Relative file paths such as LICENSE must stay inside this snapshot.
                    if not urlparse(link).scheme and link.strip().lstrip('./') in files:
                        path = link.strip().lstrip('./')
                        linked, display = ref.raw(sha, path), ref.blob(sha, path)
                    else:
                        linked, display = '', ''
                else:
                    display = linked
                if linked and not any(item[2] == display for item in evidence):
                    try:
                        text = await _fetch(session, linked, external=True)
                        if '<html' in text[:300].lower():
                            soup = BeautifulSoup(text, 'html.parser')
                            text = (soup.find('main') or soup.find('article') or soup).get_text(' ', strip=True)
                        if text.strip():
                            evidence.append(('Declared license_link', text, display))
                    except (aiohttp.ClientError, TimeoutError, ValueError) as exc:
                        failures.append(f'license_link: {type(exc).__name__}')
            if not evidence:
                raise ValueError('No readable HF license/card evidence')
        primary = next((item for item in evidence if not PurePosixPath(item[0]).name.lower().startswith(('readme', 'notice', 'copyright'))), evidence[0])
        content = (f'Hugging Face {ref.kind}: {ref.repo}\nCommit: {sha}\nInput path: {ref.path or "repository"}\n'
                   'Audit the input artifact only. Preserve the publisher declaration and explicit scope. '
                   'Do not substitute licenses of linked code, base models, training datasets or examples. '
                   'Do not infer AND/OR from a license list without an explicit relationship.\n\n' +
                   '\n\n'.join(f'{path} ({source}):\n{text}' for path, text, source in evidence))
        # Reuse the existing model-based license and copyright extraction unchanged.
        analysis = await analyze_license_content_async(content, primary[2])
        expression = (analysis or {}).get('spdx_expression') or next(iter((analysis or {}).get('licenses') or []), None)
        if not expression or expression.upper() in {'UNKNOWN', 'NONE', 'NOASSERTION', 'UNLICENSED'}:
            raise ValueError('No explicit HF license identified from the selected evidence')
        notice = await extract_copyright_info_async(content)
        analysis.update(huggingface_repo_type=ref.kind, huggingface_commit=sha,
                        huggingface_input_path=ref.path, huggingface_license_declaration=declaration,
                        huggingface_download_policy='Metadata and text evidence only; no weights or archives',
                        huggingface_evidence_urls=[v[2] for v in evidence],
                        huggingface_evidence_failures=failures)
        reasons.insert(0, 'Hugging Face repository/file identity verified; license evidence pinned to commit ' + sha)
        if failures:
            reasons.append('Some evidence was inaccessible: ' + '; '.join(failures))
        result.update(status='success', error=None, license_type=expression, license_file_license=expression,
                      license_files=primary[2], license_analysis=analysis,
                      has_license_conflict=bool(analysis.get('has_license_conflict', False)),
                      copyright_notice=notice, license_text=prepare_license_text(content),
                      license_determination_reason='; '.join(reasons))
    except Exception as exc:
        result['error'] = f'Hugging Face audit unavailable: {type(exc).__name__}: {exc}'
        result['license_determination_reason'] += '; ' + '; '.join(reasons)
        logger.warning('[HF] %s (%s)', ref.repo, result['error'])
    return result
