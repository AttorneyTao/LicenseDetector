"""HF identity, pinned evidence and absence of cross-ecosystem fallbacks."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import aiohttp
import pandas as pd
import pytest

from core import huggingface_utils as h

SHA = 'a' * 40
PAGE = 'https://huggingface.co/org/demo'
FILE = PAGE + '/resolve/main/sub/model.safetensors'
CARD = '---\nlicense: mit\n---\nModel license: MIT.\nLinked code example: BSD-3-Clause.'


@pytest.mark.parametrize('url,kind,repo,revision,path,file', [
    (PAGE, 'model', 'org/demo', '', '', False),
    (PAGE + '/', 'model', 'org/demo', '', '', False),
    (FILE, 'model', 'org/demo', 'main', 'sub/model.safetensors', True),
    (PAGE + '/blob/v1/LICENSE', 'model', 'org/demo', 'v1', 'LICENSE', True),
    (PAGE + '/raw/' + SHA + '/README.md', 'model', 'org/demo', SHA, 'README.md', True),
    (PAGE + '/tree/v1/sub', 'model', 'org/demo', 'v1', 'sub', False),
    (PAGE + '/tree/refs%2Fpr%2F3/sub', 'model', 'org/demo', 'refs/pr/3', 'sub', False),
    ('https://huggingface.co/datasets/org/demo', 'dataset', 'org/demo', '', '', False),
    ('https://huggingface.co/datasets/org/demo/resolve/v2/data/file.parquet', 'dataset', 'org/demo', 'v2', 'data/file.parquet', True),
])
def test_reference(url, kind, repo, revision, path, file):
    assert h.parse_huggingface_reference(url) == h.HFReference(kind, repo, revision, path, file)


@pytest.mark.parametrize('url', [None, float('nan'), 'https://[broken',
    'http://huggingface.co/org/demo', 'https://huggingface.co.evil/org/demo',
    'https://github.com/org/demo', 'https://pypi.org/project/demo',
    'https://huggingface.co/org', 'https://huggingface.co/spaces/org/demo',
    'https://huggingface.co/collections/org/demo', 'https://huggingface.co/docs/hub',
    PAGE + '/discussions', PAGE + '/blob/main', PAGE + '/resolve/main/sub/../LICENSE',
    PAGE + '/resolve/main/%2e%2e/LICENSE', PAGE + '/tree/%2e%2e',
    'https://user@huggingface.co/org/demo', 'https://huggingface.co/org/demo/extra'])
def test_invalid_reference(url):
    assert h.parse_huggingface_reference(url) is None


@pytest.fixture
def isolated(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Unexpected external request')
    monkeypatch.setattr(aiohttp.ClientSession, '_request', forbidden)
    fetch = AsyncMock()
    analysis = AsyncMock(return_value={'licenses': ['MIT'], 'spdx_expression': 'MIT'})
    notice = AsyncMock(return_value='Copyright (c) Actual Authors')
    monkeypatch.setattr(h, '_fetch', fetch)
    monkeypatch.setattr(h, 'analyze_license_content_async', analysis)
    monkeypatch.setattr(h, 'extract_copyright_info_async', notice)
    return fetch, analysis, notice


def serve(fetch, *, repo='org/demo', kind='model', files=None, card=CARD, error=None, sha=SHA,
          metadata_extra=None):
    info = dict(id=repo, sha=sha, siblings=[{'rfilename': x} for x in (
        files if files is not None else ['README.md', 'LICENSE', 'sub/model.safetensors'])])
    info.update(metadata_extra or {})
    async def get(session, url, **kwargs):
        if '/api/' in url:
            if error:
                raise aiohttp.ClientResponseError(SimpleNamespace(real_url=url), (), status=error)
            return deepcopy(info)
        assert '/resolve/' + SHA + '/' in url, 'Evidence must use the selected SHA'
        if url.endswith('/README.md'):
            return card
        if url.endswith('/LICENSE'):
            return 'MIT License\nCopyright (c) Actual Authors'
        raise AssertionError('Unexpected evidence URL: ' + url)
    fetch.side_effect = get


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['model', 'dataset'])
@pytest.mark.parametrize('version', [None, 'v1', SHA])
async def test_pinned_evidence(isolated, kind, version):
    fetch, analyze, notice = isolated
    serve(fetch, kind=kind)
    page = PAGE.replace('/org/', '/datasets/org/') if kind == 'dataset' else PAGE
    result = await h.process_huggingface_repository(page, version, 'input alias')
    assert result['status'] == 'success'
    assert result['resolved_version'] == SHA
    assert result['license_files'] == page + '/blob/' + SHA + '/LICENSE'
    assert result['license_type'] == 'MIT'
    assert result['copyright_notice'] == 'Copyright (c) Actual Authors'
    assert result['license_analysis']['huggingface_repo_type'] == kind
    assert result['metadata_licenses'] == '"mit"'
    assert all('model.safetensors' not in call.args[1] for call in fetch.await_args_list)
    assert 'Commit: ' + SHA in analyze.call_args.args[0]
    notice.assert_awaited_once()


@pytest.mark.asyncio
async def test_card_only_and_url_revision_precedence(isolated):
    fetch, _, _ = isolated
    serve(fetch, files=['README.md', 'sub/model.safetensors'])
    result = await h.process_huggingface_repository(FILE, 'different-tag')
    assert result['status'] == 'success'
    assert result['license_files'] == PAGE + '/blob/' + SHA + '/README.md'
    assert 'takes precedence' in result['license_determination_reason']
    assert fetch.call_args_list[0].args[1].endswith('/revision/main')


@pytest.mark.asyncio
async def test_missing_revision_uses_only_same_repo_main(isolated):
    fetch, _, _ = isolated
    serve(fetch)
    original = fetch.side_effect
    async def get(session, url, **kwargs):
        if url.endswith('/revision/absent'):
            raise aiohttp.ClientResponseError(SimpleNamespace(real_url=url), (), status=404)
        return await original(session, url, **kwargs)
    fetch.side_effect = get
    result = await h.process_huggingface_repository(PAGE, 'absent')
    assert result['status'] == 'success'
    assert result['used_default_branch'] is True
    assert 'same repository main' in result['license_determination_reason']


@pytest.mark.asyncio
@pytest.mark.parametrize('status', [401, 403, 404, 429, 500])
async def test_http_failures_are_not_unlicensed(isolated, status):
    fetch, analyze, _ = isolated
    serve(fetch, error=status)
    result = await h.process_huggingface_repository(PAGE)
    assert result['status'] == 'error'
    assert result['license_type'] is None
    analyze.assert_not_awaited()
    assert fetch.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('case', ['missing_file', 'missing_directory', 'wrong_repo', 'bad_sha', 'empty_files', 'empty_model'])
async def test_bad_identity_or_missing_evidence(isolated, case):
    fetch, analyze, _ = isolated
    serve(fetch, files=[] if case == 'empty_files' else None,
          repo='other/repo' if case == 'wrong_repo' else 'org/demo',
          sha='bad' if case == 'bad_sha' else SHA)
    url = (PAGE + '/resolve/main/absent.safetensors' if case == 'missing_file' else
           PAGE + '/tree/main/absent' if case == 'missing_directory' else PAGE)
    if case == 'empty_model':
        analyze.return_value = {'licenses': []}
    result = await h.process_huggingface_repository(url)
    assert result['status'] == 'error'
    assert result['license_type'] is None
    if case != 'empty_model':
        analyze.assert_not_awaited()


@pytest.mark.asyncio
async def test_official_redirect_required_for_alias(isolated):
    fetch, _, _ = isolated
    canonical = 'org/renamed'
    serve(fetch, repo=canonical, metadata_extra={
        '_hf_response_url': 'https://huggingface.co/api/models/org/renamed/revision/main'})
    result = await h.process_huggingface_repository(PAGE)
    assert result['status'] == 'success'
    assert result['repo_url'].endswith('/org/renamed')
    assert 'Official HF redirect' in result['license_determination_reason']


@pytest.mark.asyncio
@pytest.mark.parametrize('license_value', ['other', ['mit', 'apache-2.0']])
async def test_custom_or_list_declaration_preserved(isolated, license_value):
    fetch, analyze, _ = isolated
    import yaml
    card = '---\n' + yaml.safe_dump({'license': license_value}) + '---\nExplicit custom terms or choices.'
    serve(fetch, card=card, files=['README.md'])
    analyze.return_value = {'licenses': ['LicenseRef-Custom'], 'spdx_expression': 'LicenseRef-Custom', 'has_license_conflict': True}
    result = await h.process_huggingface_repository(PAGE)
    assert result['license_type'] == 'LicenseRef-Custom'
    assert result['has_license_conflict'] is True
    assert result['license_analysis']['huggingface_license_declaration'] == license_value


@pytest.mark.asyncio
@pytest.mark.parametrize('entry', ['cli', 'api'])
@pytest.mark.parametrize('mode', ['success', 'error', 'exception', 'unsupported'])
async def test_exclusive_dispatch_and_original_metadata(monkeypatch, tmp_path, entry, mode):
    import main
    import api
    module = main if entry == 'cli' else api
    monkeypatch.chdir(tmp_path)
    url = 'https://huggingface.co/org/demo/resolve/main/model.zip'
    if mode == 'unsupported':
        url = 'https://huggingface.co/spaces/org/demo'
    result = dict(status='success' if mode == 'success' else 'error', license_type='MIT' if mode == 'success' else None,
                  license_files=PAGE + '/blob/' + SHA + '/LICENSE', resolved_version=SHA)
    hf = AsyncMock(return_value=result)
    if mode == 'exception':
        hf.side_effect = RuntimeError('network failure')
    monkeypatch.setattr(module, 'process_huggingface_repository', hf)
    boundaries = []
    for target in ['process_github_repository', 'process_direct_archive_url', 'retry_with_purl', 'prefer_license_webpage', 'process_r_package', 'process_conda_package']:
        mock = AsyncMock(side_effect=AssertionError('Unexpected fallback: ' + target))
        monkeypatch.setattr(module, target, mock)
        boundaries.append(mock)
    # A deliberately wrong manager must not send failed HF input to PyPI.
    rows = pd.DataFrame([dict(name='original filename', version='main', github_url=url,
                              package_manager='pypi', serial_number='00001')])
    results = await (main.process_all_repos(None, rows) if entry == 'cli' else api._process_repositories(None, rows))
    assert results[0]['input_url'] == url
    assert results[0]['input_name'] == 'original filename'
    assert results[0]['serial_number'] == '00001'
    for mock in boundaries:
        mock.assert_not_awaited()
    hf.assert_awaited_once_with(url, 'main', 'original filename')


@pytest.mark.asyncio
@pytest.mark.parametrize('url', [PAGE + '/resolve/main/model.safetensors',
    'https://localhost/LICENSE', 'https://127.0.0.1/LICENSE',
    'http://example.org/LICENSE', 'https://example.org/model.tar.gz'])
async def test_fetch_rejects_weights_or_invalid_evidence(url):
    session = AsyncMock()
    with pytest.raises(ValueError):
        await h._fetch(session, url, external=True)
    session.get.assert_not_called()


class Response:
    def __init__(self, body=b'License text', headers=None, status=200):
        self.headers = headers or {'Content-Type': 'text/plain'}
        self.status, self.body, self.reads = status, body, 0
        self.content = self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def raise_for_status(self):
        pass

    async def iter_chunked(self, size):
        self.reads += 1
        yield self.body


@pytest.mark.asyncio
@pytest.mark.parametrize('headers', [
    {'Content-Type': 'application/octet-stream'},
    {'Content-Type': 'application/x-safetensors'},
    {'Content-Type': 'text/plain', 'Content-Length': str(10**12)},
    {},
])
async def test_binary_or_oversized_response_body_is_never_read(headers):
    response = Response(headers={'placeholder': 'x'} if not headers else headers)
    session = SimpleNamespace(get=lambda *args, **kwargs: response)
    with pytest.raises(ValueError):
        await h._fetch(session, PAGE + '/resolve/' + SHA + '/LICENSE')
    assert response.reads == 0


@pytest.mark.asyncio
async def test_chunked_text_has_size_limit(monkeypatch):
    monkeypatch.setattr(h, 'MAX_BYTES', 4)
    response = Response(body=b'12345')
    session = SimpleNamespace(get=lambda *args, **kwargs: response)
    with pytest.raises(ValueError, match='read limit'):
        await h._fetch(session, PAGE + '/resolve/' + SHA + '/LICENSE')


@pytest.mark.asyncio
async def test_redirect_to_weights_is_not_requested():
    calls = []
    response = Response(headers={'Location': FILE}, status=302)
    def get(url, **kwargs):
        calls.append(url)
        return response
    with pytest.raises(ValueError, match='downloads are outside'):
        await h._fetch(SimpleNamespace(get=get), PAGE + '/resolve/' + SHA + '/LICENSE', external=True)
    assert len(calls) == 1
    assert response.reads == 0


@pytest.mark.asyncio
async def test_external_license_redirect_does_not_receive_hf_token(monkeypatch):
    monkeypatch.setenv('HF_TOKEN', 'test-token')
    calls = []
    redirect = Response(headers={'Location': 'https://example.org/LICENSE'}, status=302)
    final = Response()
    def get(url, **kwargs):
        calls.append((url, kwargs['headers']))
        return redirect if len(calls) == 1 else final
    assert await h._fetch(SimpleNamespace(get=get), PAGE + '/resolve/' + SHA + '/LICENSE', external=True) == 'License text'
    assert calls[0][1]['Authorization'] == 'Bearer test-token'
    assert 'Authorization' not in calls[1][1]


def test_api_upload_exports_hf_result(isolated, monkeypatch):
    import io
    from fastapi.testclient import TestClient
    import api
    serve(isolated[0])
    monkeypatch.setattr(api, 'get_github_api', AsyncMock(return_value=object()))
    stream = io.BytesIO()
    pd.DataFrame([dict(name='original model', github_url=FILE, version='main',
                       package_manager='huggingface', serial_number='00001')]).to_excel(stream, index=False)
    with TestClient(api.app) as client:
        response = client.post('/api/v1/analyze-and-download', files={'file': ('input.xlsx', stream.getvalue())})
    assert response.status_code == 200
    result = pd.read_excel(io.BytesIO(response.content), dtype=str, keep_default_na=False)
    assert result.iloc[0]['concluded_license'] == 'MIT'
    assert result.iloc[0]['serial_number'] == '00001'
    assert result.iloc[0]['license_files'] == PAGE + '/blob/' + SHA + '/LICENSE'
    assert result.iloc[0]['input_url'] == FILE


@pytest.mark.asyncio
async def test_cli_excel_exports_hf_result(isolated, monkeypatch, tmp_path):
    import main
    serve(isolated[0])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main, 'initialize_api', AsyncMock(return_value=object()))
    pd.DataFrame([dict(name='original model', github_url=FILE, version='main',
                       package_manager='huggingface', serial_number='00001')]).to_excel('input.xlsx', index=False)
    await main.main_async()
    result = pd.read_excel('outputs/output_latest.xlsx', dtype=str, keep_default_na=False)
    assert result.iloc[0]['concluded_license'] == 'MIT'
    assert result.iloc[0]['serial_number'] == '00001'
    assert result.iloc[0]['license_files'] == PAGE + '/blob/' + SHA + '/LICENSE'
    assert result.iloc[0]['input_url'] == FILE
