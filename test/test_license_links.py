"""Link selection boundaries: no audit rerun, no loss of existing evidence."""
import asyncio
import copy
from unittest.mock import AsyncMock

import httpx
import pandas as pd
import pytest

from core import license_links as links


TEXT = 'MIT License\nCopyright (c) 2026 Example Author\nPermission is hereby granted.'
ARCHIVE = 'https://github.com/example/demo/archive/refs/tags/v1.2.3.tar.gz'
RAW = 'https://raw.githubusercontent.com/example/demo/v1.2.3/LICENSE'
BLOB = 'https://github.com/example/demo/blob/v1.2.3/LICENSE'
PAGE = 'https://crates.io/crates/demo/1.2.3'


def row(**kwargs):
    return dict(dict(name='demo', version='1.2.3', package_manager='cargo', github_url=ARCHIVE,
                     serial_number='0001'), **kwargs)


def result(**kwargs):
    return dict(dict(status='success', license_type='MIT', license_file_license='MIT',
                     resolved_version='1.2.3', used_default_branch=False, license_files=ARCHIVE,
                     copyright_notice='Copyright Example', license_text=TEXT,
                     license_analysis={'licenses': ['MIT'], 'thirdparty_dirs': ['vendor']},
                     input_name='demo'), **kwargs)


def evidence(path='LICENSE', text=TEXT):
    return {'path': path, 'digest': links.content_digest(text)}


@pytest.fixture
def network(monkeypatch):
    calls = []
    responses = {}
    real_client = httpx.AsyncClient

    def handle(request):
        key = (request.method, str(request.url))
        calls.append(key)
        response = responses.get(key, responses.get(str(request.url), (404, '', {})))
        if isinstance(response, Exception):
            raise response
        code, body, headers = response
        return httpx.Response(code, text=body, headers={'content-type': 'text/plain', **headers})

    monkeypatch.setattr(links.httpx, 'AsyncClient', lambda **kwargs: real_client(
        **kwargs, transport=httpx.MockTransport(handle)))
    analyze = AsyncMock(return_value={'licenses': ['MIT'], 'spdx_expression': 'MIT'})
    monkeypatch.setattr(links, 'analyze_license_content_async', analyze)
    return responses, calls, analyze


@pytest.mark.parametrize('url', [
    ARCHIVE, 'https://host/pkg.crate', 'https://crates.io/api/v1/crates/demo/1/download',
    'https://codeload.github.com/a/b/tar.gz/v1', 'https://host/pkg.whl?x=1',
    'https://host/get?file=package-1.0.tar.gz', 'https://host/a.nupkg',
    'https://host/a.jar', 'https://host/a.deb', 'https://host/a.rpm',
])
def test_download_url_shapes(url):
    assert links.is_package_download(url)


@pytest.mark.parametrize('url', [None, '', 'https://host/LICENSE', 'https://host/COPYING',
    'https://host/1.0/demo-1.0.pom', 'https://host/copyright', PAGE])
def test_readable_evidence_is_not_archive(url):
    assert not links.is_package_download(url)


@pytest.mark.asyncio
@pytest.mark.parametrize('url', [BLOB, RAW, PAGE, 'https://github.com/a/b/blob/v1/README.md',
    'https://host/demo-1.2.3.pom', 'https://host/copyright',
    'https://pypi.org/project/demo/1.2.3', 'https://www.npmjs.com/package/demo/v/1.2.3',
    'https://pub.dev/packages/demo/versions/1.2.3', 'https://www.nuget.org/packages/demo/1.2.3',
    'https://pkg.go.dev/example.org/demo@v1.2.3?tab=licenses',
    'https://mvnrepository.com/artifact/org.example/demo/1.2.3'])
async def test_existing_evidence_does_not_trigger_lookup(network, url):
    original = result(license_files=url)
    assert await links.prefer_license_webpage(row(), original) is original
    assert network[1] == []
    network[2].assert_not_awaited()


@pytest.mark.asyncio
async def test_exact_archive_file_precedes_registry_and_preserves_all_fields(network):
    responses, calls, analyze = network
    responses[RAW] = (200, TEXT, {})
    responses[PAGE] = (200, 'demo 1.2.3 MIT', {})
    original = result(**{links.EVIDENCE_KEY: evidence()})
    snapshot = copy.deepcopy(original)
    actual = await links.prefer_license_webpage(row(), original)
    expected = {k: v for k, v in original.items() if k != links.EVIDENCE_KEY}
    expected['license_files'] = RAW
    assert actual == expected
    assert original == snapshot
    assert calls == [('GET', BLOB), ('GET', RAW)]
    analyze.assert_not_awaited()


@pytest.mark.asyncio
async def test_matching_browser_file_preferred_to_raw(network):
    network[0][BLOB] = (200, f'<textarea data-testid="read-only-cursor-text-area">{TEXT}</textarea>',
                        {'content-type': 'text/html'})
    actual = await links.prefer_license_webpage(row(), result(**{links.EVIDENCE_KEY: evidence()}))
    assert actual['license_files'] == BLOB
    assert network[1] == [('GET', BLOB)]


@pytest.mark.asyncio
async def test_registry_page_verified_with_existing_llm(network):
    network[0][PAGE] = (200, '<main>demo 1.2.3 licensed MIT</main>', {'content-type': 'text/html'})
    original = result()
    actual = await links.prefer_license_webpage(row(), original)
    assert actual == {**original, 'license_files': PAGE}
    network[2].assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize('body', ['Sign in', 'Error: not found', 'another 1.2.3 MIT', 'demo 9.9.9 MIT',
                                  'demo-plus 1.2.3 MIT', 'demo 1.2.30 MIT'])
async def test_http_200_is_not_evidence(network, body):
    network[0][PAGE] = (200, body, {})
    original = result()
    assert await links.prefer_license_webpage(row(), original) is original
    network[2].assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('expression', ['Apache-2.0', 'MIT OR GPL-2.0-only', '', None])
async def test_conflicting_or_absent_license_rejected(network, expression):
    network[0][PAGE] = (200, 'demo 1.2.3 license information', {})
    network[2].return_value = {'spdx_expression': expression, 'licenses': []}
    original = result()
    assert await links.prefer_license_webpage(row(), original) is original


@pytest.mark.asyncio
@pytest.mark.parametrize('destination', [
    'https://crates.io/crates/demo/9.9.9', 'https://crates.io/crates/another/1.2.3',
    'https://crates.io/api/v1/crates/demo/1.2.3/download', 'https://opensource.org/license/mit',
    'https://crates.io/login',
])
async def test_redirect_cannot_bypass_identity_or_download_checks(network, destination):
    network[0][PAGE] = (302, '', {'location': destination})
    network[0][destination] = (200, 'demo 1.2.3 MIT', {})
    original = result()
    assert await links.prefer_license_webpage(row(), original) is original
    network[2].assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('status', [403, 404, 429, 500])
async def test_network_failure_preserves_original(network, status):
    network[0][PAGE] = (status, 'demo 1.2.3 MIT', {})
    original = result()
    assert await links.prefer_license_webpage(row(), original) is original


@pytest.mark.asyncio
async def test_timeout_and_cancellation(network, monkeypatch):
    async def slow(*args):
        await asyncio.sleep(10)
    monkeypatch.setattr(links, '_fetch_page', slow)
    monkeypatch.setattr(links, 'LOOKUP_TIMEOUT', 0.01)
    original = result()
    assert await asyncio.wait_for(links.prefer_license_webpage(row(), original), 0.5) is original
    monkeypatch.setattr(links, '_fetch_page', AsyncMock(side_effect=asyncio.CancelledError()))
    with pytest.raises(asyncio.CancelledError):
        await links.prefer_license_webpage(row(), original)


@pytest.mark.asyncio
async def test_extensionless_redirect_or_headers_detected_without_downloading_archive(network):
    url = 'https://host/fetch/123'
    network[0][('HEAD', url)] = (302, '', {'location': 'https://host/package.tar.gz'})
    network[0][('HEAD', 'https://host/package.tar.gz')] = (200, '', {'content-type': 'application/gzip'})
    network[0][PAGE] = (200, 'demo 1.2.3 MIT', {})
    actual = await links.prefer_license_webpage(row(), result(license_files=url))
    assert actual['license_files'] == PAGE
    assert ('GET', 'https://host/package.tar.gz') not in network[1]


@pytest.mark.asyncio
@pytest.mark.parametrize('headers', [
    {'content-type': 'application/gzip'},
    {'content-disposition': 'attachment; filename="pkg.zip"'},
])
async def test_download_disguised_as_page_rejected(network, headers):
    network[0][PAGE] = (200, 'demo 1.2.3 MIT', headers)
    original = result()
    assert await links.prefer_license_webpage(row(), original) is original


@pytest.mark.asyncio
async def test_no_recursive_lookup_for_purl(network):
    network[0][PAGE] = (200, 'demo 1.2.3 MIT', {})
    actual = await links.prefer_license_webpage(row(github_url='pkg:cargo/demo@1.2.3'), result())
    assert actual['license_files'] == PAGE
    assert network[1] == [('GET', PAGE)]
    assert await links.prefer_license_webpage(row(), actual) is actual
    assert network[1] == [('GET', PAGE)]


@pytest.mark.asyncio
@pytest.mark.parametrize('changes', [dict(package_manager=None), dict(name=None), dict(version='9.9.9'),
                                     dict(package_manager='unsupported'), dict(package_manager=pd.NA)])
async def test_missing_or_malformed_coordinates_fail_closed(network, changes):
    original = result()
    assert await links.prefer_license_webpage(row(**changes), original) is original
    assert network[1] == []


@pytest.mark.asyncio
@pytest.mark.parametrize('source', [
    'https://github.com/example/demo/archive/refs/tags/v9.9.9.tar.gz',
    'https://host/demo-9.9.9.tar.gz',
    'https://crates.io/api/v1/crates/demo/9.9.9/download',
    'https://crates.io/api/v1/crates/another/1.2.3/download',
])
async def test_download_coordinates_cannot_be_masked_by_input_version(network, source):
    network[0][PAGE] = (200, 'demo 1.2.3 MIT', {})
    original = result(license_files=source)
    assert await links.prefer_license_webpage(row(), original) is original
    assert network[1] == []


@pytest.mark.asyncio
async def test_earlier_exact_repository_reference_survives_archive_fallback(network):
    original = result(license_files='https://host/demo-1.2.3.tar.gz', **{links.EVIDENCE_KEY: evidence()})
    prior = result(repo_url='https://github.com/example/demo', resolved_version='v1.2.3', license_files='')
    network[0][RAW] = (200, TEXT, {})
    actual = await links.prefer_license_webpage(row(), original, prior)
    assert actual['license_files'] == RAW
    prior['used_default_branch'] = True
    network[1].clear()
    actual = await links.prefer_license_webpage(row(), original, prior)
    assert actual['license_files'] == original['license_files']
    assert ('GET', RAW) not in network[1]


@pytest.mark.asyncio
async def test_selected_readme_can_be_used_but_different_text_cannot(network):
    text = '# demo\nLicensed under MIT.'
    raw = RAW.replace('LICENSE', 'README.md')
    network[0][raw] = (200, text, {})
    original = result(**{links.EVIDENCE_KEY: evidence('README.md', text)})
    assert (await links.prefer_license_webpage(row(), original))['license_files'] == raw
    network[0][raw] = (200, 'Licensed under Apache-2.0.', {})
    assert (await links.prefer_license_webpage(row(), original))['license_files'] == ARCHIVE


@pytest.mark.asyncio
async def test_large_html_rejected(network):
    network[0][PAGE] = (200, 'demo 1.2.3 MIT ' + 'x' * links.MAX_PAGE_BYTES, {})
    original = result()
    assert await links.prefer_license_webpage(row(), original) is original


@pytest.mark.asyncio
@pytest.mark.parametrize('entry', ['cli', 'api'])
async def test_entry_runs_lookup_once_without_reauditing_and_keeps_output(network, monkeypatch, tmp_path, entry):
    import main
    import api
    module = main if entry == 'cli' else api
    original = result(**{links.EVIDENCE_KEY: evidence()})
    github = AsyncMock(return_value={'status': 'skipped'})
    archive = AsyncMock(return_value=original)
    crate = AsyncMock(side_effect=AssertionError('must not rerun package audit'))
    monkeypatch.setattr(module, 'process_github_repository', github)
    monkeypatch.setattr(module, 'process_direct_archive_url', archive)
    monkeypatch.setattr(module, 'process_crate_repository', crate)
    network[0][RAW] = (200, TEXT, {})
    monkeypatch.chdir(tmp_path)
    if entry == 'cli':
        output = await asyncio.wait_for(main.process_all_repos(None, pd.DataFrame([row()]), max_concurrency=1), 2)
    else:
        monkeypatch.setattr(api, 'MAX_CONCURRENCY', 1)
        q = asyncio.Queue()
        output = await asyncio.wait_for(api._process_repositories(None, pd.DataFrame([row()]), q), 2)
        messages = []
        while not q.empty():
            messages.append(q.get_nowait())
        assert len([m for m in messages if m.startswith('[PROGRESS]')]) == 1
    assert output[0]['license_files'] == RAW
    assert links.EVIDENCE_KEY not in output[0]
    assert output[0]['serial_number'] == '0001'
    github.assert_awaited_once()
    archive.assert_awaited_once()
    crate.assert_not_awaited()
    # Verify every exported field, including risk/dedup/copyright, against the
    # unchanged original result passed through the existing output formatter.
    before = {**output[0], 'license_files': ARCHIVE}
    expected_df = api._generate_output([before])
    actual_df = api._generate_output(output)
    expected_df['license_files'] = RAW
    pd.testing.assert_frame_equal(actual_df, expected_df)


@pytest.mark.asyncio
@pytest.mark.parametrize('manager,name,url', [
    ('cargo', 'demo', PAGE),
    ('npm', '@scope/demo', 'https://www.npmjs.com/package/@scope/demo/v/1.2.3'),
    ('pypi', 'demo', 'https://pypi.org/project/demo/1.2.3'),
    ('pub', 'demo', 'https://pub.dev/packages/demo/versions/1.2.3'),
    ('nuget', 'Demo.Core', 'https://www.nuget.org/packages/Demo.Core/1.2.3'),
    ('maven', 'org.example:demo', 'https://mvnrepository.com/artifact/org.example/demo/1.2.3'),
    ('golang', 'github.com/example/demo', 'https://pkg.go.dev/github.com/example/demo@1.2.3?tab=licenses'),
    ('deb', 'debian/demo', 'https://sources.debian.org/src/demo/1.2.3/'),
])
async def test_supported_registry_links(network, manager, name, url):
    network[0][url] = (200, f'{name} demo 1.2.3 licensed MIT', {})
    original = result()
    actual = await links.prefer_license_webpage(row(package_manager=manager, name=name), original)
    assert actual == {**original, 'license_files': url}


@pytest.mark.asyncio
@pytest.mark.parametrize('entry', ['cli', 'api'])
async def test_mixed_lookup_batch_keeps_slots_order_and_progress(network, monkeypatch, tmp_path, entry):
    import main
    import api
    module = main if entry == 'cli' else api
    active = peak = 0
    finished = []
    async def fetch(client, url):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        name = url.split('/')[-2]
        try:
            await asyncio.sleep(0.04 if name == 'demo0' else 0.001)
            if name == 'demo2':
                raise httpx.ReadTimeout('controlled failure')
            finished.append(name)
            return f'{name} 1.2.3 licensed MIT'
        finally:
            active -= 1
    monkeypatch.setattr(links, '_fetch_page', fetch)
    github = AsyncMock(return_value={'status': 'skipped'})
    async def archive(url, version, name, **kwargs):
        return result(license_files=url)
    monkeypatch.setattr(module, 'process_github_repository', github)
    monkeypatch.setattr(module, 'process_direct_archive_url', archive)
    monkeypatch.chdir(tmp_path)
    rows = pd.DataFrame([row(name=f'demo{i}', serial_number=f'00{i}',
        github_url=f'https://host/demo{i}-1.2.3.tar.gz') for i in range(5)])
    if entry == 'cli':
        output = await asyncio.wait_for(main.process_all_repos(None, rows, max_concurrency=2), 2)
    else:
        monkeypatch.setattr(api, 'MAX_CONCURRENCY', 2)
        q = asyncio.Queue()
        output = await asyncio.wait_for(api._process_repositories(None, rows, q), 2)
        progress = []
        while not q.empty():
            message = q.get_nowait()
            if message.startswith('[PROGRESS]'):
                progress.append(message)
        assert len(progress) == 5 and '5/5 (100.0%)' in progress[-1]
    assert peak == 2
    assert finished.index('demo1') < finished.index('demo0')
    assert github.await_count == 5
    assert [r['serial_number'] for r in output] == rows.serial_number.tolist()
    for i, actual in enumerate(output):
        expected_link = rows.iloc[i].github_url if i == 2 else f'https://crates.io/crates/demo{i}/1.2.3'
        assert actual['license_files'] == expected_link
        assert actual['copyright_notice'] == 'Copyright Example'


@pytest.mark.parametrize('route', [
    'analyze', 'analyze-and-download', 'analyze-stream', 'analyze-stream-email', 'analyze-stream-download',
])
def test_all_upload_routes_export_improved_link(network, monkeypatch, route):
    import io
    from pathlib import Path
    import api
    from fastapi.testclient import TestClient
    network[0][RAW] = (200, TEXT, {})
    monkeypatch.setattr(api, 'get_github_api', AsyncMock(return_value=None))
    monkeypatch.setattr(api, 'process_github_repository', AsyncMock(return_value={'status': 'skipped'}))
    monkeypatch.setattr(api, 'process_direct_archive_url', AsyncMock(return_value=result(**{links.EVIDENCE_KEY: evidence()})))
    captured = []
    process = api._process_repositories
    async def observe(*args, **kwargs):
        results = await process(*args, **kwargs)
        captured.extend(results)
        return results
    monkeypatch.setattr(api, '_process_repositories', observe)
    email_outputs = []
    def email(**kwargs):
        email_outputs.append(Path(kwargs['output_file_path']).read_bytes())
        return True
    monkeypatch.setattr(api, 'send_analysis_result', email)
    stream = io.BytesIO()
    pd.DataFrame([row()]).to_excel(stream, index=False)
    with TestClient(api.app) as client:
        response = client.post('/api/v1/' + route,
            files={'file': ('input.xlsx', stream.getvalue(), 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')},
            data={'email': 'test@example.com'})
    assert response.status_code == 200
    assert len(captured) == 1 and captured[0]['license_files'] == RAW
    assert links.EVIDENCE_KEY not in captured[0]
    for content in email_outputs + ([response.content] if route.endswith('download') else []):
        output = pd.read_excel(io.BytesIO(content), dtype=str)
        assert output.license_files.tolist() == [RAW]
        assert output.copyright_notice.tolist() == ['Copyright Example']
        assert output.serial_number.tolist() == ['0001']
        assert not any(c.startswith('_archive') for c in output.columns)
