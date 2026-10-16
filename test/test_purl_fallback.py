import asyncio
from unittest.mock import AsyncMock

import pandas as pd
import pytest

from core.purl_fallback import build_fallback_purl, retry_with_purl


def row(**kwargs):
    return dict(name='demo', version='1.2.3', package_manager='cargo',
                github_url='https://github.com/example/demo', serial_number='0007', **kwargs)


def github_result(**kwargs):
    result = dict(status='success', license_type='MIT', repo_url='https://github.com/example/demo',
                  used_default_branch=True, resolved_version='main',
                  license_files='https://github.com/example/demo/blob/main/LICENSE')
    result.update(kwargs)
    return result


def crate_result(**kwargs):
    result = dict(status='success', license_type='Apache-2.0', resolved_version='1.2.3',
                  used_default_branch=True, repo_url='https://github.com/example/demo',
                  license_files='https://crates.io/crates/demo/1.2.3')
    result.update(kwargs)
    return result


@pytest.mark.parametrize('manager,name,version,expected', [
    ('cargo', 'demo', '1.2.3', 'pkg:cargo/demo@1.2.3'),
    ('NPM', '@scope/name', '2.0', 'pkg:npm/%40scope/name@2.0'),
    ('pip', 'some_package', None, 'pkg:pypi/some_package'),
    ('gradle', 'org.example:demo', '1.0', 'pkg:maven/org.example/demo@1.0'),
    ('maven', 'org.example/demo', '1.0', 'pkg:maven/org.example/demo@1.0'),
    ('go', 'github.com/example/demo/v2', 'v2.0.0', 'pkg:golang/github.com/example/demo/v2@v2.0.0'),
    ('pub', 'demo', '1.0', 'pkg:pub/demo@1.0'),
    ('nuget', 'Example.Core', '1.0', 'pkg:nuget/Example.Core@1.0'),
    ('apt', 'debian/demo', '1:2.0+dfsg-1', 'pkg:deb/debian/demo@1%3A2.0%2Bdfsg-1'),
    ('maven', 'demo', '1.0', None),
    ('go', 'demo', '1.0', None),
    (None, 'demo', '1.0', None),
    (float('nan'), 'demo', '1.0', None),
    ('cargo', None, '1.0', None),
    ('unsupported', 'demo', '1.0', None),
])
def test_build_purl(manager, name, version, expected):
    assert build_fallback_purl(dict(package_manager=manager, name=name, version=version)) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize('original', [
    dict(status='error'), dict(status='skipped'),
    dict(status='success', license_type=None),
    dict(status='success', license_type='NOASSERTION'), github_result(),
])
async def test_failure_or_unmatched_github_retries(original):
    candidate = crate_result()
    retry = AsyncMock(return_value=candidate)
    assert await retry_with_purl(row(), original, retry) is candidate
    retry.assert_awaited_once_with('pkg:cargo/demo@1.2.3')


@pytest.mark.asyncio
@pytest.mark.parametrize('candidate', [
    dict(status='error'), dict(status='skipped'), crate_result(license_type=None),
    crate_result(resolved_version='9.9.9', license_files='https://crates.io/crates/demo/9.9.9'),
    crate_result(license_files='https://crates.io/crates/demo'),
    crate_result(license_files='https://github.com/example/demo/blob/main/LICENSE'),
    github_result(used_default_branch=False, resolved_version='9.9.9',
                  license_files='https://github.com/example/demo/blob/9.9.9/LICENSE'),
])
async def test_bad_or_wrong_version_retry_does_not_replace_original(candidate):
    original = github_result()
    assert await retry_with_purl(row(), original, AsyncMock(return_value=candidate)) is original


@pytest.mark.asyncio
async def test_retry_exception_keeps_original():
    original = github_result()
    assert await retry_with_purl(row(), original, AsyncMock(side_effect=RuntimeError('offline'))) is original


@pytest.mark.asyncio
@pytest.mark.parametrize('change', ['exact_github', 'no_version', 'already_purl', 'no_manager'])
async def test_unnecessary_retry_is_skipped(change):
    original, input_row = github_result(), row()
    if change == 'exact_github':
        original.update(used_default_branch=False, resolved_version='v1.2.3',
                        license_files='https://github.com/example/demo/blob/v1.2.3/LICENSE')
    elif change == 'no_version':
        input_row['version'] = None
    elif change == 'already_purl':
        input_row['github_url'] = 'pkg:cargo/demo@1.2.3'
    else:
        input_row['package_manager'] = None
    retry = AsyncMock()
    assert await retry_with_purl(input_row, original, retry) is original
    retry.assert_not_awaited()


@pytest.mark.asyncio
async def test_original_exact_link_is_preserved():
    original = crate_result(license_type='MIT')
    retry = AsyncMock(return_value=crate_result())
    assert await retry_with_purl(row(), original, retry) is original
    retry.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize('manager,name,url,resolved', [
    ('npm', '@scope/demo', 'https://www.npmjs.com/package/@scope/demo/v/1.2.3?activeTab=code', '1.2.3'),
    ('pypi', 'demo', 'https://pypi.org/project/demo/1.2.3/', '1.2.3'),
    ('pub', 'demo', 'https://pub.dev/packages/demo/versions/1.2.3', '1.2.3'),
    ('nuget', 'demo', 'https://www.nuget.org/packages/demo/1.2.3', '1.2.3'),
    ('maven', 'org.example:demo', 'https://repo.maven.apache.org/maven2/org/example/demo/1.2.3/demo-1.2.3.pom', '1.2.3'),
    ('go', 'github.com/example/demo', 'https://pkg.go.dev/github.com/example/demo@v1.2.3?tab=licenses', 'main'),
    ('cargo', 'demo', 'https://github.com/example/demo/blob/demo-v1.2.3/LICENSE', 'demo-v1.2.3'),
])
async def test_versioned_links_from_existing_handlers(manager, name, url, resolved):
    input_row = row()
    input_row.update(package_manager=manager, name=name)
    candidate = github_result(used_default_branch=resolved == 'main', resolved_version=resolved,
                              license_files=url)
    assert await retry_with_purl(input_row, github_result(), AsyncMock(return_value=candidate)) is candidate


@pytest.mark.asyncio
@pytest.mark.parametrize('entry', ['cli', 'api'])
@pytest.mark.parametrize('failure', ['default_branch', 'error', 'exception', 'no_license', 'archive_failure'])
async def test_existing_entry_reused_once_and_preserves_rows(monkeypatch, tmp_path, entry, failure):
    import api
    import main
    module = main if entry == 'cli' else api
    calls = []

    async def github(client, url, version, name=None):
        calls.append(('github', url))
        await asyncio.sleep(0.01)
        if failure in ('exception', 'archive_failure'):
            raise RuntimeError('controlled failure')
        if failure == 'error':
            return dict(status='error')
        if failure == 'no_license':
            return dict(status='success', license_type=None)
        return github_result()

    async def crate(url, version):
        calls.append(('crate', url))
        await asyncio.sleep(0.01)
        return crate_result()

    async def archive(*args, **kwargs):
        calls.append(('archive', args[0]))
        return None

    monkeypatch.setattr(module, 'process_github_repository', github)
    monkeypatch.setattr(module, 'process_crate_repository', crate)
    monkeypatch.setattr(module, 'process_direct_archive_url', archive)
    monkeypatch.chdir(tmp_path)
    original_row = row()
    if failure == 'archive_failure':
        original_row['github_url'] = 'https://example.com/demo.tar.gz'
    # Fill every semaphore slot: retries must not acquire another slot or deadlock.
    count = 2
    df = pd.DataFrame([dict(original_row, serial_number=f'000{i}') for i in range(count)])
    if entry == 'cli':
        results = await asyncio.wait_for(main.process_all_repos(None, df, max_concurrency=count), 3)
    else:
        monkeypatch.setattr(api, 'MAX_CONCURRENCY', count)
        progress = asyncio.Queue()
        results = await asyncio.wait_for(api._process_repositories(None, df, progress), 3)
        messages = []
        while not progress.empty():
            messages.append(progress.get_nowait())
        assert sum(message.startswith('[PROGRESS]') for message in messages) == count
    assert sum(kind == 'github' for kind, _ in calls) == count
    assert sum(kind == 'crate' for kind, _ in calls) == count
    if failure == 'archive_failure':
        kinds = [kind for kind, _ in calls]
        assert kinds.count('archive') == count
        assert kinds.index('archive') < kinds.index('crate')
    for index, result in enumerate(results):
        assert result['input_url'] == original_row['github_url']
        assert result['input_name'] == 'demo'
        assert result['package_manager'] == 'cargo'
        assert result['serial_number'] == f'000{index}'
        assert result['license_files'] == 'https://crates.io/crates/demo/1.2.3'
        assert result['license_type'] == 'Apache-2.0'


@pytest.mark.asyncio
@pytest.mark.parametrize('entry', ['cli', 'api'])
async def test_failed_purl_does_not_loop_or_lose_original(monkeypatch, tmp_path, entry):
    import api
    import main
    module = main if entry == 'cli' else api
    original = github_result()
    github = AsyncMock(return_value=original.copy())
    crate = AsyncMock(return_value=dict(status='error', error='registry offline'))
    monkeypatch.setattr(module, 'process_github_repository', github)
    monkeypatch.setattr(module, 'process_crate_repository', crate)
    monkeypatch.chdir(tmp_path)
    df = pd.DataFrame([row()])
    results = await (main.process_all_repos(None, df) if entry == 'cli'
                     else api._process_repositories(None, df))
    github.assert_awaited_once()
    crate.assert_awaited_once()
    assert results[0]['license_files'] == original['license_files']
    assert results[0]['license_type'] == 'MIT'


@pytest.mark.parametrize('entry', ['cli', 'api'])
def test_fallback_result_in_exported_excel(monkeypatch, tmp_path, entry):
    import io
    import api
    import main
    from fastapi.testclient import TestClient
    from openpyxl import load_workbook

    module = main if entry == 'cli' else api
    monkeypatch.setattr(module, 'process_github_repository', AsyncMock(return_value=github_result()))
    monkeypatch.setattr(module, 'process_crate_repository', AsyncMock(return_value=crate_result()))
    monkeypatch.chdir(tmp_path)
    buffer = io.BytesIO()
    pd.DataFrame([row()]).to_excel(buffer, index=False)
    if entry == 'cli':
        monkeypatch.setattr(main, 'initialize_api', AsyncMock(return_value=object()))
        (tmp_path / 'input.xlsx').write_bytes(buffer.getvalue())
        asyncio.run(main.main_async())
        content = (tmp_path / 'outputs/output_latest.xlsx').read_bytes()
    else:
        monkeypatch.setattr(api, 'get_github_api', AsyncMock(return_value=object()))
        with TestClient(api.app) as client:
            response = client.post('/api/v1/analyze-and-download', files={
                'file': ('input.xlsx', buffer.getvalue(),
                         'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')})
        assert response.status_code == 200
        content = response.content
    wb = load_workbook(io.BytesIO(content))
    rows = list(wb['分析结果'].values)
    output = dict(zip(rows[0], rows[1]))
    assert output['license_files'] == 'https://crates.io/crates/demo/1.2.3'
    assert output['concluded_license'] == 'Apache-2.0'
    assert output['serial_number'] == '0007'
    assert output['package_manager'] == 'cargo'
    assert output['input_url'] == row()['github_url']
    wb.close()
