"""Regression tests comparing original routing with the optional final retry."""
import asyncio
from copy import deepcopy
from unittest.mock import AsyncMock

import pandas as pd
import pytest

from core.purl_fallback import retry_with_purl


def result(**changes):
    value = dict(status='success', license_type='MIT', license_file_license='MIT',
                 resolved_version='1.2.3', used_default_branch=False,
                 license_files='https://github.com/example/demo/blob/1.2.3/LICENSE',
                 repo_url='https://github.com/example/demo',
                 copyright_notice='Copyright Example', license_text='Original license text',
                 license_analysis={'licenses': ['MIT'], 'thirdparty_dirs': ['vendor']})
    value.update(changes)
    return value


def input_row(**changes):
    value = dict(name='demo', version='1.2.3', github_url='https://github.com/example/demo',
                 package_manager='cargo', serial_number='0001')
    value.update(changes)
    return value


@pytest.mark.asyncio
@pytest.mark.parametrize('entry', ['cli', 'api'])
@pytest.mark.parametrize('bad_url', ['https://[broken', pd.NA])
async def test_bad_metadata_cannot_abort_batch(monkeypatch, tmp_path, entry, bad_url):
    import api
    import main
    module = main if entry == 'cli' else api
    original = result(repo_url=bad_url, used_default_branch=True, resolved_version='main')
    monkeypatch.setattr(module, 'process_github_repository', AsyncMock(return_value=original))
    monkeypatch.setattr(module, 'process_crate_repository', AsyncMock(return_value={'status': 'error'}))
    monkeypatch.chdir(tmp_path)
    rows = pd.DataFrame([input_row(), input_row(serial_number='0002')])
    results = await (main.process_all_repos(None, rows) if entry == 'cli'
                     else api._process_repositories(None, rows))
    assert len(results) == 2
    assert all(item['license_files'] == original['license_files'] for item in results)


@pytest.mark.asyncio
async def test_preparation_failure_preserves_original(monkeypatch):
    import core.purl_fallback as fallback
    original = result(used_default_branch=True)
    def fail(row):
        raise ValueError('unusable coordinates')
    monkeypatch.setattr(fallback, 'build_fallback_purl', fail)
    assert await retry_with_purl(input_row(), original, AsyncMock()) is original


@pytest.mark.asyncio
@pytest.mark.parametrize('entry', ['cli', 'api'])
@pytest.mark.parametrize('route', ['github', 'npm', 'crate', 'deb', 'go', 'pub', 'maven',
                                  'pypi', 'nuget', 'archive', 'purl', 'old_input', 'no_version'])
async def test_successful_original_routing_and_outputs_are_unchanged(monkeypatch, tmp_path, entry, route):
    import api
    import main
    module = main if entry == 'cli' else api
    monkeypatch.chdir(tmp_path)
    calls = []
    current = result()
    async def handler(*args, **kwargs):
        calls.append((args, kwargs))
        return deepcopy(current)
    for name in ('process_github_repository', 'process_npm_repository', 'process_crate_repository',
                 'process_deb_package', 'process_pubdev_package', 'process_direct_archive_url'):
        monkeypatch.setattr(module, name, handler)
    monkeypatch.setattr(module, 'get_github_url_from_pkggo', AsyncMock(return_value={
        'github_url': 'https://github.com/example/demo'}))
    monkeypatch.setattr(module, 'get_github_url_from_pubdev', AsyncMock(return_value={
        'github_url': 'https://github.com/example/demo'}))
    monkeypatch.setattr(module, 'analyze_maven_repository_url', lambda *args: {})
    monkeypatch.setattr(module, 'build_maven_repository_result', lambda *args: deepcopy(current))
    urls = {
        'github': 'https://github.com/example/demo', 'npm': 'https://www.npmjs.com/package/demo',
        'crate': 'https://crates.io/crates/demo', 'deb': 'https://sources.debian.org/src/demo/1.2.3/',
        'go': 'https://pkg.go.dev/github.com/example/demo', 'pub': 'https://pub.dev/packages/demo',
        'maven': 'https://mvnrepository.com/artifact/org.example/demo',
        'pypi': 'https://pypi.org/project/demo', 'nuget': 'https://www.nuget.org/packages/demo',
        'archive': 'https://example.com/demo-1.2.3.tar.gz', 'purl': 'pkg:cargo/demo@1.2.3',
    }
    row = input_row(github_url=urls.get(route, 'https://github.com/example/demo'))
    if route == 'old_input':
        row.pop('package_manager')
        row.pop('serial_number')
        current.update(used_default_branch=True, resolved_version='main')
    if route == 'no_version':
        row['version'] = None
        current.update(used_default_branch=True, resolved_version='main')
    df = pd.DataFrame([row])
    async def run():
        return await (main.process_all_repos(None, df) if entry == 'cli'
                      else api._process_repositories(None, df))
    async def disabled(row, original, retry):
        return original
    with monkeypatch.context() as baseline:
        baseline.setattr(module, 'retry_with_purl', disabled)
        before = await run()
    before_calls = deepcopy(calls)
    calls.clear()
    after = await run()
    assert after == before
    assert calls == before_calls
    pd.testing.assert_frame_equal(api._generate_output(after), api._generate_output(before))


@pytest.mark.asyncio
@pytest.mark.parametrize('entry', ['cli', 'api'])
async def test_retry_failure_restores_every_original_field(monkeypatch, tmp_path, entry):
    import api
    import main
    module = main if entry == 'cli' else api
    monkeypatch.chdir(tmp_path)
    current = result(used_default_branch=True, resolved_version='main',
                     license_files='https://github.com/example/demo/blob/main/LICENSE')
    monkeypatch.setattr(module, 'process_github_repository', AsyncMock(side_effect=lambda *a, **k: deepcopy(current)))
    monkeypatch.setattr(module, 'process_crate_repository', AsyncMock(side_effect=RuntimeError('offline')))
    rows = pd.DataFrame([input_row()])
    async def run():
        return await (main.process_all_repos(None, rows) if entry == 'cli'
                      else api._process_repositories(None, rows))
    async def disabled(row, original, retry):
        return original
    with monkeypatch.context() as baseline:
        baseline.setattr(module, 'retry_with_purl', disabled)
        before = await run()
    assert await run() == before


@pytest.mark.asyncio
async def test_task_cancellation_is_not_swallowed():
    original = result(used_default_branch=True)
    with pytest.raises(asyncio.CancelledError):
        await retry_with_purl(input_row(), original, AsyncMock(side_effect=asyncio.CancelledError()))


@pytest.mark.asyncio
@pytest.mark.parametrize('entry', ['cli', 'api'])
async def test_mixed_batch_preserves_order_concurrency_and_progress(monkeypatch, tmp_path, entry):
    import api
    import main
    module = main if entry == 'cli' else api
    monkeypatch.chdir(tmp_path)
    active = peak = 0
    retries = []
    finished = []

    async def delay(seconds):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(seconds)
        finally:
            active -= 1

    async def github(client, url, version, name=None):
        await delay(0.04 if name == 'demo0' else 0.001)
        current = result(input_name=name, copyright_notice=f'Copyright {name}')
        if name != 'demo0':
            current.update(used_default_branch=True, resolved_version='main',
                           license_files=f'https://github.com/example/{name}/blob/main/LICENSE')
        if name in ('demo2', 'demo4'):
            current.update(status='error', license_type=None, license_file_license=None)
        return current

    async def crate(url, version):
        name = url.rsplit('/', 1)[-1]
        retries.append(name)
        await delay(0.001)
        finished.append(name)
        if name == 'demo3':
            raise RuntimeError('registry offline')
        if name in ('demo2', 'demo5'):
            return {'status': 'error', 'error': 'registry offline'}
        return result(license_type='Apache-2.0', license_file_license='Apache-2.0',
                      copyright_notice=f'Copyright registry {name}',
                      license_files=f'https://crates.io/crates/{name}/1.2.3')

    monkeypatch.setattr(module, 'process_github_repository', github)
    monkeypatch.setattr(module, 'process_crate_repository', crate)
    rows = pd.DataFrame([input_row(name=f'demo{i}', serial_number=f'00{i}') for i in range(6)])
    rows.loc[5, 'github_url'] = 'pkg:cargo/demo5@1.2.3'
    if entry == 'cli':
        output = await asyncio.wait_for(main.process_all_repos(None, rows, max_concurrency=2), 3)
        saved = pd.read_csv(tmp_path / 'temp/temp_results_latest.csv', dtype=str)
        assert saved.serial_number.tolist() == [f'00{i}' for i in range(6)]
    else:
        monkeypatch.setattr(api, 'MAX_CONCURRENCY', 2)
        queue = asyncio.Queue()
        output = await asyncio.wait_for(api._process_repositories(None, rows, queue), 3)
        messages = []
        while not queue.empty():
            messages.append(queue.get_nowait())
        progress = [message for message in messages if message.startswith('[PROGRESS]')]
        assert len(progress) == 6
        assert '6/6 (100.0%)' in progress[-1]
    assert peak == 2
    assert sorted(retries) == ['demo1', 'demo2', 'demo3', 'demo4', 'demo5']
    assert [r['serial_number'] for r in output] == [f'00{i}' for i in range(6)]
    assert [r['status'] for r in output] == ['success', 'success', 'error', 'success', 'success', 'error']
    assert output[0]['license_type'] == 'MIT'
    assert output[1]['license_files'] == 'https://crates.io/crates/demo1/1.2.3'
    assert output[2]['copyright_notice'] == 'Copyright demo2'
    assert output[3]['copyright_notice'] == 'Copyright demo3'
    assert output[3]['license_type'] == 'MIT'
    assert output[4]['license_files'] == 'https://crates.io/crates/demo4/1.2.3'
    assert [r['input_url'] for r in output] == rows.github_url.tolist()


@pytest.mark.asyncio
@pytest.mark.parametrize('manager,url', [
    ('cargo', 'https://crates.io/crates/demo/1.2.3'),
    ('npm', 'https://www.npmjs.com/package/demo/v/1.2.3'),
    ('pypi', 'https://pypi.org/project/demo/1.2.3/'),
    ('pub', 'https://pub.dev/packages/demo/versions/1.2.3'),
    ('nuget', 'https://www.nuget.org/packages/demo/1.2.3'),
])
async def test_requested_link_cannot_mask_a_different_resolved_package_version(manager, url):
    original = result(used_default_branch=True, resolved_version='main')
    candidate = result(license_type='GPL-3.0-only', license_file_license='GPL-3.0-only',
                       resolved_version='9.9.9', used_default_branch=True, license_files=url)
    assert await retry_with_purl(input_row(package_manager=manager), original,
                                 AsyncMock(return_value=candidate)) is original
