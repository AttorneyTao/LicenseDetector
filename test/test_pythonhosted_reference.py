from unittest.mock import Mock, AsyncMock

import pytest

from core import pypi_utils, github_utils


@pytest.mark.asyncio
@pytest.mark.parametrize('filename,name,version', [
    ('mofapy2-0.7.5.tar.gz', 'mofapy2', '0.7.5'),
    ('some_pkg-1.0.zip', 'some-pkg', '1.0'),
    ('some_pkg-1.0rc1.post2.dev3+cpu-py3-none-any.whl', 'some-pkg', '1.0rc1.post2.dev3+cpu'),
    ('some_pkg-1.0-2-py3-none-any.whl', 'some-pkg', '1.0'),
])
async def test_confirmed_file(monkeypatch, filename, name, version):
    url = 'https://files.pythonhosted.org/packages/ab/hash/' + filename
    fetch = Mock(return_value={'info': {'name': name, 'version': version},
                              'urls': [{'filename': filename, 'url': url}]})
    monkeypatch.setattr(pypi_utils, '_fetch_pypi_metadata', fetch)
    assert await pypi_utils.resolve_pythonhosted_reference(url + '?download=1#hash', None) == (
        f'https://pypi.org/project/{name}/{version}/', version)
    fetch.assert_called_once_with(name, version=version)


@pytest.mark.asyncio
@pytest.mark.parametrize('case', ['host', 'http', 'filename', 'version', 'missing_file',
                                 'different_path', 'name_mismatch', 'release_mismatch', 'api_error'])
async def test_unconfirmed_reference_does_not_convert(monkeypatch, case):
    url = 'https://files.pythonhosted.org/packages/ab/hash/demo-1.0.tar.gz'
    metadata = {'info': {'name': 'demo', 'version': '1.0'},
                'urls': [{'filename': 'demo-1.0.tar.gz', 'url': url}]}
    version = None
    if case == 'host': url = url.replace('files.pythonhosted.org', 'files.pythonhosted.org.evil.test')
    if case == 'http': url = url.replace('https:', 'http:')
    if case == 'filename': url = url.replace('demo-1.0.tar.gz', 'bad.tar.gz')
    if case == 'version': version = '2.0'
    if case == 'missing_file': metadata['urls'] = []
    if case == 'different_path': metadata['urls'][0]['url'] = url.replace('/hash/', '/other/')
    if case == 'name_mismatch': metadata['info']['name'] = 'unrelated'
    if case == 'release_mismatch': metadata['info']['version'] = '2.0'
    fetch = Mock(return_value=metadata)
    if case == 'api_error': fetch.side_effect = pypi_utils.PyPIAPIError('unavailable')
    monkeypatch.setattr(pypi_utils, '_fetch_pypi_metadata', fetch)
    assert await pypi_utils.resolve_pythonhosted_reference(url, version) is None
    if case in ('host', 'http', 'filename', 'version'): fetch.assert_not_called()


@pytest.mark.asyncio
async def test_unconfirmed_file_keeps_existing_dispatch(monkeypatch):
    url = 'https://files.pythonhosted.org/packages/hash/demo-1.0.tar.gz'
    monkeypatch.setattr(github_utils, 'resolve_pythonhosted_reference', AsyncMock(return_value=None))
    pypi = AsyncMock()
    lookup = AsyncMock(return_value=None)
    monkeypatch.setattr(github_utils, 'process_pypi_repository', pypi)
    monkeypatch.setattr(github_utils, 'find_github_url_from_package_url', lookup)
    await github_utils.process_github_repository(object(), url, '1.0', name='demo')
    pypi.assert_not_called()
    lookup.assert_awaited_once_with(url, 'demo')
