"""npm copyright fallbacks must never invent the current year."""

import pytest

from core import npm_utils, utils


@pytest.mark.asyncio
@pytest.mark.parametrize('publication_date,single_version,expected_year', [
    (None, False, ''),
    ('invalid-date', False, ''),
    (None, True, ''),
    ('2018-04-02T12:00:00.000Z', False, '2018 '),
])
async def test_npm_fallback_year(monkeypatch, publication_date, single_version, expected_year):
    version = {'name': 'demo', 'version': '1.0.0', 'license': 'MIT', 'readme': '# Demo'}
    packument = version if single_version else {
        'versions': {'1.0.0': version},
        'dist-tags': {'latest': '1.0.0'},
        'time': {'1.0.0': publication_date},
    }
    monkeypatch.setattr(npm_utils, '_fetch_packument', lambda name: packument)

    async def no_copyright(*args, **kwargs):
        return 'None'

    monkeypatch.setattr(utils, 'complete_async', no_copyright)
    result = await npm_utils.process_npm_repository('https://www.npmjs.com/package/demo', '1.0.0')
    assert result['copyright_notice'] == f'Copyright (c) {expected_year}demo original author and authors'


@pytest.mark.asyncio
@pytest.mark.parametrize('notice', ['Copyright 2014-2020 Demo Authors', 'Copyright Demo Authors'])
async def test_extracted_notice_is_preserved_without_adding_year(monkeypatch, notice):
    async def extract(*args, **kwargs):
        return notice

    monkeypatch.setattr(utils, 'complete_async', extract)
    result = await utils.construct_copyright_notice_async(
        year='', owner='', repo='demo', ref='1.0.0', component_name='demo', license_content=notice,
    )
    assert result == notice


@pytest.mark.asyncio
async def test_empty_sources_omit_missing_year():
    result = await utils.construct_copyright_notice_async(
        year='', owner='', repo='demo', ref='1.0.0', component_name='demo',
    )
    assert result == 'Copyright (c) demo original author and authors'
