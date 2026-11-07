"""R web evidence boundaries and reuse of the existing archive fallback."""
from unittest.mock import AsyncMock

import aiohttp
import pandas as pd
import pytest

from core import r_utils as r

CRAN = 'https://cran.r-project.org/web/packages/readr/index.html'
TAR = 'https://cran.r-project.org/src/contrib/readr_2.2.0.tar.gz'
BIOC = 'https://bioconductor.org/packages/release/bioc/html/tradeSeq.html'
PIN = BIOC.replace('/release/', '/3.23/')
BIOC_TAR = 'https://bioconductor.org/packages/3.23/bioc/src/contrib/tradeSeq_1.26.0.tar.gz'


def page(name='readr', version='2.2.0', license='MIT + file LICENSE', bioc=False,
         release='3.23', extra='', source=None):
    title = f'Bioconductor - {name}' if bioc else f'CRAN: Package {name}'
    heading = f'<h1>{name}</h1>Bioconductor version: Release ({release})' if bioc else f'<h2>{name}: Package title</h2>'
    license = license.replace('file LICENSE', 'file <a href="' + (f'/packages/{release}/bioc/licenses/{name}/LICENSE' if bioc else 'LICENSE') + '">LICENSE</a>')
    if source is None:
        source = f'/packages/{release}/bioc/src/contrib/{name}_{version}.tar.gz' if bioc else f'/src/contrib/{name}_{version}.tar.gz'
    return f'''<html><title>{title}</title>{heading}<table>
      <tr><td>Version:</td><td>{version}</td></tr>
      <tr><td>License:</td><td>{license}</td></tr>
      <tr><td>Author:</td><td>Package Author</td></tr>
      <tr><td>Imports:</td><td>foreign-package (BSD-3-Clause)</td></tr>
      </table><a href="{source}">Source Package</a>{extra}
      <footer>Copyright 2026 Registry Operator, GPL-3.0</footer></html>'''


@pytest.mark.parametrize('url,registry,name,version', [
    (CRAN, 'cran', 'readr', ''), (TAR, 'cran', 'readr', '2.2.0'),
    ('https://cloud.r-project.org/src/contrib/Archive/pwr/pwr_1.3-0.tar.gz', 'cran', 'pwr', '1.3-0'),
    ('https://cran.r-project.org/package=readr', 'cran', 'readr', ''),
    ('https://cran.r-project.org/?package=readr', 'cran', 'readr', ''),
    (BIOC, 'bioconductor', 'tradeSeq', ''),
    (BIOC_TAR, 'bioconductor', 'tradeSeq', '1.26.0'),
    ('https://bioconductor.org/packages/3.20/data/annotation/src/contrib/org.Hs.eg.db_3.20.0.tar.gz', 'bioconductor', 'org.Hs.eg.db', '3.20.0'),
    ('https://bioconductor.org/packages/tradeSeq/', 'bioconductor', 'tradeSeq', ''),
])
def test_reference(url, registry, name, version):
    ref = r.parse_r_reference(url)
    assert (ref.registry, ref.name, ref.version) == (registry, name, version)


@pytest.mark.parametrize('url', [None, float('nan'), 'ftp://cran.r-project.org/src/contrib/readr_2.2.0.tar.gz',
    'https://cran.r-project.org.evil/web/packages/readr/index.html',
    'https://example.org/readr_2.2.0.tar.gz', 'https://bioconductor.org/about/',
    'https://cran.r-project.org/web/packages/readr/LICENSE',
    'https://www.npmjs.com/package/readr', 'https://github.com/tidyverse/readr', 'https://[broken'])
def test_unrelated_url_is_not_r(url):
    assert r.parse_r_reference(url) is None


@pytest.mark.parametrize('manager,registry', [('cran','cran'), ('bioconductor','bioconductor'), ('bioc','bioconductor')])
def test_explicit_manager(manager, registry):
    assert r.parse_r_reference('https://example.org/project', manager, 'demo').registry == registry
    assert r.parse_r_reference('https://example.org/project', manager, 'alias/demo') is None


@pytest.mark.parametrize('a,b,equal', [('1.3-0','1.3.0',True), ('0.01','0.1-0',True),
    ('1.0','1',True), ('1.20','1.2',False), ('1.3','1.3.1',False), ('v1.3','1.3',False), ('main','main',False)])
def test_r_version_comparison(a,b,equal):
    assert r._same_version(a,b) is equal


@pytest.fixture
def isolated(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Unexpected network/model call')
    monkeypatch.setattr(aiohttp.ClientSession, '_request', forbidden)
    fetch = AsyncMock()
    analyze = AsyncMock(return_value={'licenses':['MIT'], 'spdx_expression':'MIT'})
    notice = AsyncMock(return_value='Original LLM copyright notice')
    monkeypatch.setattr(r, '_fetch_text', fetch)
    monkeypatch.setattr(r, 'analyze_license_content_async', analyze)
    monkeypatch.setattr(r, 'construct_copyright_notice_async', notice)
    return fetch, analyze, notice


def serve(fetch, mapping):
    async def one(session, url):
        value = mapping.get(url, aiohttp.ClientResponseError(None, (), status=404))
        if isinstance(value, BaseException):
            raise value
        return url, value
    fetch.side_effect = one


def description(license='MIT + file LICENSE'):
    return f'Package: readr\nVersion: 2.2.0\nLicense: {license}\nAuthor: Package Author\n'


@pytest.mark.asyncio
@pytest.mark.parametrize('url', [CRAN,TAR])
async def test_web_success_preserves_package_license_and_llm_notice(isolated, url):
    fetch, analyze, notice = isolated
    serve(fetch, {CRAN.replace('index.html','DESCRIPTION'):description(), CRAN.replace('index.html','LICENSE'):'YEAR: 2020\nCOPYRIGHT HOLDER: Package Authors'})
    result = await r.process_r_package(url,'2.2.0','readr')
    assert result['license_type'] == 'MIT'
    assert result['status'] == 'success'
    assert result['resolved_version'] == '2.2.0'
    assert result['license_files'] == CRAN.replace('index.html','DESCRIPTION')
    assert result['copyright_notice'] == 'Original LLM copyright notice'
    evidence = analyze.await_args.args[0]
    assert 'License: MIT + file LICENSE' in evidence
    assert 'COPYRIGHT HOLDER: Package Authors' in evidence
    assert 'Registry Operator' not in evidence and 'foreign-package' not in evidence
    assert notice.await_args.kwargs['license_content'] == evidence
    assert fetch.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize('problem', ['newer','wrong_name','missing_version','missing_license','http','timeout','missing_file','foreign_file','redirect_file','model_empty','model_error'])
async def test_bad_web_evidence_returns_miss_without_borrowing_other_license(isolated, problem):
    fetch, analyze, notice = isolated
    html = page(name='tradeSeq',version='1.26.0',bioc=True)
    content = 'YEAR: 2020\nCOPYRIGHT HOLDER: Authors'
    if problem == 'newer': html = page(name='tradeSeq',version='1.27.0',bioc=True)
    if problem == 'wrong_name': html = page(name='foreign',version='1.26.0',bioc=True)
    if problem == 'missing_version': html = html.replace('Version:', 'Unknown:')
    if problem == 'missing_license': html = html.replace('License:', 'Unknown:')
    if problem == 'foreign_file': html = html.replace('href="/packages/3.23/bioc/licenses/tradeSeq/LICENSE"', 'href="https://example.org/LICENSE"')
    if problem == 'redirect_file': content = '<html><title>Login</title></html>'
    if problem == 'http': html = aiohttp.ClientResponseError(None, (), status=404)
    if problem == 'timeout': html = TimeoutError()
    mapping = {BIOC:html,PIN:html}
    if problem != 'missing_file': mapping[PIN.replace('/html/tradeSeq.html','/licenses/tradeSeq/LICENSE')] = content
    serve(fetch,mapping)
    if problem == 'model_empty': analyze.return_value = None
    if problem == 'model_error': analyze.side_effect = RuntimeError('provider unavailable')
    result = await r.process_r_package(BIOC,'1.26.0','tradeSeq')
    assert result['license_type'] is None and not result['resolved_version']
    assert result['status'] == 'error'
    assert result['license_files'] is None
    notice.assert_not_awaited()


@pytest.mark.asyncio
async def test_conflicting_input_and_archive_version_cannot_use_web(isolated):
    fetch, analyze, _ = isolated
    assert (await r.process_r_package(TAR,'1.0','readr'))['license_type'] is None
    fetch.assert_not_awaited()
    analyze.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('version',[None,'default','latest'])
async def test_default_uses_actual_page_version(isolated,version):
    fetch, _, _ = isolated
    serve(fetch,{CRAN.replace('index.html','DESCRIPTION'):description('GPL-2')})
    result = await r.process_r_package(CRAN,version,'readr')
    assert result['resolved_version'] == '2.2.0'
    assert result['input_version'] == version


@pytest.mark.asyncio
async def test_bioc_alias_is_pinned_and_rechecked(isolated):
    fetch, _, _ = isolated
    serve(fetch,{BIOC:page(name='tradeSeq',version='1.26.0',bioc=True),
                 PIN:page(name='tradeSeq',version='1.26.0',bioc=True),
                 PIN.replace('/html/tradeSeq.html','/licenses/tradeSeq/LICENSE'):'YEAR: 2019\nCOPYRIGHT HOLDER: Author'})
    result = await r.process_r_package(BIOC,'1.26.0','tradeSeq')
    assert result['license_files'] == PIN and result['resolved_version'] == '1.26.0'
    assert result['_r_archive_fallback_url'] == BIOC_TAR
    assert fetch.await_count == 3


@pytest.mark.asyncio
async def test_bioc_alias_race_does_not_combine_versions(isolated):
    fetch, analyze, _ = isolated
    serve(fetch,{BIOC:page(name='tradeSeq',version='1.26.0',bioc=True),
                 PIN:page(name='tradeSeq',version='1.26.1',bioc=True)})
    assert (await r.process_r_package(BIOC,'1.26.0','tradeSeq'))['license_type'] is None
    analyze.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('license',['GPL (>= 3)','LGPL (>= 2)','GPL','MIT + file LICENSE'])
async def test_original_cran_description_precedes_expanded_html(isolated,license):
    fetch, analyze, notice = isolated
    description=f'Package: readr\nVersion: 2.2.0\nLicense: {license}\nAuthor: Package Authors\n'
    desc_url=CRAN.replace('index.html','DESCRIPTION')
    serve(fetch,{desc_url:description,CRAN.replace('index.html','LICENSE'):'YEAR: 2020\nCOPYRIGHT HOLDER: Authors'})
    value=await r.process_r_package(TAR,'2.2.0','readr')
    assert value['license_type'] == 'MIT'
    assert value['license_files'] == desc_url
    assert value['_r_archive_fallback_url'] == TAR
    assert f'License: {license}' in analyze.await_args.args[0]
    assert not any(c.args[1] == CRAN for c in fetch.await_args_list)
    assert notice.await_args.kwargs['license_content'] == analyze.await_args.args[0]


@pytest.mark.asyncio
@pytest.mark.parametrize('description',[
    'Package: foreign\nVersion: 2.2.0\nLicense: BSD-3-Clause',
    'Package: readr\nVersion: 2.3.0\nLicense: BSD-3-Clause',
    'Package: readr\nVersion: main\nLicense: BSD-3-Clause',
    'Package: readr\nVersion: 2.2.0',
    '<html><title>login</title></html>',
])
async def test_wrong_description_identity_does_not_borrow_current_html(isolated,description):
    fetch,analyze,_=isolated
    serve(fetch,{CRAN.replace('index.html','DESCRIPTION'):description,CRAN:page()})
    assert (await r.process_r_package(TAR,'2.2.0','readr'))['license_type'] is None
    analyze.assert_not_awaited()
    fetch.assert_awaited_once()


@pytest.mark.asyncio
async def test_optional_copyright_file_does_not_change_license_analysis(isolated):
    fetch,analyze,notice=isolated
    html=page(name='tradeSeq',version='1.26.0',license='MIT',bioc=True)
    html=html.replace('</table>','<tr><td>Copyright:</td><td><a href="/packages/3.23/bioc/licenses/tradeSeq/COPYRIGHTS">COPYRIGHTS</a></td></tr></table>')
    serve(fetch,{BIOC:html,PIN:html,PIN.replace('/html/tradeSeq.html','/licenses/tradeSeq/COPYRIGHTS'):'Copyright 2015 Package Author. Foreign subcomponent uses BSD-3-Clause.'})
    result=await r.process_r_package(BIOC,'1.26.0','tradeSeq')
    assert result['license_type']=='MIT'
    assert 'BSD-3-Clause' not in analyze.await_args.args[0]
    assert 'Copyright 2015 Package Author' in notice.await_args.kwargs['license_content']


@pytest.mark.asyncio
async def test_expanded_cran_index_alone_is_not_original_license_evidence(isolated):
    fetch,analyze,_=isolated
    serve(fetch,{CRAN:page(license='GPL-2 | GPL-3')})
    result=await r.process_r_package(CRAN,'2.2.0','readr')
    assert result['license_type'] is None
    assert result['_r_archive_fallback_url']==TAR
    analyze.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('issue',['chunks','oversized','foreign_redirect'])
async def test_transport_reads_complete_bounded_official_response(issue):
    class Content:
        async def iter_chunked(self,size):
            for chunk in ([b'a'*500000,b'b'*500000,b'c'] if issue=='oversized' else [b'Package: ',b'readr\n',b'License: MIT']):
                yield chunk
    class Response:
        url='https://evil.org/DESCRIPTION' if issue=='foreign_redirect' else CRAN.replace('index.html','DESCRIPTION')
        charset='utf-8'
        content=Content()
        def raise_for_status(self): pass
        async def __aenter__(self): return self
        async def __aexit__(self,*args): pass
    class Session:
        def get(self,*args,**kwargs): return Response()
    if issue=='chunks':
        _,body=await r._fetch_text(Session(),CRAN)
        assert body=='Package: readr\nLicense: MIT'
    else:
        with pytest.raises(ValueError): await r._fetch_text(Session(),CRAN)


@pytest.mark.asyncio
@pytest.mark.parametrize('explicit_release',[False,True])
async def test_history_uses_only_published_same_package_links(isolated,explicit_release):
    fetch, _, _ = isolated
    old = PIN.replace('/3.23/','/3.20/')
    extra = f'<a href="{old}">3.20</a><a href="https://evil.org/packages/3.19/bioc/html/tradeSeq.html">3.19</a><a href="/packages/3.18/bioc/html/foreign.html">3.18</a>'
    serve(fetch,{BIOC:page(name='tradeSeq',version='1.26.0',license='GPL-3',bioc=True,extra=extra),
                 PIN:page(name='tradeSeq',version='1.26.0',license='GPL-3',bioc=True,extra=extra),
                 old:page(name='tradeSeq',version='1.20.0',license='MIT',bioc=True,release='3.20')})
    result = await r.process_r_package(PIN if explicit_release else BIOC,'1.20.0','tradeSeq')
    assert result['resolved_version'] == ('' if explicit_release else '1.20.0')
    assert result['license_files'] == (None if explicit_release else old)
    assert all('evil.org' not in c.args[1] and 'foreign.html' not in c.args[1] for c in fetch.await_args_list)


@pytest.mark.parametrize('url,allowed', [
    ('https://cran.r-project.org/web/packages/readr/LICENSE',True),
    ('https://cran.r-project.org/web/packages/foreign/LICENSE',False),
    ('https://cran.r-project.org/web/packages/readr/../LICENSE',False),
    ('https://evil.org/web/packages/readr/LICENSE',False),
    ('https://cran.r-project.org/web/packages/readr/%2e%2e',False),
])
def test_file_identity(url,allowed):
    assert r._same_package_file(url,r.RReference('cran','readr'),None) is allowed


@pytest.mark.asyncio
@pytest.mark.parametrize('entry',['cli','api'])
@pytest.mark.parametrize('outcome',['web','miss','exception','archive_miss','page_miss'])
async def test_entrypoints_use_existing_archive_once_and_no_other_audit(monkeypatch,tmp_path,entry,outcome):
    import api
    import main
    module = main if entry == 'cli' else api
    result = dict(status='success',license_type='MIT' if outcome == 'web' else None,
                  license_files=CRAN if outcome == 'web' else None, resolved_version='2.2.0',component_name='readr')
    if outcome == 'page_miss': result['_r_archive_fallback_url'] = TAR
    web = AsyncMock(side_effect=RuntimeError('web failed')) if outcome == 'exception' else AsyncMock(return_value=result)
    archive = AsyncMock(return_value=None if outcome == 'archive_miss' else dict(status='success',license_type='GPL-2',license_files=TAR))
    github = AsyncMock(side_effect=AssertionError('R input must not discover arbitrary GitHub repositories'))
    monkeypatch.setattr(module,'process_r_package',web)
    monkeypatch.setattr(module,'process_direct_archive_url',archive)
    monkeypatch.setattr(module,'process_github_repository',github)
    keep = AsyncMock(side_effect=lambda row,value,context:value)
    monkeypatch.setattr(module,'prefer_license_webpage',keep)
    monkeypatch.chdir(tmp_path)
    rows = pd.DataFrame([dict(name='readr',version='2.2.0',github_url=CRAN if outcome == 'page_miss' else TAR,
                              package_manager='generic',serial_number='00041')])
    values = await (main.process_all_repos(None,rows) if entry == 'cli' else api._process_repositories(None,rows))
    value = values[0]
    assert value['input_name'] == 'readr' and value['serial_number'] == '00041'
    assert '_r_archive_fallback_url' not in value
    assert value['license_type'] == ('MIT' if outcome == 'web' else None if outcome == 'archive_miss' else 'GPL-2')
    web.assert_awaited_once()
    github.assert_not_awaited()
    assert archive.await_count == (0 if outcome == 'web' else 1)
    if outcome != 'web': assert archive.await_args.args[0] == TAR


@pytest.mark.asyncio
@pytest.mark.parametrize('entry',['cli','api'])
async def test_cran_description_failure_reaches_shared_archive_with_original_expression(isolated,monkeypatch,tmp_path,entry):
    import api
    import main
    fetch,analyze,_=isolated
    serve(fetch,{CRAN:page(license='GPL-2 | GPL-3')})
    module=main if entry=='cli' else api
    monkeypatch.setattr(module,'process_r_package',r.process_r_package)
    archive=AsyncMock(return_value=dict(status='success',license_type='GPL-2.0-or-later',resolved_version='2.2.0',license_files=TAR))
    monkeypatch.setattr(module,'process_direct_archive_url',archive)
    monkeypatch.setattr(module,'process_github_repository',AsyncMock(side_effect=AssertionError('Unexpected GitHub fallback')))
    monkeypatch.setattr(module,'prefer_license_webpage',AsyncMock(side_effect=lambda row,result,context:result))
    monkeypatch.chdir(tmp_path)
    rows=pd.DataFrame([dict(name='readr',version='2.2.0',github_url=TAR,package_manager='generic')])
    values=await (main.process_all_repos(None,rows) if entry=='cli' else api._process_repositories(None,rows))
    assert values[0]['license_type']=='GPL-2.0-or-later'
    archive.assert_awaited_once()
    analyze.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('entry',['cli','api'])
async def test_unavailable_r_source_exports_as_unreachable_not_unlicensed(isolated,monkeypatch,tmp_path,entry):
    import api
    import main
    fetch,analyze,_=isolated
    serve(fetch,{})
    module=main if entry=='cli' else api
    monkeypatch.setattr(module,'process_r_package',r.process_r_package)
    archive=AsyncMock(return_value=None)
    monkeypatch.setattr(module,'process_direct_archive_url',archive)
    monkeypatch.setattr(module,'process_github_repository',AsyncMock(side_effect=AssertionError('Unexpected GitHub discovery')))
    monkeypatch.setattr(module,'prefer_license_webpage',AsyncMock(side_effect=lambda row,result,context:result))
    monkeypatch.chdir(tmp_path)
    rows=pd.DataFrame([dict(name='readr',version='2.2.0',github_url=TAR,package_manager='generic')])
    values=await (main.process_all_repos(None,rows) if entry=='cli' else api._process_repositories(None,rows))
    assert values[0]['status']=='error'
    assert api._generate_output(values).iloc[0]['concluded_license']=='Unreachable'
    archive.assert_awaited_once()
    analyze.assert_not_awaited()
