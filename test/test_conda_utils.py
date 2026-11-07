"""Registry identity, version/build scope, SPDX aggregation and bounded archive fallback."""
from copy import deepcopy
from io import BytesIO
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
from unittest.mock import AsyncMock
import zipfile

import aiohttp
import pandas as pd
import pytest

from core import conda_utils as c
from core import archive_utils as archives

PAGE='https://anaconda.org/bioconda/demo/files'
PACKAGE='https://api.anaconda.org/package/bioconda/demo'
RELEASE='https://api.anaconda.org/release/bioconda/demo/1.0'
DOWNLOAD='https://conda.anaconda.org/bioconda/linux-64/demo-1.0-h123_0.tar.bz2'


def build(license='MIT',version='1.0',build='h123_0',subdir='linux-64',**extra):
    return dict(version=version,basename=f'{subdir}/demo-{version}-{build}.tar.bz2',owner='bioconda',distribution_type='conda',labels=['main'],attrs={'license':license,'build':build,'subdir':subdir},**extra)


def analysis(expression='MIT'):
    return dict(licenses=[expression],spdx_expression=expression)


@pytest.fixture
def isolated(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('Unexpected external network/model call')
    monkeypatch.setattr(aiohttp.ClientSession,'_request',forbidden)
    fetch=AsyncMock();model=AsyncMock(return_value=analysis());notice=AsyncMock(return_value='Copyright (c) 2020 Upstream Authors')
    archive=AsyncMock(return_value=None)
    monkeypatch.setattr(c,'_fetch_json',fetch)
    monkeypatch.setattr(c,'analyze_license_content_async',model)
    monkeypatch.setattr(c,'construct_copyright_notice_async',notice)
    monkeypatch.setattr(c,'process_direct_archive_url',archive)
    return fetch,model,notice,archive


def serve(fetch,builds=None,versions=None,latest='1.0',package_extra=None):
    p=dict(name='demo',full_name='bioconda/demo',versions=versions or ['1.0'],latest_version=latest,license='Unrelated latest homepage license')
    p.update(package_extra or {})
    releases=builds if builds is not None else [build()]
    async def get(session,url):
        if url==PACKAGE:return deepcopy(p)
        if '/release/bioconda/demo/' in url:return {'distributions':deepcopy(releases)}
        raise AssertionError(url)
    fetch.side_effect=get


@pytest.mark.parametrize('url,expected',[
    (PAGE,('bioconda','demo','',False)),
    ('https://anaconda.org/bioconda/DEMO',('bioconda','demo','',False)),
    (RELEASE,('bioconda','demo','1.0',False)),
    (PACKAGE,('bioconda','demo','',False)),
    (DOWNLOAD,('bioconda','demo','1.0',True)),
    ('https://conda.anaconda.org/bioconda/label/old/linux-64/demo-1.0-h123_0.conda',('bioconda','demo','1.0',True)),
    ('https://api.anaconda.org/download/bioconda/demo/1.0/linux-64/demo-1.0-h123_0.tar.bz2',('bioconda','demo','1.0',True)),
    ('https://anaconda.org/bioconda/demo/files?version=1.0',('bioconda','demo','1.0',False)),
])
def test_reference(url,expected):
    ref=c.parse_conda_reference(url)
    assert (ref.channel,ref.name,ref.version,bool(ref.basename))==expected


@pytest.mark.parametrize('url',[None,float('nan'),'https://[broken','https://anaconda.org.evil/bioconda/demo/files','https://github.com/bioconda/demo','https://pypi.org/project/demo/','https://cran.r-project.org/src/contrib/demo_1.0.tar.gz','https://conda.anaconda.org/bioconda','https://anaconda.org/bioconda/demo/issues','https://api.anaconda.org/download/bioconda/foreign/1.0/linux-64/demo-1.0-h123_0.tar.bz2','https://conda.anaconda.org/bioconda/../demo-1.0-h123_0.tar.bz2'])
def test_unrelated_or_invalid_url(url):assert c.parse_conda_reference(url) is None


@pytest.mark.asyncio
@pytest.mark.parametrize('entry',['cli','api'])
async def test_invalid_registry_name_does_not_fall_into_github_discovery(isolated,monkeypatch,tmp_path,entry):
    import api
    import main
    module=main if entry=='cli' else api
    url='https://anaconda.org/bioconda/AutoDock%20Vina/files'
    assert c.is_conda_package_url(url) and c.parse_conda_reference(url) is None
    github=AsyncMock(side_effect=AssertionError('Unexpected GitHub discovery'))
    monkeypatch.setattr(module,'process_github_repository',github)
    monkeypatch.chdir(tmp_path)
    rows=pd.DataFrame([dict(name='AutoDock Vina',version=None,github_url=url,package_manager='conda')])
    values=await (main.process_all_repos(None,rows) if entry=='cli' else api._process_repositories(None,rows))
    assert values[0]['status']=='error'
    github.assert_not_awaited()
    isolated[0].assert_not_awaited()


@pytest.mark.asyncio
async def test_all_raw_declarations_retained_only_pair_narrowed(isolated):
    fetch,model,notice,archive=isolated
    serve(fetch,[build('GPL-3.0-only'),build('GPL-3.0-or-later',build='h456_1'),build('GPL-3.0-only',subdir='osx-64')])
    model.return_value=analysis('GPL-3.0-only')
    result=await c.process_conda_package(PAGE,'1.0','Display Name')
    assert result['metadata_licenses']=='GPL-3.0-only; GPL-3.0-or-later'
    assert result['license_type']==result['license_file_license']=='GPL-3.0-only'
    assert result['license_files']==RELEASE and result['resolved_version']=='1.0'
    assert len(result['license_analysis']['conda_build_licenses'])==3
    assert result['license_analysis']['conda_metadata_licenses']==['GPL-3.0-only','GPL-3.0-or-later']
    content=model.await_args.args[0]
    assert 'License: GPL-3.0-only' in content and 'or-later' not in content
    assert 'Unrelated latest homepage license' not in content
    assert 'GPL-3.0-or-later' in notice.await_args.kwargs['license_content']
    archive.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('declarations,expression',[
    (['MIT','BSD-3-Clause'],'MIT AND BSD-3-Clause'),
    (['GPL-2.0-or-later','GPL-3.0-or-later'],'GPL-2.0-or-later AND GPL-3.0-or-later'),
    (['GPL-2.0-only','GPL-3.0-only'],'GPL-2.0-only AND GPL-3.0-only'),
    (['MIT OR Apache-2.0','BSD-3-Clause'],'(MIT OR Apache-2.0) AND BSD-3-Clause'),
    (['GPL-3.0-only WITH Classpath-exception-2.0','MIT'],'GPL-3.0-only WITH Classpath-exception-2.0 AND MIT'),
])
async def test_other_cases_use_existing_llm_spdx_structure(isolated,declarations,expression):
    fetch,model,_,archive=isolated
    serve(fetch,[build(value,build=f'h{i}_0') for i,value in enumerate(declarations)])
    model.return_value=analysis(expression)
    result=await c.process_conda_package(PAGE,'1.0')
    assert result['license_type']==expression
    assert 'Do not replace different license versions or families with a set intersection' in model.await_args.args[0]
    for value in declarations:assert f'License: {value}' in model.await_args.args[0]
    archive.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('requested',[None,'','default','latest','无','9.9'])
async def test_default_and_version_miss_record_actual_version(isolated,requested):
    fetch,_,_,_=isolated;serve(fetch)
    result=await c.process_conda_package(PAGE,requested)
    assert result['status']=='success' and result['input_version']==requested
    assert result['resolved_version']=='1.0' and result['license_files']==RELEASE
    if requested=='9.9':assert 'requested version 9.9 unavailable' in result['license_determination_reason']


@pytest.mark.asyncio
async def test_explicit_file_selects_one_build_not_other_platform(isolated):
    fetch,model,_,_=isolated
    serve(fetch,[build('MIT'),build('BSD-3-Clause',subdir='osx-64')])
    result=await c.process_conda_package(DOWNLOAD,'1.0')
    assert result['metadata_licenses']=='MIT'
    assert len(result['license_analysis']['conda_build_licenses'])==1
    assert 'BSD' not in model.await_args.args[0]


@pytest.mark.asyncio
@pytest.mark.parametrize('bad',['package_name','package_owner','distribution_name','distribution_owner','distribution_version','distribution_build','traversal'])
async def test_wrong_identity_is_rejected_before_llm(isolated,bad):
    fetch,model,_,archive=isolated
    d=build();extra={}
    if bad=='package_name':extra={'name':'foreign'}
    if bad=='package_owner':extra={'full_name':'foreign/demo'}
    if bad=='distribution_name':d['basename']='linux-64/foreign-1.0-h123_0.tar.bz2'
    if bad=='distribution_owner':d['owner']='foreign'
    if bad=='distribution_version':d['version']='2.0'
    if bad=='distribution_build':d['attrs']['build']='other_0'
    if bad=='traversal':d['basename']='../demo-1.0-h123_0.tar.bz2'
    serve(fetch,[d],package_extra=extra)
    assert (await c.process_conda_package(PAGE,'1.0'))['status']=='error'
    model.assert_not_awaited();archive.assert_not_awaited()


@pytest.mark.asyncio
async def test_conflicting_explicit_url_version_has_no_download(isolated):
    fetch,model,_,archive=isolated
    assert (await c.process_conda_package(DOWNLOAD,'2.0'))['status']=='error'
    fetch.assert_not_awaited();model.assert_not_awaited();archive.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('failure',['empty','exception','invalid_expression'])
async def test_model_failure_falls_back_once_with_verified_build(isolated,failure):
    fetch,model,_,archive=isolated;serve(fetch)
    if failure=='empty':model.return_value=None
    if failure=='exception':model.side_effect=RuntimeError('provider unavailable')
    if failure=='invalid_expression':model.return_value=analysis('MIT AND')
    archive.return_value=dict(status='success',license_type='BSD-3-Clause',license_file_license='BSD-3-Clause',license_analysis={},resolved_version='1.0',license_files=DOWNLOAD)
    result=await c.process_conda_package(PAGE,'1.0')
    assert result['license_type']=='BSD-3-Clause' and result['metadata_licenses']=='MIT'
    archive.assert_awaited_once()
    assert archive.await_args.args[0]==DOWNLOAD
    assert archive.await_args.kwargs['directory_analyzer']


@pytest.mark.asyncio
async def test_missing_license_multiple_builds_does_not_choose_one(isolated):
    fetch,model,_,archive=isolated;serve(fetch,[build(None),build(None,subdir='osx-64')])
    result=await c.process_conda_package(PAGE,'1.0')
    assert result['status']=='error' and 'multiple unspecified builds' in result['error']
    model.assert_not_awaited();archive.assert_not_awaited()


@pytest.mark.asyncio
async def test_direct_url_can_use_manifest_when_api_unavailable(isolated):
    fetch,model,_,archive=isolated
    fetch.side_effect=TimeoutError('API unavailable')
    archive.return_value=dict(status='success',license_type='MIT',resolved_version='1.0')
    assert (await c.process_conda_package(DOWNLOAD,'1.0'))['license_type']=='MIT'
    archive.assert_awaited_once();model.assert_not_awaited()


@pytest.mark.asyncio
async def test_archive_exception_is_not_retried(isolated):
    fetch,model,_,archive=isolated;serve(fetch)
    model.return_value=None;archive.side_effect=RuntimeError('downloader failed')
    assert (await c.process_conda_package(DOWNLOAD,'1.0'))['status']=='error'
    archive.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize('entry',['cli','api'])
@pytest.mark.parametrize('outcome',['success','error','exception'])
async def test_dispatcher_never_bypasses_conda_identity_checks(monkeypatch,tmp_path,entry,outcome):
    import api
    import main
    module=main if entry=='cli' else api
    expected=dict(status='success' if outcome=='success' else 'error',license_type='MIT' if outcome=='success' else None,metadata_licenses='MIT',license_files=RELEASE)
    handler=AsyncMock(side_effect=RuntimeError('handler failed')) if outcome=='exception' else AsyncMock(return_value=expected)
    archive=AsyncMock(side_effect=AssertionError('Unexpected second archive fallback'))
    github=AsyncMock(side_effect=AssertionError('Unexpected GitHub discovery'))
    monkeypatch.setattr(module,'process_conda_package',handler)
    monkeypatch.setattr(module,'process_direct_archive_url',archive)
    monkeypatch.setattr(module,'process_github_repository',github)
    monkeypatch.setattr(module,'prefer_license_webpage',AsyncMock(side_effect=lambda row,value,context:value))
    monkeypatch.chdir(tmp_path)
    df=pd.DataFrame([dict(name='demo',version='1.0',github_url=DOWNLOAD,package_manager='conda',serial_number='0001')])
    result=(await (main.process_all_repos(None,df) if entry=='cli' else api._process_repositories(None,df)))[0]
    assert result['serial_number']=='0001' and result['input_url']==DOWNLOAD
    assert result['status']==('success' if outcome=='success' else 'error')
    handler.assert_awaited_once();archive.assert_not_awaited();github.assert_not_awaited()


@pytest.mark.asyncio
async def test_native_output_keeps_raw_metadata_and_final_risk(isolated):
    import api
    fetch,model,_,_=isolated
    serve(fetch,[build('GPL-3.0-only'),build('GPL-3.0-or-later',build='other_0')])
    model.return_value=analysis('GPL-3.0-only')
    row=await c.process_conda_package(PAGE,'1.0');row['input_name']='demo'
    frame=api._generate_output([row])
    assert frame.iloc[0]['metadata_licenses']=='GPL-3.0-only; GPL-3.0-or-later'
    assert frame.iloc[0]['concluded_license']=='GPL-3.0-only'
    assert frame.iloc[0]['concluded_license_semicolon']=='GPL-3.0-only'
    assert frame.iloc[0]['risk_level']=='高风险'


def make_archive(tmp_path,*,index=None,conda=False):
    files={'info/index.json':json.dumps(index or dict(name='demo',version='1.0',build='h123_0',license='MIT')),'info/licenses/LICENSE':'MIT License\nCopyright (c) 2020 Upstream Authors\n'}
    stream=BytesIO()
    with tarfile.open(fileobj=stream,mode='w') as tar:
        for name,value in files.items():
            data=value.encode();info=tarfile.TarInfo(name);info.size=len(data);tar.addfile(info,BytesIO(data))
    if conda:
        import zstandard
        path=tmp_path/'demo-1.0-h123_0.conda'
        with zipfile.ZipFile(path,'w') as z:
            z.writestr('metadata.json','{}');z.writestr('info-demo.tar.zst',zstandard.ZstdCompressor().compress(stream.getvalue()))
            z.writestr('pkg-demo.tar.zst',b'Payload must not be read for metadata auditing')
    else:
        path=tmp_path/'demo-1.0-h123_0.tar.bz2'
        with tarfile.open(path,'w:bz2') as target,tarfile.open(fileobj=BytesIO(stream.getvalue())) as source:
            for member in source.getmembers():target.addfile(member,source.extractfile(member))
    return path


@pytest.mark.asyncio
@pytest.mark.parametrize('conda',[False,True])
@pytest.mark.parametrize('problem',['none','wrong_name','wrong_version','wrong_build','checksum'])
async def test_real_shared_archive_lifecycle_validates_conda_manifest_and_cleanup(tmp_path,monkeypatch,conda,problem):
    index=dict(name='demo',version='1.0',build='h123_0',license='MIT')
    if problem.startswith('wrong_'):index[{'wrong_name':'name','wrong_version':'version','wrong_build':'build'}[problem]]='foreign'
    path=make_archive(tmp_path,index=index,conda=conda)
    async def download(url,dest_path,log_queue=None):shutil.copyfile(path,dest_path)
    downloader=AsyncMock(side_effect=download);model=AsyncMock(return_value=analysis())
    monkeypatch.setattr(archives,'download_archive_with_progress',downloader)
    monkeypatch.setattr('core.utils.analyze_license_content_async',model)
    monkeypatch.setattr('core.utils.construct_copyright_notice_async',AsyncMock(return_value='Copyright (c) 2020 Upstream Authors'))
    called=[]
    async def directory(root_dir,*args,**kwargs):
        called.append(Path(root_dir));return await c._archive_analysis(root_dir,*args,**kwargs,distribution=build(),ref=c.CondaReference('bioconda','demo'))
    suffix='.conda' if conda else '.tar.bz2'
    url=DOWNLOAD.removesuffix('.tar.bz2')+suffix
    result=await archives.process_direct_archive_url(url,'1.0','demo',directory_analyzer=directory,expected_sha256='0'*64 if problem=='checksum' else hashlib.sha256(path.read_bytes()).hexdigest())
    assert bool(result)==(problem=='none')
    if result:assert result['license_type']=='MIT' and result['copyright_notice']=='Copyright (c) 2020 Upstream Authors'
    if problem!='none':model.assert_not_awaited()
    assert all(not root.exists() for root in called)
    downloader.assert_awaited_once()


def test_conda_extracts_only_info_not_payload(tmp_path):
    path=make_archive(tmp_path,conda=True)
    root=Path(archives.extract_archive(str(path),str(tmp_path/'extract')))
    assert (root/'index.json').exists() and (root/'licenses/LICENSE').exists()
    assert not list((tmp_path/'extract').rglob('pkg-*'))


@pytest.mark.parametrize('url,download',[(DOWNLOAD,True),(DOWNLOAD.replace('.tar.bz2','.conda'),True),(RELEASE,False)])
def test_final_links_distinguish_conda_download_and_versioned_evidence(url,download):
    from core.license_links import is_package_download
    assert is_package_download(url) is download


@pytest.mark.asyncio
@pytest.mark.parametrize('entry',['cli','api'])
@pytest.mark.parametrize('url',['https://github.com/org/demo','https://pypi.org/project/demo/'])
async def test_conda_manager_does_not_reroute_non_conda_sources(monkeypatch,tmp_path,entry,url):
    import api
    import main
    module=main if entry=='cli' else api
    github=AsyncMock(return_value=dict(status='success',license_type='MIT',license_files='https://github.com/org/demo/blob/1.0/LICENSE',resolved_version='1.0',used_default_branch=False))
    conda=AsyncMock(side_effect=AssertionError('Unexpected Conda routing'))
    monkeypatch.setattr(module,'process_github_repository',github)
    monkeypatch.setattr(module,'process_conda_package',conda)
    monkeypatch.chdir(tmp_path)
    rows=pd.DataFrame([dict(name='demo',version='1.0',github_url=url,package_manager='conda')])
    result=(await (main.process_all_repos(None,rows) if entry=='cli' else api._process_repositories(None,rows)))[0]
    assert result['license_type']=='MIT'
    github.assert_awaited_once();conda.assert_not_awaited()
