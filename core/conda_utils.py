"""Version/build-scoped Anaconda metadata, with the shared archive lifecycle."""
import asyncio
from dataclasses import dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
import re
from urllib.parse import parse_qs, quote, unquote, urlparse

import aiohttp

from .archive_utils import analyze_extracted_directory, process_direct_archive_url
from .conda_spdx import narrow_metadata_licenses, reduce_only_or_later
from .utils import analyze_license_content_async, construct_copyright_notice_async, prepare_license_text

logger = logging.getLogger('main')
API = 'https://api.anaconda.org'
NAME = r'[a-z0-9][a-z0-9_.-]*'
MAX_METADATA_BYTES = 12_000_000


@dataclass(frozen=True)
class CondaReference:
    channel: str
    name: str
    version: str = ''
    basename: str = ''
    label: str = ''


def is_conda_package_url(url):
    """Keep malformed registry coordinates within this handler, without repository guessing."""
    if not isinstance(url,str):
        return False
    try:
        parsed=urlparse(url)
        return parsed.scheme in {'http','https'} and parsed.hostname in {'anaconda.org','www.anaconda.org','api.anaconda.org','conda.anaconda.org'}
    except ValueError:
        return False


def _filename_reference(channel, basename, label=''):
    filename = basename.rsplit('/',1)[-1]
    suffix = next((value for value in ('.tar.bz2','.conda') if filename.endswith(value)),None)
    if not suffix or len(basename.split('/')) != 2:
        return None
    parts = filename[:-len(suffix)].rsplit('-',2)
    subdir=basename.split('/')[0]
    if (len(parts) != 3 or not re.fullmatch(NAME,parts[0])
            or not re.fullmatch(r'[a-z0-9][a-z0-9_-]*',subdir)
            or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.+!]*',parts[1])
            or not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.]*',parts[2])):
        return None
    return CondaReference(channel,parts[0],parts[1],basename,label)


def parse_conda_reference(url):
    """URL coordinates only: a conda manager label never reroutes GitHub/PyPI input."""
    if not isinstance(url,str):
        return None
    try:
        parsed = urlparse(url)
        if parsed.scheme not in {'https','http'}:
            return None
        parts = [unquote(part) for part in parsed.path.strip('/').split('/')]
        if any(not part or part in {'.','..'} or '/' in part for part in parts):
            return None
        host = (parsed.hostname or '').lower()
        if host in {'anaconda.org','www.anaconda.org'}:
            if len(parts) in {2,3} and (len(parts)==2 or parts[2]=='files'):
                owner,name = parts[:2]
                version = parse_qs(parsed.query).get('version',[''])[0]
                ref = CondaReference(owner.lower(),name.lower(),version)
            elif len(parts)==6 and parts[3]=='download':
                ref = _filename_reference(parts[0].lower(),'/'.join(parts[4:]))
                if not ref or ref.name != parts[1].lower() or ref.version != parts[2]:
                    return None
            else:
                return None
        elif host=='api.anaconda.org' and parts[0] in {'package','release','download'}:
            if parts[0]=='package' and len(parts)==3:
                ref=CondaReference(parts[1].lower(),parts[2].lower())
            elif parts[0]=='release' and len(parts)==4:
                ref=CondaReference(parts[1].lower(),parts[2].lower(),parts[3])
            elif parts[0]=='download' and len(parts)==6:
                ref=_filename_reference(parts[1].lower(),'/'.join(parts[4:]))
                if not ref or ref.name != parts[2].lower() or ref.version != parts[3]:
                    return None
            else:
                return None
        elif host=='conda.anaconda.org':
            owner=parts[0].lower();remaining=parts[1:];label=''
            if len(remaining)==4 and remaining[0]=='label':
                label=remaining[1];remaining=remaining[2:]
            if len(remaining)!=2:
                return None
            ref=_filename_reference(owner,'/'.join(remaining),label)
        else:
            return None
        return ref if ref and re.fullmatch(NAME,ref.channel) and re.fullmatch(NAME,ref.name) else None
    except (ValueError,IndexError):
        return None


async def _fetch_json(session,url):
    async with session.get(url,allow_redirects=True) as response:
        response.raise_for_status()
        final=urlparse(str(response.url))
        if final.scheme!='https' or final.hostname!='api.anaconda.org':
            raise ValueError('Conda metadata redirected outside the registry API')
        chunks=[];size=0
        async for chunk in response.content.iter_chunked(65536):
            size+=len(chunk)
            if size>MAX_METADATA_BYTES:
                raise ValueError('Conda metadata exceeds size limit')
            chunks.append(chunk)
        return json.loads(b''.join(chunks))


def _declarations(builds):
    return list(dict.fromkeys(value.strip() for build in builds
                             if isinstance(value:=build.get('attrs',{}).get('license'),str) and value.strip()))


async def _analyze_declarations(ref,version,declarations,source):
    effective=narrow_metadata_licenses(declarations)
    text=f'Conda package: {ref.channel}/{ref.name}\nVersion: {version}\n'
    text+='Represent the remaining declarations with an SPDX expression. Do not replace different license versions or families with a set intersection. Preserve choices and exceptions within each declaration.\n'
    text+='Explicit license declarations recorded for the selected version/builds:\n'
    text+='\n'.join(f'License: {value}' for value in effective)
    try:
        analysis=await analyze_license_content_async(text,source)
    except Exception:
        return None
    if not analysis or not analysis.get('licenses'):
        return None
    try:
        expression=reduce_only_or_later(analysis.get('spdx_expression') or analysis['licenses'][0])
    except ValueError:
        return None
    if expression.upper() in {'NOASSERTION','NONE','UNKNOWN','UNLICENSED'}:
        return None
    analysis['spdx_expression']=expression
    analysis['conda_effective_declarations']=effective
    return analysis


async def _archive_analysis(root_dir,source_url,component_name,input_version=None,resolved_version=None,font_mode=False,*,distribution,ref):
    root=Path(root_dir);info=root/'info' if (root/'info').is_dir() else root
    index=json.loads((info/'index.json').read_text())
    expected=distribution.get('attrs') or {}
    if (index.get('name')!=ref.name or str(index.get('version'))!=resolved_version
            or index.get('build')!=expected.get('build')):
        raise ValueError('Conda archive package/version/build does not match selected distribution')
    license_root=info/'licenses'
    if not license_root.is_dir():
        license_root=info if any(p.is_file() and p.name.lower().startswith(('license','licence','copying','copyright')) for p in info.iterdir()) else None
    if license_root:
        result=await analyze_extracted_directory(str(license_root),source_url,ref.name,input_version,resolved_version)
        if result:
            return result
    about_path=info/'about.json'
    about=json.loads(about_path.read_text()) if about_path.exists() else {}
    declarations=list(dict.fromkeys(str(value).strip() for value in [index.get('license'),about.get('license')] if value))
    analysis=await _analyze_declarations(ref,resolved_version,declarations,source_url) if declarations else None
    if not analysis:
        return None
    content=json.dumps({'index':index,'about':about},ensure_ascii=False,indent=2)
    expression=analysis['spdx_expression']
    return dict(status='success',input_url=source_url,repo_url=source_url,component_name=ref.name,
                input_version=input_version,resolved_version=resolved_version,used_default_branch=False,
                license_type=expression,license_file_license=expression,readme_license=None,
                license_files=source_url,license_analysis=analysis,has_license_conflict=False,
                copyright_notice=await construct_copyright_notice_async(str(datetime.now().year),ref.name,ref.name,resolved_version,ref.name,license_content=content),
                license_text=prepare_license_text(content),
                license_determination_reason='Verified Conda archive info/index.json package, version and build; analyzed its license metadata')


async def _archive_fallback(ref,builds,version,input_version):
    # A listing with several unresolved builds does not authorize picking one.
    if len(builds)!=1:
        return None
    distribution=builds[0];basename=distribution['basename']
    label=f'/label/{quote(ref.label,safe="")}' if ref.label else ''
    url=f'https://conda.anaconda.org/{quote(ref.channel,safe="")}{label}/{basename}'
    async def analyzer(*args,**kwargs):
        return await _archive_analysis(*args,**kwargs,distribution=distribution,ref=ref)
    return await process_direct_archive_url(url,version,ref.name,directory_analyzer=analyzer,
                                            expected_sha256=distribution.get('sha256'))


async def process_conda_package(url,version=None,name=None):
    ref=parse_conda_reference(url)
    result=dict(input_url=url,input_version=version,component_name=ref.name if ref else name,
                resolved_version='',used_default_branch=False,status='error',license_type=None,
                license_file_license=None,readme_license=None,license_files=None,metadata_licenses='',
                error='Conda package metadata could not be verified')
    if not ref:
        return result
    requested=str(version).strip() if version is not None else ''
    requested='' if requested.lower() in {'','default','latest','none','nan','无'} else requested
    if requested and ref.version and requested!=ref.version:
        result['error']='Input version conflicts with the explicit Conda URL version'
        return result
    requested=requested or ref.version
    builds=[];declarations=[];archive_attempted=False
    try:
        async with asyncio.timeout(60),aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=25)) as session:
            package=await _fetch_json(session,f'{API}/package/{ref.channel}/{ref.name}')
            if (package.get('name')!=ref.name or package.get('full_name',f'{ref.channel}/{ref.name}').lower()!=f'{ref.channel}/{ref.name}'):
                raise ValueError('Conda package identity mismatch')
            versions=[str(v) for v in package.get('versions',[])]
            chosen=requested if requested in versions else str(package.get('latest_version') or '')
            if not chosen or chosen not in versions or ref.version and chosen!=ref.version:
                raise ValueError('The explicit Conda URL version is unavailable')
            source=f'{API}/release/{ref.channel}/{ref.name}/{quote(chosen,safe="")}'
            release=await _fetch_json(session,source)
            for build in release.get('distributions',[]):
                if build.get('distribution_type',build.get('type'))!='conda':
                    continue
                basename=build.get('basename') or ''
                candidate=_filename_reference(ref.channel,basename)
                filename=basename.rsplit('/',1)[-1]
                suffix='.tar.bz2' if filename.endswith('.tar.bz2') else '.conda'
                filename_build=filename[:-len(suffix)].rsplit('-',1)[-1]
                if (not candidate or candidate.name!=ref.name or candidate.version!=chosen
                        or str(build.get('version'))!=chosen
                        or build.get('owner',ref.channel).lower()!=ref.channel
                        or (build.get('attrs') or {}).get('build')!=filename_build):
                    raise ValueError('Conda release contains mismatched package/version coordinates')
                if ref.basename and basename!=ref.basename or ref.label and ref.label not in build.get('labels',[]):
                    continue
                builds.append(build)
            if not builds:
                raise ValueError('No matching Conda distribution')
            declarations=_declarations(builds)
            result.update(resolved_version=chosen,metadata_licenses=prepare_license_text('; '.join(declarations)) or '')
        analysis=await _analyze_declarations(ref,chosen,declarations,source) if declarations else None
        if analysis:
            raw_content=f'Conda package: {ref.channel}/{ref.name}\nVersion: {chosen}\n'+'\n'.join(f'License: {value}' for value in declarations)
            expression=analysis['spdx_expression']
            reason=f'Conda release metadata: {len(builds)} builds; all {len(declarations)} distinct license declarations retained; only identical-version only/or-later pairs may narrow'
            if requested and chosen!=requested:
                reason+=f'; requested version {requested} unavailable, registry latest {chosen} used'
            result.update(status='success',error=None,repo_url=source,license_type=expression,license_file_license=expression,
                          license_files=source,license_analysis=analysis,has_license_conflict=False,
                          license_text=prepare_license_text(raw_content),license_determination_reason=reason,
                          copyright_notice=await construct_copyright_notice_async(str(datetime.now().year),ref.name,ref.name,chosen,ref.name,license_content=raw_content))
        else:
            archive_attempted=True
            archive=await _archive_fallback(ref,builds,chosen,version)
            if archive:
                result.update(archive,input_url=url,input_version=version,metadata_licenses=prepare_license_text('; '.join(declarations)) or '')
            else:
                result['error']='No recognizable Conda license evidence; multiple unspecified builds are not reduced to one arbitrary archive' if len(builds)>1 else 'Conda metadata and shared archive analysis did not yield a license'
        details=result.setdefault('license_analysis',{})
        details['conda_metadata_licenses']=declarations
        details['conda_build_licenses']=[dict(basename=b['basename'],version=b['version'],build=b.get('attrs',{}).get('build'),subdir=b.get('attrs',{}).get('subdir'),labels=b.get('labels',[]),license=b.get('attrs',{}).get('license')) for b in builds]
        return result
    except Exception as exc:
        logger.info('[CONDA] Package audit unavailable (%s)',type(exc).__name__)
        result['error']=f'Conda audit unavailable: {type(exc).__name__}: {exc}'
        if ref.basename and not archive_attempted:
            filename=ref.basename.rsplit('/',1)[-1]
            suffix='.tar.bz2' if filename.endswith('.tar.bz2') else '.conda'
            build=filename[:-len(suffix)].rsplit('-',1)[-1]
            candidate=dict(basename=ref.basename,version=ref.version,attrs={'build':build})
            archive=await _archive_fallback(ref,[candidate],ref.version,version)
            if archive:
                result.update(archive,input_url=url,input_version=version,metadata_licenses=prepare_license_text('; '.join(declarations)) or '')
        return result
