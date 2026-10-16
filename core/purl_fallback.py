"""One final retry through the existing purl entry point; no package audit logic."""
import logging
from urllib.parse import quote, unquote, urlparse

from core.utils import (
    find_matching_version,
    get_concluded_license,
    is_blank_value,
    is_purl,
    parse_purl,
    purl_to_url,
)

logger = logging.getLogger('main')
_MANAGER_TYPES = {
    'cargo': 'cargo', 'crates.io': 'cargo',
    'npm': 'npm', 'yarn': 'npm', 'pnpm': 'npm',
    'pypi': 'pypi', 'pip': 'pypi',
    'maven': 'maven', 'gradle': 'maven',
    'golang': 'golang', 'go': 'golang',
    'pub': 'pub', 'dart': 'pub', 'flutter': 'pub',
    'nuget': 'nuget', 'deb': 'deb', 'apt': 'deb',
}


def build_fallback_purl(row):
    """Use supplied package coordinates only; never guess a namespace."""
    manager, name = row.get('package_manager'), row.get('name')
    if is_blank_value(manager) or is_blank_value(name):
        return None
    package_type = _MANAGER_TYPES.get(str(manager).strip().lower())
    if not package_type:
        return None
    name = str(name).strip()
    if package_type == 'maven':
        name = name.replace(':', '/')
        if len(name.split('/')) != 2:
            return None
    elif package_type == 'golang':
        if '/' not in name or '.' not in name.split('/')[0]:
            return None
    elif package_type == 'npm':
        if '/' in name and (not name.startswith('@') or len(name.split('/')) != 2):
            return None
    elif package_type == 'deb':
        if len(name.split('/')) > 2:
            return None
    elif '/' in name:
        return None
    if any(not part or part in ('.', '..') for part in name.split('/')):
        return None
    encoded_name = '/'.join(quote(part, safe='') for part in name.split('/'))
    version = row.get('version')
    suffix = '' if is_blank_value(version) else '@' + quote(str(version).strip(), safe='')
    purl = f'pkg:{package_type}/{encoded_name}{suffix}'
    parsed = parse_purl(purl)
    return purl if parsed and purl_to_url(parsed) else None


def _has_license(result):
    if result.get('status') != 'success':
        return False
    expression = get_concluded_license(
        result.get('license_type'), result.get('readme_license'),
        result.get('license_file_license'), result.get('status'),
    )
    return expression.strip().lower() not in {
        '', 'none', 'nan', 'noassertion', 'unknown', 'unreachable', 'unlicensed',
    }


def _matches_version(requested, resolved):
    if is_blank_value(requested) or is_blank_value(resolved):
        return False
    return find_matching_version(str(requested).strip(), [str(resolved).strip()]) is not None


def _has_versioned_link(result, purl):
    """Accept version evidence returned by existing processors, never invent a URL."""
    link = result.get('license_files')
    if not isinstance(link, str) or not link.strip() or not purl.version:
        return False
    parsed = urlparse(link.strip())
    if parsed.hostname == 'github.com':
        # The existing GitHub version resolver also supports component-prefixed tags.
        resolved = result.get('resolved_version')
        if is_blank_value(resolved) or result.get('used_default_branch') is not False:
            return False
        refs = [str(resolved)]
        for separator in ('-', '_', '/', '@'):
            prefix = purl.component_name + separator
            if str(resolved).startswith(prefix):
                refs.append(str(resolved)[len(prefix):])
        return (any(_matches_version(purl.version, ref) for ref in refs)
                and f'/blob/{resolved}/' in unquote(parsed.path))

    target = urlparse(purl_to_url(purl))
    path = unquote(parsed.path).rstrip('/')
    target_path = unquote(target.path).rstrip('/')
    if parsed.hostname == target.hostname:
        if purl.type == 'golang':
            prefix = target_path + '@'
        elif purl.type == 'npm':
            prefix = target_path + '/v/'
        elif purl.type == 'pub':
            prefix = target_path + '/versions/'
        elif purl.type == 'deb':
            # The existing deb mapping already includes the requested version.
            prefix = target_path.rsplit('/', 1)[0] + '/'
        else:
            prefix = target_path + '/'
        if path.startswith(prefix):
            linked_version = path[len(prefix):].split('/')[0]
            if _matches_version(purl.version, linked_version):
                if _matches_version(purl.version, result.get('resolved_version')):
                    return True
                # Go/Maven 的既有分支会校验注册表版本后替换链接，
                # resolved_version 仍记录 GitHub 默认分支；其他包流程
                # 的 resolved_version 记录实际包版本，必须与请求一致。
                return (purl.type in {'golang', 'maven'}
                        and result.get('used_default_branch') is True)

    # Maven returns the fetched POM URL; Debian may return a source copyright URL.
    # Require both the processor's matched version and that version in the URL.
    if purl.type in {'maven', 'deb'} and result.get('used_default_branch') is False:
        return (_matches_version(purl.version, result.get('resolved_version'))
                and any(_matches_version(purl.version, part) for part in path.split('/')))
    return False


async def retry_with_purl(row, original, retry):
    """Retry once after failures or GitHub default-branch fallback; keep better evidence."""
    purl = None
    try:
        if is_purl(row.get('github_url')):
            return original  # This row has already entered the same purl workflow.
        has_license = _has_license(original)
        requested = row.get('version')
        github_default = (
            not is_blank_value(requested)
            and original.get('used_default_branch') is True
            and any(urlparse(str(original.get(key) or '')).hostname == 'github.com'
                    for key in ('repo_url', 'license_files', 'input_url'))
        )
        if has_license and not github_default:
            return original
        purl = build_fallback_purl(row)
        if not purl:
            logger.info('PURL fallback skipped: missing or unsupported package coordinates (%s)', row.get('name'))
            return original
        logger.info('PURL fallback: %s (%s)', purl,
                    'GitHub has no matching version' if github_default else 'no license from earlier workflows')
        candidate = await retry(purl)
        parsed = parse_purl(purl)
        if not _has_license(candidate):
            logger.info('PURL fallback found no license; retaining original result: %s', purl)
            return original
        if parsed.version:
            if not _has_versioned_link(candidate, parsed):
                logger.info('PURL fallback did not confirm the requested version link; retaining original result: %s', purl)
                return original
            if has_license and _has_versioned_link(original, parsed):
                return original
        logger.info('PURL fallback selected: %s', candidate.get('license_files'))
        return candidate
    except Exception:
        logger.warning('PURL fallback failed; retaining original result: %s', purl, exc_info=True)
        return original
