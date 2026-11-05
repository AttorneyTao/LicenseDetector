"""PyPI package/version evidence boundaries and bounded failure fallbacks.

All transport and model entry points are isolated. Fixtures deliberately give
historical releases, latest releases, and GitHub different licences and notices
so replacing only a displayed link cannot accidentally satisfy the assertions.
"""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core import github_utils, pypi_utils


PACKAGE = "x-transformers"
PAGE = f"https://pypi.org/project/{PACKAGE}/"
OLD = "1.31.14"
LATEST = "2.31.11"
OLD_REPO = "https://github.com/lucidrains/x-transformers"
FOREIGN_REPO = "https://github.com/hazyresearch/flash-attention"
CODEBERG = "https://codeberg.org/lucidrains/x-transformers"
OLD_NOTICE = "Copyright (c) 2020 Historical Package Author"
LATEST_NOTICE = "Copyright (c) 2026 Latest Package Author"
FOREIGN_NOTICE = "Copyright (c) 2022 Foreign Project Contributors"


def _info(version=LATEST, license="MIT", project_urls=None, **extra):
    result = {
        "name": PACKAGE,
        "version": version,
        "license_expression": license,
        "license": license,
        "classifiers": [],
        "project_urls": {"Repository": CODEBERG} if project_urls is None else project_urls,
        "home_page": "",
        "author": "Historical Author" if version == OLD else "Latest Author",
        "description": "historical-readme" if version == OLD else "latest-readme",
    }
    result.update(extra)
    return result


def _project(info=None, releases=None):
    return {
        "info": info or _info(),
        "releases": {
            version: [{"packagetype": "sdist", "url": f"https://files.pythonhosted.org/{PACKAGE}-{version}.tar.gz"}]
            for version in (releases or [OLD, LATEST])
        },
    }


def _version(info):
    return {"info": info, "urls": [{"packagetype": "sdist"}]}


def _github(license="BSD-3-Clause", default=True):
    ref = "main" if default else f"v{OLD}"
    return {
        "status": "success",
        "repo_url": OLD_REPO,
        "resolved_version": ref,
        "used_default_branch": default,
        "license_type": license,
        "license_file_license": license,
        "readme_license": license,
        "license_analysis": {"licenses": [license], "spdx_expression": license},
        "has_license_conflict": False,
        "copyright_notice": FOREIGN_NOTICE,
        "license_files": f"{OLD_REPO}/blob/{ref}/LICENSE",
    }


@pytest.fixture
def isolated(monkeypatch):
    """Unexpected real HTTP or model calls must never escape these tests."""
    import httpx
    import requests
    from core import llm_service, utils

    def forbidden(*args, **kwargs):
        raise AssertionError("Unexpected network/model call in PyPI boundary test")

    monkeypatch.setattr(requests.Session, "request", forbidden)
    monkeypatch.setattr(httpx.AsyncClient, "request", forbidden)
    monkeypatch.setattr(llm_service, "get_llm_provider", forbidden)
    monkeypatch.setattr(pypi_utils, "complete_sync", forbidden)
    monkeypatch.setattr(github_utils, "GitHubAPI", lambda: object())
    github = AsyncMock(return_value=_github())
    monkeypatch.setattr(github_utils, "process_github_repository", github)
    readme = AsyncMock(return_value=None)
    monkeypatch.setattr(utils, "analyze_license_content_async", readme)
    reach = AsyncMock(return_value=True)
    monkeypatch.setattr(utils, "is_url_reachable", reach)
    sleep = AsyncMock()
    monkeypatch.setattr(pypi_utils.asyncio, "sleep", sleep)

    def notice(content):
        if "historical-readme" in content:
            return OLD_NOTICE
        if "latest-readme" in content:
            return LATEST_NOTICE
        return "Copyright (c) Package Authors"

    copyright = AsyncMock(side_effect=notice)
    monkeypatch.setattr(pypi_utils, "extract_copyright_info_async", copyright)
    monkeypatch.setattr(pypi_utils, "extract_copyright_info", notice)
    calls = []

    def metadata(project, version_response=None):
        def fetch(name, max_retries=3, *, version=None):
            calls.append((name, version))
            if version is None:
                return deepcopy(project)
            if isinstance(version_response, BaseException):
                raise version_response
            if version_response is None:
                raise AssertionError("Unexpected release-specific metadata request")
            return deepcopy(version_response)
        monkeypatch.setattr(pypi_utils, "_fetch_pypi_metadata", fetch)

    return SimpleNamespace(metadata=metadata, calls=calls, github=github,
                           copyright=copyright, readme=readme, reach=reach, sleep=sleep)


@pytest.mark.parametrize("repo", [CODEBERG, "https://gitlab.com/lucidrains/x-transformers"])
@pytest.mark.asyncio
async def test_non_github_source_does_not_promote_readme_dependency(repo, isolated):
    info = _info(project_urls={"Repository": repo},
                 description=f"latest-readme: Uses CUDA kernel from {FOREIGN_REPO}")
    isolated.metadata(_project(info))
    result = await pypi_utils.process_pypi_repository(PAGE, LATEST)
    assert result["status"] == "success"
    assert result["license_type"] == "MIT"
    assert result["copyright_notice"] == LATEST_NOTICE
    assert result["license_files"] == f"{PAGE}{LATEST}/"
    assert result["repo_url"] is None
    isolated.github.assert_not_awaited()


@pytest.mark.parametrize("project_urls", [{}, {"Homepage": PAGE}, {"Funding": "https://github.com/sponsors/lucidrains"}])
@pytest.mark.asyncio
async def test_description_only_links_cannot_supply_package_identity(project_urls, isolated):
    info = _info(project_urls=project_urls,
                 description=f"latest-readme: {FOREIGN_REPO}; {OLD_REPO}")
    isolated.metadata(_project(info))
    result = await pypi_utils.process_pypi_repository(PAGE)
    assert result["license_type"] == "MIT"
    assert result["license_files"] == f"{PAGE}{LATEST}/"
    isolated.github.assert_not_awaited()


@pytest.mark.parametrize("historical_license,latest_license", [("MIT", "BSD-3-Clause"), ("BSD-3-Clause", "MIT")])
@pytest.mark.asyncio
async def test_historical_release_uses_its_own_license_readme_and_notice(historical_license, latest_license, isolated):
    isolated.metadata(_project(_info(license=latest_license)),
                      _version(_info(OLD, historical_license)))
    result = await pypi_utils.process_pypi_repository(PAGE, OLD)
    assert result["status"] == "success"
    assert result["resolved_version"] == OLD
    assert result["license_type"] == historical_license
    assert result["license_files"] == f"{PAGE}{OLD}/"
    assert result["copyright_notice"] == OLD_NOTICE
    assert result["readme"] == "historical-readme"
    assert isolated.calls == [(PACKAGE, None), (PACKAGE, OLD)]
    isolated.github.assert_not_awaited()
    assert all("latest-readme" not in call.args[0] for call in isolated.copyright.await_args_list)


@pytest.mark.asyncio
async def test_incident_historical_repo_overrides_latest_codeberg_and_foreign_reference(isolated):
    latest = _info(description=f"latest-readme: {FOREIGN_REPO}")
    old = _info(OLD, project_urls={"Homepage": OLD_REPO})
    isolated.metadata(_project(latest), _version(old))
    isolated.github.return_value = _github("MIT", default=False)
    result = await pypi_utils.process_pypi_repository(PAGE, OLD)
    assert result["license_type"] == "MIT"
    assert result["resolved_version"] == OLD
    assert result["license_files"] == f"{OLD_REPO}/blob/v{OLD}/LICENSE"
    isolated.github.assert_awaited_once()
    call = isolated.github.await_args
    assert call.args[1:3] == (OLD_REPO, OLD)
    assert call.kwargs["name"] == PACKAGE


@pytest.mark.asyncio
async def test_historical_non_github_release_does_not_inherit_latest_github_repo(isolated):
    isolated.metadata(_project(_info(project_urls={"Source": FOREIGN_REPO}, license="BSD-3-Clause")),
                      _version(_info(OLD, "MIT")))
    result = await pypi_utils.process_pypi_repository(PAGE, OLD)
    assert result["license_type"] == "MIT"
    assert result["repo_url"] is None
    isolated.github.assert_not_awaited()


@pytest.mark.parametrize("repo_field,repo", [
    ("Source", OLD_REPO),
    ("Homepage", OLD_REPO),
    ("Repository", "https://github.com/acme/mono/tree/main/python/x-transformers"),
])
@pytest.mark.asyncio
async def test_explicit_github_and_monorepo_sources_remain_usable(repo_field, repo, isolated):
    isolated.metadata(_project(_info(project_urls={repo_field: repo})))
    isolated.github.return_value = _github("MIT", default=False)
    result = await pypi_utils.process_pypi_repository(PAGE, LATEST)
    assert result["status"] == "success"
    assert isolated.github.await_args.args[1] == repo
    assert isolated.github.await_args.kwargs["name"] == PACKAGE


@pytest.mark.parametrize("requested", [None, "", " ", "default", "latest", "999.0", float("nan")])
@pytest.mark.asyncio
async def test_existing_default_and_missing_version_policy_keeps_latest_value(requested, isolated):
    isolated.metadata(_project())
    result = await pypi_utils.process_pypi_repository(PAGE, requested)
    assert result["status"] == "success"
    assert result["resolved_version"] == LATEST
    assert result["license_type"] == "MIT"
    assert result["license_files"] == f"{PAGE}{LATEST}/"
    assert isolated.calls == [(PACKAGE, None)]


@pytest.mark.parametrize("requested,resolved", [(OLD, OLD), ("1.31.14.0", OLD), ("v1.31.14", OLD), ("1.0", "1.0.0")])
@pytest.mark.asyncio
async def test_exact_and_numeric_equivalent_releases_are_not_replaced_by_latest(requested, resolved, isolated):
    isolated.metadata(_project(releases=[resolved, LATEST]), _version(_info(resolved, "BSD-3-Clause")))
    result = await pypi_utils.process_pypi_repository(PAGE, requested)
    assert result["status"] == "success"
    assert result["resolved_version"] == resolved
    assert result["license_type"] == "BSD-3-Clause"
    assert result["license_files"] == f"{PAGE}{resolved}/"
    assert isolated.calls == [(PACKAGE, None), (PACKAGE, resolved)]


@pytest.mark.parametrize("unavailable", [
    pypi_utils.PyPIAPIError("version endpoint HTTP 404"),
    pypi_utils.PyPIAPIError("version endpoint timed out"),
    {},
    {"info": None},
    {"info": {"name": PACKAGE}},
    {"info": {"version": OLD}},
    _version(_info(OLD, name="other-package")),
    _version(_info("0.0.1")),
])
@pytest.mark.asyncio
async def test_unusable_historical_metadata_falls_back_to_honest_latest_evidence(unavailable, isolated):
    isolated.metadata(_project(_info(license="Apache-2.0")), unavailable)
    result = await pypi_utils.process_pypi_repository(PAGE, OLD)
    assert result["status"] == "success"
    assert result["input_version"] == OLD
    assert result["resolved_version"] == LATEST
    assert result["license_type"] == "Apache-2.0"
    assert result["license_files"] == f"{PAGE}{LATEST}/"
    assert result["copyright_notice"] == LATEST_NOTICE
    assert "latest" in result["license_determination_reason"].lower()
    isolated.github.assert_not_awaited()


@pytest.mark.parametrize("name", ["x_transformers", "X.TRANSFORMERS", "X---Transformers"])
@pytest.mark.asyncio
async def test_pep503_equivalent_release_name_is_accepted(name, isolated):
    isolated.metadata(_project(_info(license="BSD-3-Clause")),
                      _version(_info(OLD, "MIT", name=name)))
    result = await pypi_utils.process_pypi_repository(PAGE, OLD)
    assert result["status"] == "success"
    assert result["resolved_version"] == OLD
    assert result["license_type"] == "MIT"


@pytest.mark.parametrize("reachable", [True, False])
@pytest.mark.asyncio
async def test_github_default_branch_cannot_mix_wrong_license_with_versioned_pypi_page(reachable, isolated):
    isolated.metadata(_project(), _version(_info(OLD, "MIT", project_urls={"Source": OLD_REPO})))
    isolated.reach.return_value = reachable
    isolated.github.return_value = _github("BSD-3-Clause", default=True)
    result = await pypi_utils.process_pypi_repository(PAGE, OLD)
    assert result["status"] == "success"
    assert result["resolved_version"] == OLD
    assert result["license_type"] == "MIT"
    assert result["license_file_license"] is None
    assert result["copyright_notice"] == OLD_NOTICE
    assert result["license_files"] == f"{PAGE}{OLD}/"
    assert result["used_default_branch"] is False
    assert result["license_analysis"] is None
    assert result["readme_license"] is None


@pytest.mark.asyncio
async def test_github_unknown_version_value_survives_when_pypi_has_no_license(isolated):
    isolated.metadata(_project(), _version(_info(OLD, None, project_urls={"Source": OLD_REPO})))
    isolated.github.return_value = _github("BSD-3-Clause", default=True)
    result = await pypi_utils.process_pypi_repository(PAGE, OLD)
    assert result["status"] == "success"
    assert result["license_type"] == "BSD-3-Clause"
    assert result["license_file_license"] == "BSD-3-Clause"
    assert result["copyright_notice"] == FOREIGN_NOTICE
    assert result["resolved_version"] == "main"
    assert result["used_default_branch"] is True
    assert result["license_files"] == f"{OLD_REPO}/blob/main/LICENSE"


@pytest.mark.parametrize("failure", [{"status": "error", "error": "not found"}, None])
@pytest.mark.asyncio
async def test_github_non_success_uses_selected_release_metadata_without_repeated_attempts(failure, isolated):
    isolated.metadata(_project(), _version(_info(OLD, "MIT", project_urls={"Source": OLD_REPO})))
    isolated.github.return_value = failure
    result = await pypi_utils.process_pypi_repository(PAGE, OLD)
    assert result["status"] == "success"
    assert result["license_type"] == "MIT"
    assert result["resolved_version"] == OLD
    assert result["copyright_notice"] == OLD_NOTICE
    assert result["license_files"] == f"{PAGE}{OLD}/"
    isolated.github.assert_awaited_once()
    isolated.sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_github_transient_exception_retry_is_bounded_and_falls_back_to_same_release(isolated):
    isolated.metadata(_project(), _version(_info(OLD, "MIT", project_urls={"Source": OLD_REPO})))
    isolated.github.side_effect = RuntimeError("GitHub timeout")
    result = await pypi_utils.process_pypi_repository(PAGE, OLD)
    assert result["status"] == "success"
    assert result["license_type"] == "MIT"
    assert result["resolved_version"] == OLD
    assert result["license_files"] == f"{PAGE}{OLD}/"
    assert isolated.github.await_count == 3
    assert [call.args for call in isolated.sleep.await_args_list] == [(1,), (2,)]


@pytest.mark.parametrize("source_key", ["Source", "Repository"])
@pytest.mark.parametrize("other_key", ["Documentation", "Issue Tracker", "Download", "Homepage"])
@pytest.mark.asyncio
async def test_explicit_non_github_source_blocks_lower_priority_github_metadata(source_key, other_key, isolated):
    info = _info(project_urls={other_key: FOREIGN_REPO, source_key: CODEBERG},
                 home_page=FOREIGN_REPO, download_url=FOREIGN_REPO)
    isolated.metadata(_project(info))
    result = await pypi_utils.process_pypi_repository(PAGE, LATEST)
    assert result["status"] == "success"
    assert result["repo_url"] is None
    assert result["license_type"] == "MIT"
    isolated.github.assert_not_awaited()


@pytest.mark.asyncio
async def test_explicit_github_source_remains_valid_with_non_github_mirror(isolated):
    info = _info(project_urls={"Repository": CODEBERG, "Source": OLD_REPO,
                              "Documentation": FOREIGN_REPO})
    isolated.metadata(_project(info))
    isolated.github.return_value = _github("MIT", default=False)
    result = await pypi_utils.process_pypi_repository(PAGE, LATEST)
    assert result["status"] == "success"
    assert result["repo_url"] == OLD_REPO
    assert isolated.github.await_args.args[1] == OLD_REPO


@pytest.fixture
def metadata_transport(monkeypatch, isolated):
    """Exercise the real fetch helper with only its HTTP session substituted."""
    from unittest.mock import Mock
    session = Mock()
    factory = Mock(return_value=session)
    monkeypatch.setattr(pypi_utils, "_create_session", factory)
    pause = Mock()
    monkeypatch.setattr(pypi_utils.time, "sleep", pause)
    return SimpleNamespace(session=session, factory=factory, pause=pause)


@pytest.mark.parametrize("name,version,expected_url", [
    (PACKAGE, None, f"https://pypi.org/pypi/{PACKAGE}/json"),
    (PACKAGE, OLD, f"https://pypi.org/pypi/{PACKAGE}/{OLD}/json"),
    ("some-pkg", "1.0+cpu", "https://pypi.org/pypi/some-pkg/1.0%2Bcpu/json"),
    ("some/pkg", "1.0/a", "https://pypi.org/pypi/some%2Fpkg/1.0%2Fa/json"),
])
def test_fetch_helper_uses_exact_url_and_returns_version_payload(name, version, expected_url, metadata_transport):
    payload = _version(_info(OLD))
    response = metadata_transport.session.get.return_value
    response.json.return_value = payload
    result = pypi_utils._fetch_pypi_metadata(name, version=version)
    assert result == payload
    response.raise_for_status.assert_called_once()
    metadata_transport.session.get.assert_called_once_with(expected_url, timeout=(5, 15))
    metadata_transport.factory.assert_called_once_with(retries=3)
    metadata_transport.pause.assert_not_called()


@pytest.mark.parametrize("kind", ["timeout", "connection", "http_404", "http_503", "invalid_json"])
def test_fetch_helper_failures_are_bounded_on_the_selected_version_url(kind, metadata_transport):
    import requests
    session = metadata_transport.session
    if kind == "timeout":
        session.get.side_effect = requests.exceptions.Timeout("upstream timed out")
    elif kind == "connection":
        session.get.side_effect = requests.exceptions.ConnectionError("offline")
    elif kind in {"http_404", "http_503"}:
        session.get.return_value.raise_for_status.side_effect = requests.exceptions.HTTPError(kind)
    else:
        session.get.return_value.json.side_effect = requests.exceptions.JSONDecodeError("invalid", "oops", 0)
    with pytest.raises(pypi_utils.PyPIAPIError):
        pypi_utils._fetch_pypi_metadata(PACKAGE, max_retries=2, version=OLD)
    assert session.get.call_count == 2
    assert all(call.args[0] == f"https://pypi.org/pypi/{PACKAGE}/{OLD}/json"
               for call in session.get.call_args_list)
    assert [call.args for call in metadata_transport.pause.call_args_list] == [(0,)]


def test_fetch_helper_transient_failure_recovers_without_changing_version(metadata_transport):
    import requests
    from unittest.mock import Mock
    response = Mock()
    response.json.return_value = _version(_info(OLD))
    metadata_transport.session.get.side_effect = [requests.exceptions.Timeout("temporary"), response]
    result = pypi_utils._fetch_pypi_metadata(PACKAGE, max_retries=2, version=OLD)
    assert result["info"]["version"] == OLD
    assert metadata_transport.session.get.call_count == 2
    assert len({call.args[0] for call in metadata_transport.session.get.call_args_list}) == 1


def test_fetch_helper_existing_ssl_retry_keeps_the_same_release(metadata_transport):
    import requests
    from unittest.mock import Mock
    response = Mock()
    response.json.return_value = _version(_info(OLD))
    metadata_transport.session.get.side_effect = [requests.exceptions.SSLError("certificate"), response]
    result = pypi_utils._fetch_pypi_metadata(PACKAGE, version=OLD)
    assert result["info"]["version"] == OLD
    calls = metadata_transport.session.get.call_args_list
    assert len(calls) == 2
    assert all(call.args[0] == f"https://pypi.org/pypi/{PACKAGE}/{OLD}/json" for call in calls)
    assert calls[1].kwargs == {"verify": False, "timeout": (5, 15)}


@pytest.mark.parametrize("encoded_name,decoded_name", [
    ("x%2Dtransformers", "x-transformers"),
    ("x%2dtransformers", "x-transformers"),
    ("x%5Ftransformers", "x_transformers"),
    ("x%2Etransformers", "x.transformers"),
    ("X%2DTransformers", "X-Transformers"),
    ("X-Transformers", "X-Transformers"),
])
@pytest.mark.asyncio
async def test_encoded_package_url_fetches_decoded_identity_without_double_encoding(encoded_name, decoded_name, metadata_transport):
    from unittest.mock import Mock
    project_response = Mock()
    project_response.json.return_value = _project(_info(license="BSD-3-Clause"))
    release_response = Mock()
    release_response.json.return_value = _version(_info(OLD, "MIT"))
    metadata_transport.session.get.side_effect = [project_response, release_response]

    result = await pypi_utils.process_pypi_repository(
        f"https://pypi.org/project/{encoded_name}/", OLD,
    )

    assert result["status"] == "success"
    assert result["resolved_version"] == OLD
    assert result["component_name"] == decoded_name
    assert result["license_type"] == "MIT"
    assert result["copyright_notice"] == OLD_NOTICE
    assert result["license_files"] == f"https://pypi.org/project/{decoded_name}/{OLD}/"
    requested_urls = [call.args[0] for call in metadata_transport.session.get.call_args_list]
    assert requested_urls == [
        f"https://pypi.org/pypi/{decoded_name}/json",
        f"https://pypi.org/pypi/{decoded_name}/{OLD}/json",
    ]
    assert all("%25" not in url for url in requested_urls)
