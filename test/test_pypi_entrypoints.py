"""Exercise PyPI evidence selection through real CLI, API and purl routing.

Only external metadata/GitHub/model boundaries are mocked. The version resolver,
PyPI processor, purl retry, final link selection and output builders remain real.
"""
import io
from copy import deepcopy
from unittest.mock import AsyncMock

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api
import main
from core import github_utils, pypi_utils


PACKAGE = "x-transformers"
VERSION = "1.31.14"
PAGE = f"https://pypi.org/project/{PACKAGE}/"
VERSION_PAGE = f"{PAGE}{VERSION}/"
SOURCE = "https://github.com/lucidrains/x-transformers"
PURL = f"pkg:pypi/{PACKAGE}@{VERSION}"
OLD_NOTICE = "Copyright (c) 2020 Phil Wang"


def _info(version, source, license_expression="MIT"):
    return {
        "name": PACKAGE,
        "version": version,
        "project_urls": {"Repository": source},
        "home_page": "",
        "license_expression": license_expression,
        "license": license_expression or "",
        "classifiers": [],
        "author": "Phil Wang",
        "description": (
            "Package documentation. Third-party kernel: "
            "https://github.com/hazyresearch/flash-attention. " + OLD_NOTICE
        ),
    }


def _install(monkeypatch, mode):
    """Modes cover authoritative tags, stale branches and non-GitHub sources."""
    github_calls, metadata_calls, dispatch_calls = [], [], []
    actual_dispatch = github_utils.process_github_repository
    source = "https://codeberg.org/lucidrains/x-transformers" if mode == "codeberg" else SOURCE
    old_info = _info(VERSION, source, None if mode == "registry_unknown" else "MIT")
    metadata = {
        "info": _info("2.31.11", "https://codeberg.org/lucidrains/x-transformers"),
        "releases": {
            VERSION: [{"packagetype": "sdist"}],
            "2.31.11": [{"packagetype": "sdist"}],
        },
    }

    def fetch(name, max_retries=3, *, version=None):
        assert name == PACKAGE
        assert version in (None, VERSION)
        metadata_calls.append(version)
        return deepcopy(metadata if version is None else {"info": old_info})

    async def github(client, url, version, **kwargs):
        github_calls.append((url, version, kwargs.get("name")))
        assert url == SOURCE, "Third-party README links must never be audited"
        if mode == "github_error":
            return {"status": "error", "error": "Repository unavailable"}
        tag_matched = mode in ("tag", "exact_tag")
        selected = "MIT" if tag_matched else "BSD-3-Clause"
        ref = "v" + VERSION if mode == "tag" else VERSION if mode == "exact_tag" else "main"
        return {
            "status": "success", "repo_url": url,
            "resolved_version": ref, "used_default_branch": not tag_matched,
            "license_files": f"{url}/blob/{ref}/LICENSE",
            "license_type": selected, "license_file_license": selected,
            "license_analysis": {"licenses": [selected]},
            "readme_license": selected, "has_license_conflict": False,
            "copyright_notice": OLD_NOTICE if tag_matched else "Copyright Unrelated Latest Authors",
        }

    async def dispatch(client, url, version, **kwargs):
        dispatch_calls.append(url)
        if url == "https://example.invalid/component":
            return {"status": "error", "error": "Original lookup failed"}
        return await actual_dispatch(client, url, version, **kwargs)

    monkeypatch.setattr(pypi_utils, "_fetch_pypi_metadata", fetch)
    monkeypatch.setattr(github_utils, "GitHubAPI", lambda: object())
    monkeypatch.setattr(github_utils, "process_github_repository", github)
    monkeypatch.setattr(api, "process_github_repository", dispatch)
    monkeypatch.setattr(main, "process_github_repository", dispatch)
    monkeypatch.setattr("core.utils.analyze_license_content_async", AsyncMock(return_value={"licenses": []}))
    monkeypatch.setattr(pypi_utils, "extract_copyright_info_async", AsyncMock(return_value=OLD_NOTICE))

    async def unexpected_network(*args, **kwargs):
        raise AssertionError("Registry evidence must not require another network fallback")

    monkeypatch.setattr("core.utils.is_url_reachable", unexpected_network)
    monkeypatch.setattr(api, "get_github_api", AsyncMock(return_value=object()))
    monkeypatch.setattr(main, "initialize_api", AsyncMock(return_value=object()))
    return github_calls, metadata_calls, dispatch_calls


def _row(route="url"):
    return {
        "name": PACKAGE, "version": VERSION,
        "github_url": {"url": PAGE, "purl": PURL, "retry": "https://example.invalid/component"}[route],
        "package_manager": "pypi", "serial_number": "00017",
    }


def _assert_result(result, mode, route):
    assert result["status"] == "success"
    assert result["component_name"] == PACKAGE
    assert result["resolved_version"] == VERSION
    assert result["license_type"] == "MIT"
    assert result["copyright_notice"] == OLD_NOTICE
    assert result["input_url"] == _row(route)["github_url"]
    assert result["serial_number"] == "00017"
    assert result["package_manager"] == "pypi"
    if mode in ("tag", "exact_tag"):
        ref = "v" + VERSION if mode == "tag" else VERSION
        expected_link = f"{SOURCE}/blob/{ref}/LICENSE"
    else:
        expected_link = VERSION_PAGE
    assert result["license_files"] == expected_link
    assert "BSD-3-Clause" not in str(result)
    assert "Unrelated Latest Authors" not in str(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["cli", "api"])
@pytest.mark.parametrize(("route", "mode"), [
    pytest.param(route, mode, marks=pytest.mark.xfail(
        condition=route == "retry" and mode == "tag",
        strict=True,
        reason="Pre-existing purl gate requires /blob/release-version/ and rejects the v-prefixed GitHub tag",
    ))
    for route in ("url", "purl", "retry")
    for mode in ("tag", "exact_tag", "default_branch", "codeberg", "github_error")
])
async def test_pypi_final_evidence_survives_all_entrypoints(monkeypatch, tmp_path, entry, route, mode):
    github_calls, metadata_calls, dispatch_calls = _install(monkeypatch, mode)
    monkeypatch.chdir(tmp_path)
    rows = pd.DataFrame([_row(route)])
    results = await (main.process_all_repos(None, rows) if entry == "cli"
                     else api._process_repositories(None, rows))
    assert len(results) == 1
    _assert_result(results[0], mode, route)
    assert metadata_calls == [None, VERSION]
    assert github_calls == ([] if mode == "codeberg" else [(SOURCE, VERSION, PACKAGE)])
    assert len(dispatch_calls) == (2 if route == "retry" else 1)
    output = api._generate_output(results)
    assert output.iloc[0]["concluded_license"] == "MIT"
    assert output.iloc[0]["concluded_license_semicolon"] == "MIT"
    assert output.columns[:2].tolist() == ["serial_number", "package_manager"]
    assert output.columns.get_loc("concluded_license_semicolon") == output.columns.get_loc("concluded_license") + 1
    assert not any(column.startswith("_") for column in output.columns)


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["cli", "api"])
@pytest.mark.parametrize("route", ["url", "purl"])
async def test_registry_without_license_keeps_honest_github_fallback_and_bounded_retry(
    monkeypatch, tmp_path, entry, route,
):
    github_calls, metadata_calls, dispatch_calls = _install(monkeypatch, "registry_unknown")
    monkeypatch.chdir(tmp_path)
    rows = pd.DataFrame([_row(route)])
    results = await (main.process_all_repos(None, rows) if entry == "cli"
                     else api._process_repositories(None, rows))
    result = results[0]
    assert result["status"] == "success"
    assert result["license_type"] == "BSD-3-Clause"
    assert result["license_file_license"] == "BSD-3-Clause"
    assert result["resolved_version"] == "main"
    assert result["used_default_branch"] is True
    assert result["license_files"] == f"{SOURCE}/blob/main/LICENSE"
    assert result["copyright_notice"] == "Copyright Unrelated Latest Authors"
    assert "default-branch" in result["license_determination_reason"]
    expected_attempts = 2 if route == "url" else 1
    assert len(github_calls) == expected_attempts
    assert metadata_calls == [None, VERSION] * expected_attempts
    assert len(dispatch_calls) == expected_attempts
    assert api._generate_output(results).iloc[0]["concluded_license"] == "BSD-3-Clause"


def _xlsx(rows):
    stream = io.BytesIO()
    pd.DataFrame(rows).to_excel(stream, index=False)
    return stream.getvalue()


def _assert_excel(content):
    output = pd.read_excel(io.BytesIO(content), dtype={"serial_number": str})
    assert output.shape[0] == 2
    assert output["input_url"].tolist() == [PAGE, PURL]
    assert output["resolved_version"].tolist() == [VERSION, VERSION]
    assert output["concluded_license"].tolist() == ["MIT", "MIT"]
    assert output["concluded_license_semicolon"].tolist() == ["MIT", "MIT"]
    assert output["license_files"].tolist() == [VERSION_PAGE, VERSION_PAGE]
    assert output["copyright_notice"].tolist() == [OLD_NOTICE, OLD_NOTICE]
    assert output["serial_number"].tolist() == ["00017", "00017"]
    assert output["package_manager"].tolist() == ["pypi", "pypi"]
    assert not any(column.startswith("_") for column in output.columns)


def test_api_upload_exports_matching_license_evidence(monkeypatch):
    _install(monkeypatch, "default_branch")
    with TestClient(api.app) as client:
        response = client.post(
            "/api/v1/analyze-and-download",
            files={"file": ("input.xlsx", _xlsx([_row(), _row("purl")]),
                             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        )
    assert response.status_code == 200
    _assert_excel(response.content)


@pytest.mark.asyncio
async def test_cli_local_excel_exports_matching_license_evidence(monkeypatch, tmp_path):
    _install(monkeypatch, "default_branch")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "input.xlsx").write_bytes(_xlsx([_row(), _row("purl")]))
    await main.main_async()
    _assert_excel((tmp_path / "outputs/output_latest.xlsx").read_bytes())
