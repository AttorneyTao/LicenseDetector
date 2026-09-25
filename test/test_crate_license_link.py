"""The crate result displays a versioned registry page without changing its audit."""

import pytest

from core import crate_utils, github_utils


@pytest.mark.asyncio
@pytest.mark.parametrize("github_used_default_branch", [False, True, None])
async def test_versioned_crate_page_wins_over_github_license_link(
    monkeypatch, github_used_default_branch
):
    github_calls = []

    monkeypatch.setattr(crate_utils, "_fetch_crate_info", lambda name: {
        "crate": {
            "repository": "https://github.com/example/demo",
            "max_stable_version": "1.2.3",
        },
        "versions": [{"num": "1.2.3"}],
    })
    monkeypatch.setattr(crate_utils, "_fetch_version_info", lambda name, version: {
        "version": {"license": "MIT"}
    })
    monkeypatch.setattr(crate_utils, "_fetch_crate_readme", lambda name, version: None)
    monkeypatch.setattr(crate_utils, "_fetch_crate_owners", lambda name: {"users": []})
    monkeypatch.setattr(crate_utils, "_fetch_github_readme", lambda *args: None)
    monkeypatch.setattr(github_utils, "GitHubAPI", lambda: object())

    async def fake_resolve(**kwargs):
        return "1.2.3", False

    async def fake_github(api, url, version):
        github_calls.append((url, version))
        return {
            "license_files": "https://github.com/example/demo/blob/v1.2.3/LICENSE",
            "license_file_license": "Apache-2.0",
            "license_analysis": {"licenses": ["Apache-2.0"]},
            "copyright_notice": "Copyright 2024 Example",
            "used_default_branch": github_used_default_branch,
        }

    monkeypatch.setattr(crate_utils, "resolve_crate_version", fake_resolve)
    monkeypatch.setattr(github_utils, "process_github_repository", fake_github)

    result = await crate_utils.process_crate_repository(
        "https://crates.io/crates/demo", "1.2.3"
    )

    assert github_calls == [("https://github.com/example/demo", "1.2.3")]
    assert result["status"] == "success"
    assert result["license_files"] == "https://crates.io/crates/demo/1.2.3"
    assert result["license_file_license"] == "Apache-2.0"
    assert result["license_analysis"] == {"licenses": ["Apache-2.0"]}
    assert result["copyright_notice"] == "Copyright 2024 Example"
    assert result["used_default_branch"] is (
        github_used_default_branch if github_used_default_branch is not None else False
    )


@pytest.mark.asyncio
async def test_crate_without_github_still_uses_versioned_registry_page(monkeypatch):
    monkeypatch.setattr(crate_utils, "_fetch_crate_info", lambda name: {
        "crate": {"max_stable_version": "0.1.0"},
        "versions": [{"num": "0.1.0"}],
    })
    monkeypatch.setattr(crate_utils, "_fetch_version_info", lambda name, version: {
        "version": {"license": "MIT"}
    })
    monkeypatch.setattr(crate_utils, "_fetch_crate_readme", lambda name, version: None)
    monkeypatch.setattr(crate_utils, "_fetch_crate_owners", lambda name: {"users": []})

    async def fake_resolve(**kwargs):
        return "0.1.0", False

    async def fake_copyright(text):
        return None

    monkeypatch.setattr(crate_utils, "resolve_crate_version", fake_resolve)
    monkeypatch.setattr(crate_utils, "extract_copyright_info_async", fake_copyright)
    result = await crate_utils.process_crate_repository(
        "https://crates.io/api/v1/crates/demo/0.1.0/download"
    )

    assert result["status"] == "success"
    assert result["license_files"] == "https://crates.io/crates/demo/0.1.0"
    assert result["license_type"] == "MIT"
