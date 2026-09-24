"""Crate README copyright extraction is only needed if GitHub cannot supply it."""

import pytest

from core import crate_utils, github_utils


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("github_notice", "expected_notice", "expected_calls"),
    [
        ("Copyright 2024 GitHub Owner", "Copyright 2024 GitHub Owner", 0),
        ("Copyright 2024 original author and authors", "Copyright 2023 Crate Owner", 1),
        (None, "Copyright 2023 Crate Owner", 1),
    ],
)
async def test_crate_copyright_extraction_is_fallback(
    monkeypatch, github_notice, expected_notice, expected_calls
):
    monkeypatch.setattr(crate_utils, "_fetch_crate_info", lambda name: {
        "crate": {
            "repository": "https://github.com/example/demo",
            "license": "MIT",
            "max_stable_version": "1.0.0",
            "updated_at": "2023-01-01T00:00:00Z",
        },
        "versions": [{"num": "1.0.0"}],
    })
    monkeypatch.setattr(crate_utils, "_fetch_version_info", lambda name, version: {"version": {}})
    monkeypatch.setattr(crate_utils, "_fetch_crate_readme", lambda name, version: "README copyright data")
    monkeypatch.setattr(crate_utils, "_fetch_crate_owners", lambda name: {"users": []})
    monkeypatch.setattr(github_utils, "GitHubAPI", lambda: object())

    async def fake_github(*args):
        return {"copyright_notice": github_notice}

    monkeypatch.setattr(github_utils, "process_github_repository", fake_github)
    calls = []

    async def fake_extract(content):
        calls.append(content)
        return "Copyright 2023 Crate Owner"

    monkeypatch.setattr(crate_utils, "extract_copyright_info_async", fake_extract)
    result = await crate_utils.process_crate_repository("https://crates.io/crates/demo", "1.0.0")
    assert result["status"] == "success"
    assert result["copyright_notice"] == expected_notice
    assert calls == ["README copyright data"] * expected_calls


@pytest.mark.asyncio
async def test_crate_copyright_fallback_when_github_fails(monkeypatch):
    monkeypatch.setattr(crate_utils, "_fetch_crate_info", lambda name: {
        "crate": {"repository": "https://github.com/example/demo", "license": "MIT", "max_stable_version": "1.0.0"},
        "versions": [{"num": "1.0.0"}],
    })
    monkeypatch.setattr(crate_utils, "_fetch_version_info", lambda name, version: {"version": {}})
    monkeypatch.setattr(crate_utils, "_fetch_crate_readme", lambda name, version: "README copyright data")
    monkeypatch.setattr(crate_utils, "_fetch_crate_owners", lambda name: {"users": []})
    monkeypatch.setattr(github_utils, "GitHubAPI", lambda: object())

    async def failing_github(*args):
        raise RuntimeError("GitHub unavailable")

    monkeypatch.setattr(github_utils, "process_github_repository", failing_github)
    calls = []

    async def fake_extract(content):
        calls.append(content)
        return "Copyright 2023 Crate Owner"

    monkeypatch.setattr(crate_utils, "extract_copyright_info_async", fake_extract)
    result = await crate_utils.process_crate_repository("https://crates.io/crates/demo", "1.0.0")
    assert result["copyright_notice"] == "Copyright 2023 Crate Owner"
    assert calls == ["README copyright data"]
