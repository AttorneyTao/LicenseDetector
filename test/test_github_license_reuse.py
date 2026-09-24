"""Only reuse a verified analysis of the exact root LICENSE in one scan."""

from unittest.mock import AsyncMock

import pytest

from core import github_utils


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("selected_path", "selected_content", "first_licenses", "expected_calls"),
    [
        ("LICENSE", "MIT license text", ["MIT"], 1),
        ("LICENSE", "changed license text", ["MIT"], 2),
        ("LICENSE.md", "MIT license text", ["MIT"], 2),
        ("LICENSE", "MIT license text", [], 2),
    ],
)
async def test_step_10_reuses_only_identical_valid_step_5_analysis(
    monkeypatch, selected_path, selected_content, first_licenses, expected_calls
):
    api = AsyncMock()
    api.get_repo_info.return_value = {"name": "repo", "default_branch": "v1"}
    api.get_license.return_value = {
        "path": "LICENSE",
        "content": "MIT license text",
        "_links": {"html": "https://github.com/owner/repo/blob/v1/LICENSE"},
    }
    api.get_tree.return_value = {"tree": [
        {"path": "LICENSE", "type": "blob"},
        {"path": selected_path, "type": "blob"},
    ]}
    api.get_file_content.return_value = selected_content
    monkeypatch.setattr(github_utils, "resolve_github_version", AsyncMock(return_value=("v1", False)))
    monkeypatch.setattr(github_utils, "save_github_tree_to_file", AsyncMock())
    monkeypatch.setattr(github_utils, "find_readme", lambda tree, sub_path=None: None)
    monkeypatch.setattr(github_utils, "select_primary_license_file", AsyncMock(return_value={
        "path": selected_path,
        "filename": "LICENSE",
        "directory": selected_path.rsplit("/", 1)[0] if "/" in selected_path else "",
        "url": f"https://github.com/owner/repo/blob/v1/{selected_path}",
    }))
    monkeypatch.setattr(github_utils, "construct_copyright_notice_async", AsyncMock(return_value="Copyright owner"))
    monkeypatch.setattr(github_utils, "get_github_last_update_time", AsyncMock(return_value="2024"))
    analyses = []

    async def fake_analyze(content, url):
        analyses.append((content, url))
        licenses = first_licenses if len(analyses) == 1 else ["MIT"]
        return {"licenses": licenses, "source_url": url}

    monkeypatch.setattr(github_utils, "analyze_license_content_async", fake_analyze)
    result = await github_utils.process_github_repository(
        api, "https://github.com/owner/repo", "v1", name="repo-package"
    )
    assert result["status"] == "success"
    assert len(analyses) == expected_calls
    assert result["license_analysis"]["source_url"] == f"https://github.com/owner/repo/blob/v1/{selected_path}"
    assert result["license_type"] == "MIT"
