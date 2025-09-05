import pytest
import asyncio
from unittest.mock import AsyncMock, patch

from core import github_utils

@pytest.mark.asyncio
async def test_process_github_repository_github_license_file(monkeypatch):
    # 1. mock外部依赖
    api = AsyncMock()
    api.get_repo_info.return_value = {"name": "repo1", "default_branch": "main"}
    api.get_license.return_value = {
        "content": "MIT license text",
        "license": {"spdx_id": "MIT"},
        "_links": {"html": "https://github.com/owner/repo/blob/main/LICENSE"}
    }
    api.get_tree.return_value = {"tree": [{"path": "LICENSE", "type": "blob"}]}
    api.get_file_content.return_value = "MIT license text"
    api.get_branches.return_value = [{"name": "main"}]
    api.get_tags.return_value = []
    api._make_request.return_value = [{"commit": {"author": {"date": "2023-01-01T00:00:00Z"}}}]

    # 2. patch依赖函数
    monkeypatch.setattr(github_utils, "find_github_url_from_package_url", lambda url: None)
    monkeypatch.setattr(github_utils, "parse_github_url", lambda url: ("https://github.com/owner/repo", "", github_utils.Kind.REPO))
    monkeypatch.setattr(github_utils, "resolve_github_version", AsyncMock(return_value=("main", True)))
    monkeypatch.setattr(github_utils, "find_top_level_thirdparty_dirs", lambda tree: [])
    monkeypatch.setattr(github_utils, "save_github_tree_to_file", AsyncMock())
    monkeypatch.setattr(github_utils, "find_license_files", lambda path_map, sub_path, keywords: ["LICENSE"])
    monkeypatch.setattr(github_utils, "analyze_license_content_async", AsyncMock(return_value={"licenses": ["MIT"]}))
    monkeypatch.setattr(github_utils, "deduplicate_license_files", lambda files, owner, repo, ref: ["https://github.com/owner/repo/blob/main/LICENSE"])
    monkeypatch.setattr(github_utils, "construct_copyright_notice_async", AsyncMock(return_value="Copyright 2023 owner"))
    monkeypatch.setattr(github_utils, "get_github_last_update_time", AsyncMock(return_value="2023"))
    monkeypatch.setattr(github_utils, "find_readme", lambda tree, sub_path=None: None)

    # 3. 调用主函数
    result = await github_utils.process_github_repository(
        api=api,
        github_url="https://github.com/owner/repo",
        version=None
    )

    # 4. 验证返回结构
    assert result["status"] == "success"
    assert result["license_type"] == "MIT"
    assert "MIT" in result["license_analysis"]["licenses"]
    assert "github.com/owner/repo/blob/main/LICENSE" in result["license_files"]

@pytest.mark.asyncio
async def test_process_github_repository_not_github_url(monkeypatch):
    api = AsyncMock()
    monkeypatch.setattr(github_utils, "find_github_url_from_package_url", lambda url: None)
    monkeypatch.setattr(
        github_utils,
        "process_pypi_repository",
        AsyncMock(return_value={"status": "skipped", "license_determination_reason": "Not a GitHub repository and could not find corresponding GitHub URL"})
    )
    result = await github_utils.process_github_repository(
        api=api,
        github_url="https://pypi.org/project/xxx",
        version=None
    )
    assert result["status"] == "skipped"
    assert result["license_determination_reason"].startswith("Not a GitHub repository")

@pytest.mark.asyncio
async def test_process_github_repository_license_in_readme(monkeypatch):
    api = AsyncMock()
    api.get_repo_info.return_value = {"name": "repo2", "default_branch": "main"}
    api.get_license.return_value = None
    api.get_tree.return_value = {"tree": [{"path": "README.md", "type": "blob"}]}
    api.get_file_content.return_value = "BSD license in readme"
    api.get_branches.return_value = [{"name": "main"}]
    api.get_tags.return_value = []
    api._make_request.return_value = [{"commit": {"author": {"date": "2022-01-01T00:00:00Z"}}}]

    monkeypatch.setattr(github_utils, "find_github_url_from_package_url", lambda url: None)
    monkeypatch.setattr(github_utils, "parse_github_url", lambda url: ("https://github.com/owner/repo2", "", github_utils.Kind.REPO))
    monkeypatch.setattr(github_utils, "resolve_github_version", AsyncMock(return_value=("main", True)))
    monkeypatch.setattr(github_utils, "find_top_level_thirdparty_dirs", lambda tree: [])
    monkeypatch.setattr(github_utils, "save_github_tree_to_file", AsyncMock())
    monkeypatch.setattr(github_utils, "find_license_files", lambda path_map, sub_path, keywords: [])
    monkeypatch.setattr(github_utils, "find_readme", lambda tree, sub_path=None: "README.md")
    monkeypatch.setattr(github_utils, "analyze_license_content_async", AsyncMock(return_value={"licenses": ["BSD"]}))
    monkeypatch.setattr(github_utils, "deduplicate_license_files", lambda files, owner, repo, ref: [])
    monkeypatch.setattr(github_utils, "construct_copyright_notice_async", AsyncMock(return_value="Copyright 2022 owner"))
    monkeypatch.setattr(github_utils, "get_github_last_update_time", AsyncMock(return_value="2022"))

    result = await github_utils.process_github_repository(
        api=api,
        github_url="https://github.com/owner/repo2",
        version=None
    )
    assert result["status"] == "success"
    assert result["license_type"] == "BSD"
    assert "BSD" in result["license_analysis"]["licenses"]
    assert "README.md" in result["license_files"]

@pytest.mark.asyncio
async def test_process_github_repository_error(monkeypatch):
    api = AsyncMock()
    api.get_repo_info.side_effect = Exception("repo error")
    monkeypatch.setattr(github_utils, "find_github_url_from_package_url", lambda url: None)
    monkeypatch.setattr(github_utils, "parse_github_url", lambda url: ("https://github.com/owner/repo", "", github_utils.Kind.REPO))
    result = await github_utils.process_github_repository(
        api=api,
        github_url="https://github.com/owner/repo",
        version=None
    )
    assert result["status"] == "error"
    assert result["license_determination_reason"].startswith("Failed to get repo_info")

from core.github_utils import deduplicate_license_files

def test_deduplicate_license_files():
    license_files = [
        "LICENSE",
        "subdir/LICENSE",
        "https://raw.githubusercontent.com/owner1/repo1/main/LICENSE",
        "https://raw.githubusercontent.com/owner1/repo1/main/LICENSE",  # 重复
        "https://raw.githubusercontent.com/owner2/repo2/main/LICENSE.txt",
        "LICENSE",  # 重复
    ]
    owner = "owner1"
    repo = "repo1"
    resolved_version = "main"

    result = deduplicate_license_files(license_files, owner, repo, resolved_version)

    assert result.count("https://github.com/owner1/repo1/blob/main/LICENSE") == 1
    assert result.count("https://github.com/owner1/repo1/blob/main/subdir/LICENSE") == 1
    assert result.count("https://github.com/owner2/repo2/blob/main/LICENSE.txt") == 1  # 这里修正
    assert len(result) == 3

@pytest.mark.asyncio
async def test_find_github_url_from_package_url_success(monkeypatch):
    # 模拟 USE_LLM 为 True
    monkeypatch.setattr(github_utils, "USE_LLM", True)
    # 构造 mock response
    class MockResponse:
        text = '{"github_url": "https://github.com/owner/repo", "confidence": 0.85}'
    mock_model = AsyncMock()
    mock_model.generate_content = AsyncMock(return_value=MockResponse())
    # patch genai.GenerativeModel 返回 mock_model
    monkeypatch.setattr(github_utils.genai, "GenerativeModel", lambda model: mock_model)
    # 调用
    result = await github_utils.find_github_url_from_package_url("https://pypi.org/project/xxx", name="xxx")
    assert result == "https://github.com/owner/repo"

@pytest.mark.asyncio
async def test_find_github_url_from_package_url_low_confidence(monkeypatch):
    monkeypatch.setattr(github_utils, "USE_LLM", True)
    class MockResponse:
        text = '{"github_url": "https://github.com/owner/repo", "confidence": 0.5}'
    mock_model = AsyncMock()
    mock_model.generate_content = AsyncMock(return_value=MockResponse())
    monkeypatch.setattr(github_utils.genai, "GenerativeModel", lambda model: mock_model)
    result = await github_utils.find_github_url_from_package_url("https://pypi.org/project/xxx", name="xxx")
    assert result is None

@pytest.mark.asyncio
async def test_find_github_url_from_package_url_no_json(monkeypatch):
    monkeypatch.setattr(github_utils, "USE_LLM", True)
    class MockResponse:
        text = "not a json"
    mock_model = AsyncMock()
    mock_model.generate_content = AsyncMock(return_value=MockResponse())
    monkeypatch.setattr(github_utils.genai, "GenerativeModel", lambda model: mock_model)
    result = await github_utils.find_github_url_from_package_url("https://pypi.org/project/xxx", name="xxx")
    assert result is None

@pytest.mark.asyncio
async def test_find_github_url_from_package_url_llm_disabled(monkeypatch):
    monkeypatch.setattr(github_utils, "USE_LLM", False)
    result = await github_utils.find_github_url_from_package_url("https://pypi.org/project/xxx", name="xxx")
    assert result is None