# -*- coding: utf-8 -*-
"""PyPI 元数据的 GitHub 仓库 URL 提取。

PyPI 的 project_urls 里混杂 Funding / Issue Tracker / Changelog 等非仓库页面，
直接取「第一个含 github.com 的值」会把赞助页当仓库（pydantic、attrs、starlette
都中招），GitHub 流程解析失败后静默回落 PyPI 元数据，丢失 LICENSE 全文与版权声明。
"""
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pypi_utils import (
    _extract_github_repo_url,
    _normalize_github_repo_url,
    _pypi_project_page,
    process_pypi_repository,
)


class TestNormalizeGithubRepoUrl:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("https://github.com/foo/bar", "https://github.com/foo/bar"),
            ("https://github.com/foo/bar/", "https://github.com/foo/bar"),
            ("https://github.com/foo/bar.git", "https://github.com/foo/bar"),
            ("git+https://github.com/foo/bar.git", "https://github.com/foo/bar"),
            ("git@github.com:foo/bar.git", "https://github.com/foo/bar"),
            ("git://github.com/foo/bar", "https://github.com/foo/bar"),
            ("github.com/foo/bar", "https://github.com/foo/bar"),
            ("https://www.github.com/foo/bar", "https://github.com/foo/bar"),
        ],
    )
    def test_accepts_common_repo_url_forms(self, raw, expected):
        assert _normalize_github_repo_url(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        [
            "https://github.com/sponsors/samuelcolvin",  # 赞助页（pydantic 曾中招）
            "https://github.com/sponsors/hynek",          # attrs 曾中招
            "https://github.com/topics/python",
            "https://github.com/samuelcolvin",            # 用户页，单段
            "https://docs.github.com/en/actions",         # 子域
            "https://raw.githubusercontent.com/foo/bar/main/x",
            "https://gitlab.com/foo/bar",
            "https://pypi.org/project/foo/",
            "",
            None,
        ],
    )
    def test_rejects_non_repository_urls(self, raw):
        assert _normalize_github_repo_url(raw) is None

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("https://github.com/foo/bar/issues", "https://github.com/foo/bar"),
            ("https://github.com/foo/bar/issues/123", "https://github.com/foo/bar"),
            ("https://github.com/foo/bar/wiki", "https://github.com/foo/bar"),
            ("https://github.com/foo/bar/releases", "https://github.com/foo/bar"),
            ("https://github.com/foo/bar/tags", "https://github.com/foo/bar"),
            ("https://github.com/foo/bar/blob/main/CHANGES.md", "https://github.com/foo/bar"),
            ("https://github.com/foo/bar/actions?query=workflow%3ACI", "https://github.com/foo/bar"),
        ],
    )
    def test_strips_non_repo_suffixes(self, raw, expected):
        assert _normalize_github_repo_url(raw) == expected

    def test_keeps_monorepo_subdirectory(self):
        """tree 子目录路径要保留，供 GitHub 流程定位 monorepo 子包。"""
        url = "https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/core/azure-core"
        assert _normalize_github_repo_url(url) == url


class TestExtractGithubRepoUrl:
    def test_prefers_source_over_funding(self):
        """pydantic：Funding 排在 Source 前，且指向赞助页。"""
        info = {
            "project_urls": {
                "Funding": "https://github.com/sponsors/samuelcolvin",
                "Source": "https://github.com/pydantic/pydantic",
            }
        }
        assert _extract_github_repo_url(info) == "https://github.com/pydantic/pydantic"

    def test_falls_back_to_home_page(self):
        info = {"project_urls": {}, "home_page": "https://github.com/psf/requests"}
        assert _extract_github_repo_url(info) == "https://github.com/psf/requests"

    def test_falls_back_to_description(self):
        """只在 README 里放了仓库链接的包。"""
        info = {
            "project_urls": {"Funding": "https://github.com/sponsors/someone"},
            "description": "Read the code: https://github.com/foo/bar for details.",
        }
        assert _extract_github_repo_url(info) == "https://github.com/foo/bar"

    def test_deprecated_key_still_usable_when_it_is_repo_root(self):
        """sqlalchemy 的唯一 GitHub 链接挂在 Issue Tracker 上，指向仓库根。"""
        info = {"project_urls": {"Issue Tracker": "https://github.com/sqlalchemy/sqlalchemy/"}}
        assert _extract_github_repo_url(info) == "https://github.com/sqlalchemy/sqlalchemy"

    def test_deprecated_key_not_preferred_over_real_source(self):
        info = {
            "project_urls": {
                "Issue Tracker": "https://github.com/foo/bar/issues",
                "Repository": "https://github.com/foo/baz",
            }
        }
        assert _extract_github_repo_url(info) == "https://github.com/foo/baz"

    def test_returns_none_when_no_github_repo(self):
        info = {"project_urls": {"Homepage": "https://www.sqlalchemy.org"}, "home_page": ""}
        assert _extract_github_repo_url(info) is None


class TestPypiProjectPage:
    @pytest.mark.parametrize(
        "package,version,expected",
        [
            ("foo-pkg", "1.0.0", "https://pypi.org/project/foo-pkg/1.0.0/"),
            ("foo-pkg", None, "https://pypi.org/project/foo-pkg/"),
        ],
    )
    def test_builds_description_page_url(self, package, version, expected):
        """Description 页（带许可证分类器与 README），不是只列构件的 #files 页。"""
        assert _pypi_project_page(package, version) == expected


def _metadata(project_urls=None, description="readme", home_page=""):
    return {
        "info": {
            "version": "1.0.0",
            "license_expression": "MIT",
            "license": "MIT",
            "classifiers": [],
            "project_urls": project_urls or {},
            "home_page": home_page,
            "author": "Foo Author",
            "description": description,
        },
        "releases": {"1.0.0": [{"packagetype": "sdist"}]},
    }


class TestProcessPypiRepositoryPassesNameAndUrl:
    """端到端：传给 GitHub 流程的 URL 与 name 必须正确。"""

    async def _run(self, metadata):
        captured = {}

        async def fake_github(api, url, version, **kwargs):
            captured["url"] = url
            captured["name"] = kwargs.get("name")
            return {
                "status": "success",
                "used_default_branch": False,
                "license_files": "https://github.com/pydantic/pydantic/blob/v1.0.0/LICENSE",
                "license_type": "MIT",
                "license_file_license": "MIT",
                "copyright_notice": "Copyright (c) 2020 Foo",
            }

        with patch("core.pypi_utils._fetch_pypi_metadata", return_value=metadata), \
             patch("core.github_utils.process_github_repository", new=fake_github), \
             patch("core.github_utils.GitHubAPI", return_value=MagicMock()):
            result = await process_pypi_repository("https://pypi.org/project/pkg/", "1.0.0")
        return result, captured

    @pytest.mark.asyncio
    async def test_funding_page_does_not_derail_github_analysis(self):
        """pydantic 场景：赞助页不能顶掉真正的 Source 仓库。"""
        result, captured = await self._run(
            _metadata(
                {
                    "Funding": "https://github.com/sponsors/samuelcolvin",
                    "Source": "https://github.com/pydantic/pydantic",
                }
            )
        )
        assert captured["url"] == "https://github.com/pydantic/pydantic"
        assert captured["name"] == "pkg"  # 供 monorepo 子目录定位兜底
        assert result["repo_url"] == "https://github.com/pydantic/pydantic"
        assert result["license_determination_reason"] == "Analyzed via GitHub repository (primary source)"

    @pytest.mark.asyncio
    async def test_monorepo_tree_url_preserved_and_name_passed(self):
        metadata = _metadata({"Source": "https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/core/azure-core"})
        result, captured = await self._run(metadata)
        assert captured["url"] == "https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/core/azure-core"
        assert captured["name"] == "pkg"

    @pytest.mark.asyncio
    async def test_no_repo_falls_back_to_pypi_description_page(self):
        result, captured = await self._run(_metadata({}))
        assert captured.get("url") is None
        assert result["license_files"] == "https://pypi.org/project/pkg/1.0.0/"
        assert "#files" not in result["license_files"]

    @pytest.mark.asyncio
    async def test_non_success_does_not_retry_three_times(self):
        """拿到结果即退出：确定性失败（404 等）重试只会浪费 GitHub 配额。"""
        calls = []

        async def failing_github(api, url, version, **kwargs):
            calls.append(url)
            return {"status": "error", "error": "repo not found"}

        with patch("core.pypi_utils._fetch_pypi_metadata", return_value=_metadata({"Source": "https://github.com/foo/bar"})), \
             patch("core.github_utils.process_github_repository", new=failing_github), \
             patch("core.github_utils.GitHubAPI", return_value=MagicMock()):
            result = await process_pypi_repository("https://pypi.org/project/pkg/", "1.0.0")

        assert len(calls) == 1
        assert result["status"] == "success"
        assert result["license_files"] == "https://pypi.org/project/pkg/1.0.0/"
        assert result["license_determination_reason"] == "Fetched from PyPI registry"
