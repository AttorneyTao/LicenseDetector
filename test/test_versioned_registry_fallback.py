# -*- coding: utf-8 -*-
"""Tests for versioned registry link fallback when GitHub has no matching tag."""
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.go_utils import extract_module_path, build_versioned_pkggo_license_url
from core.maven_utils import build_versioned_maven_license_url


class TestExtractModulePath:
    def test_pkggo_url_with_protocol(self):
        assert extract_module_path("https://pkg.go.dev/go.uber.org/atomic") == "go.uber.org/atomic"

    def test_bare_module_path_with_query(self):
        assert extract_module_path("go.uber.org/atomic?tab=doc") == "go.uber.org/atomic"

    def test_godev_prefix(self):
        assert extract_module_path("https://go.dev/golang.org/x/net") == "golang.org/x/net"


class _FakeResp:
    def __init__(self, status):
        self.status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class _FakeSession:
    """Fake aiohttp.ClientSession: maps URL -> status, 404 otherwise."""

    def __init__(self, status_map):
        self.status_map = status_map
        self.requested = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def get(self, url, **kwargs):
        self.requested.append(url)
        return _FakeResp(self.status_map.get(url, 404))


class TestBuildVersionedPkggoLicenseUrl:
    @pytest.mark.asyncio
    async def test_version_found_with_v_prefix(self):
        session = _FakeSession({
            "https://proxy.golang.org/go.uber.org/atomic/@v/v1.9.0.info": 200,
        })
        with patch("core.go_utils.aiohttp.ClientSession", return_value=session):
            url = await build_versioned_pkggo_license_url(
                "https://pkg.go.dev/go.uber.org/atomic", "1.9.0"
            )
        assert url == "https://pkg.go.dev/go.uber.org/atomic@v1.9.0?tab=licenses"

    @pytest.mark.asyncio
    async def test_version_found_as_is(self):
        session = _FakeSession({
            "https://proxy.golang.org/go.uber.org/atomic/@v/v1.9.0.info": 200,
        })
        with patch("core.go_utils.aiohttp.ClientSession", return_value=session):
            url = await build_versioned_pkggo_license_url(
                "https://pkg.go.dev/go.uber.org/atomic", "v1.9.0"
            )
        assert url == "https://pkg.go.dev/go.uber.org/atomic@v1.9.0?tab=licenses"

    @pytest.mark.asyncio
    async def test_version_not_on_proxy_returns_none(self):
        session = _FakeSession({})
        with patch("core.go_utils.aiohttp.ClientSession", return_value=session):
            url = await build_versioned_pkggo_license_url(
                "https://pkg.go.dev/go.uber.org/atomic", "99.99.99"
            )
        assert url is None

    @pytest.mark.asyncio
    async def test_no_version_returns_none(self):
        url = await build_versioned_pkggo_license_url(
            "https://pkg.go.dev/go.uber.org/atomic", None
        )
        assert url is None


class TestBuildVersionedMavenLicenseUrl:
    @pytest.mark.asyncio
    async def test_explicit_version_reachable(self):
        with patch("core.utils.is_url_reachable", new=AsyncMock(return_value=True)) as mock_check:
            url = await build_versioned_maven_license_url(
                "https://mvnrepository.com/artifact/org.slf4j/slf4j-api", "1.7.36"
            )
        assert url == "https://mvnrepository.com/artifact/org.slf4j/slf4j-api/1.7.36"
        # 校验走的是 repo1.maven.org 源站，而不是 mvnrepository.com
        mock_check.assert_awaited_once_with(
            "https://repo1.maven.org/maven2/org/slf4j/slf4j-api/1.7.36/"
        )

    @pytest.mark.asyncio
    async def test_version_from_url_path(self):
        with patch("core.utils.is_url_reachable", new=AsyncMock(return_value=True)):
            url = await build_versioned_maven_license_url(
                "https://mvnrepository.com/artifact/org.slf4j/slf4j-api/1.7.36"
            )
        assert url == "https://mvnrepository.com/artifact/org.slf4j/slf4j-api/1.7.36"

    @pytest.mark.asyncio
    async def test_repo1_maven_central_url_converted(self):
        with patch("core.utils.is_url_reachable", new=AsyncMock(return_value=True)):
            url = await build_versioned_maven_license_url(
                "https://repo1.maven.org/maven2/org/slf4j/slf4j-api/1.7.36/slf4j-api-1.7.36.jar"
            )
        assert url == "https://mvnrepository.com/artifact/org.slf4j/slf4j-api/1.7.36"

    @pytest.mark.asyncio
    async def test_version_missing_returns_none(self):
        url = await build_versioned_maven_license_url(
            "https://mvnrepository.com/artifact/org.slf4j/slf4j-api"
        )
        assert url is None

    @pytest.mark.asyncio
    async def test_unreachable_version_returns_none(self):
        with patch("core.utils.is_url_reachable", new=AsyncMock(return_value=False)):
            url = await build_versioned_maven_license_url(
                "https://mvnrepository.com/artifact/org.slf4j/slf4j-api", "0.0.0-nonexistent"
            )
        assert url is None

    @pytest.mark.asyncio
    async def test_unparseable_url_returns_none(self):
        url = await build_versioned_maven_license_url("https://example.com/not-maven", "1.0.0")
        assert url is None


class TestBuildVersionedPkggoLicenseUrlEdgeCases:
    @pytest.mark.asyncio
    async def test_candidate_order_raw_then_v_prefixed(self):
        session = _FakeSession({})
        with patch("core.go_utils.aiohttp.ClientSession", return_value=session):
            await build_versioned_pkggo_license_url("https://pkg.go.dev/go.uber.org/atomic", "1.9.0")
        assert session.requested == [
            "https://proxy.golang.org/go.uber.org/atomic/@v/1.9.0.info",
            "https://proxy.golang.org/go.uber.org/atomic/@v/v1.9.0.info",
        ]

    @pytest.mark.asyncio
    async def test_v_prefixed_version_tried_once(self):
        session = _FakeSession({})
        with patch("core.go_utils.aiohttp.ClientSession", return_value=session):
            await build_versioned_pkggo_license_url("https://pkg.go.dev/go.uber.org/atomic", "v1.9.0")
        assert session.requested == [
            "https://proxy.golang.org/go.uber.org/atomic/@v/v1.9.0.info",
        ]

    @pytest.mark.asyncio
    async def test_proxy_request_exception_returns_none(self):
        class _RaisingSession(_FakeSession):
            def get(self, url, **kwargs):
                raise RuntimeError("boom")

        with patch("core.go_utils.aiohttp.ClientSession", return_value=_RaisingSession({})):
            url = await build_versioned_pkggo_license_url(
                "https://pkg.go.dev/go.uber.org/atomic", "1.9.0"
            )
        assert url is None


class TestBuildVersionedMavenLicenseUrlEdgeCases:
    @pytest.mark.asyncio
    async def test_explicit_version_overrides_url_version(self):
        with patch("core.utils.is_url_reachable", new=AsyncMock(return_value=True)) as mock_check:
            url = await build_versioned_maven_license_url(
                "https://mvnrepository.com/artifact/org.slf4j/slf4j-api/1.0.0", "2.0.0"
            )
        assert url == "https://mvnrepository.com/artifact/org.slf4j/slf4j-api/2.0.0"
        mock_check.assert_awaited_once_with(
            "https://repo1.maven.org/maven2/org/slf4j/slf4j-api/2.0.0/"
        )

    @pytest.mark.asyncio
    async def test_deep_group_id_path_conversion(self):
        with patch("core.utils.is_url_reachable", new=AsyncMock(return_value=True)) as mock_check:
            url = await build_versioned_maven_license_url(
                "https://mvnrepository.com/artifact/com.fasterxml.jackson.core/jackson-databind",
                "2.15.2",
            )
        assert url == "https://mvnrepository.com/artifact/com.fasterxml.jackson.core/jackson-databind/2.15.2"
        mock_check.assert_awaited_once_with(
            "https://repo1.maven.org/maven2/com/fasterxml/jackson/core/jackson-databind/2.15.2/"
        )

    @pytest.mark.asyncio
    async def test_repo1_url_with_too_few_parts_returns_none(self):
        url = await build_versioned_maven_license_url(
            "https://repo1.maven.org/maven2/org/slf4j", "1.7.36"
        )
        assert url is None


class _ReachResp:
    def __init__(self, status=200, exc=None):
        self.status = status
        self._exc = exc

    async def __aenter__(self):
        if self._exc is not None:
            raise self._exc
        return self

    async def __aexit__(self, *args):
        return False


class _ReachSession:
    """Fake session for is_url_reachable: configurable HEAD/GET behaviour."""

    def __init__(self, head_status=200, get_status=200, head_exc=None):
        self.head_status = head_status
        self.get_status = get_status
        self.head_exc = head_exc
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def head(self, url, **kwargs):
        self.calls.append("head")
        return _ReachResp(self.head_status, self.head_exc)

    def get(self, url, **kwargs):
        self.calls.append("get")
        return _ReachResp(self.get_status)


class TestIsUrlReachable:
    @pytest.mark.asyncio
    async def test_head_ok(self):
        from core.utils import is_url_reachable

        session = _ReachSession(head_status=200)
        with patch("aiohttp.ClientSession", return_value=session):
            assert await is_url_reachable("https://example.com/") is True
        assert session.calls == ["head"]

    @pytest.mark.asyncio
    async def test_head_redirect_status_ok(self):
        from core.utils import is_url_reachable

        session = _ReachSession(head_status=302)
        with patch("aiohttp.ClientSession", return_value=session):
            assert await is_url_reachable("https://example.com/") is True

    @pytest.mark.asyncio
    async def test_head_404_no_get_fallback(self):
        from core.utils import is_url_reachable

        session = _ReachSession(head_status=404)
        with patch("aiohttp.ClientSession", return_value=session):
            assert await is_url_reachable("https://example.com/") is False
        assert session.calls == ["head"]

    @pytest.mark.asyncio
    async def test_head_403_falls_back_to_get_ok(self):
        from core.utils import is_url_reachable

        session = _ReachSession(head_status=403, get_status=200)
        with patch("aiohttp.ClientSession", return_value=session):
            assert await is_url_reachable("https://example.com/") is True
        assert session.calls == ["head", "get"]

    @pytest.mark.asyncio
    async def test_head_405_get_500_returns_false(self):
        from core.utils import is_url_reachable

        session = _ReachSession(head_status=405, get_status=500)
        with patch("aiohttp.ClientSession", return_value=session):
            assert await is_url_reachable("https://example.com/") is False

    @pytest.mark.asyncio
    async def test_head_client_error_falls_back_to_get(self):
        from core.utils import is_url_reachable

        session = _ReachSession(head_exc=aiohttp.ClientError("HEAD not supported"), get_status=200)
        with patch("aiohttp.ClientSession", return_value=session):
            assert await is_url_reachable("https://example.com/") is True
        assert session.calls == ["head", "get"]

    @pytest.mark.asyncio
    async def test_session_creation_failure_returns_false(self):
        from core.utils import is_url_reachable

        with patch("aiohttp.ClientSession", side_effect=RuntimeError("no network stack")):
            assert await is_url_reachable("https://example.com/") is False

    @pytest.mark.asyncio
    async def test_invalid_host_returns_false(self):
        from core.utils import is_url_reachable

        assert await is_url_reachable("http://nonexistent.invalid/", timeout=3) is False


def _pypi_metadata_fixture():
    return {
        "info": {
            "version": "1.0.0",
            "license_expression": "MIT",
            "license": "MIT",
            "classifiers": [],
            "project_urls": {"Source": "https://github.com/foo/bar"},
            "home_page": "",
            "author": "Foo Author",
            "description": "some readme",
        },
        "releases": {"1.0.0": [{"packagetype": "sdist"}]},
    }


def _github_result_fixture(used_default_branch, license_files):
    return {
        "status": "success",
        "used_default_branch": used_default_branch,
        "license_files": license_files,
        "license_type": "MIT",
        "license_analysis": {"licenses": ["MIT"]},
        "has_license_conflict": False,
        "readme_license": "MIT",
        "license_file_license": "MIT",
        "copyright_notice": "Copyright (c) 2020 Foo",
    }


class TestPypiVersionedFallbackGating:
    """process_pypi_repository 的 license_files 门控：只在 GitHub 无 tag 时换成 PyPI 带版本链接。"""

    async def _run(self, github_result, reachable=True):
        from core.pypi_utils import process_pypi_repository

        with patch("core.pypi_utils._fetch_pypi_metadata", return_value=_pypi_metadata_fixture()), \
             patch("core.github_utils.process_github_repository", new=AsyncMock(return_value=github_result)), \
             patch("core.github_utils.GitHubAPI", return_value=MagicMock()), \
             patch("core.utils.is_url_reachable", new=AsyncMock(return_value=reachable)) as mock_reach:
            result = await process_pypi_repository("https://pypi.org/project/foo-pkg/", "1.0.0")
        return result, mock_reach

    @pytest.mark.asyncio
    async def test_tag_matched_keeps_github_blob_url(self):
        blob = "https://github.com/foo/bar/blob/v1.0.0/LICENSE"
        result, mock_reach = await self._run(_github_result_fixture(False, blob))
        assert result["status"] == "success"
        assert result["license_files"] == blob
        mock_reach.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_tag_uses_versioned_pypi_link(self):
        blob = "https://github.com/foo/bar/blob/master/LICENSE"
        result, _ = await self._run(_github_result_fixture(True, blob))
        assert result["status"] == "success"
        assert result["license_files"] == "https://pypi.org/project/foo-pkg/1.0.0/#files"
        # 其余字段仍取自 GitHub 分析
        assert result["license_type"] == "MIT"
        assert result["license_file_license"] == "MIT"
        assert result["copyright_notice"] == "Copyright (c) 2020 Foo"
        assert result["license_determination_reason"] == "Analyzed via GitHub repository (primary source)"

    @pytest.mark.asyncio
    async def test_no_tag_but_unreachable_keeps_github_link(self):
        blob = "https://github.com/foo/bar/blob/master/LICENSE"
        result, _ = await self._run(_github_result_fixture(True, blob), reachable=False)
        assert result["license_files"] == blob

    @pytest.mark.asyncio
    async def test_no_github_repo_uses_pypi_link(self):
        from core.pypi_utils import process_pypi_repository

        metadata = _pypi_metadata_fixture()
        metadata["info"]["project_urls"] = {}
        with patch("core.pypi_utils._fetch_pypi_metadata", return_value=metadata), \
             patch("core.utils.is_url_reachable", new=AsyncMock(return_value=True)):
            result = await process_pypi_repository("https://pypi.org/project/foo-pkg/", "1.0.0")
        assert result["status"] == "success"
        assert result["license_files"] == "https://pypi.org/project/foo-pkg/1.0.0/#files"


class TestMainDispatchGating:
    """通过 main.process_all_repos 驱动真实分发路径，验证 Maven / Go 分支的门控。"""

    async def _run_row(self, row, github_result, reachable=True, proxy_status_map=None):
        import pandas as pd
        from main import process_all_repos

        df = pd.DataFrame([row])
        session = _FakeSession(proxy_status_map or {})
        with patch("main.process_github_repository", new=AsyncMock(return_value=dict(github_result))), \
             patch("main.get_github_url_from_pkggo", new=AsyncMock(
                 return_value={"github_url": "https://github.com/uber-go/atomic"})), \
             patch("core.utils.is_url_reachable", new=AsyncMock(return_value=reachable)), \
             patch("core.go_utils.aiohttp.ClientSession", return_value=session):
            results = await process_all_repos(api=MagicMock(), df=df, max_concurrency=1)
        assert len(results) == 1
        return results[0]

    @pytest.mark.asyncio
    async def test_maven_no_tag_uses_versioned_registry_link(self):
        result = await self._run_row(
            {"github_url": "https://mvnrepository.com/artifact/org.slf4j/slf4j-api", "version": "1.7.36"},
            _github_result_fixture(True, "https://github.com/qos-ch/slf4j/blob/master/LICENSE.txt"),
        )
        assert result["license_files"] == "https://mvnrepository.com/artifact/org.slf4j/slf4j-api/1.7.36"

    @pytest.mark.asyncio
    async def test_maven_tag_matched_keeps_github_blob_url(self):
        blob = "https://github.com/qos-ch/slf4j/blob/v_1.7.36/LICENSE.txt"
        result = await self._run_row(
            {"github_url": "https://mvnrepository.com/artifact/org.slf4j/slf4j-api", "version": "1.7.36"},
            _github_result_fixture(False, blob),
        )
        assert result["license_files"] == blob

    @pytest.mark.asyncio
    async def test_maven_no_tag_version_missing_in_central_keeps_github_link(self):
        blob = "https://github.com/qos-ch/slf4j/blob/master/LICENSE.txt"
        result = await self._run_row(
            {"github_url": "https://mvnrepository.com/artifact/org.slf4j/slf4j-api", "version": "999.0.0"},
            _github_result_fixture(True, blob),
            reachable=False,
        )
        assert result["license_files"] == blob

    @pytest.mark.asyncio
    async def test_go_no_tag_uses_versioned_pkggo_link(self):
        result = await self._run_row(
            {"github_url": "https://pkg.go.dev/go.uber.org/atomic", "version": "1.9.0"},
            _github_result_fixture(True, "https://github.com/uber-go/atomic/blob/master/LICENSE.txt"),
            proxy_status_map={
                "https://proxy.golang.org/go.uber.org/atomic/@v/v1.9.0.info": 200,
            },
        )
        assert result["license_files"] == "https://pkg.go.dev/go.uber.org/atomic@v1.9.0?tab=licenses"

    @pytest.mark.asyncio
    async def test_go_tag_matched_keeps_github_blob_url(self):
        blob = "https://github.com/uber-go/atomic/blob/v1.9.0/LICENSE.txt"
        result = await self._run_row(
            {"github_url": "https://pkg.go.dev/go.uber.org/atomic", "version": "1.9.0"},
            _github_result_fixture(False, blob),
        )
        assert result["license_files"] == blob

    @pytest.mark.asyncio
    async def test_go_no_tag_version_not_on_proxy_keeps_github_link(self):
        blob = "https://github.com/uber-go/atomic/blob/master/LICENSE.txt"
        result = await self._run_row(
            {"github_url": "https://pkg.go.dev/go.uber.org/atomic", "version": "99.0.0"},
            _github_result_fixture(True, blob),
            proxy_status_map={},
        )
        assert result["license_files"] == blob

    @pytest.mark.asyncio
    async def test_go_no_input_version_keeps_github_link(self):
        blob = "https://github.com/uber-go/atomic/blob/master/LICENSE.txt"
        result = await self._run_row(
            {"github_url": "https://pkg.go.dev/go.uber.org/atomic", "version": None},
            _github_result_fixture(True, blob),
        )
        assert result["license_files"] == blob
