# -*- coding: utf-8 -*-
"""Tests for versioned registry link fallback when GitHub has no matching tag."""
import os
import sys
from unittest.mock import AsyncMock, patch

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


class TestIsUrlReachable:
    @pytest.mark.asyncio
    async def test_invalid_host_returns_false(self):
        from core.utils import is_url_reachable

        assert await is_url_reachable("http://nonexistent.invalid/", timeout=3) is False
