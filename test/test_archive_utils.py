# -*- coding: utf-8 -*-
"""Tests for core/archive_utils.py — direct archive URL fallback analysis."""
import asyncio
import os
import sys
import tarfile
import tempfile
import zipfile
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.archive_utils import (
    build_local_tree,
    extract_archive,
    guess_name_version_from_url,
    is_direct_archive_url,
    process_direct_archive_url,
)


# ----------------------------------------------------------------------------
# URL 识别
# ----------------------------------------------------------------------------

class TestIsDirectArchiveUrl:
    def test_tar_gz(self):
        assert is_direct_archive_url("https://download.redis.io/releases/redis-7.2.7.tar.gz")

    def test_tar_xz(self):
        assert is_direct_archive_url("https://ftp.exim.org/pub/exim/exim4/old/exim-4.99.3.tar.xz")

    def test_tar_bz2(self):
        assert is_direct_archive_url("https://matt.ucc.asn.au/dropbear/releases/dropbear-2020.81.tar.bz2")

    def test_github_archive(self):
        assert is_direct_archive_url("https://github.com/FFmpeg/FFmpeg/archive/refs/tags/n8.0.1.tar.gz")

    def test_zip(self):
        assert is_direct_archive_url("https://example.com/pkg/foo-1.0.zip")

    def test_query_params_ignored(self):
        assert is_direct_archive_url("https://example.com/foo-1.0.tar.gz?mirror=cn&x=1")

    def test_regular_repo_url(self):
        assert not is_direct_archive_url("https://github.com/redis/redis")

    def test_registry_url(self):
        assert not is_direct_archive_url("https://pypi.org/project/requests/")

    def test_non_string(self):
        assert not is_direct_archive_url(None)
        assert not is_direct_archive_url(float("nan"))

    def test_no_scheme(self):
        assert not is_direct_archive_url("ftp://example.com/foo.tar.gz")


# ----------------------------------------------------------------------------
# 文件名 -> 组件名/版本猜测
# ----------------------------------------------------------------------------

class TestGuessNameVersion:
    def test_standard(self):
        assert guess_name_version_from_url(
            "https://download.redis.io/releases/redis-7.2.7.tar.gz"
        ) == ("redis", "7.2.7")

    def test_year_version(self):
        assert guess_name_version_from_url(
            "https://matt.ucc.asn.au/dropbear/releases/dropbear-2020.81.tar.bz2"
        ) == ("dropbear", "2020.81")

    def test_multi_dash_name(self):
        name, version = guess_name_version_from_url(
            "https://downloads.isc.org/isc/bind9/9.20.17/bind-9.20.17.tar.xz"
        )
        assert name == "bind"
        assert version == "9.20.17"

    def test_github_archive_tag_only(self):
        name, version = guess_name_version_from_url(
            "https://github.com/FFmpeg/FFmpeg/archive/refs/tags/n8.0.1.tar.gz"
        )
        assert name == "FFmpeg"
        assert version == "n8.0.1"

    def test_imagemagick_tag(self):
        name, version = guess_name_version_from_url(
            "https://github.com/ImageMagick/ImageMagick/archive/refs/tags/7.1.2-12.tar.gz"
        )
        assert version == "7.1.2-12"


# ----------------------------------------------------------------------------
# 解压（含路径穿越防护）
# ----------------------------------------------------------------------------

def _make_tarball(tmp_path, arcname_prefix="pkg-1.0"):
    """构造一个带顶层目录、LICENSE 和 README 的 tar.gz 样本。"""
    src = os.path.join(tmp_path, "src")
    os.makedirs(src, exist_ok=True)
    with open(os.path.join(src, "LICENSE"), "w", encoding="utf-8") as f:
        f.write("MIT License\n\nCopyright (c) 2024 Test Author\n")
    with open(os.path.join(src, "README.md"), "w", encoding="utf-8") as f:
        f.write("# Test Package\nLicensed under MIT.\n")
    archive_path = os.path.join(tmp_path, "pkg-1.0.tar.gz")
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(src, arcname=arcname_prefix)
    return archive_path


class TestExtractArchive:
    def test_tarball_single_top_dir(self, tmp_path):
        tmp_path = str(tmp_path)
        archive_path = _make_tarball(tmp_path)
        dest = os.path.join(tmp_path, "extracted")
        root = extract_archive(archive_path, dest)
        # 唯一顶层目录时应返回该目录
        assert os.path.basename(root) == "pkg-1.0"
        assert os.path.isfile(os.path.join(root, "LICENSE"))

    def test_zip(self, tmp_path):
        tmp_path = str(tmp_path)
        archive_path = os.path.join(tmp_path, "pkg.zip")
        with zipfile.ZipFile(archive_path, "w") as zf:
            zf.writestr("pkg-2.0/LICENSE", "Apache License 2.0")
            zf.writestr("pkg-2.0/main.c", "int main(){}")
        dest = os.path.join(tmp_path, "extracted")
        root = extract_archive(archive_path, dest)
        assert os.path.basename(root) == "pkg-2.0"
        assert os.path.isfile(os.path.join(root, "LICENSE"))

    def test_zip_path_traversal_blocked(self, tmp_path):
        tmp_path = str(tmp_path)
        archive_path = os.path.join(tmp_path, "evil.zip")
        with zipfile.ZipFile(archive_path, "w") as zf:
            zf.writestr("../evil.txt", "pwned")
            zf.writestr("ok.txt", "fine")
        dest = os.path.join(tmp_path, "extracted")
        extract_archive(archive_path, dest)
        assert not os.path.exists(os.path.join(tmp_path, "evil.txt"))
        assert os.path.isfile(os.path.join(dest, "ok.txt"))

    def test_tar_path_traversal_blocked(self, tmp_path):
        tmp_path = str(tmp_path)
        archive_path = os.path.join(tmp_path, "evil.tar.gz")
        content_file = os.path.join(tmp_path, "payload.txt")
        with open(content_file, "w") as f:
            f.write("pwned")
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(content_file, arcname="../evil.txt")
            tar.add(content_file, arcname="ok.txt")
        dest = os.path.join(tmp_path, "extracted")
        extract_archive(archive_path, dest)
        assert not os.path.exists(os.path.join(tmp_path, "evil.txt"))


# ----------------------------------------------------------------------------
# 本地目录 -> GitHub tree 结构
# ----------------------------------------------------------------------------

class TestBuildLocalTree:
    def test_tree_structure(self, tmp_path):
        tmp_path = str(tmp_path)
        os.makedirs(os.path.join(tmp_path, "docs"))
        os.makedirs(os.path.join(tmp_path, ".git"))
        open(os.path.join(tmp_path, "LICENSE"), "w").close()
        open(os.path.join(tmp_path, "docs", "guide.md"), "w").close()
        open(os.path.join(tmp_path, ".git", "config"), "w").close()

        tree = build_local_tree(tmp_path)
        paths = sorted(item["path"] for item in tree)
        assert paths == ["LICENSE", "docs/guide.md"]  # 隐藏目录被跳过
        assert all(item["type"] == "blob" for item in tree)


# ----------------------------------------------------------------------------
# 端到端（mock 下载 + mock LLM）
# ----------------------------------------------------------------------------

class TestProcessDirectArchiveUrl:
    def test_full_flow_with_mocked_download(self, tmp_path):
        """下载被 mock 成本地构造的 tar.gz；LLM 分析被 mock；验证结果结构与临时文件清理。"""
        tmp_path = str(tmp_path)
        archive_path = _make_tarball(tmp_path, arcname_prefix="redis-7.2.7")

        created_tmp_dirs = []
        real_mkdtemp = tempfile.mkdtemp

        def tracking_mkdtemp(*args, **kwargs):
            d = real_mkdtemp(*args, **kwargs)
            created_tmp_dirs.append(d)
            return d

        async def fake_download(url, dest_path, log_queue=None):
            import shutil
            shutil.copyfile(archive_path, dest_path)
            return os.path.getsize(dest_path)

        fake_analysis = {
            "licenses": ["MIT"],
            "spdx_expression": "MIT",
            "is_dual_licensed": False,
            "confidence": 0.99,
        }

        with patch("core.archive_utils.download_archive_with_progress", side_effect=fake_download), \
             patch("core.archive_utils.tempfile.mkdtemp", side_effect=tracking_mkdtemp), \
             patch("core.utils.analyze_license_content_async", new=AsyncMock(return_value=dict(fake_analysis))), \
             patch("core.utils.construct_copyright_notice_async", new=AsyncMock(return_value="Copyright (c) 2024 Test Author")):
            result = asyncio.run(process_direct_archive_url(
                "https://download.redis.io/releases/redis-7.2.7.tar.gz"
            ))

        assert result is not None
        assert result["status"] == "success"
        assert result["license_type"] == "MIT"
        assert result["license_file_license"] == "MIT"
        assert result["component_name"] == "redis"
        assert result["resolved_version"] == "7.2.7"
        assert "LICENSE" in result["license_files"]
        assert result["copyright_notice"] == "Copyright (c) 2024 Test Author"
        assert "Downloaded source archive" in result["license_determination_reason"]
        # license 原文应完整保留在 license_text 字段
        assert "MIT License" in result["license_text"]
        assert "Copyright (c) 2024 Test Author" in result["license_text"]
        # 必备字段与 GitHub 流程结果兼容
        for field in ("input_url", "repo_url", "input_version", "used_default_branch",
                      "license_analysis", "has_license_conflict", "readme_license"):
            assert field in result
        # 临时目录必须被清理
        assert created_tmp_dirs, "应通过 tempfile.mkdtemp 创建临时目录"
        for d in created_tmp_dirs:
            assert not os.path.exists(d), f"临时目录未清理: {d}"

    def test_download_failure_returns_none_and_cleans_up(self):
        created_tmp_dirs = []
        real_mkdtemp = tempfile.mkdtemp

        def tracking_mkdtemp(*args, **kwargs):
            d = real_mkdtemp(*args, **kwargs)
            created_tmp_dirs.append(d)
            return d

        async def failing_download(url, dest_path, log_queue=None):
            raise RuntimeError("network down")

        with patch("core.archive_utils.download_archive_with_progress", side_effect=failing_download), \
             patch("core.archive_utils.tempfile.mkdtemp", side_effect=tracking_mkdtemp):
            result = asyncio.run(process_direct_archive_url(
                "https://example.com/foo-1.0.tar.gz"
            ))

        assert result is None
        for d in created_tmp_dirs:
            assert not os.path.exists(d), f"临时目录未清理: {d}"

    def test_non_archive_url_returns_none(self):
        result = asyncio.run(process_direct_archive_url("https://github.com/redis/redis"))
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
