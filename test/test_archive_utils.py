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
    analyze_extracted_directory,
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
        assert result["license_files"] == "https://download.redis.io/releases/redis-7.2.7.tar.gz"
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


R_DESCRIPTION_CASES = [
    ("tximport", "1.40.0", "LGPL (>=2)"),
    ("survminer", "0.5.2", "GPL-2"),
    ("survival", "3.8-12", "LGPL (>= 2)"),
    ("tradeSeq", "1.26.0", "MIT + file LICENSE"),
    ("SingleCellExperiment", "1.34.0", "GPL-3"),
    ("slingshot", "2.20.0", "Artistic-2.0"),
    ("RNASeqPower", "1.52.0", "LGPL (>=2)"),
    ("pwr", "1.3-0", "GPL (>= 3)"),
    ("platetools", "0.1.7", "MIT + file LICENSE"),
    ("ReactomePA", "1.56.0", "GPL-2"),
    ("muscat", "1.26.0", "GPL-3"),
    ("gsDesign", "3.11.0", "GPL (>= 3)"),
    ("FlowSOM", "2.20.0", "GPL (>= 2)"),
    ("DEqMS", "1.30.0", "LGPL"),
    ("coloc", "5.2.3", "GPL"),
    ("CATALYST", "1.36.0", "GPL (>=2)"),
    ("BiocManager", "1.30.27", "Artistic-2.0"),
    ("ALDEx2", "1.44.0", "GPL (>=3)"),
    ("WGCNA", "1.74", "GPL (>= 2)"),
    ("harmony", "2.0.5", "GPL-3"),
    ("readr", "2.2.0", "MIT + file LICENSE"),
    ("tidyr", "1.3.2", "MIT + file LICENSE"),
    ("dplyr", "1.2.1", "MIT + file LICENSE"),
    ("ggplot2", "4.0.3", "MIT + file LICENSE"),
]


def _description(name="tradeSeq", version="1.26.0", license="MIT + file LICENSE"):
    return f"Package: {name}\nVersion: {version}\nLicense: {license}\nAuthors@R: Sample Author\n"


def _empty_analysis():
    return {"licenses": [], "spdx_expression": None}


@pytest.mark.parametrize("name,version,declared", R_DESCRIPTION_CASES)
def test_r_description_recovers_all_24_observed_package_shapes(tmp_path, name, version, declared):
    description = _description(name, version, declared)
    (tmp_path / "DESCRIPTION").write_text(description)
    stub = "YEAR: 2019\nCOPYRIGHT HOLDER: Package Authors\n"
    has_license = "file LICENSE" in declared
    if has_license:
        (tmp_path / "LICENSE").write_text(stub)
    seen = []

    async def analyze(content, source_url=None):
        seen.append(content)
        if "DESCRIPTION:" in content:
            return {"licenses": ["MIT"], "spdx_expression": "MIT"}
        return _empty_analysis()

    with patch("core.utils.analyze_license_content_async", new=AsyncMock(side_effect=analyze)), \
         patch("core.utils.construct_copyright_notice_async", new=AsyncMock(return_value="Copyright supplied by existing LLM")) as copyright:
        result = asyncio.run(analyze_extracted_directory(
            str(tmp_path), f"https://example.org/{name}_{version}.tar.gz", name, version, version
        ))
    assert result["license_type"] == "MIT"  # Only the mocked model decides SPDX.
    assert result["resolved_version"] == version
    assert result["license_file_license"] == "MIT"
    assert result["copyright_notice"] == "Copyright supplied by existing LLM"
    assert description in seen[-1]
    assert (stub in seen[-1]) == has_license
    assert "DESCRIPTION" in result["license_determination_reason"]
    assert result["_archive_license_link_evidence"]["path"] == "DESCRIPTION"
    assert copyright.await_args.args[-1] == seen[-1]
    assert len(seen) == (2 if has_license else 1)


@pytest.mark.parametrize("successful_source", ["LICENSE", "README.md"])
def test_existing_license_and_readme_success_does_not_read_description(tmp_path, successful_source):
    (tmp_path / successful_source).write_text("Existing Apache license evidence")
    (tmp_path / "DESCRIPTION").write_text(_description())
    with patch("core.utils.analyze_license_content_async", new=AsyncMock(return_value={"licenses": ["Apache-2.0"]})) as analyze, \
         patch("core.utils.construct_copyright_notice_async", new=AsyncMock(return_value="Original notice")), \
         patch("core.archive_utils._read_r_description_evidence", new=AsyncMock()) as read_description:
        result = asyncio.run(analyze_extracted_directory(str(tmp_path), "https://example.org/a.tar.gz", "tradeSeq", "1.26.0", "1.26.0"))
    assert result["license_type"] == "Apache-2.0"
    assert result["copyright_notice"] == "Original notice"
    analyze.assert_awaited_once()
    read_description.assert_not_awaited()


@pytest.mark.parametrize("content,path", [
    (_description(name="foreign"), "DESCRIPTION"),
    (_description(version="9.0.0"), "DESCRIPTION"),
    ("Version: 1.26.0\nLicense: MIT\n", "DESCRIPTION"),
    ("Package: tradeSeq\nLicense: MIT\n", "DESCRIPTION"),
    (_description(license=""), "DESCRIPTION"),
    (_description(), "vendor/DESCRIPTION"),
    ("This prose mentions MIT", "DESCRIPTION"),
])
def test_unrelated_or_incomplete_description_is_not_used(tmp_path, content, path):
    destination = tmp_path / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content)
    with patch("core.utils.analyze_license_content_async", new=AsyncMock()) as analyze:
        result = asyncio.run(analyze_extracted_directory(str(tmp_path), "https://example.org/a.tar.gz", "tradeSeq", "1.26.0", "1.26.0"))
    assert result is None
    analyze.assert_not_awaited()


@pytest.mark.parametrize("reference", ["../outside", "/tmp/outside", "missing.txt"])
def test_description_cannot_use_unreadable_or_external_license_reference(tmp_path, reference):
    (tmp_path / "DESCRIPTION").write_text(_description(license=f"MIT + file {reference}"))
    with patch("core.utils.analyze_license_content_async", new=AsyncMock()) as analyze:
        result = asyncio.run(analyze_extracted_directory(str(tmp_path), "https://example.org/a.tar.gz", "tradeSeq", "1.26.0", "1.26.0"))
    assert result is None
    analyze.assert_not_awaited()


def test_multiline_description_and_referenced_license_stay_together(tmp_path):
    (tmp_path / "DESCRIPTION").write_text(_description(name="TRADESEQ", license="MIT +\n    file LICENSE.note"))
    (tmp_path / "LICENSE.note").write_text("YEAR: 2019\nCOPYRIGHT HOLDER: Authors\n")
    with patch("core.utils.analyze_license_content_async", new=AsyncMock(side_effect=[_empty_analysis(), {"licenses": ["MIT"]}])) as analyze, \
         patch("core.utils.construct_copyright_notice_async", new=AsyncMock(return_value="Original LLM notice")):
        result = asyncio.run(analyze_extracted_directory(str(tmp_path), "https://example.org/a.tar.gz", "tradeSeq", "1.26.0", "1.26.0"))
    assert result["license_type"] == "MIT"
    assert "file LICENSE.note" in analyze.await_args.args[0]
    assert "COPYRIGHT HOLDER: Authors" in analyze.await_args.args[0]


@pytest.mark.parametrize("model_response", [None, {"licenses": []}, {"licenses": [], "spdx_expression": "MIT"}])
def test_description_without_confirmed_model_result_preserves_failure(tmp_path, model_response):
    (tmp_path / "DESCRIPTION").write_text(_description(license="MIT"))
    with patch("core.utils.analyze_license_content_async", new=AsyncMock(return_value=model_response)), \
         patch("core.utils.construct_copyright_notice_async", new=AsyncMock()) as copyright:
        result = asyncio.run(analyze_extracted_directory(str(tmp_path), "https://example.org/a.tar.gz", "tradeSeq", "1.26.0", "1.26.0"))
    assert result is None
    copyright.assert_not_awaited()


def test_archive_model_error_returns_none_and_cleans_temp_directory(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "DESCRIPTION").write_text(_description(license="MIT"))
    package = tmp_path / "pkg.tar.gz"
    with tarfile.open(package, "w:gz") as tar:
        tar.add(source, arcname="tradeSeq")

    async def download(url, dest_path, log_queue=None):
        import shutil
        shutil.copyfile(package, dest_path)
        return package.stat().st_size

    created = []
    real_mkdtemp = tempfile.mkdtemp

    def track(*args, **kwargs):
        directory = real_mkdtemp(*args, **kwargs)
        created.append(directory)
        return directory

    with patch("core.archive_utils.download_archive_with_progress", side_effect=download), \
         patch("core.archive_utils.tempfile.mkdtemp", side_effect=track), \
         patch("core.utils.analyze_license_content_async", new=AsyncMock(side_effect=RuntimeError("model unavailable"))):
        result = asyncio.run(process_direct_archive_url("https://example.org/tradeSeq_1.26.0.tar.gz"))
    assert result is None
    assert created and all(not os.path.exists(directory) for directory in created)


def test_font_mode_does_not_use_r_description(tmp_path):
    (tmp_path / "DESCRIPTION").write_text(_description(license="MIT"))
    with patch("core.utils.analyze_license_content_async", new=AsyncMock()) as analyze:
        result = asyncio.run(analyze_extracted_directory(str(tmp_path), "https://example.org/a.tar.gz", "tradeSeq", "1.26.0", "1.26.0", font_mode=True))
    assert result is None
    analyze.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["cli", "api"])
@pytest.mark.parametrize("previous", ["no_license", "error", "exception"])
async def test_description_fallback_through_existing_entrypoints(tmp_path, monkeypatch, entry, previous):
    import pandas as pd
    import api
    import main
    import shutil

    source = tmp_path / "package"
    source.mkdir()
    (source / "DESCRIPTION").write_text(_description(license="Artistic-2.0"))
    archive = tmp_path / "package.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(source, arcname="tradeSeq")
    module = main if entry == "cli" else api
    upstream = AsyncMock(side_effect=RuntimeError("upstream unavailable")) if previous == "exception" else AsyncMock(return_value={
        "status": "error" if previous == "error" else "success", "license_type": None,
    })
    monkeypatch.setattr(module, "process_github_repository", upstream)

    async def download(url, dest_path, log_queue=None):
        shutil.copyfile(archive, dest_path)
        return archive.stat().st_size

    downloader = AsyncMock(side_effect=download)
    monkeypatch.setattr("core.archive_utils.download_archive_with_progress", downloader)
    analyze = AsyncMock(return_value={"licenses": ["Artistic-2.0"], "spdx_expression": "Artistic-2.0"})
    monkeypatch.setattr("core.utils.analyze_license_content_async", analyze)
    monkeypatch.setattr("core.utils.construct_copyright_notice_async", AsyncMock(return_value="LLM original notice"))

    async def keep_link(row, result, context):
        return result

    monkeypatch.setattr(module, "prefer_license_webpage", keep_link)
    monkeypatch.chdir(tmp_path)
    url = "https://example.org/tradeSeq_1.26.0.tar.gz"
    rows = pd.DataFrame([dict(name="tradeSeq", version="1.26.0", github_url=url,
                              package_manager="generic", serial_number="00041")])
    results = await (main.process_all_repos(None, rows) if entry == "cli" else api._process_repositories(None, rows))
    assert len(results) == 1
    result = results[0]
    assert result["license_type"] == "Artistic-2.0"
    assert result["copyright_notice"] == "LLM original notice"
    assert result["input_url"] == url
    assert result["resolved_version"] == "1.26.0"
    assert result["serial_number"] == "00041"
    assert result["package_manager"] == "generic"
    upstream.assert_awaited_once()
    downloader.assert_awaited_once()
    analyze.assert_awaited_once()
    assert "License: Artistic-2.0" in analyze.await_args.args[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
