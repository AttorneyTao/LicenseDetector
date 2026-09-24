"""The published crate, rather than a guessed repository, supplies the evidence."""

import io
import tarfile

import pytest

from core import crate_archive, crate_utils


def _crate_file(tmp_path, name, version, manifest_fields, files):
    path = tmp_path / f"{name}-{version}.crate"
    manifest = f'[package]\nname = "{name}"\nversion = "{version}"\n{manifest_fields}\n'
    with tarfile.open(path, "w:gz") as archive:
        for relative, content in {"Cargo.toml": manifest, **files}.items():
            data = content.encode()
            info = tarfile.TarInfo(f"{name}-{version}/{relative}")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return path


@pytest.mark.asyncio
async def test_versioned_download_reads_license_file_in_crate(monkeypatch, tmp_path):
    source = _crate_file(
        tmp_path, "zerocopy", "0.6.1", 'license-file = "LICENSE"',
        {"LICENSE": "Copyright 2019 The Fuchsia Authors.\nRedistribution and use are permitted."},
    )
    calls = []

    async def fake_download(url, dest, **kwargs):
        from shutil import copyfile
        copyfile(source, dest)
        calls.append((url, kwargs["headers"]["User-Agent"]))

    async def fake_analyze(content, source_url):
        assert "Fuchsia Authors" in content
        assert source_url.endswith("(LICENSE)")
        return {"licenses": ["BSD-2-Clause"], "spdx_expression": "BSD-2-Clause"}

    monkeypatch.setattr(crate_archive, "download_archive_with_progress", fake_download)
    monkeypatch.setattr(crate_archive, "analyze_license_content_async", fake_analyze)
    monkeypatch.setattr(crate_utils, "_fetch_crate_info", lambda name: pytest.fail("registry must not replace package"))

    url = "https://crates.io/api/v1/crates/zerocopy/0.6.1/download"
    result = await crate_utils.process_crate_repository(url, "0.6.1")
    assert result["status"] == "success"
    assert result["license_type"] == "BSD-2-Clause"
    assert result["license_file_license"] == "BSD-2-Clause"
    assert result["package_license_files"] == ["LICENSE"]
    assert "Downloaded crates.io package" in result["license_determination_reason"]
    assert calls == [(url, crate_utils.CRATES_IO_HEADERS["User-Agent"])]


@pytest.mark.asyncio
async def test_manifest_expression_preserved_and_nested_vendor_license_excluded(monkeypatch, tmp_path):
    source = _crate_file(
        tmp_path, "libssh2-sys", "0.3.0", 'license = "MIT/Apache-2.0"',
        {"libssh2/COPYING": "Copyright vendor; licensed under BSD-3-Clause"},
    )

    async def fake_download(url, dest, **kwargs):
        from shutil import copyfile
        copyfile(source, dest)

    monkeypatch.setattr(crate_archive, "download_archive_with_progress", fake_download)
    monkeypatch.setattr(crate_archive, "analyze_license_content_async", lambda *args: pytest.fail("manifest governs"))
    url = "https://crates.io/api/v1/crates/libssh2-sys/0.3.0/download"
    result = await crate_utils.process_crate_repository(url, "0.3.0")
    assert result["license_type"] == "MIT/Apache-2.0"
    assert result["license_file_license"] is None
    assert "nested license files (not used for package license): libssh2/COPYING" in result["license_determination_reason"]
    assert "nested third party: libssh2/COPYING" in result["license_text"]


@pytest.mark.asyncio
async def test_mismatched_version_and_failed_audit_cannot_succeed(monkeypatch):
    url = "https://crates.io/api/v1/crates/zerocopy/0.6.1/download"
    mismatch = await crate_utils.process_crate_repository(url, "0.8.0")
    assert mismatch["status"] == "error"
    assert "不一致" in mismatch["error"]

    async def broken(*args, **kwargs):
        raise OSError("download unavailable")

    monkeypatch.setattr(crate_utils, "process_crate_download", broken)
    monkeypatch.setattr(crate_utils, "_fetch_crate_info", lambda name: pytest.fail("must not silently use registry"))
    failed = await crate_utils.process_crate_repository(url, "0.6.1")
    assert failed["status"] == "error"
    assert "download unavailable" in failed["error"]
