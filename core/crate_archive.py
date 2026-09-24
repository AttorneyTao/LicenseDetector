"""Inspect the exact crate published at a versioned crates.io download URL."""

import asyncio
import logging
import os
import re
import tempfile
import tomllib
from pathlib import Path
from typing import Any, Dict, Optional

from .archive_utils import (
    MAX_CONTENT_CHARS,
    build_local_tree,
    download_archive_with_progress,
    extract_archive,
)
from .utils import (
    analyze_license_content_async,
    extract_copyright_info_async,
    prepare_license_text,
)

logger = logging.getLogger(__name__)

_LICENSE_NAME = re.compile(r"^(?:LICEN[CS]E|COPYING|NOTICE)(?:[._-].*)?$", re.I)


def _read_package_file(root: Path, relative: str) -> Optional[str]:
    """Read a file only when its path stays within the extracted crate."""
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        return None
    return path.read_text(encoding="utf-8", errors="replace")[:MAX_CONTENT_CHARS]


def _inspect_extracted_crate(
    root: Path, name: str, version: str
) -> Dict[str, Any]:
    manifest_text = _read_package_file(root, "Cargo.toml")
    if not manifest_text:
        raise ValueError("下载的 crate 中没有 Cargo.toml")
    package = tomllib.loads(manifest_text).get("package", {})
    if not isinstance(package, dict):
        raise ValueError("crate 的 Cargo.toml 缺少 [package]")
    if package.get("name") != name or package.get("version") != version:
        raise ValueError(
            f"下载包身份与 URL 不符: {package.get('name')}@{package.get('version')} != {name}@{version}"
        )

    declared_license = package.get("license")
    license_file = package.get("license-file")
    paths = sorted(
        item["path"] for item in build_local_tree(str(root))
        if _LICENSE_NAME.fullmatch(Path(item["path"]).name)
    )
    root_license_paths = [path for path in paths if "/" not in path]
    evidence_paths = list(root_license_paths)
    if license_file and license_file not in evidence_paths:
        evidence_paths.insert(0, license_file)

    texts = {}
    for path in evidence_paths:
        content = _read_package_file(root, path)
        if content:
            texts[path] = content
    if license_file and license_file not in texts:
        raise ValueError(f"Cargo.toml 声明的 license-file 不存在或不可读: {license_file}")
    nested_texts = {}
    for path in paths:
        if "/" in path:
            content = _read_package_file(root, path)
            if content:
                nested_texts[path] = content

    return {
        "package": package,
        "declared_license": declared_license.strip() if isinstance(declared_license, str) else None,
        "license_file": license_file,
        "license_paths": paths,
        "texts": texts,
        "nested_texts": nested_texts,
    }


async def process_crate_download(
    url: str,
    name: str,
    version: str,
    *,
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Download and inspect the published .crate, using its own manifest as primary evidence."""
    with tempfile.TemporaryDirectory(prefix="crate_audit_") as temp_dir:
        archive_path = os.path.join(temp_dir, f"{name}-{version}.crate")
        logger.info("[CRATE_PACKAGE] 开始下载发布包: %s", url)
        await download_archive_with_progress(url, archive_path, headers=headers)
        extracted = await asyncio.to_thread(
            extract_archive, archive_path, os.path.join(temp_dir, "extracted")
        )
        inspected = _inspect_extracted_crate(Path(extracted), name, version)

        declared = inspected["declared_license"]
        license_file = inspected["license_file"]
        texts = inspected["texts"]
        license_analysis = None
        file_license = None
        if not declared and license_file:
            license_analysis = await analyze_license_content_async(
                texts[license_file], f"{url} ({license_file})"
            )
            if license_analysis and license_analysis.get("licenses"):
                file_license = (
                    license_analysis.get("spdx_expression")
                    or license_analysis["licenses"][0]
                )

        readme_content = _read_package_file(Path(extracted), "README.md")
        copyright_content = "\n\n".join(
            [content for content in texts.values()]
            + ([readme_content] if readme_content else [])
        )
        notice = await extract_copyright_info_async(copyright_content)

        root_files = [path for path in inspected["license_paths"] if "/" not in path]
        nested_files = [path for path in inspected["license_paths"] if "/" in path]
        reason = (
            f"Downloaded crates.io package {name}@{version}; inspected Cargo.toml"
            f" and package license files: {', '.join(root_files) or 'none'}"
        )
        if license_file:
            reason += f"; Cargo.toml license-file: {license_file}"
        if nested_files:
            reason += f"; nested license files (not used for package license): {', '.join(nested_files)}"
        logger.info("[CRATE_PACKAGE] %s; license=%s", reason, file_license or declared)

        license_text = "\n\n".join(
            [f"===== package: {path} =====\n{content}" for path, content in texts.items()]
            + [
                f"===== nested third party: {path} =====\n{content}"
                for path, content in inspected["nested_texts"].items()
            ]
        )
        return {
            "input_url": url,
            "repo_url": inspected["package"].get("repository") or url,
            "input_version": version,
            "normalized_input_version": version,
            "resolved_version": version,
            "used_default_branch": False,
            "component_name": name,
            "license_files": url,
            "license_analysis": license_analysis,
            "license_type": file_license or declared,
            "has_license_conflict": None,
            "readme_license": None,
            "license_file_license": file_license,
            "copyright_notice": notice,
            "license_text": prepare_license_text(license_text),
            "status": "success",
            "license_determination_reason": reason,
            "package_license_files": inspected["license_paths"],
            "package_manifest_license": declared,
        }
