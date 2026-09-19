"""npm 包声明的许可证与其 GitHub 仓库根 LICENSE 冲突时的处理。

背景：monorepo（如 fontsource/font-files）一个仓库发布上千个包，仓库根
LICENSE 描述的是构建工具（MIT），而各个包分发的字体是 OFL-1.1。此前流程
把仓库级结论直接当作包的 license_file_license，导致 concluded_license 被
MIT 覆盖。

这里既覆盖冲突场景的修复，也覆盖「不冲突」场景保持原样，防止误伤。
"""

import pytest

from core.utils import extract_spdx_license_ids, licenses_disagree


# ---------------------------------------------------------------------------
# SPDX 标识符提取
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("expr,expected", [
    ("MIT", {"mit"}),
    ("OFL-1.1", {"ofl-1.1"}),
    ("Apache-2.0 AND MIT", {"apache-2.0", "mit"}),
    ("(MIT OR Apache-2.0) AND BSD-3-Clause", {"mit", "apache-2.0", "bsd-3-clause"}),
    # WITH 后面是例外条款，不是许可证
    ("GPL-2.0 WITH Classpath-exception-2.0", {"gpl-2.0"}),
    # 结构性标记与未判定值都不算许可证
    ("MIT AND Others", {"mit"}),
    ("NOASSERTION", set()),
    ("", set()),
    (None, set()),
    # -only / -or-later 归一到同一标识符
    ("GPL-3.0-or-later", {"gpl-3.0"}),
])
def test_extract_spdx_license_ids(expr, expected):
    assert extract_spdx_license_ids(expr) == expected


# ---------------------------------------------------------------------------
# 冲突判定：只有"完全没有交集"才算冲突
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("declared,observed", [
    ("OFL-1.1", "MIT"),                     # fontsource 的实际情况
    ("Apache-2.0", "GPL-3.0-only"),
    ("MIT OR Apache-2.0", "BSD-3-Clause"),
])
def test_disagree_when_no_common_license(declared, observed):
    assert licenses_disagree(declared, observed) is True


@pytest.mark.parametrize("declared,observed", [
    ("MIT", "MIT"),
    ("MIT", "mit"),                          # 大小写不敏感
    ("GPL-3.0", "GPL-3.0-or-later"),         # 同一许可证的不同写法
    ("MIT", "MIT AND Apache-2.0"),           # 仓库结论只是更细，不算冲突
    ("MIT", "MIT AND Others"),               # 追加的结构性标记不算冲突
    ("MIT", "MIT OR Apache-2.0"),
    # 任一侧无法解析出标识符时不判冲突，避免把"信息缺失"当成"矛盾"
    ("MIT", None),
    (None, "MIT"),
    ("MIT", "NOASSERTION"),
    ("", "MIT"),
])
def test_no_disagreement(declared, observed):
    assert licenses_disagree(declared, observed) is False


# ---------------------------------------------------------------------------
# 端到端：process_npm_repository 在冲突/不冲突下的取值
# ---------------------------------------------------------------------------

OFL_TEXT = (
    "Copyright 2020 The JetBrains Mono Project Authors\n\n"
    "This Font Software is licensed under the SIL Open Font License, Version 1.1.\n"
)

MIT_REPO_RESULT = {
    "status": "success",
    "used_default_branch": True,
    "license_files": "https://github.com/fontsource/font-files/blob/main/LICENSE",
    "license_analysis": {"licenses": ["MIT"], "spdx_expression": "MIT"},
    "has_license_conflict": False,
    "readme_license": None,
    "license_file_license": "MIT",
    "license_text": "MIT License\n\nCopyright (c) 2023 fontsource\n",
    "copyright_notice": "Copyright (c) 2023 fontsource",
}


def _packument(declared_license):
    return {
        "versions": {
            "5.3.0": {
                "name": "@fontsource-variable/jetbrains-mono",
                "version": "5.3.0",
                "license": declared_license,
                "repository": {"url": "git+https://github.com/fontsource/font-files.git"},
                "dist": {"tarball": "https://registry.npmjs.org/fake.tgz"},
                "readme": "# jetbrains-mono",
            }
        },
        "dist-tags": {"latest": "5.3.0"},
        "time": {"5.3.0": "2026-07-19T03:50:26.648Z"},
    }


async def _run_npm(monkeypatch, declared_license, repo_license, tarball_license=OFL_TEXT):
    """跑一遍 process_npm_repository，外部依赖全部替换成可控假实现。

    返回 (result, analyzed_contents)，analyzed_contents 记录包内 LICENSE 是否被真正分析过。
    """
    from core import npm_utils
    import core.github_utils as github_utils

    github_result = dict(MIT_REPO_RESULT, license_file_license=repo_license)
    analyzed = []

    async def fake_github(api, url, version, **kwargs):
        return github_result

    async def fake_tarball(tarball_url):
        return [], tarball_license

    async def fake_analyze(content, source_url=None):
        analyzed.append(content)
        return {"licenses": ["OFL-1.1"], "spdx_expression": "OFL-1.1", "source_url": source_url}

    async def fake_copyright(**kwargs):
        return "Copyright 2020 The JetBrains Mono Project Authors"

    monkeypatch.setattr(npm_utils, "_fetch_packument", lambda name: _packument(declared_license))
    monkeypatch.setattr(npm_utils, "async_analyze_npm_tarball", fake_tarball)
    monkeypatch.setattr(npm_utils, "analyze_license_content_async", fake_analyze)
    monkeypatch.setattr(npm_utils, "construct_copyright_notice_async", fake_copyright)
    monkeypatch.setattr(github_utils, "process_github_repository", fake_github)
    monkeypatch.setattr(github_utils, "GitHubAPI", lambda *a, **k: object())

    result = await npm_utils.process_npm_repository(
        "https://www.npmjs.com/package/@fontsource-variable/jetbrains-mono", "5.3.0"
    )
    return result, analyzed


@pytest.mark.asyncio
async def test_repo_license_does_not_override_conflicting_npm_declaration(monkeypatch):
    """仓库根 LICENSE 是 MIT、包声明 OFL-1.1 时，结论必须是 OFL-1.1。"""
    from core.utils import get_concluded_license

    result, analyzed = await _run_npm(monkeypatch, "OFL-1.1", "MIT")

    assert result["status"] == "success"
    assert result["license_type"] == "OFL-1.1"
    # 仓库的 MIT 不能进入任何一个参与结论的字段
    assert result["license_file_license"] == "OFL-1.1"
    assert result["readme_license"] != "MIT"
    concluded = get_concluded_license(
        result["license_type"], result["readme_license"], result["license_file_license"]
    )
    assert concluded == "OFL-1.1"

    # 包内 LICENSE 被真正读取并分析过
    assert analyzed and "SIL Open Font License" in analyzed[0]
    # 佐证材料也应指向 npm 包本身，而不是仓库
    assert result["license_files"].startswith("https://www.npmjs.com/package/")
    assert "MIT License" not in (result["license_text"] or "")
    assert result["copyright_notice"] == "Copyright 2020 The JetBrains Mono Project Authors"
    assert result["has_license_conflict"] is True
    assert "conflict" in result["license_determination_reason"]


@pytest.mark.asyncio
async def test_tarball_without_license_falls_back_to_npm_declaration(monkeypatch):
    """包内没有 LICENSE 时留空 license_file_license，让结论回落到 npm 声明。"""
    from core.utils import get_concluded_license

    result, analyzed = await _run_npm(monkeypatch, "OFL-1.1", "MIT", tarball_license=None)

    assert analyzed == []
    assert result["license_file_license"] is None
    concluded = get_concluded_license(
        result["license_type"], result["readme_license"], result["license_file_license"]
    )
    assert concluded == "OFL-1.1"


@pytest.mark.asyncio
async def test_matching_repo_license_keeps_github_result(monkeypatch):
    """不冲突时行为保持不变：仍以 GitHub 仓库扫描结果为准。"""
    result, analyzed = await _run_npm(monkeypatch, "MIT", "MIT")

    assert result["license_file_license"] == "MIT"
    assert result["license_determination_reason"] == "Fetched from GitHub repository"
    assert result["copyright_notice"] == "Copyright (c) 2023 fontsource"
    assert result["license_text"].startswith("MIT License")
    # 不冲突就不该多下载一次 tarball 去分析
    assert analyzed == []


@pytest.mark.asyncio
async def test_repo_license_superset_is_not_a_conflict(monkeypatch):
    """仓库结论只是比 npm 声明更细时，保留信息更完整的仓库结论。"""
    result, _ = await _run_npm(monkeypatch, "MIT", "MIT AND Apache-2.0")

    assert result["license_file_license"] == "MIT AND Apache-2.0"
    assert result["license_determination_reason"] == "Fetched from GitHub repository"


# ---------------------------------------------------------------------------
# monorepo 子目录定位：repository.directory / homepage tree URL 应传给
# GitHub 流程，而不是只传仓库根地址（否则子包会拿到根 LICENSE，张冠李戴）
# ---------------------------------------------------------------------------

def _monorepo_packument(directory=None, homepage=None, repo_url="git+https://github.com/aws/aws-sdk-js-v3.git"):
    repository = {"type": "git", "url": repo_url}
    if directory:
        repository["directory"] = directory
    version_obj = {
        "name": "@aws-sdk/credential-provider-node",
        "version": "3.972.75",
        "license": "Apache-2.0",
        "repository": repository,
        "author": {"name": "AWS"},
    }
    if homepage:
        version_obj["homepage"] = homepage
    return {
        "dist-tags": {"latest": "3.972.75"},
        "time": {"3.972.75": "2026-09-01T00:00:00Z"},
        "versions": {"3.972.75": version_obj},
    }


async def _capture_github_call(monkeypatch, packument):
    """跑一遍 npm 流程，捕获传给 process_github_repository 的 URL 与 name。"""
    from core import npm_utils
    import core.github_utils as github_utils

    captured = {}

    async def fake_github(api, url, version, **kwargs):
        captured["url"] = url
        captured["name"] = kwargs.get("name")
        return {
            "status": "success",
            "used_default_branch": False,
            "license_files": "https://github.com/aws/aws-sdk-js-v3/blob/v3.973.0/packages-internal/credential-provider-node/LICENSE",
            "license_file_license": "Apache-2.0",
            "copyright_notice": "Copyright Amazon.com",
        }

    monkeypatch.setattr(npm_utils, "_fetch_packument", lambda name: packument)
    monkeypatch.setattr(npm_utils, "fetch_npm_readme_simple", lambda *a, **k: "")
    monkeypatch.setattr(github_utils, "process_github_repository", fake_github)
    monkeypatch.setattr(github_utils, "GitHubAPI", lambda *a, **k: object())

    result = await npm_utils.process_npm_repository(
        "https://www.npmjs.com/package/@aws-sdk/credential-provider-node/v/3.972.75",
        "3.972.75",
    )
    return result, captured


@pytest.mark.asyncio
async def test_monorepo_directory_from_repository_field(monkeypatch):
    """repository.directory 存在时，应拼成 tree URL 传入 GitHub 子目录流程。"""
    result, captured = await _capture_github_call(
        monkeypatch, _monorepo_packument(directory="packages-internal/credential-provider-node")
    )
    assert captured["url"].endswith("/tree/HEAD/packages-internal/credential-provider-node")
    assert captured["name"] == "@aws-sdk/credential-provider-node"
    # 子目录 LICENSE 成为最终 license_files
    assert "packages-internal/credential-provider-node/LICENSE" in result["license_files"]


@pytest.mark.asyncio
async def test_monorepo_homepage_tree_url_preferred(monkeypatch):
    """homepage 自带 tree 路径时优先于 repository.directory（含真实分支名）。"""
    result, captured = await _capture_github_call(
        monkeypatch,
        _monorepo_packument(
            directory="packages-internal/credential-provider-node",
            homepage="https://github.com/aws/aws-sdk-js-v3/tree/main/packages-internal/credential-provider-node",
        ),
    )
    assert captured["url"] == (
        "https://github.com/aws/aws-sdk-js-v3/tree/main/packages-internal/credential-provider-node"
    )


@pytest.mark.asyncio
async def test_plain_repo_without_directory_unchanged(monkeypatch):
    """无 directory / homepage tree 时保持仓库根 URL，行为不变（回归保护）。"""
    result, captured = await _capture_github_call(monkeypatch, _monorepo_packument())
    assert captured["url"] == "https://github.com/aws/aws-sdk-js-v3"
    assert captured["name"] == "@aws-sdk/credential-provider-node"
