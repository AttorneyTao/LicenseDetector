"""purl 输入支持的单元测试。

覆盖三层：
1. parse_purl —— 规范语法的解析（percent-encoding、qualifiers、subpath、版本切分）
2. purl_to_url —— 各生态的 URL 映射
3. resolve_input_ref —— 入口适配层（version/name 回填、非 purl 输入零影响、
   以及翻译结果确实能命中各生态既有的路由判定）
"""

import pytest

from core.utils import is_blank_value, is_purl, parse_purl, purl_to_url
from core.github_utils import normalize_github_url, resolve_input_ref


# ---------------------------------------------------------------------------
# is_purl
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", [
    "pkg:npm/lodash@4.17.21",
    "  pkg:pypi/requests  ",
    "PKG:MAVEN/org.x/y",
])
def test_is_purl_true(value):
    assert is_purl(value) is True


@pytest.mark.parametrize("value", [
    "https://github.com/foo/bar",
    "github.com/foo/bar",
    "",
    None,
    123,
    float("nan"),
])
def test_is_purl_false(value):
    assert is_purl(value) is False


# ---------------------------------------------------------------------------
# parse_purl
# ---------------------------------------------------------------------------

def test_parse_scoped_npm_decodes_at_sign():
    purl = parse_purl("pkg:npm/%40babel/core@7.24.0")
    assert purl.type == "npm"
    assert purl.namespace == "@babel"
    assert purl.name == "core"
    assert purl.version == "7.24.0"
    assert purl.full_name == "@babel/core"
    assert purl.component_name == "@babel/core"


def test_parse_maven_namespace_is_group_id():
    purl = parse_purl("pkg:maven/org.apache.commons/commons-lang3@3.12.0")
    assert purl.namespace == "org.apache.commons"
    assert purl.name == "commons-lang3"
    # maven 的 namespace 是 groupId，不属于组件名
    assert purl.component_name == "commons-lang3"


def test_parse_golang_multi_segment_namespace():
    purl = parse_purl("pkg:golang/github.com/gin-gonic/gin@v1.9.1")
    assert purl.namespace == "github.com/gin-gonic"
    assert purl.name == "gin"
    assert purl.version == "v1.9.1"


def test_parse_pypi_normalizes_name():
    # 规范要求 pypi 名称小写、下划线转连字符
    purl = parse_purl("pkg:pypi/Django_Rest_Framework@3.15.1")
    assert purl.name == "django-rest-framework"


def test_parse_qualifiers_and_subpath():
    purl = parse_purl(
        "pkg:generic/openssl@3.0.0"
        "?download_url=https%3A%2F%2Fexample.com%2Fopenssl.tar.gz&arch=amd64"
        "#src/crypto"
    )
    assert purl.qualifiers["download_url"] == "https://example.com/openssl.tar.gz"
    assert purl.qualifiers["arch"] == "amd64"
    assert purl.subpath == "src/crypto"
    assert purl.version == "3.0.0"


def test_parse_version_is_optional():
    purl = parse_purl("pkg:cargo/serde")
    assert purl.name == "serde"
    assert purl.version is None


def test_parse_tolerates_legacy_double_slash():
    purl = parse_purl("pkg://npm/lodash@4.17.21")
    assert purl.type == "npm"
    assert purl.name == "lodash"


@pytest.mark.parametrize("value", [
    "pkg:npm",          # 只有 type，缺 name
    "pkg:",             # 空
    "https://github.com/foo/bar",
])
def test_parse_invalid_returns_none(value):
    assert parse_purl(value) is None


# ---------------------------------------------------------------------------
# purl_to_url
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("purl_str,expected", [
    ("pkg:npm/lodash@4.17.21", "https://www.npmjs.com/package/lodash"),
    ("pkg:npm/%40babel/core@7.24.0", "https://www.npmjs.com/package/@babel/core"),
    ("pkg:pypi/requests@2.31.0", "https://pypi.org/project/requests"),
    ("pkg:maven/org.apache.commons/commons-lang3@3.12.0",
     "https://mvnrepository.com/artifact/org.apache.commons/commons-lang3"),
    ("pkg:golang/github.com/gin-gonic/gin@v1.9.1",
     "https://pkg.go.dev/github.com/gin-gonic/gin"),
    ("pkg:cargo/serde@1.0.197", "https://crates.io/crates/serde"),
    ("pkg:pub/http@1.2.0", "https://pub.dev/packages/http"),
    ("pkg:nuget/Newtonsoft.Json@13.0.3", "https://www.nuget.org/packages/Newtonsoft.Json"),
    ("pkg:github/torvalds/linux@v6.1", "https://github.com/torvalds/linux"),
])
def test_purl_to_url_mapping(purl_str, expected):
    assert purl_to_url(parse_purl(purl_str)) == expected


def test_purl_to_url_maven_without_group_id_is_unmappable():
    assert purl_to_url(parse_purl("pkg:maven/commons-lang3@3.12.0")) is None


def test_purl_to_url_generic_uses_download_url():
    purl = parse_purl(
        "pkg:generic/openssl@3.0.0?download_url=https://example.com/openssl-3.0.0.tar.gz"
    )
    assert purl_to_url(purl) == "https://example.com/openssl-3.0.0.tar.gz"


def test_purl_to_url_unknown_type_falls_back_to_vcs_url():
    purl = parse_purl(
        "pkg:cocoapods/AFNetworking@4.0.1"
        "?vcs_url=git%2Bhttps://github.com/AFNetworking/AFNetworking.git%40abc123"
    )
    # git+ 前缀、.git 后缀与 @revision 都应被剥掉
    assert purl_to_url(purl) == "https://github.com/AFNetworking/AFNetworking"


def test_purl_to_url_unknown_type_without_hints_returns_none():
    assert purl_to_url(parse_purl("pkg:rpm/fedora/curl@7.50.3-1")) is None


# ---------------------------------------------------------------------------
# resolve_input_ref —— 非 purl 输入必须与原行为逐字节一致
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw", [
    "https://github.com/foo/bar",
    "github.com/foo/bar",
    "https://mvnrepository.com/artifact/org.x/y",
    "https://github.com/foo/bar_x000D_",
    "",
    None,
])
def test_non_purl_input_unchanged(raw):
    url, version, name = resolve_input_ref(raw, "1.0.0", "bar")
    assert url == normalize_github_url(raw)
    assert version == "1.0.0"
    assert name == "bar"


# ---------------------------------------------------------------------------
# resolve_input_ref —— purl 输入的 version / name 回填
# ---------------------------------------------------------------------------

def test_purl_backfills_empty_version_and_name():
    url, version, name = resolve_input_ref("pkg:cargo/serde@1.0.197", None, None)
    assert url == "https://crates.io/crates/serde"
    assert version == "1.0.197"
    assert name == "serde"


@pytest.mark.parametrize("blank", [None, "", "   ", float("nan")])
def test_purl_backfills_over_blank_columns(blank):
    _, version, name = resolve_input_ref("pkg:cargo/serde@1.0.197", blank, blank)
    assert version == "1.0.197"
    assert name == "serde"


def test_purl_version_wins_over_conflicting_column():
    # 约定：purl 自带版本优先于 version 列
    _, version, _ = resolve_input_ref("pkg:cargo/serde@1.0.197", "0.9.0", None)
    assert version == "1.0.197"


def test_purl_name_wins_over_conflicting_column():
    _, _, name = resolve_input_ref("pkg:cargo/serde@1.0.197", None, "serde-old")
    assert name == "serde"


def test_purl_without_version_keeps_column_value():
    _, version, _ = resolve_input_ref("pkg:cargo/serde", "1.0.197", None)
    assert version == "1.0.197"


def test_unmappable_purl_passes_through_untouched():
    # 未覆盖的 type 原样透传，交给下游 LLM 兜底；version / name 仍然回填
    url, version, name = resolve_input_ref("pkg:rpm/fedora/curl@7.50.3-1", None, None)
    assert url == "pkg:rpm/fedora/curl@7.50.3-1"
    assert version == "7.50.3-1"
    assert name == "curl"


def test_nuget_purl_supplies_name_and_version_required_by_handler():
    # NuGet 分支的判定是 `if name and version`，两者必须由 purl 补齐
    url, version, name = resolve_input_ref("pkg:nuget/Newtonsoft.Json@13.0.3", None, None)
    assert url.startswith("https://www.nuget.org/")
    assert name == "Newtonsoft.Json"
    assert version == "13.0.3"


# ---------------------------------------------------------------------------
# 翻译结果必须命中各生态既有的路由判定
# ---------------------------------------------------------------------------

def test_npm_purl_hits_existing_npm_router():
    from core.npm_utils import is_npm_package_url
    url, _, _ = resolve_input_ref("pkg:npm/%40babel/core@7.24.0", None, None)
    assert is_npm_package_url(url) is True


def test_generic_archive_purl_hits_archive_router():
    from core.archive_utils import is_direct_archive_url
    url, _, _ = resolve_input_ref(
        "pkg:generic/openssl@3.0.0?download_url=https://example.com/openssl-3.0.0.tar.gz",
        None, None,
    )
    assert is_direct_archive_url(url) is True


@pytest.mark.parametrize("purl_str,marker", [
    ("pkg:maven/org.apache.commons/commons-lang3@3.12.0", "mvnrepository.com/artifact"),
    ("pkg:golang/github.com/gin-gonic/gin@v1.9.1", "https://pkg.go.dev/"),
    ("pkg:cargo/serde@1.0.197", "crates.io/crates"),
    ("pkg:pub/http@1.2.0", "https://pub.dev/packages/"),
    ("pkg:pypi/requests@2.31.0", "https://pypi.org/"),
])
def test_translated_url_matches_router_condition(purl_str, marker):
    url, _, _ = resolve_input_ref(purl_str, None, None)
    assert marker in url


# ---------------------------------------------------------------------------
# 集成级：purl 输入经 API 入口后确实落到对应生态的 handler
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_purl_rows_reach_expected_handlers(monkeypatch):
    """把一批 purl 喂给 api._process_repositories，断言路由与参数回填。"""
    import pandas as pd
    import api as api_module

    calls = {"npm": [], "maven": [], "github": []}

    async def fake_process_npm_repository(url, version):
        calls["npm"].append((url, version))
        return {"status": "success", "license_type": "MIT"}

    async def fake_process_github_repository(api_obj, url, version, name=None, **kwargs):
        calls["github"].append((url, version, name))
        return {"status": "success", "license_type": "Apache-2.0"}

    def fake_analyze_maven_repository_url(url, version=None):
        calls["maven"].append((url, version))
        return {
            "artifact_id": "commons-lang3",
            "version": version,
            "license": "Apache-2.0",
            "pom_url": (
                "https://repo1.maven.org/maven2/org/apache/commons/"
                "commons-lang3/3.12.0/commons-lang3-3.12.0.pom"
            ),
            "license_source": "maven_central",
        }

    monkeypatch.setattr(api_module, "process_npm_repository", fake_process_npm_repository)
    monkeypatch.setattr(api_module, "process_github_repository", fake_process_github_repository)
    monkeypatch.setattr(
        api_module,
        "analyze_maven_repository_url",
        fake_analyze_maven_repository_url,
    )

    df = pd.DataFrame([
        {"github_url": "pkg:npm/%40babel/core@7.24.0", "version": None, "name": None},
        {"github_url": "pkg:maven/org.apache.commons/commons-lang3@3.12.0",
         "version": None, "name": None},
        {"github_url": "https://github.com/foo/bar", "version": "1.0.0", "name": "bar"},
    ])

    results = await api_module._process_repositories(api=None, df=df)

    assert len(results) == 3
    # npm purl -> npm handler，版本来自 purl
    assert calls["npm"] == [("https://www.npmjs.com/package/@babel/core", "7.24.0")]
    # maven purl -> 原仓库 POM 优先，版本来自 purl
    assert calls["maven"] == [(
        "https://mvnrepository.com/artifact/org.apache.commons/commons-lang3",
        "3.12.0",
    )]
    # 普通 URL 行为不变
    assert ("https://github.com/foo/bar", "1.0.0", "bar") in calls["github"]
    # 输出里的 input_url 仍是用户原始输入（purl 原文）
    assert results[0]["input_url"] == "pkg:npm/%40babel/core@7.24.0"


# ---------------------------------------------------------------------------
# 未覆盖 type 透传后的收尾行为：必须干净地 skipped，不能抛异常
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unmappable_purl_returns_skipped_when_llm_finds_nothing(monkeypatch):
    """pkg:deb/... 这类透传输入在 LLM 查不到仓库时应返回 skipped。

    回归用例：原先查找失败会回落到原始输入，把 purl 送进 parse_github_url
    并抛 "Not a GitHub URL"，整行变成 error。
    """
    import core.github_utils as gh

    async def fake_lookup(package_url, name=None):
        return None

    monkeypatch.setattr(gh, "find_github_url_from_package_url", fake_lookup)

    result = await gh.process_github_repository(
        None, "pkg:deb/adduser@3.137ubuntu1", "3.137ubuntu1", name="adduser"
    )

    assert result["status"] == "skipped"
    assert result["input_url"] == "pkg:deb/adduser@3.137ubuntu1"
    assert result["repo_url"] is None
    assert "could not find" in result["license_determination_reason"].lower()


@pytest.mark.asyncio
async def test_non_github_llm_answer_is_treated_as_not_found(monkeypatch):
    """LLM 返回非 GitHub 地址时同样按未找到处理，而不是继续解析。"""
    import core.github_utils as gh

    async def fake_lookup(package_url, name=None):
        return "https://gitlab.com/foo/bar"

    monkeypatch.setattr(gh, "find_github_url_from_package_url", fake_lookup)

    result = await gh.process_github_repository(
        None, "pkg:conan/zlib@1.3", None, name="zlib"
    )

    assert result["status"] == "skipped"


# ---------------------------------------------------------------------------
# is_blank_value
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    (None, True),
    (float("nan"), True),
    ("", True),
    ("   ", True),
    ("1.0.0", False),
    (0, False),
])
def test_is_blank_value(value, expected):
    assert is_blank_value(value) is expected
