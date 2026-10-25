"""Maven output regressions using the affected batch's original POMs."""
import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pandas as pd
import pytest

from core import maven_utils as maven, utils


FIXTURES = Path(__file__).parent / "maven_samples"
MEDIA3_URL = "https://dl.google.com/dl/android/maven2/androidx/media3/media3-common/1.0.0/media3-common-1.0.0-sources.jar"
MEDIA3_PAGE = "https://mvnrepository.com/artifact/androidx.media3/media3-common/1.0.0"


@pytest.fixture
def llm(monkeypatch):
    mock = Mock(return_value=json.dumps({"copyright_notice": None}))
    monkeypatch.setattr(utils, "USE_LLM", True)
    monkeypatch.setattr(utils, "complete_sync", mock)
    return mock


@pytest.fixture
def pom_http(monkeypatch):
    def fetch(url):
        path = FIXTURES / url.rsplit("/", 1)[-1]
        return (path.read_text(), 200) if path.exists() else (None, 404)

    mock = Mock(side_effect=fetch)
    monkeypatch.setattr(maven, "_http_get", mock)
    monkeypatch.setattr(maven, "_convert_licenses_to_spdx", lambda licenses: "Apache-2.0")
    return mock


def test_media3_no_fabricated_year_or_owner(llm, pom_http):
    analysis = maven.analyze_maven_repository_url(MEDIA3_URL)
    assert analysis["license"] == "Apache-2.0"
    assert analysis["pom_url"].endswith("media3-common-1.0.0.pom")
    assert analysis.get("copyright") is None
    result = maven.build_maven_repository_result(analysis, MEDIA3_URL, "1.0.0", "androidx.media3:media3-common")
    assert result["license_files"] == MEDIA3_PAGE
    output = utils.fill_blank_copyright_notices(pd.DataFrame([result]))
    assert output.loc[0, "copyright_notice"] == "Copyright androidx.media3:media3-common Original author and authors"
    llm.assert_called_once()
    assert llm.call_args.args[0] == "copyright_extract"
    pom_http.assert_called_once_with(analysis["pom_url"])


def test_slf4j_parent_javadoc_template_is_not_copyright(llm, pom_http):
    url = "https://repo.maven.apache.org/maven2/org/slf4j/slf4j-api/2.0.16/slf4j-api-2.0.16.pom"
    analysis = maven.analyze_maven_repository_url(url)
    assert analysis.get("copyright") is None
    assert len(pom_http.call_args_list) == 3
    assert llm.call_count == 3
    for call in llm.call_args_list:
        content = call.kwargs["context"]["content"]
        assert "{currentYear}" not in content
        assert "&copy;" not in content
    assert analysis["pom_url"] == url
    assert maven.build_maven_repository_result(analysis, url)["license_files"] == "https://mvnrepository.com/artifact/org.slf4j/slf4j-api/2.0.16"


@pytest.mark.parametrize("namespace", ["", ' xmlns="http://maven.apache.org/POM/4.0.0"'])
def test_real_xml_comment_and_description_use_existing_llm_unchanged(llm, namespace):
    notice = "© 2017–2022 Example Authors; Portions © Another Contributor"
    llm.return_value = json.dumps({"copyright_notice": notice})
    pom = f'<project{namespace}><!-- © 2017–2022 Example Authors --><description>Portions © Another Contributor</description></project>'
    assert maven._extract_pom_copyright(pom) == notice
    content = llm.call_args.kwargs["context"]["content"]
    assert "© 2017–2022 Example Authors" in content
    assert "Portions © Another Contributor" in content


def test_pom_header_outside_project_is_preserved(llm):
    notice = "Copyright 2001-2020 Example Foundation"
    llm.return_value = json.dumps({"copyright_notice": notice})
    pom = f'<?xml version="1.0" encoding="UTF-8"?><!-- {notice} --><project><name>Example</name></project>'
    assert maven._extract_pom_copyright(pom) == notice
    assert notice in llm.call_args.kwargs["context"]["content"]


@pytest.mark.parametrize("section", ["build", "reporting"])
def test_plugin_templates_inside_profiles_are_excluded(llm, section):
    pom = f'<project><profiles><profile><{section}><plugins><plugin><configuration><bottom>Copyright TEMPLATE</bottom></configuration></plugin></plugins></{section}></profile></profiles></project>'
    assert maven._extract_pom_copyright(pom) is None
    assert "TEMPLATE" not in llm.call_args.kwargs["context"]["content"]


@pytest.mark.parametrize("reply", [None, "", "  ", "None", "null"])
def test_empty_llm_result_stays_empty(llm, reply):
    llm.return_value = json.dumps({"copyright_notice": reply})
    assert maven._extract_pom_copyright("<project><name>Example</name></project>") is None


def test_llm_failure_does_not_break_pom_license(llm, pom_http):
    llm.side_effect = RuntimeError("model unavailable")
    result = maven.analyze_maven_repository_url(MEDIA3_URL)
    assert result["license"] == "Apache-2.0"
    assert result.get("copyright") is None


def test_legacy_pom_parser_uses_llm_instead_of_first_copyright_line(llm):
    llm.return_value = json.dumps({"copyright_notice": "© Example Contributors"})
    licenses, _, notice = maven._extract_license_from_pom("<project><description>© Example Contributors</description></project>")
    assert notice == "© Example Contributors"
    llm.assert_called_once()


@pytest.mark.parametrize("url", [
    "https://repo.maven.apache.org/maven2/org/example/demo/1.2.3/demo-1.2.3.pom",
    "https://central.sonatype.com/artifact/org.example/demo/1.2.3",
    "https://mvnrepository.com/artifact/org.example/demo/1.2.3",
    "https://mirrors.tencent.com/repository/maven/tencent_public/org/example/demo/1.2.3/demo-1.2.3.jar",
    "https://repo.example.test/repository/releases/org/example/demo/1.2.3/demo-1.2.3.pom",
])
@pytest.mark.asyncio
async def test_all_repository_hosts_normalize_to_mvnrepository(monkeypatch, url):
    reachable = AsyncMock(return_value=True)
    monkeypatch.setattr(utils, "is_url_reachable", reachable)
    expected = "https://mvnrepository.com/artifact/org.example/demo/1.2.3"
    assert await maven.build_versioned_maven_license_url(url) == expected
    location = maven.parse_maven_repository_location(url)
    analysis = {"group_id": "org.example", "artifact_id": "demo", "version": "1.2.3", "license": "MIT", "pom_url": location.pom_url}
    assert maven.build_maven_repository_result(analysis, url)["license_files"] == expected


@pytest.mark.parametrize("entry", ["cli", "api"])
@pytest.mark.parametrize("input_kind", ["url", "purl"])
@pytest.mark.asyncio
async def test_cli_and_api_output_preserve_purl_and_metadata(monkeypatch, llm, pom_http, entry, input_kind, tmp_path):
    import main
    import api

    monkeypatch.chdir(tmp_path)
    module = main if entry == "cli" else api
    # purl resolves to Maven Central; the exact POM response is supplied locally.
    input_url = MEDIA3_URL if input_kind == "url" else "pkg:maven/androidx.media3/media3-common@1.0.0"
    df = pd.DataFrame([{"github_url": input_url, "version": "1.0.0", "name": "androidx.media3:media3-common", "package_manager": "maven", "serial_number": "00027"}])
    github = AsyncMock(side_effect=AssertionError("POM succeeded; GitHub must not run"))
    monkeypatch.setattr(module, "process_github_repository", github)
    results = await (main.process_all_repos(None, df, 1) if entry == "cli" else api._process_repositories(None, df))
    assert results[0]["license_files"] == MEDIA3_PAGE
    assert results[0]["input_url"] == input_url
    output = api._generate_output(results)
    expected_name = "media3-common" if input_kind == "purl" else "androidx.media3:media3-common"
    assert output.loc[0, "copyright_notice"] == f"Copyright {expected_name} Original author and authors"
    assert output.loc[0, "serial_number"] == "00027"
    assert output.loc[0, "package_manager"] == "maven"
    github.assert_not_awaited()


@pytest.mark.parametrize("entry", ["cli", "api"])
@pytest.mark.parametrize("default_branch", [True, False])
@pytest.mark.asyncio
async def test_maven_github_fallback_also_normalizes_link(monkeypatch, tmp_path, entry, default_branch):
    import main
    import api

    monkeypatch.chdir(tmp_path)
    module = main if entry == "cli" else api
    monkeypatch.setattr(module, "analyze_maven_repository_url", lambda *args: {"license": None})
    monkeypatch.setattr(utils, "is_url_reachable", AsyncMock(return_value=True))
    github_result = {
        "status": "success", "license_type": "Apache-2.0",
        "license_file_license": "Apache-2.0", "used_default_branch": default_branch,
        "resolved_version": "main" if default_branch else "1.0.0",
        "license_files": "https://github.com/androidx/media/blob/main/LICENSE",
        "copyright_notice": "Copyright 2020 The Android Open Source Project",
    }
    monkeypatch.setattr(module, "process_github_repository", AsyncMock(return_value=github_result))
    df = pd.DataFrame([{"github_url": MEDIA3_URL, "version": "1.0.0", "name": "androidx.media3:media3-common"}])
    results = await (main.process_all_repos(None, df, 1) if entry == "cli" else api._process_repositories(None, df))
    assert results[0]["license_files"] == MEDIA3_PAGE
    assert results[0]["copyright_notice"] == github_result["copyright_notice"]
    assert results[0]["license_type"] == "Apache-2.0"
