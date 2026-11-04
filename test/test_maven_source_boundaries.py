"""Maven source/version evidence must survive every public dispatch path."""
from copy import deepcopy
from io import BytesIO
from unittest.mock import AsyncMock, Mock

import pandas as pd
import pytest

from core import maven_utils as maven, utils
from core.purl_fallback import retry_with_purl


CENTRAL = "https://repo1.maven.org/maven2"
GOOGLE = "https://dl.google.com/dl/android/maven2"
PRIVATE = "https://packages.example.test/repository/releases"
WEBKIT = ("androidx.webkit", "webkit", "1.7.0")
MEDIA3 = ("androidx.media3", "media3-common", "1.0.0")


def pom_url(base, coordinates=WEBKIT):
    group, artifact, version = coordinates
    return f"{base}/{group.replace('.', '/')}/{artifact}/{version}/{artifact}-{version}.pom"


def page_url(coordinates=WEBKIT):
    return "https://mvnrepository.com/artifact/" + "/".join(coordinates)


def pom(coordinates=WEBKIT, license_name="Apache License, Version 2.0", extra=""):
    group, artifact, version = coordinates
    license_xml = (
        f"<licenses><license><name>{license_name}</name></license></licenses>"
        if license_name else ""
    )
    return (
        '<project xmlns="http://maven.apache.org/POM/4.0.0">'
        f"<modelVersion>4.0.0</modelVersion><groupId>{group}</groupId>"
        f"<artifactId>{artifact}</artifactId><version>{version}</version>"
        f"{extra}{license_xml}</project>"
    )


@pytest.fixture(autouse=True)
def no_external_models(monkeypatch):
    monkeypatch.setattr(utils, "extract_copyright_info", lambda content: None)

    def spdx(licenses):
        names = [item.get("name") for item in licenses]
        assert names == ["Apache License, Version 2.0"]
        return "Apache-2.0"

    monkeypatch.setattr(maven, "_convert_licenses_to_spdx", spdx)


def http_responses(monkeypatch, responses):
    # Exact URL lookup is intentional: a Google POM must never appear at Central.
    http = Mock(side_effect=lambda url: responses.get(url, (None, 404)))
    monkeypatch.setattr(maven, "_http_get", http)
    return http


@pytest.mark.parametrize("coordinates", [WEBKIT, MEDIA3])
@pytest.mark.parametrize("host", ["mvnrepository.com", "central.sonatype.com"])
def test_androidx_pages_use_google_pom(monkeypatch, coordinates, host):
    url = page_url(coordinates).replace("mvnrepository.com", host)
    http = http_responses(monkeypatch, {pom_url(GOOGLE, coordinates): (pom(coordinates), 200)})
    analysis = maven.analyze_maven_repository_url(url)
    assert analysis["version_verified"] is True
    assert analysis["pom_url"] == pom_url(GOOGLE, coordinates)
    assert analysis["repository_base_url"] == GOOGLE
    assert analysis["license"] == "Apache-2.0"
    http.assert_called_once_with(pom_url(GOOGLE, coordinates))


@pytest.mark.parametrize("base", [CENTRAL, GOOGLE, PRIVATE])
def test_direct_repository_remains_first_source(monkeypatch, base):
    http = http_responses(monkeypatch, {pom_url(base): (pom(), 200)})
    analysis = maven.analyze_maven_repository_url(pom_url(base))
    assert analysis["pom_url"] == pom_url(base)
    assert analysis["version_verified"] is True
    http.assert_called_once_with(pom_url(base))


@pytest.mark.parametrize("base", [CENTRAL, PRIVATE])
def test_androidx_missing_at_input_source_can_use_google(monkeypatch, base):
    http = http_responses(monkeypatch, {pom_url(GOOGLE): (pom(), 200)})
    analysis = maven.analyze_maven_repository_url(pom_url(base))
    assert analysis["license"] == "Apache-2.0"
    assert analysis["pom_url"] == pom_url(GOOGLE)
    assert http.call_args_list[0].args == (pom_url(base),)
    assert [c.args[0] for c in http.call_args_list].count(pom_url(GOOGLE)) == 1


def test_androidx_google_miss_retains_central_fallback(monkeypatch):
    http = http_responses(monkeypatch, {pom_url(CENTRAL): (pom(), 200)})
    analysis = maven.analyze_maven_repository_url(page_url())
    assert analysis["license"] == "Apache-2.0"
    assert analysis["pom_url"] == pom_url(CENTRAL)
    assert [call.args[0] for call in http.call_args_list] == [pom_url(GOOGLE), pom_url(CENTRAL)]


@pytest.mark.asyncio
async def test_explicit_version_column_retains_precedence_over_page_version(monkeypatch):
    selected = ("androidx.webkit", "webkit", "1.8.0")
    http = http_responses(monkeypatch, {pom_url(GOOGLE, selected): (pom(selected), 200)})
    analysis = maven.analyze_maven_repository_url(page_url(), "1.8.0")
    assert analysis["version"] == "1.8.0"
    assert await maven.build_versioned_maven_license_url(page_url(), "1.8.0", analysis=analysis) == page_url(selected)
    http.assert_called_once_with(pom_url(GOOGLE, selected))


@pytest.mark.asyncio
async def test_directory_404_does_not_override_valid_pom(monkeypatch):
    http = http_responses(monkeypatch, {pom_url(GOOGLE): (pom(), 200)})
    directory_check = AsyncMock(side_effect=AssertionError("Directory status is not version evidence"))
    monkeypatch.setattr(utils, "is_url_reachable", directory_check)
    assert await maven.build_versioned_maven_license_url(page_url()) == page_url()
    http.assert_called_once_with(pom_url(GOOGLE))
    directory_check.assert_not_awaited()


@pytest.mark.asyncio
async def test_verified_pom_reused_when_license_metadata_missing(monkeypatch):
    http = http_responses(monkeypatch, {pom_url(GOOGLE): (pom(license_name=None), 200)})
    analysis = maven.analyze_maven_repository_url(page_url())
    assert not analysis.get("license")
    assert analysis["version_verified"] is True
    assert analysis["pom_url"] == pom_url(GOOGLE)
    count = http.call_count
    http.side_effect = AssertionError("Existing version evidence must be reused")
    assert await maven.build_versioned_maven_license_url(page_url(), analysis=analysis) == page_url()
    assert http.call_count == count


@pytest.mark.parametrize("body", [
    pom(("androidx.other", "webkit", "1.7.0")),
    pom(("androidx.webkit", "other", "1.7.0")),
    pom(("androidx.webkit", "webkit", "1.8.0")),
    pom(("androidx.webkit", "webkit", "${unresolved}")),
    "<html><body>Sign in</body></html>",
    "<project><broken>",
    "<project><licenses><license><name>Apache License, Version 2.0</name></license></licenses></project>",
])
@pytest.mark.asyncio
async def test_http_200_without_matching_pom_is_not_version_evidence(monkeypatch, body):
    http_responses(monkeypatch, {pom_url(GOOGLE): (body, 200)})
    analysis = maven.analyze_maven_repository_url(page_url())
    assert not analysis.get("license")
    assert not analysis.get("version_verified")
    assert await maven.build_versioned_maven_license_url(page_url(), analysis=analysis) is None


@pytest.mark.parametrize("status", [None, 403, 404, 500])
@pytest.mark.asyncio
async def test_unavailable_sources_do_not_claim_version_missing_or_confirmed(monkeypatch, status):
    http = http_responses(monkeypatch, {
        pom_url(GOOGLE): (None, status),
        pom_url(CENTRAL): (None, status),
    })
    analysis = maven.analyze_maven_repository_url(page_url())
    assert analysis["version"] == "1.7.0"
    assert not analysis.get("version_verified")
    assert not analysis.get("license")
    request_count = http.call_count
    assert await maven.build_versioned_maven_license_url(page_url(), analysis=analysis) is None
    assert http.call_count == request_count


@pytest.mark.parametrize("field,value", [("group_id", "androidx.other"), ("artifact_id", "other"), ("version", "1.8.0")])
@pytest.mark.asyncio
async def test_other_components_cached_evidence_is_not_reused(monkeypatch, field, value):
    http_responses(monkeypatch, {pom_url(GOOGLE): (pom(), 200)})
    analysis = maven.analyze_maven_repository_url(page_url())
    analysis[field] = value
    http_responses(monkeypatch, {})
    assert await maven.build_versioned_maven_license_url(page_url(), analysis=analysis) is None


@pytest.mark.parametrize("error", [TimeoutError, ConnectionError])
@pytest.mark.parametrize("central_available", [False, True])
@pytest.mark.asyncio
async def test_transport_exception_preserves_remaining_sources(monkeypatch, error, central_available):
    def fetch(url):
        if central_available and url == pom_url(CENTRAL):
            return pom(), 200
        raise error("simulated transport failure")

    http = Mock(side_effect=fetch)
    monkeypatch.setattr(maven, "_http_get", http)
    analysis = maven.analyze_maven_repository_url(page_url())
    assert analysis["version_verified"] is central_available
    assert bool(analysis.get("license")) is central_available
    count = http.call_count
    assert await maven.build_versioned_maven_license_url(page_url(), analysis=analysis) == (
        page_url() if central_available else None
    )
    assert http.call_count == count


def test_inherited_group_and_version_are_valid_coordinates(monkeypatch):
    body = pom().replace("<groupId>androidx.webkit</groupId>", "", 1).replace("<version>1.7.0</version>", "", 1)
    body = body.replace("<modelVersion>", "<parent><groupId>androidx.webkit</groupId><artifactId>parent</artifactId><version>1.7.0</version></parent><modelVersion>")
    http_responses(monkeypatch, {pom_url(GOOGLE): (body, 200)})
    analysis = maven.analyze_maven_repository_url(page_url())
    assert analysis["version_verified"] is True
    assert analysis["license"] == "Apache-2.0"


def test_local_revision_property_is_valid_coordinate(monkeypatch):
    body = pom(("androidx.webkit", "webkit", "${revision}"), extra="<properties><revision>1.7.0</revision></properties>")
    http_responses(monkeypatch, {pom_url(GOOGLE): (body, 200)})
    analysis = maven.analyze_maven_repository_url(page_url())
    assert analysis["version_verified"] is True
    assert analysis["version"] == "1.7.0"


def test_parent_license_keeps_child_version_link(monkeypatch):
    parent = ("androidx.webkit", "parent", "2.0")
    body = pom(license_name=None, extra="<parent><groupId>androidx.webkit</groupId><artifactId>parent</artifactId><version>2.0</version></parent>")
    http = http_responses(monkeypatch, {pom_url(GOOGLE): (body, 200), pom_url(GOOGLE, parent): (pom(parent), 200)})
    analysis = maven.analyze_maven_repository_url(page_url())
    result = maven.build_maven_repository_result(analysis, page_url())
    assert result["license_files"] == page_url()
    assert result["resolved_version"] == "1.7.0"
    assert [call.args[0] for call in http.call_args_list] == [pom_url(GOOGLE), pom_url(GOOGLE, parent)]


@pytest.mark.asyncio
async def test_timestamped_snapshot_pom_confirms_snapshot_version(monkeypatch):
    coordinates = ("org.example", "demo", "1.0-SNAPSHOT")
    version_dir = f"{PRIVATE}/org/example/demo/1.0-SNAPSHOT"
    timestamped = f"{version_dir}/demo-1.0-20261103.010203-4.pom"
    metadata = "<metadata><versioning><snapshotVersions><snapshotVersion><extension>pom</extension><value>1.0-20261103.010203-4</value></snapshotVersion></snapshotVersions></versioning></metadata>"
    http = http_responses(monkeypatch, {
        f"{version_dir}/maven-metadata.xml": (metadata, 200),
        timestamped: (pom(coordinates), 200),
    })
    analysis = maven.analyze_maven_repository_url(pom_url(PRIVATE, coordinates))
    assert analysis["version_verified"] is True
    assert analysis["pom_url"] == timestamped
    assert await maven.build_versioned_maven_license_url(pom_url(PRIVATE, coordinates), analysis=analysis) == page_url(coordinates)
    assert [c.args[0] for c in http.call_args_list] == [pom_url(PRIVATE, coordinates), f"{version_dir}/maven-metadata.xml", timestamped]


async def run_entry(monkeypatch, tmp_path, entry, row, github_result=None):
    import api
    import main

    monkeypatch.chdir(tmp_path)
    module = main if entry == "cli" else api
    github = AsyncMock(return_value=github_result) if github_result else AsyncMock(side_effect=AssertionError("Confirmed POM must finish without GitHub"))
    monkeypatch.setattr(module, "process_github_repository", github)
    retries = []

    async def watch_retry(row, original, retry):
        callback = AsyncMock(wraps=retry)
        retries.append(callback)
        return await retry_with_purl(row, original, callback)

    monkeypatch.setattr(module, "retry_with_purl", watch_retry)
    frame = pd.DataFrame([row])
    results = await (main.process_all_repos(None, frame, 1) if entry == "cli" else api._process_repositories(None, frame))
    return results[0], api._generate_output(results), github, retries


@pytest.mark.parametrize("entry", ["cli", "api"])
@pytest.mark.parametrize("kind", ["page", "purl", "google_pom", "google_short_pom", "google_archive"])
@pytest.mark.parametrize("version", [
    pytest.param(None, id="none"),
    pytest.param(float("nan"), id="excel-nan"),
    pytest.param(pd.NA, id="pandas-na"),
    pytest.param("", id="empty-string"),
    pytest.param(" \t ", id="whitespace"),
    pytest.param("1.7.0", id="explicit-version"),
])
@pytest.mark.asyncio
async def test_confirmed_version_survives_dispatch_and_export(monkeypatch, tmp_path, entry, kind, version):
    url = {
        "page": page_url(),
        "purl": "pkg:maven/androidx.webkit/webkit@1.7.0",
        "google_pom": pom_url(GOOGLE),
        "google_short_pom": pom_url("https://maven.google.com"),
        "google_archive": pom_url(GOOGLE).replace(".pom", "-sources.jar"),
    }[kind]
    expected_pom = url if kind == "google_short_pom" else pom_url(GOOGLE)
    http = http_responses(monkeypatch, {expected_pom: (pom(), 200)})
    row = {"github_url": url, "version": version, "name": "androidx.webkit:webkit", "package_manager": "maven", "serial_number": "00017"}
    result, output, github, retries = await run_entry(monkeypatch, tmp_path, entry, row)
    assert result["status"] == "success"
    assert result["license_files"] == page_url()
    assert result["resolved_version"] == "1.7.0"
    assert result["used_default_branch"] is False
    assert output.loc[0, "concluded_license"] == "Apache-2.0"
    assert output.loc[0, "license_files"] == page_url()
    assert output.loc[0, "input_url"] == url
    assert output.loc[0, "serial_number"] == "00017"
    assert output.loc[0, "package_manager"] == "maven"
    assert "version_verified" not in output.columns
    assert "pom_url" not in output.columns
    assert "repository_base_url" not in output.columns
    github.assert_not_awaited()
    for retry in retries:
        retry.assert_not_awaited()
    http.assert_called_once_with(expected_pom)


def test_download_endpoint_reads_actual_empty_excel_version_cells(monkeypatch, tmp_path):
    """Exercise upload, read_input_excel, dispatch and the standard XLSX writer."""
    import api
    from fastapi.testclient import TestClient

    monkeypatch.chdir(tmp_path)
    http = http_responses(monkeypatch, {pom_url(GOOGLE): (pom(), 200)})
    github = AsyncMock(wraps=api.process_github_repository)
    monkeypatch.setattr(api, "process_github_repository", github)
    # Use an existing client, as a running service does, without a startup health request.
    client_api = api.GitHubAPI()
    network = AsyncMock(side_effect=AssertionError("Maven POM succeeded; GitHub HTTP must not run"))
    monkeypatch.setattr(client_api, "_make_request", network)
    monkeypatch.setattr(api, "_api_instance", client_api)
    urls = [page_url(), pom_url(GOOGLE), "pkg:maven/androidx.webkit/webkit@1.7.0"]
    source = pd.DataFrame([
        {"github_url": url, "version": version, "name": "androidx.webkit:webkit",
         "package_manager": "maven", "serial_number": f"000{index + 1}"}
        for index, (url, version) in enumerate(zip(urls, [None, "", " \t "]))
    ])
    input_bytes = BytesIO()
    source.to_excel(input_bytes, index=False, engine="openpyxl")
    payload = input_bytes.getvalue()
    parsed = utils.read_input_excel(BytesIO(payload))
    assert isinstance(parsed.loc[0, "version"], float)
    assert pd.isna(parsed.loc[0, "version"])
    assert pd.isna(parsed.loc[1, "version"])
    assert parsed.loc[2, "version"] == " \t "

    with TestClient(api.app) as client:
        response = client.post(
            "/api/v1/analyze-and-download",
            files={"file": ("input.xlsx", payload, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        )
    assert response.status_code == 200, response.text[:500] if response.status_code != 200 else ""
    assert response.headers["content-type"].startswith("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    assert response.content[:2] == b"PK"
    with pd.ExcelFile(BytesIO(response.content)) as workbook:
        assert workbook.sheet_names == ["分析结果"]
        output = pd.read_excel(workbook, dtype={"serial_number": str, "resolved_version": str})
    assert len(output) == 3
    assert output["license_files"].tolist() == [page_url()] * 3
    assert output["resolved_version"].tolist() == ["1.7.0"] * 3
    assert output["concluded_license"].tolist() == ["Apache-2.0"] * 3
    assert output["status"].tolist() == ["success"] * 3
    assert output["used_default_branch"].tolist() == [False] * 3
    assert output["input_url"].tolist() == urls
    assert output["serial_number"].tolist() == ["0001", "0002", "0003"]
    assert not {"version_verified", "pom_url", "repository_base_url"}.intersection(output.columns)
    github.assert_not_awaited()
    network.assert_not_awaited()
    assert [call.args[0] for call in http.call_args_list] == [pom_url(GOOGLE)] * 3


@pytest.mark.parametrize("entry", ["cli", "api"])
@pytest.mark.asyncio
async def test_missing_version_keeps_registry_release_resolution(monkeypatch, tmp_path, entry):
    url = page_url().rsplit("/", 1)[0]
    metadata_url = f"{GOOGLE}/androidx/webkit/webkit/maven-metadata.xml"
    metadata = "<metadata><versioning><release>1.7.0</release><latest>1.8.0-beta01</latest></versioning></metadata>"
    http = http_responses(monkeypatch, {metadata_url: (metadata, 200), pom_url(GOOGLE): (pom(), 200)})
    row = {"github_url": url, "version": None, "name": "androidx.webkit:webkit", "package_manager": "maven"}
    result, output, github, retries = await run_entry(monkeypatch, tmp_path, entry, row)
    assert result["resolved_version"] == "1.7.0"
    assert output.loc[0, "license_files"] == page_url()
    assert [call.args[0] for call in http.call_args_list] == [metadata_url, pom_url(GOOGLE)]
    github.assert_not_awaited()
    for retry in retries:
        retry.assert_not_awaited()


def github_default_result():
    return {
        "status": "success", "license_type": "Apache-2.0", "license_file_license": "Apache-2.0",
        "repo_url": "https://github.com/androidx/androidx", "resolved_version": "androidx-main",
        "license_files": "https://github.com/androidx/androidx/blob/androidx-main/LICENSE.txt",
        "used_default_branch": True, "copyright_notice": "Copyright Example Authors",
    }


@pytest.mark.parametrize("entry", ["cli", "api"])
@pytest.mark.asyncio
async def test_github_input_purl_retry_recovers_google_version(monkeypatch, tmp_path, entry):
    http = http_responses(monkeypatch, {pom_url(GOOGLE): (pom(), 200)})
    row = {
        "github_url": "https://github.com/androidx/androidx",
        "version": "1.7.0", "name": "androidx.webkit:webkit",
        "package_manager": "maven", "serial_number": "00017",
    }
    result, output, github, retries = await run_entry(
        monkeypatch, tmp_path, entry, row, github_default_result()
    )
    assert result["status"] == "success"
    assert result["resolved_version"] == "1.7.0"
    assert result["used_default_branch"] is False
    assert result["license_files"] == page_url()
    assert output.loc[0, "concluded_license"] == "Apache-2.0"
    assert output.loc[0, "license_files"] == page_url()
    assert output.loc[0, "input_url"] == row["github_url"]
    assert output.loc[0, "serial_number"] == "00017"
    github.assert_awaited_once()
    assert sum(retry.await_count for retry in retries) == 1
    assert retries[0].await_args.args == ("pkg:maven/androidx.webkit/webkit@1.7.0",)
    http.assert_called_once_with(pom_url(GOOGLE))


@pytest.mark.parametrize("entry", ["cli", "api"])
@pytest.mark.parametrize("valid_pom", [True, False])
@pytest.mark.asyncio
async def test_github_and_purl_fallback_preserve_best_available_version(monkeypatch, tmp_path, entry, valid_pom):
    responses = {pom_url(GOOGLE): (pom(license_name=None), 200)} if valid_pom else {}
    http_responses(monkeypatch, responses)
    row = {"github_url": page_url(), "version": "1.7.0", "name": "androidx.webkit:webkit", "package_manager": "maven"}
    original = github_default_result()
    result, output, github, retries = await run_entry(monkeypatch, tmp_path, entry, row, deepcopy(original))
    assert result["status"] == "success"
    assert result["license_type"] == "Apache-2.0"
    assert result["copyright_notice"] == original["copyright_notice"]
    expected_link = page_url() if valid_pom else original["license_files"]
    assert result["license_files"] == expected_link
    assert output.loc[0, "license_files"] == expected_link
    assert output.loc[0, "concluded_license"] == "Apache-2.0"
    assert github.await_count >= 1
    if not valid_pom:
        assert sum(retry.await_count for retry in retries) == 1
        assert github.await_count == 2
