"""Coordinate verification preserves valid properties inherited from parent POMs."""
from unittest.mock import Mock

import pytest

from core import maven_utils as maven, utils


BASE = "https://packages.example.test/repository/releases"
GAV = maven.GAV("org.example", "demo", "1.2.3")


def url(artifact="demo", version="1.2.3", base=BASE):
    return f"{base}/org/example/{artifact}/{version}/{artifact}-{version}.pom"


def pom(artifact="demo", version="${revision}", parent=None, properties=None, namespace=True):
    parent_xml = ""
    if parent:
        parent_xml = ("<parent><groupId>org.example</groupId>"
                      f"<artifactId>{parent[0]}</artifactId><version>{parent[1]}</version></parent>")
    property_xml = "".join(f"<{key}>{value}</{key}>" for key, value in (properties or {}).items())
    ns = ' xmlns="http://maven.apache.org/POM/4.0.0"' if namespace else ""
    return (f"<project{ns}><groupId>org.example</groupId><artifactId>{artifact}</artifactId>"
            f"<version>{version}</version>{parent_xml}<properties>{property_xml}</properties>"
            "<licenses><license><name>MIT</name></license></licenses></project>")


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(utils, "extract_copyright_info", lambda *args: None)
    monkeypatch.setattr(maven, "_convert_licenses_to_spdx", lambda licenses: "MIT")


def http(monkeypatch, responses):
    mock = Mock(side_effect=lambda address: responses.get(address, (None, 404)))
    monkeypatch.setattr(maven, "_http_get", mock)
    return mock


@pytest.mark.parametrize("namespace", [True, False])
@pytest.mark.asyncio
async def test_parent_revision_keeps_license_and_version_page(monkeypatch, namespace):
    child = pom(parent=("parent", "1.0"), namespace=namespace)
    fetch = http(monkeypatch, {
        url(): (child, 200),
        url("parent", "1.0"): (pom("parent", "1.0", properties={"revision": "1.2.3"}, namespace=namespace), 200),
    })
    analysis = maven.analyze_maven_repository_url(url())
    assert analysis["license"] == "MIT"
    assert analysis["version_verified"] is True
    assert analysis["pom_url"] == url()
    assert await maven.build_versioned_maven_license_url(url(), analysis=analysis) == "https://mvnrepository.com/artifact/org.example/demo/1.2.3"
    assert [call.args[0] for call in fetch.call_args_list] == [url(), url("parent", "1.0")]


def test_child_properties_override_parent_values(monkeypatch):
    child = pom(version="${major}.${minor}.${patch}", parent=("parent", "1.0"),
                properties={"major": "1", "patch": "3"})
    fetch = http(monkeypatch, {
        url(): (child, 200),
        url("parent", "1.0"): (pom("parent", "1.0", properties={"major": "9", "minor": "2", "patch": "9"}), 200),
    })
    assert maven._fetch_verified_pom(GAV, BASE) == (child, url())
    assert fetch.call_count == 2


def test_local_revision_does_not_fetch_unneeded_parent(monkeypatch):
    child = pom(parent=("parent", "1.0"), properties={"revision": "1.2.3"})
    fetch = http(monkeypatch, {url(): (child, 200)})
    assert maven._fetch_verified_pom(GAV, BASE) == (child, url())
    fetch.assert_called_once_with(url())


def test_inherited_expression_can_use_child_property(monkeypatch):
    child = pom(parent=("parent", "1.0"), properties={"release": "1.2.3"})
    fetch = http(monkeypatch, {
        url(): (child, 200),
        url("parent", "1.0"): (pom("parent", "1.0", properties={"revision": "${release}"}), 200),
    })
    assert maven._fetch_verified_pom(GAV, BASE) == (child, url())
    assert fetch.call_count == 2


@pytest.mark.parametrize("parent_revision", [None, "2.0"])
def test_child_properties_cannot_validate_parent_coordinates(monkeypatch, parent_revision):
    child = pom(parent=("parent", "1.0"), properties={"parentRevision": "1.0"})
    parent_properties = {"revision": "1.2.3"}
    if parent_revision:
        parent_properties["parentRevision"] = parent_revision
    http(monkeypatch, {
        url(): (child, 200),
        url("parent", "1.0"): (pom("parent", "${parentRevision}", properties=parent_properties), 200),
    })
    assert maven._fetch_verified_pom(GAV, BASE) is None


def test_grandparent_properties_resolve_parent_property_reference(monkeypatch):
    child = pom(parent=("parent", "1.0"))
    fetch = http(monkeypatch, {
        url(): (child, 200),
        url("parent", "1.0"): (pom("parent", "1.0", parent=("grandparent", "2.0"), properties={"revision": "${release}"}), 200),
        url("grandparent", "2.0"): (pom("grandparent", "2.0", properties={"release": "1.2.3"}), 200),
    })
    assert maven._fetch_verified_pom(GAV, BASE) == (child, url())
    assert [call.args[0] for call in fetch.call_args_list] == [url(), url("parent", "1.0"), url("grandparent", "2.0")]


@pytest.mark.parametrize("parent_body,status", [
    (None, None), (None, 403), (None, 404), (None, 500),
    ("<html>Access denied</html>", 200),
    (pom("other", "1.0", properties={"revision": "1.2.3"}), 200),
    (pom("parent", "2.0", properties={"revision": "1.2.3"}), 200),
    (pom("parent", "1.0", properties={"revision": "1.2.3"}).replace("org.example", "org.other"), 200),
])
def test_failed_or_wrong_parent_cannot_supply_version(monkeypatch, parent_body, status):
    http(monkeypatch, {url(): (pom(parent=("parent", "1.0")), 200),
                       url("parent", "1.0"): (parent_body, status)})
    assert maven._fetch_verified_pom(GAV, BASE) is None


def test_parent_from_existing_central_fallback_is_verified(monkeypatch):
    child = pom(parent=("parent", "1.0"))
    central_parent = url("parent", "1.0", maven._DEFAULT_MAVEN_CENTRAL_BASE)
    fetch = http(monkeypatch, {
        url(): (child, 200),
        central_parent: (pom("parent", "1.0", properties={"revision": "1.2.3"}), 200),
    })
    assert maven._fetch_verified_pom(GAV, BASE) == (child, url())
    assert [call.args[0] for call in fetch.call_args_list] == [url(), url("parent", "1.0"), central_parent]


def test_parent_cycle_is_bounded_and_unconfirmed(monkeypatch):
    fetch = http(monkeypatch, {
        url(): (pom(parent=("parent", "1.0")), 200),
        url("parent", "1.0"): (pom("parent", "1.0", parent=("grandparent", "2.0")), 200),
        url("grandparent", "2.0"): (pom("grandparent", "2.0", parent=("parent", "1.0")), 200),
    })
    assert maven._fetch_verified_pom(GAV, BASE) is None
    requested = [call.args[0] for call in fetch.call_args_list]
    assert len(requested) <= 5
    assert len(requested) == len(set(requested))


def test_parent_chain_stops_after_five_levels(monkeypatch):
    responses = {url(): (pom(parent=("parent1", "1.0")), 200)}
    for index in range(1, 8):
        responses[url(f"parent{index}", "1.0")] = (
            pom(f"parent{index}", "1.0", parent=(f"parent{index + 1}", "1.0"),
                properties={"revision": "1.2.3"} if index == 7 else {}), 200,
        )
    fetch = http(monkeypatch, responses)
    assert maven._fetch_verified_pom(GAV, BASE) is None
    requested = [call.args[0] for call in fetch.call_args_list]
    assert url("parent5", "1.0") in requested
    assert url("parent6", "1.0") not in requested
    assert len(requested) <= 11


def test_unresolved_parent_coordinates_are_not_guessed(monkeypatch):
    fetch = http(monkeypatch, {url(): (pom(parent=("parent", "${parentRevision}")), 200)})
    assert maven._fetch_verified_pom(GAV, BASE) is None
    fetch.assert_called_once_with(url())
