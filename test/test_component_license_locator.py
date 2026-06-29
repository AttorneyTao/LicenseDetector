"""
Tests for component-aware subdirectory license location (Step 9.5).

Covers:
- find_license_files_detailed(include_all_dirs=True) now surfaces subdir licenses
  that the root-level search intentionally hides.
- locate_component_license_dir() picks the right subdir for scoped / bare names,
  declines when nothing matches (→ root fallback), and rejects hallucinated paths.
"""

import asyncio
import json

import core.github_utils as gh
from core.utils import find_license_files_detailed


# A dicebear-style monorepo: per-style packages under packages/@dicebear/<style>,
# plus a root README (no root LICENSE).
def _dicebear_tree():
    styles = ["avataaars", "adventurer-neutral", "big-ears", "fun-emoji"]
    tree = [{"type": "blob", "path": "README.md",
             "url": "https://api.github.com/repos/dicebear/dicebear/git/blobs/aaa"}]
    for s in styles:
        tree.append({"type": "tree", "path": f"packages/@dicebear/{s}"})
        tree.append({
            "type": "blob",
            "path": f"packages/@dicebear/{s}/LICENSE",
            "url": "https://api.github.com/repos/dicebear/dicebear/git/blobs/bbb",
        })
    return tree


KEYWORDS = ["license", "licenses", "copying", "notice"]


def test_root_search_hides_subdir_licenses():
    """Default root-level search must NOT pull in packages/*/LICENSE."""
    path_map = {"tree": _dicebear_tree(), "resolved_version": "v9.4.2"}
    files = find_license_files_detailed(path_map, "", KEYWORDS)
    assert files == [], f"expected no root license files, got {files}"


def test_include_all_dirs_surfaces_subdir_licenses():
    """include_all_dirs=True surfaces every subdir LICENSE as a candidate."""
    path_map = {"tree": _dicebear_tree(), "resolved_version": "v9.4.2"}
    files = find_license_files_detailed(path_map, "", KEYWORDS, include_all_dirs=True)
    dirs = {f["directory"] for f in files}
    assert dirs == {
        "packages/@dicebear/avataaars",
        "packages/@dicebear/adventurer-neutral",
        "packages/@dicebear/big-ears",
        "packages/@dicebear/fun-emoji",
    }


class _FakeProvider:
    """Returns a canned JSON response, recording the prompt it was given."""

    def __init__(self, response):
        self._response = response
        self.last_prompt = None

    def generate(self, prompt, **kwargs):
        self.last_prompt = prompt
        return self._response


def _candidate_dirs():
    return [
        {"directory": "packages/@dicebear/avataaars", "filenames": ["LICENSE"]},
        {"directory": "packages/@dicebear/adventurer-neutral", "filenames": ["LICENSE"]},
        {"directory": "packages/@dicebear/big-ears", "filenames": ["LICENSE"]},
    ]


def _run_locate(monkeypatch, response, name):
    provider = _FakeProvider(response)
    monkeypatch.setattr(gh, "USE_LLM", True)
    monkeypatch.setattr(gh, "get_llm_provider", lambda: provider)
    result = asyncio.run(
        gh.locate_component_license_dir(name, "9.4.2", _candidate_dirs())
    )
    return result, provider


def test_locate_scoped_name(monkeypatch):
    """Scoped input (@dicebear/avataaars) — the exact bug case — now resolves."""
    resp = json.dumps({
        "license_directory": "packages/@dicebear/avataaars",
        "confidence": 0.95,
        "reasoning": "scope+name match",
    })
    result, _ = _run_locate(monkeypatch, resp, "@dicebear/avataaars")
    assert result == "packages/@dicebear/avataaars"


def test_locate_bare_name(monkeypatch):
    resp = json.dumps({
        "license_directory": "packages/@dicebear/big-ears",
        "confidence": 0.9,
        "reasoning": "name match",
    })
    result, _ = _run_locate(monkeypatch, resp, "big-ears")
    assert result == "packages/@dicebear/big-ears"


def test_locate_declines_returns_root(monkeypatch):
    """Component is the whole repo / no subdir matches → null → caller uses root."""
    resp = json.dumps({
        "license_directory": None,
        "confidence": 0.2,
        "reasoning": "no subdir corresponds to this component",
    })
    result, _ = _run_locate(monkeypatch, resp, "dicebear")
    assert result is None


def test_locate_rejects_hallucinated_path(monkeypatch):
    """A path the LLM invents (not in candidates) must be ignored, not trusted."""
    resp = json.dumps({
        "license_directory": "packages/@dicebear/does-not-exist",
        "confidence": 0.99,
        "reasoning": "hallucinated",
    })
    result, _ = _run_locate(monkeypatch, resp, "@dicebear/ghost")
    assert result is None


def test_locate_string_none_treated_as_decline(monkeypatch):
    """Tolerate the LLM returning the string "none"/"root" instead of null."""
    for literal in ("none", "None", "root", "null", ""):
        resp = json.dumps({"license_directory": literal, "confidence": 0.1, "reasoning": "x"})
        result, _ = _run_locate(monkeypatch, resp, "whatever")
        assert result is None, f"literal {literal!r} should decline"


def test_locate_no_candidates_skips_llm(monkeypatch):
    """No candidate subdirs → return None without calling the LLM."""
    called = {"hit": False}

    def _boom():
        called["hit"] = True
        raise AssertionError("LLM should not be called when there are no candidates")

    monkeypatch.setattr(gh, "USE_LLM", True)
    monkeypatch.setattr(gh, "get_llm_provider", _boom)
    result = asyncio.run(gh.locate_component_license_dir("x", "1.0", []))
    assert result is None
    assert called["hit"] is False


def test_locate_disabled_llm_returns_none(monkeypatch):
    monkeypatch.setattr(gh, "USE_LLM", False)
    result = asyncio.run(
        gh.locate_component_license_dir("@dicebear/avataaars", "9.4.2", _candidate_dirs())
    )
    assert result is None
