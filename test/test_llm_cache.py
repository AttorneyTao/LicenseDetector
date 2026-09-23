"""Cache safety tests use a fake provider; no network or model credits."""

import asyncio
import ast
import json
import stat
import sqlite3
from pathlib import Path

from core import llm_cache, llm_service


class FakeProvider:
    provider_name = "Fake"
    model_name = "test-model"

    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def generate(self, prompt, **kwargs):
        self.calls += 1
        return next(self.responses)

    async def generate_async(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)


def setup_cache(monkeypatch, tmp_path, provider, mode="read_write"):
    monkeypatch.setenv("LLM_CACHE_PATH", str(tmp_path / "cache.sqlite"))
    monkeypatch.setenv("LLM_CACHE_MODE", mode)
    monkeypatch.setenv("LLM_CACHE_AUDIT_RATE", "0")
    monkeypatch.setattr(llm_service, "get_llm_provider", lambda: provider)


def version_response(version):
    return json.dumps({"resolved_version": version, "used_default_branch": False})


def test_two_independent_answers_then_hit(monkeypatch, tmp_path):
    provider = FakeProvider([version_response("v1"), version_response("v1")])
    setup_cache(monkeypatch, tmp_path, provider)
    context = {"candidates": ["main", "v1"], "default": "main"}
    for _ in range(3):
        assert json.loads(llm_service.complete_sync("crate_version", "same exact prompt", context=context))["resolved_version"] == "v1"
    assert provider.calls == 2
    assert stat.S_IMODE(llm_cache.cache_path().stat().st_mode) == 0o600


def test_disagreement_quarantines_and_never_serves(monkeypatch, tmp_path):
    provider = FakeProvider([version_response("v1"), version_response("v2"), version_response("v1")])
    setup_cache(monkeypatch, tmp_path, provider)
    context = {"candidates": ["main", "v1", "v2"], "default": "main"}
    for _ in range(3):
        llm_service.complete_sync("crate_version", "prompt", context=context)
    assert provider.calls == 3
    with sqlite3.connect(llm_cache.cache_path()) as db:
        assert db.execute("SELECT state FROM responses").fetchone()[0] == "quarantined"


def test_invalid_and_negative_answers_are_not_cached(monkeypatch, tmp_path):
    provider = FakeProvider([version_response("hallucinated"), version_response("hallucinated"), version_response("hallucinated")])
    setup_cache(monkeypatch, tmp_path, provider)
    context = {"candidates": ["main", "v1"], "default": "main"}
    for _ in range(3):
        llm_service.complete_sync("crate_version", "prompt", context=context)
    assert provider.calls == 3
    with sqlite3.connect(llm_cache.cache_path()) as db:
        assert db.execute("SELECT COUNT(*) FROM responses").fetchone()[0] == 0


def test_context_revalidation_and_manual_invalidation(monkeypatch, tmp_path):
    provider = FakeProvider([version_response("v1"), version_response("v1"), version_response("v2"), version_response("v2")])
    setup_cache(monkeypatch, tmp_path, provider)
    first = {"candidates": ["main", "v1"], "default": "main"}
    second = {"candidates": ["main", "v2"], "default": "main"}
    for _ in range(2):
        llm_service.complete_sync("npm_version", "prompt", context=first)
    assert llm_service.complete_sync("npm_version", "prompt", context=second) == version_response("v2")
    assert llm_cache.invalidate(task="npm_version") == 1
    assert llm_service.complete_sync("npm_version", "prompt", context=second) == version_response("v2")
    assert provider.calls == 4


def test_audit_detects_change_and_returns_live(monkeypatch, tmp_path):
    provider = FakeProvider([version_response("v1"), version_response("v1"), version_response("v2"), version_response("v2")])
    setup_cache(monkeypatch, tmp_path, provider)
    context = {"candidates": ["main", "v1", "v2"], "default": "main"}
    for _ in range(2):
        llm_service.complete_sync("crate_version", "prompt", context=context)
    monkeypatch.setenv("LLM_CACHE_AUDIT_RATE", "1")
    assert llm_service.complete_sync("crate_version", "prompt", context=context) == version_response("v2")
    monkeypatch.setenv("LLM_CACHE_AUDIT_RATE", "0")
    assert llm_service.complete_sync("crate_version", "prompt", context=context) == version_response("v2")
    assert provider.calls == 4


def test_async_and_sync_share_cache(monkeypatch, tmp_path):
    response = json.dumps({"copyright_notice": "Copyright 2024 Alice"})
    provider = FakeProvider([response, response])
    setup_cache(monkeypatch, tmp_path, provider)
    context = {"content": "Copyright 2024 Alice"}
    assert llm_service.complete_sync("copyright_extract", "prompt", context=context) == response
    assert asyncio.run(llm_service.complete_async("copyright_extract", "prompt", context=context)) == response
    assert asyncio.run(llm_service.complete_async("copyright_extract", "prompt", context=context)) == response
    assert provider.calls == 2


def test_off_read_only_and_unknown_task_bypass(monkeypatch, tmp_path):
    provider = FakeProvider(["live", "live", "live"])
    setup_cache(monkeypatch, tmp_path, provider, mode="off")
    llm_service.complete_sync("license_standardize", "prompt")
    monkeypatch.setenv("LLM_CACHE_MODE", "read_only")
    llm_service.complete_sync("license_standardize", "prompt")
    monkeypatch.setenv("LLM_CACHE_MODE", "read_write")
    llm_service.complete_sync("unknown_task", "prompt")
    assert provider.calls == 3
    assert not llm_cache.cache_path().exists()


def test_cache_failure_falls_back_to_live(monkeypatch, tmp_path):
    provider = FakeProvider([version_response("v1")])
    setup_cache(monkeypatch, tmp_path, provider)
    monkeypatch.setenv("LLM_CACHE_PATH", str(tmp_path))  # directory, not database
    result = llm_service.complete_sync("crate_version", "prompt", context={"candidates": ["v1"], "default": "main"})
    assert result == version_response("v1")
    assert provider.calls == 1


def test_key_isolates_prompt_model_parameters_and_epoch(monkeypatch):
    provider = FakeProvider([])
    policy = llm_service.POLICIES["crate_version"]
    key = llm_cache.make_key("crate_version", policy, "a", provider, {})
    assert key != llm_cache.make_key("crate_version", policy, "a ", provider, {})
    assert key != llm_cache.make_key("crate_version", policy, "a", provider, {"temperature": 0})
    provider.model_name = "other"
    assert key != llm_cache.make_key("crate_version", policy, "a", provider, {})
    provider.model_name = "test-model"
    monkeypatch.setenv("LLM_CACHE_EPOCH", "2")
    assert key != llm_cache.make_key("crate_version", policy, "a", provider, {})


def test_expired_entry_requires_new_confirmation(monkeypatch, tmp_path):
    provider = FakeProvider([version_response("v1"), version_response("v1"), version_response("v2")])
    setup_cache(monkeypatch, tmp_path, provider)
    context = {"candidates": ["main", "v1", "v2"], "default": "main"}
    for _ in range(2):
        llm_service.complete_sync("crate_version", "prompt", context=context)
    with sqlite3.connect(llm_cache.cache_path()) as db:
        db.execute("UPDATE responses SET expires_at=0")
    assert llm_service.complete_sync("crate_version", "prompt", context=context) == version_response("v2")
    assert provider.calls == 3
    with sqlite3.connect(llm_cache.cache_path()) as db:
        assert db.execute("SELECT state FROM responses").fetchone()[0] == "candidate"


def test_cached_license_response_uses_current_source_url(monkeypatch, tmp_path):
    from core import utils

    response = json.dumps({"main_licenses": ["MIT"], "spdx_expression": "MIT", "confidence": 0.99})
    provider = FakeProvider([response, response])
    setup_cache(monkeypatch, tmp_path, provider)
    monkeypatch.setattr(utils, "USE_LLM", True)
    first = utils.analyze_license_content("MIT License", "https://example.com/one")
    utils.analyze_license_content("MIT License", "https://example.com/two")
    third = utils.analyze_license_content("MIT License", "https://example.com/three")
    assert first["source_url"] == "https://example.com/one"
    assert third["source_url"] == "https://example.com/three"
    assert provider.calls == 2


def test_no_new_direct_provider_calls_outside_facade():
    core_dir = Path(__file__).resolve().parents[1] / "core"
    for path in core_dir.glob("*.py"):
        if path.name in {"llm_provider.py", "llm_service.py"}:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        direct = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"generate", "generate_async"}
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "provider"
        ]
        assert not direct, f"Direct LLM call outside facade: {path}"
