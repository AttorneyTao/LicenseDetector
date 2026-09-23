"""Cache hit-rate logs use fake providers and never call a real model."""

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import pytest

from core import llm_service
from core.llm_metrics import CacheMetrics
import core.llm_metrics as llm_metrics


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


def setup_cache(monkeypatch, tmp_path, provider):
    monkeypatch.setenv("LLM_CACHE_PATH", str(tmp_path / "cache.sqlite"))
    monkeypatch.setenv("LLM_CACHE_MODE", "read_write")
    monkeypatch.setenv("LLM_CACHE_AUDIT_RATE", "0")
    monkeypatch.setattr(llm_service, "get_llm_provider", lambda: provider)


def version_response(version):
    return json.dumps({"resolved_version": version, "used_default_branch": False})


def _metric_messages(caplog):
    return [record.getMessage() for record in caplog.records if "LLM_CACHE_STATS" in record.getMessage()]


def test_sync_async_hit_rate_audit_and_bypass(monkeypatch, tmp_path, caplog):
    provider = FakeProvider([
        version_response("v1"), version_response("v1"),
        version_response("v1"), "uncached",
    ])
    setup_cache(monkeypatch, tmp_path, provider)
    monkeypatch.setattr(llm_service, "cache_metrics", CacheMetrics())
    context = {"candidates": ["main", "v1"], "default": "main"}

    with caplog.at_level(logging.INFO, logger="llm_cache"):
        llm_service.complete_sync("crate_version", "same prompt", context=context)
        asyncio.run(llm_service.complete_async("crate_version", "same prompt", context=context))
        llm_service.complete_sync("crate_version", "same prompt", context=context)
        monkeypatch.setenv("LLM_CACHE_AUDIT_RATE", "1")
        asyncio.run(llm_service.complete_async("crate_version", "same prompt", context=context))
        monkeypatch.setenv("LLM_CACHE_MODE", "off")
        llm_service.complete_sync("legacy", "other prompt")

    messages = _metric_messages(caplog)
    assert len(messages) == 5
    assert all(
        f"outcome={outcome}" in message
        for outcome, message in zip(("miss", "miss", "hit", "audit", "bypass"), messages)
    )
    assert "task_requests=4 task_eligible=4 task_hits=1 task_hit_rate=25.00%" in messages[3]
    assert "global_requests=5 global_eligible=4 global_hits=1" in messages[-1]
    assert "global_hit_rate=25.00% global_avoidance_rate=20.00%" in messages[-1]
    assert "global_audits=1 global_bypasses=1 global_provider_errors=0" in messages[-1]
    assert provider.calls == 4


def test_provider_error_is_in_denominator(monkeypatch, tmp_path, caplog):
    class FailingProvider(FakeProvider):
        def generate(self, prompt, **kwargs):
            self.calls += 1
            raise RuntimeError("model unavailable")

    provider = FailingProvider([])
    setup_cache(monkeypatch, tmp_path, provider)
    monkeypatch.setattr(llm_service, "cache_metrics", CacheMetrics())
    with caplog.at_level(logging.INFO, logger="llm_cache"):
        with pytest.raises(RuntimeError, match="model unavailable"):
            llm_service.complete_sync(
                "crate_version", "prompt",
                context={"candidates": ["main", "v1"], "default": "main"},
            )
    message = _metric_messages(caplog)[-1]
    assert "outcome=provider_error" in message
    assert "global_requests=1 global_eligible=1 global_hits=0" in message
    assert "global_provider_errors=1" in message


def test_daily_counters_reset_and_events_remain_logged(caplog):
    current = [date(2026, 9, 23)]
    metrics = CacheMetrics(today=lambda: current[0])
    with caplog.at_level(logging.INFO, logger="llm_cache"):
        metrics.record("crate_version", "hit", eligible=True)
        current[0] = date(2026, 9, 24)
        metrics.record("crate_version", "miss", eligible=True)
    first, second = _metric_messages(caplog)
    assert "day=2026-09-23" in first and "global_hits=1" in first
    assert "day=2026-09-24" in second and "global_requests=1 global_eligible=1 global_hits=0" in second
    assert f"run={metrics.run_id}" in first and f"run={metrics.run_id}" in second


def test_concurrent_counters_do_not_lose_requests(caplog):
    metrics = CacheMetrics()
    with caplog.at_level(logging.INFO, logger="llm_cache"):
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda _: metrics.record("crate_version", "hit", eligible=True), range(100)))
    messages = _metric_messages(caplog)
    assert len(messages) == 100
    assert "global_requests=100 global_eligible=100 global_hits=100" in messages[-1]
    assert "global_hit_rate=100.00% global_avoidance_rate=100.00%" in messages[-1]


def test_metrics_log_failure_does_not_change_model_result(monkeypatch, tmp_path):
    provider = FakeProvider(["live response"])
    setup_cache(monkeypatch, tmp_path, provider)
    monkeypatch.setattr(llm_service, "cache_metrics", CacheMetrics())

    def fail_log(*args, **kwargs):
        raise OSError("log unavailable")

    monkeypatch.setattr(llm_metrics.logger, "info", fail_log)
    assert llm_service.complete_sync("legacy", "prompt") == "live response"
    assert provider.calls == 1
