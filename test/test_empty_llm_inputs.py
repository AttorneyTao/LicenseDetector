"""Blank source material must not consume a model call."""

import asyncio

from core import utils


def test_empty_license_analysis_skips_model_sync_and_async(monkeypatch):
    monkeypatch.setattr(utils, "USE_LLM", True)

    def unexpected_sync(*args, **kwargs):
        raise AssertionError("model called for empty license text")

    async def unexpected_async(*args, **kwargs):
        raise AssertionError("model called for empty license text")

    monkeypatch.setattr(utils, "complete_sync", unexpected_sync)
    monkeypatch.setattr(utils, "complete_async", unexpected_async)
    for content in ("", " \n\t", None):
        sync_result = utils.analyze_license_content(content, "https://example.com/package")
        async_result = asyncio.run(utils.analyze_license_content_async(content, "https://example.com/package"))
        assert sync_result == async_result
        assert sync_result["licenses"] == []
        assert sync_result["confidence"] == 0.0
        assert sync_result["source_url"] == "https://example.com/package"


def test_empty_copyright_extraction_skips_model_sync_and_async(monkeypatch):
    monkeypatch.setattr(utils, "USE_LLM", True)

    def unexpected_sync(*args, **kwargs):
        raise AssertionError("model called for empty copyright text")

    async def unexpected_async(*args, **kwargs):
        raise AssertionError("model called for empty copyright text")

    monkeypatch.setattr(utils, "complete_sync", unexpected_sync)
    monkeypatch.setattr(utils, "complete_async", unexpected_async)
    for content in ("", " \n\t", None):
        assert utils.extract_copyright_info(content) is None
        assert asyncio.run(utils.extract_copyright_info_async(content)) is None


def test_nonempty_inputs_still_use_model(monkeypatch):
    monkeypatch.setattr(utils, "USE_LLM", True)
    calls = []

    def fake_complete(task, prompt, context):
        calls.append((task, context["content"]))
        if task == "license_analysis":
            return '{"main_licenses": ["MIT"]}'
        return '{"copyright_notice": "Copyright 2024 Example"}'

    async def fake_complete_async(task, prompt, context):
        return fake_complete(task, prompt, context)

    monkeypatch.setattr(utils, "complete_sync", fake_complete)
    monkeypatch.setattr(utils, "complete_async", fake_complete_async)
    assert utils.analyze_license_content("MIT text")["licenses"] == ["MIT"]
    assert asyncio.run(utils.analyze_license_content_async("MIT text"))["licenses"] == ["MIT"]
    assert utils.extract_copyright_info("Copyright 2024 Example") == "Copyright 2024 Example"
    assert asyncio.run(utils.extract_copyright_info_async("Copyright 2024 Example")) == "Copyright 2024 Example"
    assert calls == [
        ("license_analysis", "MIT text"),
        ("license_analysis", "MIT text"),
        ("copyright_extract", "Copyright 2024 Example"),
        ("copyright_extract", "Copyright 2024 Example"),
    ]
