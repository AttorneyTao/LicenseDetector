"""Single entry point for synchronous and asynchronous model calls."""

import asyncio
import logging
import os
import random
from typing import Any, Mapping

from . import llm_cache
from .llm_policies import POLICIES
from .llm_provider import get_llm_provider

logger = logging.getLogger("llm_cache")


def _audit_due() -> bool:
    try:
        rate = float(os.getenv("LLM_CACHE_AUDIT_RATE", "0.01"))
    except ValueError:
        rate = 0.0
    return random.random() < max(0.0, min(rate, 1.0))


def _prepare(task: str, prompt: str, provider: Any, kwargs: dict) -> tuple[Any, str] | None:
    policy = POLICIES.get(task)
    if policy is None or not isinstance(prompt, str) or not prompt.strip() or llm_cache.cache_mode() == "off":
        return None
    try:
        return policy, llm_cache.make_key(task, policy, prompt, provider, kwargs)
    except (TypeError, ValueError):
        logger.warning("Uncacheable parameters for task %s", task)
        return None


def _read(prepared: tuple[Any, str], task: str, context: Mapping[str, Any]) -> str | None:
    policy, key = prepared
    try:
        response = llm_cache.read(key)
        if response is None:
            return None
        if policy.validate(response, context) is None:
            if llm_cache.cache_mode() == "read_write":
                llm_cache.quarantine(key, policy)
            logger.warning("Invalid cached response quarantined for task %s, key %s", task, key)
            return None
        return response
    except Exception as exc:
        logger.warning("Cache read failed; using live model: %s", exc)
        return None


def _record(prepared: tuple[Any, str], task: str, response: str, context: Mapping[str, Any]) -> None:
    if llm_cache.cache_mode() != "read_write":
        return
    policy, key = prepared
    try:
        semantic = policy.validate(response, context)
        if semantic is None:
            # A bad live answer must also invalidate a previously verified hit.
            llm_cache.quarantine(key, policy)
            return
        state = llm_cache.observe(key, task, response, semantic, policy)
        logger.info("LLM cache %s: task=%s key=%s", state, task, key)
    except Exception as exc:
        logger.warning("Cache write failed; live response retained: %s", exc)


def complete_sync(task: str, prompt: str, *, context: Mapping[str, Any] | None = None, **kwargs: Any) -> str:
    provider = get_llm_provider()
    context = context or {}
    prepared = _prepare(task, prompt, provider, kwargs)
    cached = _read(prepared, task, context) if prepared else None
    if cached is not None and not _audit_due():
        logger.info("LLM cache hit: task=%s key=%s", task, prepared[1])
        return cached
    response = provider.generate(prompt, **kwargs)
    if prepared:
        _record(prepared, task, response, context)
    return response


async def complete_async(task: str, prompt: str, *, context: Mapping[str, Any] | None = None, **kwargs: Any) -> str:
    provider = get_llm_provider()
    context = context or {}
    prepared = _prepare(task, prompt, provider, kwargs)
    cached = await asyncio.to_thread(_read, prepared, task, context) if prepared else None
    if cached is not None and not _audit_due():
        logger.info("LLM cache hit: task=%s key=%s", task, prepared[1])
        return cached
    response = await provider.generate_async(prompt, **kwargs)
    if prepared:
        await asyncio.to_thread(_record, prepared, task, response, context)
    return response
