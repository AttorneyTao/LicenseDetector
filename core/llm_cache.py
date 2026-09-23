"""Small, fail-open SQLite store for confirmed LLM responses.

Only a SHA-256 of the prompt is stored, never the prompt itself. Responses may
still contain sensitive text, so the database and its directory are private.
"""

import hashlib
import json
import logging
import os
import sqlite3
import sys
import time
from contextlib import closing
from pathlib import Path
from typing import Any

from .llm_policies import TaskPolicy

logger = logging.getLogger("llm_cache")


def cache_mode() -> str:
    mode = os.getenv("LLM_CACHE_MODE", "read_write").lower()
    return mode if mode in {"off", "read_only", "read_write"} else "off"


def cache_path() -> Path:
    configured = os.getenv("LLM_CACHE_PATH")
    if configured:
        return Path(configured).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library/Application Support/LicenseDetector/llm-cache.sqlite"
    return Path("/var/lib/licensedetector/llm-cache.sqlite")


def make_key(task: str, policy: TaskPolicy, prompt: str, provider: Any, kwargs: dict) -> str:
    # Exact prompt text and effective call configuration are part of the key.
    material = {
        "task": task,
        "policy_version": policy.version,
        "prompt": prompt,
        "provider": provider.provider_name,
        "model": provider.model_name,
        "model_revision": os.getenv("LLM_CACHE_MODEL_REVISION", ""),
        "endpoint": getattr(provider, "base_url", None),
        "parameters": kwargs,
        "epoch": os.getenv("LLM_CACHE_EPOCH", "1"),
    }
    payload = json.dumps(material, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _connect() -> sqlite3.Connection:
    path = cache_path()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not path.exists():
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        except FileExistsError:
            pass
        else:
            os.close(descriptor)
    connection = sqlite3.connect(path, timeout=3)
    connection.execute("PRAGMA busy_timeout=3000")
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("""
        CREATE TABLE IF NOT EXISTS responses (
            key TEXT PRIMARY KEY,
            task TEXT NOT NULL,
            state TEXT NOT NULL CHECK(state IN ('candidate','verified','quarantined')),
            response TEXT,
            semantic TEXT,
            created_at REAL NOT NULL,
            expires_at REAL NOT NULL
        )
    """)
    connection.execute("CREATE INDEX IF NOT EXISTS responses_task ON responses(task)")
    connection.execute("CREATE INDEX IF NOT EXISTS responses_expiry ON responses(expires_at)")
    return connection


def read(key: str) -> str | None:
    path = cache_path()
    if not path.exists():
        return None
    # A read-only connection must not initialize or modify the database.
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=3)) as db:
        row = db.execute("SELECT state, response, expires_at FROM responses WHERE key=?", (key,)).fetchone()
    if row and row[0] == "verified" and row[2] > time.time():
        return row[1]
    return None


def observe(key: str, task: str, response: str, semantic: str, policy: TaskPolicy) -> str:
    """Record one independent live answer; return resulting entry state."""
    now = time.time()
    with closing(_connect()) as db, db:
        db.execute("BEGIN IMMEDIATE")
        db.execute("DELETE FROM responses WHERE expires_at <= ?", (now,))
        row = db.execute(
            "SELECT state, semantic, expires_at FROM responses WHERE key=?", (key,)
        ).fetchone()
        if row and row[2] > now:
            if row[0] == "quarantined":
                return "quarantined"
            if row[0] == "verified":
                if row[1] != semantic:
                    db.execute(
                        "UPDATE responses SET state='quarantined', response=NULL, expires_at=? WHERE key=?",
                        (now + policy.ttl_seconds, key),
                    )
                    return "quarantined"
                return "verified"
            if row[1] == semantic:
                db.execute(
                    "UPDATE responses SET state='verified', response=?, created_at=?, expires_at=? WHERE key=?",
                    (response, now, now + policy.ttl_seconds, key),
                )
                return "verified"
            db.execute(
                "UPDATE responses SET state='quarantined', response=NULL, expires_at=? WHERE key=?",
                (now + policy.ttl_seconds, key),
            )
            return "quarantined"
        db.execute(
            "INSERT OR REPLACE INTO responses(key,task,state,response,semantic,created_at,expires_at) "
            "VALUES(?,?,'candidate',?,?,?,?)",
            (key, task, response, semantic, now, now + min(policy.ttl_seconds, 24 * 60 * 60)),
        )
        return "candidate"


def quarantine(key: str, policy: TaskPolicy) -> None:
    with closing(_connect()) as db, db:
        db.execute(
            "UPDATE responses SET state='quarantined', response=NULL, expires_at=? WHERE key=?",
            (time.time() + policy.ttl_seconds, key),
        )


def invalidate(*, key: str | None = None, task: str | None = None) -> int:
    """Manually remove one key, one task, or all entries."""
    with closing(_connect()) as db, db:
        if key is not None:
            cursor = db.execute("DELETE FROM responses WHERE key=?", (key,))
        elif task is not None:
            cursor = db.execute("DELETE FROM responses WHERE task=?", (task,))
        else:
            cursor = db.execute("DELETE FROM responses")
        return cursor.rowcount


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Invalidate cached LLM responses")
    parser.add_argument("invalidate", choices=["invalidate"])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--key")
    group.add_argument("--task")
    group.add_argument("--all", action="store_true")
    args = parser.parse_args()
    print(f"Removed {invalidate(key=args.key, task=args.task)} cache entries")
