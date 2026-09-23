import logging
import os
from datetime import date, datetime, timedelta

import pytest

from core.crate_utils import _parse_crate_name, parse_crates_io_reference
from core.logging_utils import WeeklyRotatingFileHandler
from scripts.prune_legacy_logs import prune_directory


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://crates.io/crates/serde", ("serde", None)),
        ("https://crates.io/crates/serde/1.0.197", ("serde", "1.0.197")),
        (
            "https://crates.io/api/v1/crates/auto-future/1.0.0/download?source=api",
            ("auto-future", "1.0.0"),
        ),
        ("https://evil.example/crates/serde", None),
    ],
)
def test_crate_reference(url, expected):
    assert parse_crates_io_reference(url) == expected
    if expected:
        assert _parse_crate_name(url) == expected[0]


@pytest.mark.asyncio
async def test_api_crate_download_uses_crate_processor(monkeypatch):
    import pandas as pd
    import api

    calls = []

    async def fake_crate(url, version):
        calls.append((url, version))
        return {"status": "success", "license_type": "MIT"}

    async def unexpected_github(*args, **kwargs):
        raise AssertionError("crate URL must not enter GitHub discovery")

    monkeypatch.setattr(api, "process_crate_repository", fake_crate)
    monkeypatch.setattr(api, "process_github_repository", unexpected_github)
    url = "https://crates.io/api/v1/crates/auto-future/1.0.0/download"
    results = await api._process_repositories(
        api=None, df=pd.DataFrame([{"github_url": url, "version": None, "name": None}])
    )
    assert calls == [(url, None)]
    assert results[0]["status"] == "success"
    assert results[0]["input_url"] == url


def test_rotation_by_size_and_date_and_pruning(tmp_path):
    path = tmp_path / "service.log"
    handler = WeeklyRotatingFileHandler(path, max_bytes=20, retention_days=7)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger("test_rotation_by_size_and_date_and_pruning")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        logger.info("first line is long")
        logger.info("second line is long")
        archives = list(tmp_path.glob("service.log.*"))
        assert len(archives) == 1
        assert archives[0].read_text() == "first line is long\n"

        handler._active_date = date.today() - timedelta(days=1)
        logger.info("next day")
        assert len(list(tmp_path.glob("service.log.*"))) == 2

        old = tmp_path / f"service.log.{date.today() - timedelta(days=8)}.001"
        old.write_text("old")
        old_time = (datetime.now() - timedelta(days=8)).timestamp()
        os.utime(old, (old_time, old_time))
        handler.prune_archives()
        assert not old.exists()
    finally:
        logger.removeHandler(handler)
        handler.close()


def test_legacy_pruning_preserves_multiline_recent_records(tmp_path):
    log = tmp_path / "service.log"
    log.write_bytes(
        b"2026-09-01 10:00:00,000 - INFO - old\nold continuation\n"
        b"2026-09-20 10:00:00,000 - INFO - recent\nrecent continuation\n"
    )
    cutoff = datetime(2026, 9, 16)
    kept, removed = prune_directory(tmp_path, cutoff)
    assert kept > 0 and removed > 0
    assert b"old" in log.read_bytes()  # dry run did not rewrite
    prune_directory(tmp_path, cutoff, apply=True)
    assert log.read_bytes() == (
        b"2026-09-20 10:00:00,000 - INFO - recent\nrecent continuation\n"
    )
