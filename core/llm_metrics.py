"""In-process cache counters, emitted as parseable log events per LLM request.

Every completed request logs its outcome. Daily counters restart with the
process, while run_id lets logs from different restarts be distinguished and
the individual events be aggregated across processes later.
"""

import logging
import os
import threading
import uuid
from dataclasses import dataclass
from datetime import date
from typing import Callable


logger = logging.getLogger("llm_cache")
OUTCOMES = frozenset({"hit", "miss", "audit", "bypass", "provider_error"})


@dataclass
class Counts:
    requests: int = 0
    eligible: int = 0
    hits: int = 0
    audits: int = 0
    bypasses: int = 0
    provider_errors: int = 0

    def add(self, outcome: str, eligible: bool) -> None:
        self.requests += 1
        self.eligible += int(eligible)
        self.hits += int(outcome == "hit")
        self.audits += int(outcome == "audit")
        self.bypasses += int(outcome == "bypass")
        self.provider_errors += int(outcome == "provider_error")

    @property
    def hit_rate(self) -> float:
        """Percentage of cache-eligible calls actually served from cache."""
        return 100 * self.hits / self.eligible if self.eligible else 0.0

    @property
    def avoidance_rate(self) -> float:
        """Percentage of all calls that avoided the model."""
        return 100 * self.hits / self.requests if self.requests else 0.0


class CacheMetrics:
    def __init__(self, *, today: Callable[[], date] = date.today):
        self.run_id = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self._today = today
        self._lock = threading.Lock()
        self._day: date | None = None
        self._global = Counts()
        self._tasks: dict[str, Counts] = {}

    def record(self, task: str, outcome: str, *, eligible: bool) -> None:
        if outcome not in OUTCOMES:
            raise ValueError(f"Unknown cache outcome: {outcome}")
        with self._lock:
            day = self._today()
            if day != self._day:
                self._day = day
                self._global = Counts()
                self._tasks = {}
            task_counts = self._tasks.setdefault(task, Counts())
            task_counts.add(outcome, eligible)
            self._global.add(outcome, eligible)
            all_counts = self._global
            # One complete, machine-readable line per request also preserves
            # history if the process exits before a scheduled summary could run.
            try:
                logger.info(
                    "LLM_CACHE_STATS run=%s day=%s task=%s outcome=%s "
                    "task_requests=%d task_eligible=%d task_hits=%d task_hit_rate=%.2f%% "
                    "global_requests=%d global_eligible=%d global_hits=%d "
                    "global_hit_rate=%.2f%% global_avoidance_rate=%.2f%% "
                    "global_audits=%d global_bypasses=%d global_provider_errors=%d",
                    self.run_id, day.isoformat(), task, outcome,
                    task_counts.requests, task_counts.eligible, task_counts.hits,
                    task_counts.hit_rate, all_counts.requests, all_counts.eligible,
                    all_counts.hits, all_counts.hit_rate, all_counts.avoidance_rate,
                    all_counts.audits, all_counts.bypasses, all_counts.provider_errors,
                )
            except Exception:
                # Observability must never replace a valid model/cache answer.
                pass


cache_metrics = CacheMetrics()
