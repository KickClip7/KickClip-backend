from __future__ import annotations

import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .artifacts import write_json_atomic


@dataclass
class CandidatePreparationWorkMetrics:
    """Candidate preparation 작업량과 단계별 시간을 기록한다."""

    schema_version: str = "kickclip.candidate_preparation_work_metrics.r1"

    counters: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    durations_seconds: dict[str, float] = field(
        default_factory=lambda: defaultdict(float)
    )

    _lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
    )
    _started_at_monotonic: float = field(
        default_factory=time.perf_counter,
        init=False,
        repr=False,
    )

    def increment(self, name: str, amount: int = 1) -> None:
        if amount == 0:
            return

        with self._lock:
            self.counters[name] = (
                int(self.counters.get(name, 0)) + int(amount)
            )

    def add_duration(self, name: str, seconds: float) -> None:
        if seconds <= 0:
            return

        with self._lock:
            self.durations_seconds[name] = round(
                float(self.durations_seconds.get(name, 0.0))
                + float(seconds),
                6,
            )

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        started = time.perf_counter()

        try:
            yield
        finally:
            self.add_duration(
                name,
                time.perf_counter() - started,
            )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            counters = {
                key: int(value)
                for key, value in sorted(self.counters.items())
            }
            durations = {
                key: round(float(value), 6)
                for key, value in sorted(
                    self.durations_seconds.items()
                )
            }

        return {
            "schema_version": self.schema_version,
            "elapsed_seconds": round(
                time.perf_counter()
                - self._started_at_monotonic,
                6,
            ),
            "counters": counters,
            "durations_seconds": durations,
        }

    def write(self, path: Path) -> str:
        return write_json_atomic(path, self.snapshot())