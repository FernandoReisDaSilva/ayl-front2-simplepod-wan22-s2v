from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import time


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def format_hhmmss(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_mmss(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    minutes = total // 60
    secs = total % 60
    return f"{minutes:02d}:{secs:02d}"


class PhaseTimer:
    def __init__(self, *, emit: bool = True) -> None:
        self.emit = emit
        self.started_monotonic = time.monotonic()
        self.phases: list[dict] = []

    def stamp(self) -> str:
        return format_mmss(time.monotonic() - self.started_monotonic)

    def log(self, message: str) -> None:
        if self.emit:
            print(f"[{self.stamp()}] {message}", flush=True)

    @contextmanager
    def phase(self, phase_name: str):
        started_at = now_iso()
        started_monotonic = time.monotonic()
        self.log(f"START phase={phase_name}")
        record = {
            "phase_name": phase_name,
            "started_at": started_at,
            "ended_at": "",
            "elapsed_seconds": None,
            "elapsed_hhmmss": "",
            "elapsed_mmss": "",
        }
        try:
            yield record
        finally:
            elapsed = time.monotonic() - started_monotonic
            record["ended_at"] = now_iso()
            record["elapsed_seconds"] = round(elapsed, 3)
            record["elapsed_hhmmss"] = format_hhmmss(elapsed)
            record["elapsed_mmss"] = format_mmss(elapsed)
            self.phases.append(record)
            self.log(f"DONE phase={phase_name} elapsed={record['elapsed_mmss']}")
