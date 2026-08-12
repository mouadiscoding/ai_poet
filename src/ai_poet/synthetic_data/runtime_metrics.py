"""Lightweight periodic endpoint-pool telemetry for long corpus runs."""

from __future__ import annotations

import json
from pathlib import Path
import threading
import time
from typing import Any, Protocol


class SnapshotSource(Protocol):
    def snapshot(self) -> dict[str, Any]: ...


class GenerationProgress:
    """Combine endpoint telemetry with accepted-sample rate and ETA."""

    def __init__(
        self,
        endpoints: SnapshotSource,
        *,
        total_samples: int,
        initial_completed: int,
        initial_repaired: int,
        initial_response_characters: int,
    ) -> None:
        self.endpoints = endpoints
        self.total_samples = total_samples
        self.initial_completed = initial_completed
        self.initial_successes = initial_completed
        self.initial_response_characters = initial_response_characters
        self.completed = initial_completed
        self.successes = initial_completed
        self.failures = 0
        self.repaired = initial_repaired
        self.response_characters = initial_response_characters
        self.started = time.monotonic()
        self._lock = threading.Lock()

    def record_success(self, record: dict[str, Any]) -> None:
        with self._lock:
            self.completed += 1
            self.successes += 1
            self.repaired += int(
                record.get("validation_status") == "passed_after_repair"
            )
            self.response_characters += len(str(record.get("response", "")))

    def record_failure(self) -> None:
        with self._lock:
            self.completed += 1
            self.failures += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            elapsed = max(0.000001, time.monotonic() - self.started)
            newly_completed = self.completed - self.initial_completed
            new_successes = self.successes - self.initial_successes
            new_response_characters = (
                self.response_characters - self.initial_response_characters
            )
            rate = newly_completed / elapsed
            remaining = max(0, self.total_samples - self.completed)
            progress = {
                "progress": {
                    "total_samples": self.total_samples,
                    "completed_samples": self.completed,
                    "successful_samples": self.successes,
                    "failed_samples": self.failures,
                    "repaired_samples": self.repaired,
                    "repair_rate": (
                        self.repaired / self.successes if self.successes else 0.0
                    ),
                    "validated_samples_per_hour": new_successes / elapsed * 3600,
                    "accepted_response_characters_per_hour": (
                        new_response_characters / elapsed * 3600
                    ),
                    "eta_seconds": remaining / rate if rate > 0 else None,
                }
            }
        return {**self.endpoints.snapshot(), **progress}


class RuntimeMetricsWriter:
    """Append pool snapshots without recording prompts or responses."""

    def __init__(
        self,
        path: Path,
        source: SnapshotSource,
        *,
        interval_seconds: float = 60.0,
    ) -> None:
        self.path = path
        self.source = source
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="gemma-runtime-metrics",
            daemon=True,
        )

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(1.0, self.interval_seconds + 1.0))
        self._write_snapshot(final=True)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._write_snapshot(final=False)

    def _write_snapshot(self, *, final: bool) -> None:
        record = {"event": "runtime_metrics", "final": final, **self.source.snapshot()}
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
