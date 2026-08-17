"""Append and replay durable sample and accepted-stage checkpoints."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import threading
from typing import Any


CHECKPOINT_VERSION = 3


@dataclass
class CheckpointState:
    successes: dict[str, dict[str, Any]] = field(default_factory=dict)
    success_fingerprints: dict[str, str] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)
    instructions: dict[str, dict[str, Any]] = field(default_factory=dict)
    reasoning_chunks: dict[str, dict[int, dict[str, Any]]] = field(
        default_factory=dict
    )
    stages: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)


def load_checkpoint_state(path: Path) -> CheckpointState:
    """Replay legacy and version-two checkpoint events in append order."""
    state = CheckpointState()
    if not path.exists():
        return state
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid checkpoint JSON on line {line_number}"
                ) from exc
            sample_id = record["sample_id"]
            event = record.get("event")
            if event is None:
                if record.get("status") == "success":
                    event = "sample_success"
                else:
                    event = "sample_failure"

            if event == "sample_success":
                state.successes[sample_id] = record["record"]
                if record.get("generation_fingerprint"):
                    state.success_fingerprints[sample_id] = str(
                        record["generation_fingerprint"]
                    )
                else:
                    state.success_fingerprints.pop(sample_id, None)
                state.failures.pop(sample_id, None)
            elif event == "sample_failure":
                if sample_id not in state.successes:
                    state.failures[sample_id] = record.get("error", "unknown error")
            elif event == "stage_success":
                stage_name = str(record.get("stage_name", ""))
                stage_key = str(record.get("stage_key", stage_name))
                payload = record.get("payload")
                if not stage_name or not isinstance(payload, dict):
                    raise ValueError(
                        f"Invalid stage checkpoint on line {line_number}"
                    )
                flattened = {
                    **record,
                    **payload,
                    "generation_fingerprint": record.get(
                        "workflow_fingerprint",
                        record.get("generation_fingerprint"),
                    ),
                }
                state.stages.setdefault(sample_id, {})[stage_key] = flattened
                if (
                    record.get("task_type", "poem-generation")
                    == "poem-generation"
                    and stage_name == "instruction"
                ):
                    state.instructions[sample_id] = flattened
                    previous = state.reasoning_chunks.get(sample_id, {})
                    state.reasoning_chunks[sample_id] = {
                        start: chunk
                        for start, chunk in previous.items()
                        if chunk.get("instruction_fingerprint")
                        == flattened.get("instruction_fingerprint")
                    }
                elif (
                    record.get("task_type", "poem-generation")
                    == "poem-generation"
                    and stage_name == "reasoning_chunk"
                ):
                    start_offset = int(flattened["start_offset"])
                    state.reasoning_chunks.setdefault(sample_id, {})[
                        start_offset
                    ] = flattened
            elif event == "instruction_success":
                state.instructions[sample_id] = record
                state.stages.setdefault(sample_id, {})["instruction"] = record
                previous = state.reasoning_chunks.get(sample_id, {})
                state.reasoning_chunks[sample_id] = {
                    start: chunk
                    for start, chunk in previous.items()
                    if chunk.get("instruction_fingerprint")
                    == record.get("instruction_fingerprint")
                }
            elif event == "reasoning_chunk_success":
                start_offset = int(record["start_offset"])
                state.reasoning_chunks.setdefault(sample_id, {})[start_offset] = record
                state.stages.setdefault(sample_id, {})[
                    f"reasoning_chunk:{start_offset}"
                ] = record
    return state


def load_checkpoint(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Return the legacy two-map checkpoint view used by existing callers."""
    state = load_checkpoint_state(path)
    return state.successes, state.failures


def _serialize(event: dict[str, Any]) -> str:
    return json.dumps(event, ensure_ascii=False) + "\n"


def append_checkpoint(path: Path, event: dict[str, Any]) -> None:
    """Append and flush one event; retained for backward compatibility."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(_serialize(event))
        handle.flush()


class CheckpointWriter:
    """Serialize checkpoint writes from concurrent sample and chunk workers."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def append(self, event: dict[str, Any]) -> None:
        record = {"checkpoint_version": CHECKPOINT_VERSION, **event}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.path.open(
            "a", encoding="utf-8", newline="\n"
        ) as handle:
            handle.write(_serialize(record))
            handle.flush()
