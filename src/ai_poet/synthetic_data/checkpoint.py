"""Append and replay durable per-sample generation checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_checkpoint(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Replay an append-only generation checkpoint into current sample state.

    Each non-blank JSONL event is processed in file order, so later events for
    a sample replace earlier ones. A success stores its record and clears any
    prior failure for that sample. A failure stores its message but deliberately
    does not remove an already recorded success; callers ultimately give the
    success mapping precedence when determining completed work.

    Args:
        path: Checkpoint JSONL path. A missing file is treated as an empty
            checkpoint.

    Returns:
        A pair ``(successes, failures)`` keyed by sample ID. Successful values
        are full SFT records; failure values are error strings.

    Raises:
        ValueError: If any non-blank line contains malformed JSON.
        KeyError: If a decoded event lacks required ``sample_id``, ``status``,
            or successful ``record`` fields.
        OSError: If an existing checkpoint cannot be read.
    """
    successes: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    if not path.exists():
        return successes, failures
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid checkpoint JSON on line {line_number}"
                ) from exc
            sample_id = event["sample_id"]
            if event["status"] == "success":
                successes[sample_id] = event["record"]
                failures.pop(sample_id, None)
            else:
                failures[sample_id] = event.get("error", "unknown error")
    return successes, failures


def append_checkpoint(path: Path, event: dict[str, Any]) -> None:
    """Append one durable JSON event to the generation checkpoint.

    Parent directories are created as needed. The event is encoded as a single
    UTF-8 JSON line with Arabic characters preserved, then the Python stream is
    flushed so progress is visible to subsequent resume attempts even while a
    larger corpus run is still active.

    Args:
        path: Destination JSONL checkpoint path.
        event: JSON-serializable success or failure event.

    Raises:
        OSError: If directories or the checkpoint file cannot be written.
        TypeError: If ``event`` contains values unsupported by ``json.dumps``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        handle.flush()
