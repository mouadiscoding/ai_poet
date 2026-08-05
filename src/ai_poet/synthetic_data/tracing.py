"""Secret-safe, synchronized generation audit tracing."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import threading
from typing import Any, Sequence
from uuid import uuid4


PRINT_LOCK = threading.Lock()


def _redact(value: Any, secrets: Sequence[str]) -> Any:
    """Recursively replace configured secrets in a JSON-compatible value."""
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[REDACTED]")
        return value
    if isinstance(value, dict):
        return {key: _redact(item, secrets) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(item, secrets) for item in value]
    return value


class GenerationTracer:
    """Write full, secret-scrubbed generation events to JSONL and stdout."""

    def __init__(self, path: Path, *, secrets: Sequence[str] = ()) -> None:
        self.path = path
        self.run_id = uuid4().hex
        self.secrets = tuple(secret for secret in secrets if secret)

    def emit(self, event: dict[str, Any]) -> None:
        record = _redact(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "run_id": self.run_id,
                **event,
            },
            self.secrets,
        )
        serialized = json.dumps(record, ensure_ascii=False)
        rendered = json.dumps(record, ensure_ascii=False, indent=2)
        label = str(record.get("event", "event"))
        sample_id = record.get("sample_id")
        if sample_id:
            label += f" sample={str(sample_id)[:12]}"
        with PRINT_LOCK:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(serialized + "\n")
            print(f"\n=== SFT TRACE {label} ===")
            print(rendered)
            print("=== END SFT TRACE ===", flush=True)
