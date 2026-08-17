"""Shared generation, repair, verdict, and provenance helpers."""

from __future__ import annotations

import json
from typing import Any, Sequence

from .errors import GenerationError
from .validation import parse_field_verdict


VALIDATOR_FORMAT_ATTEMPTS = 3


def emit_client_trace(client: Any, event: dict[str, Any]) -> None:
    tracer = getattr(client, "tracer", None)
    if tracer is not None:
        tracer.emit(event)


def repair_messages(
    base_messages: Sequence[dict[str, str]],
    raw: str,
    errors: Sequence[str],
) -> list[dict[str, str]]:
    return [
        *base_messages,
        {"role": "assistant", "content": raw},
        {
            "role": "user",
            "content": (
                "الجواب السابق غير صالح للأسباب الآتية:\n- "
                + "\n- ".join(errors)
                + "\nأعد كائن JSON مصححًا كاملًا فقط، مع الحفاظ على العقد المطلوب."
            ),
        },
    ]


def request_verdict(
    client: Any,
    messages: Sequence[dict[str, str]],
    *,
    max_tokens: int,
    seed: int,
    trace_context: dict[str, Any],
) -> tuple[str, dict[str, Any], int]:
    last_error: Exception | None = None
    raw = ""
    for attempt in range(1, VALIDATOR_FORMAT_ATTEMPTS + 1):
        retry_messages = list(messages)
        if attempt > 1:
            retry_messages.extend(
                [
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": (
                            "أعد كائن JSON واحدًا فقط بالمفتاحين passed وerrors، "
                            "بلا شرح ولا كائن ثانٍ."
                        ),
                    },
                ]
            )
        raw = client.chat(
            retry_messages,
            max_tokens=max_tokens,
            temperature=0.0,
            seed=seed + attempt - 1,
            trace_context={**trace_context, "validator_format_attempt": attempt},
        )
        try:
            return raw, parse_field_verdict(raw), attempt
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc
    raise GenerationError(f"validation response remained invalid: {last_error}")


def client_provenance(client: Any, sample_id: str) -> dict[str, Any]:
    if hasattr(client, "sample_stats"):
        return client.sample_stats(sample_id)
    return {
        "generation_endpoint_ids": ["legacy"],
        "network_attempts": 0,
        "endpoint_failover_count": 0,
        "truncated_completions": 0,
    }


def merge_provenance(
    current: dict[str, Any],
    prior_events: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    prior = [
        event.get("provenance", {})
        for event in prior_events
        if isinstance(event.get("provenance"), dict)
    ]
    endpoints = set(current.get("generation_endpoint_ids", []))
    for item in prior:
        endpoints.update(item.get("generation_endpoint_ids", []))
    return {
        "generation_endpoint_ids": sorted(endpoints),
        "network_attempts": int(current.get("network_attempts", 0))
        + max((int(item.get("network_attempts", 0)) for item in prior), default=0),
        "endpoint_failover_count": int(current.get("endpoint_failover_count", 0))
        + max(
            (int(item.get("endpoint_failover_count", 0)) for item in prior),
            default=0,
        ),
        "truncated_completions": int(current.get("truncated_completions", 0))
        + max(
            (int(item.get("truncated_completions", 0)) for item in prior),
            default=0,
        ),
    }
