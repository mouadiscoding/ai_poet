"""Parse untrusted generation results and Gemma validation verdicts."""

from __future__ import annotations

import json
import re
from typing import Any


CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def extract_json_object(raw: str) -> dict[str, Any]:
    """Extract a top-level JSON object from an untrusted model response."""
    cleaned = CODE_FENCE_RE.sub("", raw.strip()).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("response does not contain a JSON object")
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("response JSON must be an object")
    return value


def extract_generated_pair(value: dict[str, Any]) -> tuple[str, str]:
    """Extract the two strings required to assemble an SFT record.

    This is intentionally limited to a serialization invariant. Gemma, not
    Python keyword or content rules, decides whether either field is suitable.
    Extra keys remain in ``value`` so the Gemma validator can reject them.
    """
    instruction = value.get("instruction")
    reasoning = value.get("reasoning")
    if not isinstance(instruction, str) or not isinstance(reasoning, str):
        raise ValueError("instruction and reasoning must be strings")
    return instruction, reasoning


def parse_validation_verdict(raw: str) -> dict[str, dict[str, Any]]:
    """Parse Gemma's strict field-level validation verdict.

    Both field objects must contain exactly ``passed`` and ``errors``. A passed
    field must have no errors, while a rejected field must explain at least one
    error. Any malformed or internally inconsistent verdict is rejected rather
    than being interpreted as approval.
    """
    value = extract_json_object(raw)
    if set(value) != {"instruction", "reasoning"}:
        raise ValueError(
            "validation JSON must contain only instruction and reasoning"
        )
    verdict: dict[str, dict[str, Any]] = {}
    for field in ("instruction", "reasoning"):
        field_value = value[field]
        if not isinstance(field_value, dict) or set(field_value) != {
            "passed",
            "errors",
        }:
            raise ValueError(
                f"validation {field} must contain only passed and errors"
            )
        passed = field_value["passed"]
        errors = field_value["errors"]
        if not isinstance(passed, bool):
            raise ValueError(f"validation {field}.passed must be a boolean")
        if not isinstance(errors, list) or not all(
            isinstance(error, str) and error.strip() for error in errors
        ):
            raise ValueError(
                f"validation {field}.errors must be a list of non-empty strings"
            )
        if passed == bool(errors):
            raise ValueError(
                f"validation {field} passed/errors values are inconsistent"
            )
        verdict[field] = {"passed": passed, "errors": errors}
    return verdict


def verdict_errors(verdict: dict[str, dict[str, Any]]) -> list[str]:
    """Flatten rejected field feedback for the generation repair prompt."""
    return [
        f"Gemma rejected {field}: {error}"
        for field in ("instruction", "reasoning")
        for error in verdict[field]["errors"]
    ]
