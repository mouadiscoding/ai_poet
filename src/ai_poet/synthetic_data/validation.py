"""Parse and validate untrusted model generation results."""

from __future__ import annotations

import json
import re
from typing import Any

from .meters import METER_NAMES
from .poems import PoemRecord


ARABIC_RE = re.compile(r"[\u0600-\u06ff]")
LATIN_RE = re.compile(r"[A-Za-z]")
DIACRITICS_RE = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06edـ]")
CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def extract_json_object(raw: str) -> dict[str, Any]:
    """Extract a top-level JSON object from a model response.

    Leading or trailing Markdown JSON fences are stripped first. The cleaned
    response is parsed as-is; if that fails, the substring spanning its first
    opening brace through its last closing brace is parsed to tolerate prose or
    other wrapper text around the object. Arrays and scalar JSON values are
    rejected even when syntactically valid.

    Args:
        raw: Untrusted text returned by the generation endpoint.

    Returns:
        The decoded JSON object as a dictionary.

    Raises:
        ValueError: If no plausible object is present or the decoded top-level
            value is not an object.
        json.JSONDecodeError: If the selected object substring is invalid JSON.
    """
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


def _normalise_arabic(text: str) -> str:
    """Reduce Arabic text to comparable base letters.

    Arabic combining marks, Quranic annotation marks, and tatweel are removed,
    followed by every character outside the Arabic Unicode block. The result is
    intentionally aggressive: spaces and punctuation disappear so copied
    hemistichs can be detected despite superficial formatting or diacritics.

    Args:
        text: Arbitrary text to normalize for containment checks.

    Returns:
        A compact string containing only normalized Arabic-block characters.
    """
    text = DIACRITICS_RE.sub("", text)
    return re.sub(r"[^\u0600-\u06ff]", "", text)


def validate_generation(
    value: dict[str, Any], poem: PoemRecord, min_chars: int
) -> list[str]:
    """Validate a generated instruction/reasoning object for SFT use.

    Validation covers the exact two-key schema, string types, minimum lengths,
    dominance of Arabic over Latin characters, explicit numeric couplet count,
    and correct meter or prose terminology. It rejects mention of conflicting
    meters and instructions that reproduce a complete sufficiently long source
    hemistich after Arabic normalization. The reasoning must discuss semantics,
    imagery or rhetoric, and revision; metered poems must also discuss prosody,
    and every response must contain the final-result transition.

    The function accumulates independent failures instead of raising, allowing
    the caller to give the model a complete repair prompt. A schema type error
    returns early because content checks require string fields.

    Args:
        value: Decoded model JSON to validate.
        poem: Source poem providing the expected meter, count, and verse text.
        min_chars: Minimum stripped length required independently for the
            ``instruction`` and ``reasoning`` fields.

    Returns:
        Human-readable validation errors. An empty list means the generation
        passed every rule.
    """
    errors: list[str] = []
    if set(value) != {"instruction", "reasoning"}:
        errors.append("JSON must contain only instruction and reasoning")
    instruction = value.get("instruction")
    reasoning = value.get("reasoning")
    if not isinstance(instruction, str) or not isinstance(reasoning, str):
        return errors + ["instruction and reasoning must be strings"]
    if len(instruction.strip()) < min_chars:
        errors.append(f"instruction must contain at least {min_chars} characters")
    if len(reasoning.strip()) < min_chars:
        errors.append(f"reasoning must contain at least {min_chars} characters")

    combined = instruction + reasoning
    arabic = len(ARABIC_RE.findall(combined))
    latin = len(LATIN_RE.findall(combined))
    if arabic < 100 or arabic < 4 * latin:
        errors.append("content must be predominantly Arabic")
    if str(poem.couplet_count) not in instruction:
        errors.append("instruction must state the exact couplet count in digits")
    if poem.meter_name == "النثر":
        if "النثر" not in instruction and "منثور" not in instruction:
            errors.append("prose instruction must explicitly request prose poetry")
    elif poem.meter_name not in instruction:
        errors.append(f"instruction must name بحر {poem.meter_name}")

    for other_meter in METER_NAMES:
        if other_meter != poem.meter_name and f"بحر {other_meter}" in instruction:
            errors.append(f"instruction contradicts the source meter with {other_meter}")
            break

    normalized_instruction = _normalise_arabic(instruction)
    for verse in poem.verses:
        normalized_verse = _normalise_arabic(verse)
        if len(normalized_verse) >= 20 and normalized_verse in normalized_instruction:
            errors.append("instruction copies a complete source hemistich")
            break

    required_groups = (
        ("معنى", "المعاني", "دلالة"),
        ("صورة", "بلاغ", "استعار", "تشبيه"),
        ("صياغ", "تعديل", "تحرير", "محاولة"),
    )
    for group in required_groups:
        if not any(term in reasoning for term in group):
            errors.append(f"reasoning is missing required discussion: {group[0]}")
    if poem.meter_name != "النثر" and not any(
        term in reasoning for term in ("وزن", "عروض", "إيقاع", "بحر")
    ):
        errors.append("metered reasoning must discuss prosody")
    if "النتيجة النهائية" not in reasoning:
        errors.append("reasoning must end with the final-result transition")
    return errors
