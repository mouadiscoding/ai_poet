"""Parse and deterministically validate untrusted generation results."""

from __future__ import annotations

import json
import re
from typing import Any, Sequence


CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)
ARABIC_MARK_RE = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed\u0640]")

REQUIRED_INSTRUCTION_HEADINGS = (
    "الموضوع العام:",
    "الجو العاطفي المطلوب:",
    "ألفاظ وصور يُستحسن استعمالها أو الدوران حولها:",
    "القافية:",
    "شرح البحر المطلوب:",
    "الصورة الصوتية التقريبية:",
    "خطة عملية لصناعة",
)

VERSE_REASONING_FIELDS = (
    "verse_index",
    "intended_meaning",
    "connection_to_previous",
    "imagery_and_diction",
    "first_draft",
    "problem_with_first_draft",
    "revised_draft",
    "first_hemistich_scansion",
    "second_hemistich_scansion",
    "rhyme_check",
)

REASONING_META_PHRASES = (
    "instruction",
    "reasoning",
    "التعليمات",
    "عدد المحارف",
    "الحد الأدنى",
    "الحد الأدنى للمحارف",
    "النص المرجعي",
    "القصيدة المرجعية",
    "تحليل النص",
    "وضعت في الموضوع العام",
    "وضعتُ في الموضوع العام",
    "هذا الحقل",
)

DETAILED_REASONING_FIELDS = (
    "intended_meaning",
    "connection_to_previous",
    "imagery_and_diction",
    "problem_with_first_draft",
    "first_hemistich_scansion",
    "second_hemistich_scansion",
    "rhyme_check",
)
MIN_DETAIL_CHARS = 20

RETROSPECTIVE_PRE_DRAFT_PHRASES = (
    "استخدمت",
    "استعملت",
    "وظفت",
    "اخترت",
    "جعلت",
    "اثرت",
    "اعتمدت",
    "ابقيت",
    "انتقيت",
    "استعرت",
    "ربطت",
    "كررت",
    "تعمدت",
    "وازنت",
    "قابلت",
)


def _normalize_arabic_for_contract(value: str) -> str:
    """Normalize marks and alif variants for narrow wording checks."""
    unmarked = ARABIC_MARK_RE.sub("", value)
    return unmarked.translate(str.maketrans("أإآٱ", "اااا"))


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


def extract_instruction(value: dict[str, Any]) -> str:
    """Extract the sole instruction string from an instruction response."""
    if set(value) != {"instruction"} or not isinstance(value["instruction"], str):
        raise ValueError("instruction JSON must contain only a string instruction")
    return value["instruction"].strip()


def instruction_contract_errors(
    instruction: str,
    *,
    meter_name: str,
    couplet_count: int,
    minimum_chars: int,
    source_hemistichs: Sequence[str],
) -> list[str]:
    """Return deterministic instruction-contract violations."""
    errors: list[str] = []
    if len(instruction) < minimum_chars:
        errors.append(
            f"instruction has {len(instruction)} characters; minimum is {minimum_chars}"
        )
    positions = [instruction.find(heading) for heading in REQUIRED_INSTRUCTION_HEADINGS]
    missing_headings = [
        heading
        for heading, position in zip(
            REQUIRED_INSTRUCTION_HEADINGS, positions, strict=True
        )
        if position < 0
    ]
    if missing_headings:
        label = "heading" if len(missing_headings) == 1 else "headings"
        errors.append(
            f"instruction is missing required {label}: "
            + ", ".join(missing_headings)
        )
    if positions[0] > 0:
        errors.append(
            "instruction must start exactly with heading: الموضوع العام:"
        )
    if not missing_headings and positions != sorted(positions):
        errors.append("instruction headings are out of order")
    if str(couplet_count) not in instruction:
        errors.append(f"instruction does not contain numeric couplet count {couplet_count}")
    if meter_name not in instruction:
        errors.append(f"instruction does not name meter {meter_name}")
    if any(
        len(hemistich.strip()) >= 12 and hemistich.strip() in instruction
        for hemistich in source_hemistichs
    ):
        errors.append("instruction copies a complete source hemistich")
    return errors


def extract_reasoning_chunk(
    value: dict[str, Any],
    *,
    expected_indices: Sequence[int],
    expected_couplets: Sequence[str],
    include_overview: bool,
) -> tuple[str | None, list[dict[str, Any]]]:
    """Extract and validate one structured verse-reasoning chunk.

    The validation is deliberately structural and source-linked. Semantic and
    prosodic quality remain the responsibility of the separate Gemma judge.
    """
    expected_top_level = (
        {"overview", "verse_reasoning"} if include_overview else {"verse_reasoning"}
    )
    if set(value) != expected_top_level:
        raise ValueError(
            "reasoning JSON has incorrect keys; expected "
            + ", ".join(sorted(expected_top_level))
        )

    overview: str | None = None
    if include_overview:
        overview_value = value["overview"]
        if not isinstance(overview_value, str) or not overview_value.strip():
            raise ValueError("reasoning overview must be a non-empty string")
        overview = overview_value.strip()

    blocks = value["verse_reasoning"]
    if not isinstance(blocks, list):
        raise ValueError("verse_reasoning must be a list")
    if len(blocks) != len(expected_indices):
        raise ValueError(
            f"verse_reasoning has {len(blocks)} blocks; expected {len(expected_indices)}"
        )

    cleaned_blocks: list[dict[str, Any]] = []
    for position, (block, expected_index, expected_couplet) in enumerate(
        zip(blocks, expected_indices, expected_couplets, strict=True), start=1
    ):
        if not isinstance(block, dict) or set(block) != set(VERSE_REASONING_FIELDS):
            raise ValueError(
                f"reasoning block {position} must contain exactly the required fields"
            )
        if block["verse_index"] != expected_index:
            raise ValueError(
                f"reasoning block {position} has verse_index {block['verse_index']}; "
                f"expected {expected_index}"
            )

        cleaned: dict[str, Any] = {"verse_index": expected_index}
        for field in VERSE_REASONING_FIELDS[1:]:
            field_value = block[field]
            if not isinstance(field_value, str) or not field_value.strip():
                raise ValueError(
                    f"reasoning block {expected_index} field {field} must be non-empty"
                )
            cleaned[field] = field_value.strip()

        if cleaned["revised_draft"] != expected_couplet:
            raise ValueError(
                f"reasoning block {expected_index} revised_draft must exactly match "
                "its source couplet"
            )
        if cleaned["first_draft"] == cleaned["revised_draft"]:
            raise ValueError(
                f"reasoning block {expected_index} first_draft must differ from revised_draft"
            )
        if cleaned["first_draft"].count("=") != 1:
            raise ValueError(
                f"reasoning block {expected_index} first_draft must contain two "
                "hemistichs separated by exactly one ="
            )
        short_fields = [
            field
            for field in DETAILED_REASONING_FIELDS
            if len(cleaned[field]) < MIN_DETAIL_CHARS
        ]
        if short_fields:
            raise ValueError(
                f"reasoning block {expected_index} is too terse in fields: "
                + ", ".join(short_fields)
            )
        normalized_imagery = _normalize_arabic_for_contract(
            cleaned["imagery_and_diction"]
        )
        retrospective_phrases = [
            phrase
            for phrase in RETROSPECTIVE_PRE_DRAFT_PHRASES
            if phrase in normalized_imagery
        ]
        if retrospective_phrases:
            raise ValueError(
                f"reasoning block {expected_index} imagery_and_diction must "
                "describe a plan before first_draft; found retrospective wording: "
                + ", ".join(retrospective_phrases)
            )
        searchable = "\n".join(
            cleaned[field] for field in DETAILED_REASONING_FIELDS
        ).casefold()
        found_meta = [phrase for phrase in REASONING_META_PHRASES if phrase.casefold() in searchable]
        if found_meta:
            raise ValueError(
                f"reasoning block {expected_index} contains generation metatext: "
                + ", ".join(found_meta)
            )
        cleaned_blocks.append(cleaned)

    if overview is not None:
        searchable_overview = overview.casefold()
        found_meta = [
            phrase
            for phrase in REASONING_META_PHRASES
            if phrase.casefold() in searchable_overview
        ]
        if found_meta:
            raise ValueError(
                "reasoning overview contains generation metatext: "
                + ", ".join(found_meta)
            )
    return overview, cleaned_blocks


def parse_field_verdict(raw: str) -> dict[str, Any]:
    """Parse a strict verdict containing only ``passed`` and ``errors``."""
    value = extract_json_object(raw)
    if set(value) != {"passed", "errors"}:
        raise ValueError("validation JSON must contain only passed and errors")
    passed = value["passed"]
    errors = value["errors"]
    if not isinstance(passed, bool):
        raise ValueError("validation passed must be a boolean")
    if not isinstance(errors, list) or not all(
        isinstance(error, str) and error.strip() for error in errors
    ):
        raise ValueError("validation errors must be a list of non-empty strings")
    if passed == bool(errors):
        raise ValueError("validation passed/errors values are inconsistent")
    return {"passed": passed, "errors": errors}
