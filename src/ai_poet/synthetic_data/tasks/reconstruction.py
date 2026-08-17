"""Localized poem-corruption and reconstruction SFT workflow."""

from __future__ import annotations

from difflib import SequenceMatcher
import hashlib
import json
from typing import Any, Sequence

from ..assignment import sft_split
from ..errors import GenerationError
from ..poems import PoemRecord
from ..validation import extract_json_object
from ..workflow import (
    client_provenance,
    emit_client_trace,
    repair_messages,
    request_verdict,
)
from .base import TASK_POEM_RECONSTRUCTION, TaskWorkflow


TASK_VERSION = 1
MAX_FRAGMENT_WORDS = 3

SYSTEM_PROMPT = """أنت تنشئ مثالًا لتدريب نموذج على إصلاح قصيدة عربية محرّفة. ستتلقى القصيدة الأصلية وعدد التحريفات المطلوب. اكتب نسخة محرّفة كاملة تحافظ حرفيًا على عدد الأبيات وترتيبها وعلامة = في كل بيت. غيّر في كل بيت مختار كلمة واحدة أو عبارة متصلة لا تتجاوز ثلاث كلمات، ولا تحذف أو تضف أو تنقل بيتًا. لا تنشئ القصيدة الأصلية كاملة في reasoning.

أعد كائن JSON فقط بالمفتاحين corrupted_poem وrepairs. repairs قائمة بعدد التحريفات، ولكل عنصر المفاتيح couplet_index وcorrupted_fragment وcorrected_fragment وdiagnosis وcontext_evidence وrepair_reason. يجوز ذكر اللفظ أو العبارة المصححة محليًا، لكن لا تنسخ البيت الأصلي كاملًا."""

VALIDATION_SYSTEM_PROMPT = """أنت مدقق مثال لإعادة بناء قصيدة عربية محرّفة. قارن الأصل بالنسخة المحرّفة وتحقق أن كل استبدال محلي أحدث خللًا دلاليًا أو لغويًا أو تصويريًا أو صوتيًا يمكن اكتشافه من السياق، وأن كل سجل إصلاح يحدد الخلل والشاهد والتصحيح وسبب استعادة النص على نحو مفصل. لا تصلح المرشح ولا تنشئ القصيدة الأصلية.

أعد JSON فقط بهذه البنية: {"passed":true,"errors":[]}."""


def corruption_count(poem: PoemRecord) -> int:
    upper = min(3, poem.couplet_count)
    seed = int(
        hashlib.sha256(
            f"reconstruction-count:{poem.sample_id}".encode()
        ).hexdigest()[:16],
        16,
    )
    return seed % upper + 1


def build_generation_messages(poem: PoemRecord, count: int) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"أنشئ {count} تحريفات موضعية في {count} أبيات مختلفة. "
                "يجب أن تبقى سائر الكلمات والأبيات كما هي حرفيًا.\n"
                f"<original_poem>\n{poem.poem_text}\n</original_poem>"
            ),
        },
    ]


def _contains_complete_poem(value: str, poem_text: str) -> bool:
    escaped = json.dumps(poem_text, ensure_ascii=False)[1:-1]
    return poem_text in value or escaped in value


def _single_replacement(
    original_line: str,
    corrupted_line: str,
) -> tuple[str, str]:
    original_tokens = original_line.split()
    corrupted_tokens = corrupted_line.split()
    matcher = SequenceMatcher(None, original_tokens, corrupted_tokens, autojunk=False)
    changes = [opcode for opcode in matcher.get_opcodes() if opcode[0] != "equal"]
    if len(changes) != 1 or changes[0][0] != "replace":
        raise ValueError("each changed couplet must contain one replacement only")
    _, original_start, original_end, corrupted_start, corrupted_end = changes[0]
    original_fragment = original_tokens[original_start:original_end]
    corrupted_fragment = corrupted_tokens[corrupted_start:corrupted_end]
    if not 1 <= len(original_fragment) <= MAX_FRAGMENT_WORDS:
        raise ValueError("corrected fragment must contain one to three words")
    if not 1 <= len(corrupted_fragment) <= MAX_FRAGMENT_WORDS:
        raise ValueError("corrupted fragment must contain one to three words")
    if "=" in original_fragment or "=" in corrupted_fragment:
        raise ValueError("a corruption may not alter the couplet separator")
    return " ".join(corrupted_fragment), " ".join(original_fragment)


def extract_candidate(
    value: dict[str, Any],
    *,
    poem: PoemRecord,
    expected_count: int,
) -> dict[str, Any]:
    if set(value) != {"corrupted_poem", "repairs"}:
        raise ValueError("reconstruction JSON must contain corrupted_poem and repairs")
    corrupted = value["corrupted_poem"]
    repairs = value["repairs"]
    if not isinstance(corrupted, str) or not corrupted.strip():
        raise ValueError("corrupted_poem must be a non-empty string")
    corrupted = corrupted.strip()
    if corrupted == poem.poem_text:
        raise ValueError("corrupted_poem must differ from the original poem")
    original_lines = poem.poem_text.splitlines()
    corrupted_lines = corrupted.splitlines()
    if len(corrupted_lines) != len(original_lines):
        raise ValueError("corrupted_poem must preserve the exact couplet count")
    if any(line.count("=") != 1 for line in corrupted_lines):
        raise ValueError("every corrupted couplet must preserve exactly one = separator")

    diffs: dict[int, tuple[str, str]] = {}
    for index, (original_line, corrupted_line) in enumerate(
        zip(original_lines, corrupted_lines, strict=True), start=1
    ):
        if original_line == corrupted_line:
            continue
        diffs[index] = _single_replacement(original_line, corrupted_line)
    if len(diffs) != expected_count:
        raise ValueError(
            f"corrupted_poem changes {len(diffs)} couplets; expected {expected_count}"
        )

    if not isinstance(repairs, list) or len(repairs) != expected_count:
        raise ValueError(f"repairs must contain exactly {expected_count} items")
    cleaned_repairs: list[dict[str, Any]] = []
    seen: set[int] = set()
    generated_reasoning: list[str] = []
    required_fields = {
        "couplet_index",
        "corrupted_fragment",
        "corrected_fragment",
        "diagnosis",
        "context_evidence",
        "repair_reason",
    }
    for repair in repairs:
        if not isinstance(repair, dict) or set(repair) != required_fields:
            raise ValueError("each repair must contain exactly the required fields")
        index = repair["couplet_index"]
        if not isinstance(index, int) or index not in diffs or index in seen:
            raise ValueError("repair couplet indices must match changed couplets once")
        seen.add(index)
        corrupted_fragment, corrected_fragment = diffs[index]
        if not isinstance(repair["corrupted_fragment"], str) or (
            " ".join(repair["corrupted_fragment"].split()) != corrupted_fragment
        ):
            raise ValueError("repair corrupted_fragment must match the actual diff")
        if not isinstance(repair["corrected_fragment"], str) or (
            " ".join(repair["corrected_fragment"].split()) != corrected_fragment
        ):
            raise ValueError("repair corrected_fragment must match the original diff")
        cleaned = {
            "couplet_index": index,
            "corrupted_fragment": corrupted_fragment,
            "corrected_fragment": corrected_fragment,
        }
        for field in ("diagnosis", "context_evidence", "repair_reason"):
            detail = repair[field]
            if not isinstance(detail, str) or len(detail.strip()) < 20:
                raise ValueError(f"repair {index} field {field} must be detailed")
            cleaned[field] = detail.strip()
            generated_reasoning.append(detail.strip())
        generated_reasoning.extend([corrupted_fragment, corrected_fragment])
        if corrected_fragment == original_lines[index - 1]:
            raise ValueError("Gemma may not reproduce a complete corrected couplet")
        cleaned_repairs.append(cleaned)
    if set(diffs) != seen:
        raise ValueError("repairs must cover every changed couplet")
    if poem.poem_text in "\n".join(generated_reasoning):
        raise ValueError("Gemma reasoning must not contain the complete original poem")
    cleaned_repairs.sort(key=lambda item: item["couplet_index"])
    return {"corrupted_poem": corrupted, "repairs": cleaned_repairs}


def build_validation_messages(
    poem: PoemRecord,
    candidate: dict[str, Any],
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": VALIDATION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"<original_poem>\n{poem.poem_text}\n</original_poem>\n"
                f"<candidate>\n{json.dumps(candidate, ensure_ascii=False)}\n</candidate>"
            ),
        },
    ]


def render_instruction(corrupted_poem: str) -> str:
    return (
        "تعرضت القصيدة الآتية لتحريفات موضعية أفسدت بعض ألفاظها أو صورها. "
        "تتبع مواضع الخلل مستعينًا بالسياق والمعنى والجرس، واشرح إصلاح كل موضع "
        "بالتفصيل، ثم أعد القصيدة سليمة.\n\n"
        f"القصيدة المحرفة:\n{corrupted_poem}"
    )


def render_response(repairs: Sequence[dict[str, Any]], poem: PoemRecord) -> str:
    sections = [
        "التحليل والاستدلال:",
        "",
        "أقارن الألفاظ بسياق كل بيت وبالمسار الدلالي والصوتي للقصيدة، ثم أحدد "
        "المواضع التي لا تنسجم مع ما حولها وأختبر الإصلاح المحلي الأنسب.",
    ]
    for repair in repairs:
        sections.extend(
            [
                "",
                f"موضع الخلل في البيت {repair['couplet_index']}:",
                f"اللفظ المحرّف: {repair['corrupted_fragment']}",
                f"تشخيص الخلل: {repair['diagnosis']}",
                f"شاهد السياق: {repair['context_evidence']}",
                f"الإصلاح المحلي: {repair['corrected_fragment']}",
                f"سبب الإصلاح: {repair['repair_reason']}",
            ]
        )
    sections.extend(["", "القصيدة الأصلية بعد الإصلاح:", "", poem.poem_text])
    return "\n".join(sections)


def generate_one(
    poem: PoemRecord,
    client: Any,
    settings: Any,
    *,
    generation_fingerprint: str = "legacy",
    **_kwargs: Any,
) -> dict[str, Any]:
    expected_count = corruption_count(poem)
    base_messages = build_generation_messages(poem, expected_count)
    seed = int(poem.sample_id[:8], 16)
    raw = ""
    errors: list[str] = []
    candidate: dict[str, Any] | None = None
    attempts = 0
    for repair_index in range(settings.max_repairs + 1):
        attempts = repair_index + 1
        messages = base_messages if repair_index == 0 else repair_messages(
            base_messages, raw, errors
        )
        raw = client.chat(
            messages,
            seed=seed + repair_index,
            trace_context={
                "sample_id": poem.sample_id,
                "task_type": TASK_POEM_RECONSTRUCTION,
                "request_kind": (
                    "reconstruction_generation"
                    if repair_index == 0
                    else "reconstruction_repair"
                ),
                "generation_attempt": attempts,
            },
        )
        try:
            if _contains_complete_poem(raw, poem.poem_text):
                raise ValueError("Gemma output must not contain the complete original poem")
            candidate = extract_candidate(
                extract_json_object(raw),
                poem=poem,
                expected_count=expected_count,
            )
            errors = []
        except (ValueError, json.JSONDecodeError) as exc:
            candidate = None
            errors = [str(exc)]
        emit_client_trace(
            client,
            {
                "event": "reconstruction_generation_result",
                "task_type": TASK_POEM_RECONSTRUCTION,
                "sample_id": poem.sample_id,
                "generation_attempt": attempts,
                "raw_model_content": raw,
                "parsed_output": candidate,
                "passed": not errors,
                "deterministic_errors": errors,
            },
        )
        if errors or candidate is None:
            continue
        validation_raw, verdict, validator_attempts = request_verdict(
            client,
            build_validation_messages(poem, candidate),
            max_tokens=1200,
            seed=seed + 100_000 + repair_index,
            trace_context={
                "sample_id": poem.sample_id,
                "task_type": TASK_POEM_RECONSTRUCTION,
                "request_kind": "reconstruction_validation",
                "generation_attempt": attempts,
            },
        )
        errors = [
            f"Gemma rejected reconstruction: {error}"
            for error in verdict["errors"]
        ]
        if _contains_complete_poem(validation_raw, poem.poem_text):
            errors.append(
                "Gemma reconstruction validator emitted the complete original poem"
            )
        emit_client_trace(
            client,
            {
                "event": "reconstruction_validation_result",
                "task_type": TASK_POEM_RECONSTRUCTION,
                "sample_id": poem.sample_id,
                "raw_validator_content": validation_raw,
                "parsed_verdict": verdict,
                "validator_format_attempts": validator_attempts,
                "passed": not errors,
            },
        )
        if not errors:
            break
    else:
        raise GenerationError(
            "reconstruction remained invalid after repairs: " + "; ".join(errors)
        )

    if candidate is None:
        raise GenerationError("reconstruction generation produced no candidate")
    instruction = render_instruction(candidate["corrupted_poem"])
    response = render_response(candidate["repairs"], poem)
    provenance = client_provenance(client, poem.sample_id)
    if not hasattr(client, "sample_stats"):
        provenance["network_attempts"] = attempts * 2
    record = {
        "sample_id": poem.sample_id,
        "record_id": f"{TASK_POEM_RECONSTRUCTION}:{poem.sample_id}",
        "task_type": TASK_POEM_RECONSTRUCTION,
        "task_version": TASK_VERSION,
        "source_row_indices": list(poem.source_row_indices),
        "source_urls": list(poem.source_urls),
        "poet_name": poem.poet_name,
        "poem_title": poem.poem_title,
        "meter_id": poem.meter_id,
        "meter_name": poem.meter_name,
        "couplet_count": poem.couplet_count,
        "corrupted_poem": candidate["corrupted_poem"],
        "corruption_count": expected_count,
        "instruction": instruction,
        "response": response,
        "messages": [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ],
        "sft_split": sft_split(poem.sample_id),
        "metadata_conflict": poem.metadata_conflict,
        "generation_attempts": attempts,
        "validation_status": "passed_after_repair" if attempts > 1 else "passed",
        **provenance,
    }
    emit_client_trace(
        client,
        {
            "event": "final_output",
            "task_type": TASK_POEM_RECONSTRUCTION,
            "sample_id": poem.sample_id,
            "structured_repairs": candidate["repairs"],
            "final_assistant_response": response,
            "generation_fingerprint": generation_fingerprint,
            "postprocessing": {"exact_source_poem_appended": True},
            **provenance,
        },
    )
    return record


def estimate_work(_poem: PoemRecord, _run_settings: Any) -> int:
    return 2


def contract_settings(settings: Any) -> dict[str, Any]:
    return {
        "temperature": settings.temperature,
        "top_p": settings.top_p,
        "max_tokens": settings.max_tokens,
        "max_source_chars": settings.max_source_chars,
    }


def trace_metadata() -> dict[str, Any]:
    return {
        "maximum_corruptions": 3,
        "maximum_fragment_words": MAX_FRAGMENT_WORDS,
        "generation_system_prompt": SYSTEM_PROMPT,
        "validation_system_prompt": VALIDATION_SYSTEM_PROMPT,
    }


WORKFLOW = TaskWorkflow(
    task_type=TASK_POEM_RECONSTRUCTION,
    version=TASK_VERSION,
    generate_one=generate_one,
    estimate_work=estimate_work,
    contract_settings=contract_settings,
    trace_metadata=trace_metadata,
    checkpoint_stages=(),
    benchmark_profile="single-generation-validation",
    pilot_profile="bounded-corruption-count",
)
