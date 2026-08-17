"""Generate, validate, repair, and assemble one synthetic QCM SFT record."""

from __future__ import annotations

import json
import random
from typing import Any

from .assignment import sft_split
from .client import GemmaClient
from .config import GenerationSettings
from .errors import GenerationError
from .poems import PoemRecord, split_poem_chunks
from .prompts.builder import build_qcm_messages, build_qcm_validation_messages
from .prompts.qcm_templates import QCM_PROMPT_TEMPLATES, QCM_TEMPLATE_VERSION
from .responses import compose_qcm_response
from .validation import (
    extract_json_object,
    extract_qcm,
    parse_field_verdict,
    qcm_contract_errors,
)

TEMPLATE_RATIONALE = (
    "Four QCM prompt templates vary the framing and analytical entry point "
    "while every template requires a poem-grounded question, four plausible "
    "choices with one correct answer, and a demonstrative reasoning that "
    "follows the full conceptual path from question to answer."
)


def _emit_client_trace(client: Any, event: dict[str, Any]) -> None:
    tracer = getattr(client, "tracer", None)
    if tracer is not None:
        tracer.emit(event)


def _chunk_analysis(client: GemmaClient, poem: PoemRecord, max_chars: int) -> str:
    """Summarize a long poem for the global QCM-generation stage.

    The summaries cover the whole poem ordered by chunk, so a global QCM can be
    built from the entire poem text rather than from a single chunk.
    """
    summaries: list[str] = []
    chunks = split_poem_chunks(poem.verses, max_chars)
    for index, chunk in enumerate(chunks, start=1):
        messages = [
            {
                "role": "system",
                "content": (
                    "حلل مقطعًا من قصيدة عربية في 300 إلى 600 محرف. اذكر المعاني "
                    "والصور والنبرة والقافية الظاهرة فقط، ولا تنشئ تعليمات ولا قصيدة جديدة."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"البحر الأساس: {poem.meter_name}\n"
                    f"المقطع {index} من {len(chunks)}:\n{chunk}"
                ),
            },
        ]
        summaries.append(
            client.chat(
                messages,
                max_tokens=800,
                temperature=0.2,
                seed=int(poem.sample_id[:8], 16) + index,
                trace_context={
                    "sample_id": poem.sample_id,
                    "request_kind": "chunk_analysis",
                    "chunk_index": index,
                    "chunk_count": len(chunks),
                },
            ).strip()
        )
    return "\n\n".join(
        f"ملخص المقطع {index}: {summary}"
        for index, summary in enumerate(summaries, start=1)
    )


def _repair_messages(
    base_messages: list[dict[str, str]],
    raw: str,
    errors: list[str],
) -> list[dict[str, str]]:
    """Append one complete rejected answer and actionable repair feedback."""
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


def _generate_qcm(
    *,
    poem: PoemRecord,
    client: GemmaClient,
    settings: GenerationSettings,
    template: Any,
    source_material: str,
    seed: int,
) -> tuple[dict[str, Any], int]:
    """Generate and independently validate one QCM for a poem."""
    base_messages = build_qcm_messages(
        template=template,
        meter_name=poem.meter_name,
        couplet_count=poem.couplet_count,
        poem=source_material,
    )
    raw = ""
    errors: list[str] = []
    for repair in range(settings.max_repairs + 1):
        messages = (
            base_messages
            if repair == 0
            else _repair_messages(base_messages, raw, errors)
        )
        raw = client.chat(
            messages,
            seed=seed + repair,
            trace_context={
                "sample_id": poem.sample_id,
                "request_kind": "qcm_generation" if repair == 0 else "qcm_repair",
                "generation_attempt": repair + 1,
                "repair_index": repair,
                "template_id": template.template_id,
            },
        )
        try:
            qcm = extract_qcm(extract_json_object(raw))
            errors = qcm_contract_errors(qcm, poem=poem.poem_text)
        except (ValueError, json.JSONDecodeError) as exc:
            qcm = {}
            errors = [str(exc)]

        _emit_client_trace(
            client,
            {
                "event": "qcm_generation_result",
                "sample_id": poem.sample_id,
                "generation_attempt": repair + 1,
                "raw_model_content": raw,
                "parsed_qcm": qcm or None,
                "passed": not errors,
                "deterministic_errors": errors,
            },
        )
        if errors:
            continue

        validation_raw = client.chat(
            build_qcm_validation_messages(
                poem=source_material,
                candidate=qcm,
            ),
            max_tokens=1200,
            temperature=0.0,
            seed=seed + 100_000 + repair,
            trace_context={
                "sample_id": poem.sample_id,
                "request_kind": "qcm_validation",
                "generation_attempt": repair + 1,
                "template_id": template.template_id,
            },
        )
        try:
            verdict = parse_field_verdict(validation_raw)
        except (ValueError, json.JSONDecodeError) as exc:
            raise GenerationError(
                f"Gemma QCM validation response was invalid: {exc}"
            ) from exc
        errors = [f"Gemma rejected QCM: {error}" for error in verdict["errors"]]
        _emit_client_trace(
            client,
            {
                "event": "qcm_validation_result",
                "sample_id": poem.sample_id,
                "generation_attempt": repair + 1,
                "raw_validator_content": validation_raw,
                "parsed_verdict": verdict,
                "passed": not errors,
                "validation_errors": errors,
            },
        )
        if not errors:
            return qcm, repair + 1

    raise GenerationError("QCM remained invalid after repairs: " + "; ".join(errors))


def generate_one(
    poem: PoemRecord,
    client: GemmaClient,
    settings: GenerationSettings,
) -> dict[str, Any]:
    """Generate one poem-grounded multiple-choice question."""
    template = random.choice(QCM_PROMPT_TEMPLATES)
    _emit_client_trace(
        client,
        {
            "event": "template_selection",
            "sample_id": poem.sample_id,
            "source_row_indices": list(poem.source_row_indices),
            "poet_name": poem.poet_name,
            "poem_title": poem.poem_title,
            "meter_name": poem.meter_name,
            "couplet_count": poem.couplet_count,
            "source_characters": len(poem.poem_text),
            "eligible_template_ids": [
                item.template_id for item in QCM_PROMPT_TEMPLATES
            ],
            "selected_template_id": template.template_id,
            "why_used": {
                "purpose": TEMPLATE_RATIONALE,
                "eligibility": "all four QCM templates support metered and prose poems",
                "selection": "fresh uniform random choice, made once for this call",
            },
        },
    )

    oversized = len(poem.poem_text) > settings.max_source_chars
    notes = _chunk_analysis(client, poem, settings.chunk_chars) if oversized else None
    source_material = (
        "القصيدة طويلة جدًا؛ هذه ملخصات مرتبة لكل مقاطعها لبناء سؤال "
        f"عام يستند إلى القصيدة كلها من غير افتراض ما لم تذكره:\n{notes}"
        if oversized
        else poem.poem_text
    )
    seed = int(poem.sample_id[:8], 16)
    qcm, generation_attempts = _generate_qcm(
        poem=poem,
        client=client,
        settings=settings,
        template=template,
        source_material=source_material,
        seed=seed,
    )

    response = compose_qcm_response(qcm, poem)
    repaired = generation_attempts > 1
    record = {
        "sample_id": poem.sample_id,
        "source_row_indices": list(poem.source_row_indices),
        "source_urls": list(poem.source_urls),
        "poet_name": poem.poet_name,
        "poem_title": poem.poem_title,
        "meter_id": poem.meter_id,
        "meter_name": poem.meter_name,
        "couplet_count": poem.couplet_count,
        "poem": poem.poem_text,
        "question": qcm["question"],
        "choices": qcm["choices"],
        "reasoning": qcm["reasoning"],
        "correct_answer": qcm["correct_answer"],
        "correct_answer_text": qcm["choices"][qcm["correct_answer"]],
        "template_id": template.template_id,
        "template_version": QCM_TEMPLATE_VERSION,
        "response": response,
        "messages": [
            {"role": "user", "content": poem.poem_text},
            {"role": "assistant", "content": response},
        ],
        "sft_split": sft_split(poem.sample_id),
        "oversized_for_sft": oversized,
        "metadata_conflict": poem.metadata_conflict,
        "generation_attempts": generation_attempts,
        "validation_status": "passed_after_repair" if repaired else "passed",
    }
    _emit_client_trace(
        client,
        {
            "event": "final_output",
            "sample_id": poem.sample_id,
            "template_id": template.template_id,
            "parsed_qcm": qcm,
            "final_assistant_response": response,
            "postprocessing": {
                "exact_source_poem_present": poem.poem_text in response,
                "question_present": qcm["question"] in response,
                "all_choices_present": all(
                    qcm["choices"][letter] in response
                    for letter in ("A", "B", "C", "D")
                ),
                "correct_answer_present": qcm["correct_answer"] in response,
            },
            "generation_attempts": generation_attempts,
            "validation_status": record["validation_status"],
        },
    )
    return record
