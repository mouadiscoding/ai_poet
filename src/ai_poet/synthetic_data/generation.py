"""Generate, validate, repair, and assemble one synthetic SFT record."""

from __future__ import annotations

import json
import random
from typing import Any, Sequence

from .assignment import sft_split
from .client import GemmaClient
from .config import GenerationSettings
from .errors import GenerationError
from .poems import PoemRecord, split_poem_chunks
from .prompts.builder import (
    build_instruction_validation_messages,
    build_messages,
    build_reasoning_messages,
    build_reasoning_validation_messages,
)
from .prompts.templates import PROMPT_TEMPLATES, TEMPLATE_VERSION
from .responses import compose_response, render_reasoning
from .validation import (
    extract_instruction,
    extract_json_object,
    extract_reasoning_chunk,
    instruction_contract_errors,
    parse_field_verdict,
)


REASONING_CHUNK_COUPLETS = 3

TEMPLATE_RATIONALE = (
    "Six concrete prompts vary the order and framing of instruction "
    "reconstruction while every prompt requires all six poetic dimensions. "
    "Verse-level editorial work is generated separately in bounded chunks."
)


def _emit_client_trace(client: Any, event: dict[str, Any]) -> None:
    tracer = getattr(client, "tracer", None)
    if tracer is not None:
        tracer.emit(event)


def _chunk_analysis(client: GemmaClient, poem: PoemRecord, max_chars: int) -> str:
    """Summarize a long poem for the global instruction-generation stage."""
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
    base_messages: Sequence[dict[str, str]],
    raw: str,
    errors: Sequence[str],
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


def _generate_instruction(
    *,
    poem: PoemRecord,
    client: GemmaClient,
    settings: GenerationSettings,
    template: Any,
    source_material: str,
    seed: int,
) -> tuple[str, int]:
    """Generate and independently validate the writing instruction."""
    base_messages = build_messages(
        template=template,
        meter_name=poem.meter_name,
        couplet_count=poem.couplet_count,
        poem=source_material,
        minimum_chars=settings.min_chars,
    )
    raw = ""
    errors: list[str] = []
    for repair in range(settings.max_repairs + 1):
        messages = (
            base_messages if repair == 0 else _repair_messages(base_messages, raw, errors)
        )
        raw = client.chat(
            messages,
            seed=seed + repair,
            trace_context={
                "sample_id": poem.sample_id,
                "request_kind": (
                    "instruction_generation" if repair == 0 else "instruction_repair"
                ),
                "generation_attempt": repair + 1,
                "repair_index": repair,
                "template_id": template.template_id,
            },
        )
        try:
            instruction = extract_instruction(extract_json_object(raw))
            errors = instruction_contract_errors(
                instruction,
                meter_name=poem.meter_name,
                couplet_count=poem.couplet_count,
                minimum_chars=settings.min_chars,
                source_hemistichs=poem.verses,
            )
        except (ValueError, json.JSONDecodeError) as exc:
            instruction = ""
            errors = [str(exc)]

        _emit_client_trace(
            client,
            {
                "event": "instruction_generation_result",
                "sample_id": poem.sample_id,
                "generation_attempt": repair + 1,
                "raw_model_content": raw,
                "parsed_instruction": instruction or None,
                "passed": not errors,
                "deterministic_errors": errors,
            },
        )
        if errors:
            continue

        validation_raw = client.chat(
            build_instruction_validation_messages(
                instruction=instruction,
                meter_name=poem.meter_name,
                couplet_count=poem.couplet_count,
                poem=source_material,
                minimum_chars=settings.min_chars,
            ),
            max_tokens=1200,
            temperature=0.0,
            seed=seed + 100_000 + repair,
            trace_context={
                "sample_id": poem.sample_id,
                "request_kind": "instruction_validation",
                "generation_attempt": repair + 1,
                "template_id": template.template_id,
            },
        )
        try:
            verdict = parse_field_verdict(validation_raw)
        except (ValueError, json.JSONDecodeError) as exc:
            raise GenerationError(
                f"Gemma instruction validation response was invalid: {exc}"
            ) from exc
        errors = [
            f"Gemma rejected instruction: {error}" for error in verdict["errors"]
        ]
        _emit_client_trace(
            client,
            {
                "event": "instruction_validation_result",
                "sample_id": poem.sample_id,
                "generation_attempt": repair + 1,
                "raw_validator_content": validation_raw,
                "parsed_verdict": verdict,
                "passed": not errors,
                "validation_errors": errors,
            },
        )
        if not errors:
            return instruction, repair + 1

    raise GenerationError(
        "instruction remained invalid after repairs: " + "; ".join(errors)
    )


def _generate_reasoning_chunk(
    *,
    poem: PoemRecord,
    client: GemmaClient,
    settings: GenerationSettings,
    instruction: str,
    all_couplets: Sequence[str],
    start_offset: int,
    chunk_couplets: Sequence[str],
    seed: int,
) -> tuple[str | None, list[dict[str, Any]], int]:
    """Generate and independently validate one bounded verse-work chunk."""
    start_index = start_offset + 1
    expected_indices = list(
        range(start_index, start_index + len(chunk_couplets))
    )
    include_overview = start_offset == 0
    base_messages = build_reasoning_messages(
        instruction=instruction,
        meter_name=poem.meter_name,
        total_couplet_count=poem.couplet_count,
        start_index=start_index,
        couplets=chunk_couplets,
        previous_couplet=(all_couplets[start_offset - 1] if start_offset else None),
        next_couplet=(
            all_couplets[start_offset + len(chunk_couplets)]
            if start_offset + len(chunk_couplets) < len(all_couplets)
            else None
        ),
        include_overview=include_overview,
    )
    raw = ""
    errors: list[str] = []
    for repair in range(settings.max_repairs + 1):
        messages = (
            base_messages if repair == 0 else _repair_messages(base_messages, raw, errors)
        )
        raw = client.chat(
            messages,
            seed=seed + 10_000 + start_index * 100 + repair,
            trace_context={
                "sample_id": poem.sample_id,
                "request_kind": (
                    "reasoning_generation" if repair == 0 else "reasoning_repair"
                ),
                "chunk_start": start_index,
                "chunk_end": expected_indices[-1],
                "generation_attempt": repair + 1,
                "repair_index": repair,
            },
        )
        try:
            value = extract_json_object(raw)
            overview, blocks = extract_reasoning_chunk(
                value,
                expected_indices=expected_indices,
                expected_couplets=chunk_couplets,
                include_overview=include_overview,
            )
            errors = []
        except (ValueError, json.JSONDecodeError) as exc:
            value = {}
            overview = None
            blocks = []
            errors = [str(exc)]

        _emit_client_trace(
            client,
            {
                "event": "reasoning_chunk_generation_result",
                "sample_id": poem.sample_id,
                "chunk_start": start_index,
                "chunk_end": expected_indices[-1],
                "generation_attempt": repair + 1,
                "raw_model_content": raw,
                "parsed_output": value or None,
                "passed": not errors,
                "deterministic_errors": errors,
            },
        )
        if errors:
            continue

        validation_raw = client.chat(
            build_reasoning_validation_messages(
                instruction=instruction,
                meter_name=poem.meter_name,
                expected_couplets=chunk_couplets,
                candidate=value,
            ),
            max_tokens=1200,
            temperature=0.0,
            seed=seed + 200_000 + start_index * 100 + repair,
            trace_context={
                "sample_id": poem.sample_id,
                "request_kind": "reasoning_validation",
                "chunk_start": start_index,
                "chunk_end": expected_indices[-1],
                "generation_attempt": repair + 1,
            },
        )
        try:
            verdict = parse_field_verdict(validation_raw)
        except (ValueError, json.JSONDecodeError) as exc:
            raise GenerationError(
                f"Gemma reasoning validation response was invalid: {exc}"
            ) from exc
        errors = [
            f"Gemma rejected reasoning: {error}" for error in verdict["errors"]
        ]
        _emit_client_trace(
            client,
            {
                "event": "reasoning_chunk_validation_result",
                "sample_id": poem.sample_id,
                "chunk_start": start_index,
                "chunk_end": expected_indices[-1],
                "generation_attempt": repair + 1,
                "raw_validator_content": validation_raw,
                "parsed_verdict": verdict,
                "passed": not errors,
                "validation_errors": errors,
            },
        )
        if not errors:
            return overview, blocks, repair + 1

    raise GenerationError(
        f"reasoning chunk {start_index}-{expected_indices[-1]} remained invalid "
        "after repairs: "
        + "; ".join(errors)
    )


def generate_one(
    poem: PoemRecord,
    client: GemmaClient,
    settings: GenerationSettings,
) -> dict[str, Any]:
    """Generate one instruction and a validated editorial block per couplet."""
    template = random.choice(PROMPT_TEMPLATES)
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
            "eligible_template_ids": [item.template_id for item in PROMPT_TEMPLATES],
            "selected_template_id": template.template_id,
            "why_used": {
                "purpose": TEMPLATE_RATIONALE,
                "eligibility": "all six templates support metered and prose poems",
                "selection": "fresh uniform random choice, made once for this call",
            },
        },
    )

    oversized = len(poem.poem_text) > settings.max_source_chars
    notes = _chunk_analysis(client, poem, settings.chunk_chars) if oversized else None
    source_material = (
        "القصيدة طويلة؛ هذه ملخصات مرتبة لبناء التكليف العام من غير افتراض "
        f"ما لم تذكره:\n{notes}"
        if oversized
        else poem.poem_text
    )
    seed = int(poem.sample_id[:8], 16)
    instruction, instruction_attempts = _generate_instruction(
        poem=poem,
        client=client,
        settings=settings,
        template=template,
        source_material=source_material,
        seed=seed,
    )

    all_couplets = poem.poem_text.splitlines()
    overview: str | None = None
    all_blocks: list[dict[str, Any]] = []
    reasoning_attempts = 0
    chunk_count = 0
    for start_offset in range(0, len(all_couplets), REASONING_CHUNK_COUPLETS):
        chunk_count += 1
        chunk_couplets = all_couplets[
            start_offset : start_offset + REASONING_CHUNK_COUPLETS
        ]
        chunk_overview, blocks, attempts = _generate_reasoning_chunk(
            poem=poem,
            client=client,
            settings=settings,
            instruction=instruction,
            all_couplets=all_couplets,
            start_offset=start_offset,
            chunk_couplets=chunk_couplets,
            seed=seed,
        )
        if chunk_overview is not None:
            overview = chunk_overview
        all_blocks.extend(blocks)
        reasoning_attempts += attempts

    if overview is None:
        raise GenerationError("first reasoning chunk did not produce an overview")
    if [block["verse_index"] for block in all_blocks] != list(
        range(1, poem.couplet_count + 1)
    ):
        raise GenerationError("assembled reasoning does not cover every couplet once")

    editorial_reasoning = render_reasoning(
        overview,
        all_blocks,
        is_prose=poem.meter_name == "النثر",
    )
    response = compose_response(editorial_reasoning, poem)
    generation_attempts = instruction_attempts + reasoning_attempts
    repaired = instruction_attempts > 1 or reasoning_attempts > chunk_count
    record = {
        "sample_id": poem.sample_id,
        "source_row_indices": list(poem.source_row_indices),
        "source_urls": list(poem.source_urls),
        "poet_name": poem.poet_name,
        "poem_title": poem.poem_title,
        "meter_id": poem.meter_id,
        "meter_name": poem.meter_name,
        "couplet_count": poem.couplet_count,
        "template_id": template.template_id,
        "template_version": TEMPLATE_VERSION,
        "instruction": instruction,
        "response": response,
        "messages": [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ],
        "sft_split": sft_split(poem.sample_id),
        "oversized_for_sft": oversized,
        "metadata_conflict": poem.metadata_conflict,
        "generation_attempts": generation_attempts,
        "instruction_generation_attempts": instruction_attempts,
        "reasoning_generation_attempts": reasoning_attempts,
        "reasoning_chunk_count": chunk_count,
        "validation_status": "passed_after_repair" if repaired else "passed",
    }
    _emit_client_trace(
        client,
        {
            "event": "final_output",
            "sample_id": poem.sample_id,
            "template_id": template.template_id,
            "parsed_instruction": instruction,
            "reasoning_overview": overview,
            "structured_verse_reasoning": all_blocks,
            "rendered_editorial_reasoning": editorial_reasoning,
            "final_assistant_response": response,
            "postprocessing": {
                "full_poem_occurrences_removed": editorial_reasoning.count(
                    poem.poem_text
                ),
                "quoted_source_couplets_preserved": len(all_blocks),
                "canonical_result_marker_added": True,
                "exact_source_poem_appended": True,
            },
            "generation_attempts": generation_attempts,
            "validation_status": record["validation_status"],
        },
    )
    return record
