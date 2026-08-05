"""Generate, repair, and assemble one synthetic SFT record."""

from __future__ import annotations

import json
from typing import Any

from .assignment import choose_template, sft_split
from .client import GemmaClient
from .config import GenerationSettings
from .errors import GenerationError
from .poems import PoemRecord, split_poem_chunks
from .prompts.builder import build_messages, build_validation_messages
from .prompts.templates import PROMPT_TEMPLATES, TEMPLATE_VERSION
from .responses import compose_response
from .validation import (
    extract_generated_pair,
    extract_json_object,
    parse_validation_verdict,
    verdict_errors,
)


TEMPLATE_RATIONALE = (
    "Six concrete prompts vary the order and framing of the reverse-generation "
    "task while every prompt requires all six poetic dimensions. One template "
    "is chosen uniformly without a seed for each generation call."
)


def _emit_client_trace(client: Any, event: dict[str, Any]) -> None:
    tracer = getattr(client, "tracer", None)
    if tracer is not None:
        tracer.emit(event)



def _chunk_analysis(client: GemmaClient, poem: PoemRecord, max_chars: int) -> str:
    """Summarize a long poem chunk by chunk through the generation API.

    The poem is divided on couplet boundaries, then each chunk is sent in its
    own low-temperature Arabic analysis request. Requests identify the source
    meter and the chunk's position, constrain the answer to meaning, imagery,
    tone, and observable rhyme, and use a deterministic seed derived from the
    poem ID plus the one-based chunk index. The ordered summaries provide a
    compact substitute for source text that would exceed the main prompt limit.

    Args:
        client: Configured chat client used for every analysis request.
        poem: Canonical poem to split and analyze.
        max_chars: Preferred maximum size passed to :func:`split_poem_chunks`.

    Returns:
        The stripped summaries in source order, each prefixed with an Arabic
        one-based chunk label and separated by a blank line.

    Raises:
        ValueError: If ``max_chars`` is not positive.
        GenerationError: If any chunk request cannot be completed.
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



def generate_one(
    poem: PoemRecord,
    client: GemmaClient,
    settings: GenerationSettings,
) -> dict[str, Any]:
    """Generate and Gemma-validate one supervised fine-tuning record.

    One concrete prompt is chosen randomly and retained throughout this call.
    Poems over ``max_source_chars`` are summarized in chunks; the same source
    material is then supplied to generation and validation. Python performs
    only the parsing needed to extract the pair. Gemma alone judges its quality
    and supplies field-level feedback for up to ``max_repairs`` revisions.

    On success, the reasoning is cleaned and combined with the exact source
    poem. The returned dictionary contains provenance, meter and template
    metadata, a two-message SFT conversation, deterministic dataset split,
    oversize/conflict flags, attempt count, and validation status.

    Args:
        poem: Canonical poem that supplies source text and metadata.
        client: Chat-completion client used for analysis and generation calls.
        settings: Prompt limits, repair count, and other generation settings.

    Returns:
        A JSON-serializable SFT record ready for checkpointing and export.

    Raises:
        GenerationError: If network generation fails or the response remains
            invalid after the configured number of repair attempts.
        ValueError: If chunk settings or poem formatting invariants are invalid.
    """
    template = choose_template()
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
    notes = (
        _chunk_analysis(client, poem, settings.chunk_chars) if oversized else None
    )
    if oversized:
        source_material = (
            "القصيدة طويلة؛ المرجع الآتي ملخصات تحليلية لمقاطعها مرتبة. "
            "قيّم وأنشئ زوجًا يغطي القصيدة كلها من غير افتراض ما لم تذكره الملخصات:\n"
            f"{notes}"
        )
    else:
        source_material = poem.poem_text
    messages = build_messages(
        template=template,
        meter_name=poem.meter_name,
        couplet_count=poem.couplet_count,
        poem=source_material,
        minimum_chars=settings.min_chars,
    )

    raw = ""
    errors: list[str] = []
    attempts = 0
    seed = int(poem.sample_id[:8], 16)
    for repair in range(settings.max_repairs + 1):
        attempts += 1
        if repair:
            repair_messages = list(messages)
            repair_messages.extend(
                [
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": (
                            "الجواب السابق غير صالح للأسباب الآتية:\n- "
                            + "\n- ".join(errors)
                            + "\nأعد كائن JSON مصححًا كاملًا فقط."
                        ),
                    },
                ]
            )
        else:
            repair_messages = messages
        raw = client.chat(
            repair_messages,
            seed=seed + repair,
            trace_context={
                "sample_id": poem.sample_id,
                "request_kind": "initial_generation" if repair == 0 else "repair",
                "generation_attempt": attempts,
                "repair_index": repair,
                "template_id": template.template_id,
            },
        )
        try:
            value = extract_json_object(raw)
            instruction, reasoning = extract_generated_pair(value)
        except (ValueError, json.JSONDecodeError) as exc:
            errors = [str(exc)]
            _emit_client_trace(
                client,
                {
                    "event": "generation_parse_result",
                    "sample_id": poem.sample_id,
                    "generation_attempt": attempts,
                    "raw_model_content": raw,
                    "passed": False,
                    "serialization_errors": errors,
                },
            )
            continue

        instruction = instruction.strip()
        reasoning = reasoning.strip()
        candidate = dict(value)
        candidate["instruction"] = instruction
        candidate["reasoning"] = reasoning
        _emit_client_trace(
            client,
            {
                "event": "generation_parse_result",
                "sample_id": poem.sample_id,
                "generation_attempt": attempts,
                "raw_model_content": raw,
                "parsed_output": candidate,
                "passed": True,
                "serialization_errors": [],
            },
        )

        validation_raw = client.chat(
            build_validation_messages(
                candidate=candidate,
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
                "request_kind": "gemma_validation",
                "generation_attempt": attempts,
                "repair_index": repair,
                "template_id": template.template_id,
            },
        )
        try:
            verdict = parse_validation_verdict(validation_raw)
        except (ValueError, json.JSONDecodeError) as exc:
            _emit_client_trace(
                client,
                {
                    "event": "gemma_validation_result",
                    "sample_id": poem.sample_id,
                    "generation_attempt": attempts,
                    "raw_validator_content": validation_raw,
                    "passed": False,
                    "validator_protocol_error": str(exc),
                },
            )
            raise GenerationError(
                f"Gemma validation response was invalid: {exc}"
            ) from exc
        errors = verdict_errors(verdict)
        _emit_client_trace(
            client,
            {
                "event": "gemma_validation_result",
                "sample_id": poem.sample_id,
                "generation_attempt": attempts,
                "raw_validator_content": validation_raw,
                "parsed_verdict": verdict,
                "passed": not errors,
                "validation_errors": errors,
            },
        )
        if not errors:
            response = compose_response(reasoning, poem)
            source_lines = {
                line.strip() for line in poem.poem_text.splitlines() if line.strip()
            }
            source_lines.update(verse.strip() for verse in poem.verses if verse.strip())
            marker = "النتيجة النهائية:"
            postprocessing = {
                "full_poem_occurrences_removed": reasoning.count(poem.poem_text),
                "source_lines_removed": sum(
                    line.strip() in source_lines for line in reasoning.splitlines()
                ),
                "result_marker_lines_removed": sum(
                    line.strip().startswith(marker) for line in reasoning.splitlines()
                ),
                "canonical_result_marker_added": True,
                "exact_source_poem_appended": True,
            }
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
                    {
                        "role": "assistant",
                        "content": response,
                    },
                ],
                "sft_split": sft_split(poem.sample_id),
                "oversized_for_sft": oversized,
                "metadata_conflict": poem.metadata_conflict,
                "generation_attempts": attempts,
                "validation_status": (
                    "passed" if attempts == 1 else "passed_after_repair"
                ),
            }
            _emit_client_trace(
                client,
                {
                    "event": "final_output",
                    "sample_id": poem.sample_id,
                    "template_id": template.template_id,
                    "parsed_instruction": instruction,
                    "gemma_editorial_reasoning": reasoning,
                    "final_assistant_response": response,
                    "postprocessing": postprocessing,
                    "generation_attempts": attempts,
                    "validation_status": record["validation_status"],
                    "origin": "generated_in_this_run",
                },
            )
            return record
    raise GenerationError("validation failed after repairs: " + "; ".join(errors))
