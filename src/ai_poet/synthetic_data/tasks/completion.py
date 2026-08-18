"""Poem-completion workflow built on the validated poem-writing pipeline."""

from __future__ import annotations

from concurrent.futures import Executor
import hashlib
import math
import random
from typing import Any, Sequence

from ..checkpoint import CheckpointWriter
from ..client import GemmaClient
from ..config import GenerationSettings
from ..generation import TEMPLATE_RATIONALE, generate_poem_writing_record
from ..poems import PoemRecord
from ..prompts.templates import (
    ALL_FOCUS_REQUIREMENTS,
    PROMPT_TEMPLATES,
    TEMPLATE_VERSION,
)
from .base import TASK_POEM_COMPLETION, TaskWorkflow


TASK_VERSION = 2
PREFIX_POLICY = "seeded-uniform-complete-couplet-v1"


def provided_couplet_count(poem: PoemRecord) -> int:
    """Choose a reproducible random complete-couplet prefix."""
    if poem.couplet_count < 2:
        raise ValueError("poem-completion requires at least two couplets")
    seed_material = f"{TASK_POEM_COMPLETION}:v{TASK_VERSION}:{poem.sample_id}"
    seed = int.from_bytes(
        hashlib.sha256(seed_material.encode("utf-8")).digest(), "big"
    )
    return random.Random(seed).randrange(1, poem.couplet_count)


def poem_beginning(poem: PoemRecord, count: int | None = None) -> str:
    """Return the exact formatted prefix selected for a completion record."""
    provided = provided_couplet_count(poem) if count is None else count
    if not 1 <= provided < poem.couplet_count:
        raise ValueError("provided couplet count must leave at least one couplet")
    return "\n".join(poem.poem_text.splitlines()[:provided])


def compose_completion_instruction(
    instruction: str,
    *,
    beginning: str,
    total_couplets: int,
    provided_couplets: int,
) -> str:
    """Add the trusted source prefix and completion contract to an instruction."""
    remaining = total_couplets - provided_couplets
    return f"""{instruction.rstrip()}

بداية القصيدة:
---
{beginning}
---

مهمة الإكمال:
هذه بداية ملزمة من القصيدة المطلوبة؛ حافظ عليها حرفيًا ولا تغيّر ترتيب أبياتها. عدد أبيات القصيدة كاملة: {total_couplets}. عدد الأبيات المعطاة أعلاه: {provided_couplets}. عدد الأبيات المطلوب إضافتها: {remaining}. اعرض سجل التفكير والتحرير المطلوب، ثم اختم بالقصيدة كاملة بما فيها البداية المعطاة."""


def build_work_items(poems: Sequence[PoemRecord]) -> list[PoemRecord]:
    """Keep only poems that have a non-empty complete-couplet continuation."""
    return [poem for poem in poems if poem.couplet_count >= 2]


def generate_one(
    poem: PoemRecord,
    client: GemmaClient,
    settings: GenerationSettings,
    *,
    generation_fingerprint: str = "legacy",
    resume_instruction: dict[str, Any] | None = None,
    resume_chunks: dict[int, dict[str, Any]] | None = None,
    resume_stages: dict[str, dict[str, Any]] | None = None,
    checkpoint_writer: CheckpointWriter | None = None,
    chunk_executor: Executor | None = None,
    chunk_parallelism: int = 1,
) -> dict[str, Any]:
    """Generate one completion record; Python appends the trusted final poem."""
    provided = provided_couplet_count(poem)
    beginning = poem_beginning(poem, provided)
    remaining = poem.couplet_count - provided

    def augment(instruction: str) -> str:
        return compose_completion_instruction(
            instruction,
            beginning=beginning,
            total_couplets=poem.couplet_count,
            provided_couplets=provided,
        )

    return generate_poem_writing_record(
        poem,
        client,
        settings,
        task_type=TASK_POEM_COMPLETION,
        task_version=TASK_VERSION,
        instruction_augmenter=augment,
        record_metadata={
            "poem_beginning": beginning,
            "provided_couplet_count": provided,
            "remaining_couplet_count": remaining,
        },
        reasoning_start_offset=provided,
        generation_fingerprint=generation_fingerprint,
        resume_instruction=resume_instruction,
        resume_chunks=resume_chunks,
        resume_stages=resume_stages,
        checkpoint_writer=checkpoint_writer,
        chunk_executor=chunk_executor,
        chunk_parallelism=chunk_parallelism,
    )


def estimate_work(poem: PoemRecord, _run_settings: Any) -> int:
    remaining = poem.couplet_count - provided_couplet_count(poem)
    return 2 + 2 * math.ceil(remaining / 3)


def contract_settings(settings: Any) -> dict[str, Any]:
    return {
        "temperature": settings.temperature,
        "top_p": settings.top_p,
        "max_tokens": settings.max_tokens,
        "min_chars": settings.min_chars,
        "max_source_chars": settings.max_source_chars,
        "prefix_policy": PREFIX_POLICY,
    }


def trace_metadata() -> dict[str, Any]:
    return {
        "template_version": TEMPLATE_VERSION,
        "template_rationale": TEMPLATE_RATIONALE,
        "required_focuses": ALL_FOCUS_REQUIREMENTS,
        "prefix_policy": PREFIX_POLICY,
        "prompt_templates": [
            {"template_id": template.template_id, "prompt": template.prompt}
            for template in PROMPT_TEMPLATES
        ],
    }


WORKFLOW = TaskWorkflow(
    task_type=TASK_POEM_COMPLETION,
    version=TASK_VERSION,
    generate_one=generate_one,
    estimate_work=estimate_work,
    contract_settings=contract_settings,
    trace_metadata=trace_metadata,
    checkpoint_stages=("instruction", "reasoning_chunk"),
    benchmark_profile="poem-completion",
    pilot_profile="bounded-poem-completion",
    expand_work_items=build_work_items,
)
