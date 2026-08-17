"""Compatibility adapter for the original poem-generation workflow."""

from __future__ import annotations

import math
from typing import Any

from ..generation import TEMPLATE_RATIONALE, generate_one
from ..poems import PoemRecord, split_poem_chunks
from ..prompts.templates import (
    ALL_FOCUS_REQUIREMENTS,
    PROMPT_TEMPLATES,
    TEMPLATE_VERSION,
)
from .base import TASK_POEM_GENERATION, TaskWorkflow


def estimate_work(poem: PoemRecord, run_settings: Any) -> int:
    reasoning_chunks = math.ceil(poem.couplet_count / 3)
    oversized_chunks = (
        len(split_poem_chunks(poem.verses, run_settings.generation.chunk_chars))
        if len(poem.poem_text) > run_settings.generation.max_source_chars
        else 0
    )
    return 2 + 2 * reasoning_chunks + oversized_chunks


def contract_settings(settings: Any) -> dict[str, Any]:
    return {
        "temperature": settings.temperature,
        "top_p": settings.top_p,
        "max_tokens": settings.max_tokens,
        "min_chars": settings.min_chars,
        "max_source_chars": settings.max_source_chars,
        "chunk_chars": settings.chunk_chars,
    }


def trace_metadata() -> dict[str, Any]:
    return {
        "template_version": TEMPLATE_VERSION,
        "template_rationale": TEMPLATE_RATIONALE,
        "required_focuses": ALL_FOCUS_REQUIREMENTS,
        "prompt_templates": [
            {"template_id": template.template_id, "prompt": template.prompt}
            for template in PROMPT_TEMPLATES
        ],
    }


WORKFLOW = TaskWorkflow(
    task_type=TASK_POEM_GENERATION,
    version=TEMPLATE_VERSION,
    generate_one=generate_one,
    estimate_work=estimate_work,
    contract_settings=contract_settings,
    trace_metadata=trace_metadata,
    checkpoint_stages=("instruction", "reasoning_chunk"),
    benchmark_profile="poem-generation",
    pilot_profile="tail-heavy",
)
