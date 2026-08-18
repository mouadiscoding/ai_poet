"""Registry and shared interface for one-poem SFT workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence


TASK_POEM_GENERATION = "poem-generation"
TASK_POEM_COMPLETION = "poem-completion"
TASK_MCQ = "mcq"
TASK_POEM_RECONSTRUCTION = "poem-reconstruction"
TASK_TYPES = (
    TASK_POEM_GENERATION,
    TASK_POEM_COMPLETION,
    TASK_MCQ,
    TASK_POEM_RECONSTRUCTION,
)

DEFAULT_OUTPUT_DIRS = {
    TASK_POEM_GENERATION: Path("data/ashaar_sft"),
    TASK_POEM_COMPLETION: Path("data/ashaar_completion_sft"),
    TASK_MCQ: Path("data/ashaar_mcq_sft"),
    TASK_POEM_RECONSTRUCTION: Path("data/ashaar_reconstruction_sft"),
}


def _one_work_item_per_poem(poems: Sequence[Any]) -> list[Any]:
    return list(poems)


def _sample_id(work_item: Any) -> str:
    return str(work_item.sample_id)


@dataclass(frozen=True)
class TaskWorkflow:
    """Behavior owned by one SFT task while the runner stays task-neutral."""

    task_type: str
    version: int
    generate_one: Callable[..., dict[str, Any]]
    estimate_work: Callable[[Any, Any], int]
    contract_settings: Callable[[Any], dict[str, Any]]
    trace_metadata: Callable[[], dict[str, Any]]
    checkpoint_stages: tuple[str, ...]
    benchmark_profile: str
    pilot_profile: str
    expand_work_items: Callable[[Sequence[Any]], list[Any]] = _one_work_item_per_poem
    work_item_id: Callable[[Any], str] = _sample_id


def default_output_dir(task_type: str) -> Path:
    try:
        return DEFAULT_OUTPUT_DIRS[task_type]
    except KeyError as exc:
        raise ValueError(f"Unknown SFT task: {task_type}") from exc


def get_task_workflow(task_type: str) -> TaskWorkflow:
    """Load one workflow lazily to avoid task/runner import cycles."""
    if task_type == TASK_POEM_GENERATION:
        from .poem_generation import WORKFLOW
    elif task_type == TASK_POEM_COMPLETION:
        from .completion import WORKFLOW
    elif task_type == TASK_MCQ:
        from .mcq import WORKFLOW
    elif task_type == TASK_POEM_RECONSTRUCTION:
        from .reconstruction import WORKFLOW
    else:
        raise ValueError(f"Unknown SFT task: {task_type}")
    return WORKFLOW
