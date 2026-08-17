"""Run the strict, checkpoint-reusable 300-poem production pilot."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Sequence

from .capacity import (
    PILOT_REPORT_VERSION,
    generation_fingerprint,
    load_capacity_report,
)
from .checkpoint import load_checkpoint
from .config import GenerationSettings, RunSettings
from .corpus import load_poems
from .poems import PoemRecord
from .runner import file_sha256, run
from .tasks.base import (
    TASK_MCQ,
    TASK_POEM_GENERATION,
    get_task_workflow,
)
from .tasks.mcq import QUESTION_DOMAINS


PILOT_QUOTAS = {
    "couplets_1_3": 120,
    "couplets_4_9": 75,
    "couplets_10_24": 50,
    "couplets_25_74": 35,
    "couplets_75_plus": 15,
    "oversized": 5,
}

BOUNDED_TASK_PILOT_QUOTAS = {
    "couplets_1_3": 120,
    "couplets_4_9": 100,
    "couplets_10_24": 80,
}


@dataclass(frozen=True)
class PilotSettings:
    input: Path
    output_dir: Path
    capacity_report: Path
    generation: GenerationSettings
    trace: bool = False
    per_sample_chunk_cap: int = 4
    task_type: str = TASK_POEM_GENERATION


def _group_name(poem: PoemRecord, settings: GenerationSettings) -> str:
    if len(poem.poem_text) > settings.max_source_chars:
        return "oversized"
    count = poem.couplet_count
    if count <= 3:
        return "couplets_1_3"
    if count <= 9:
        return "couplets_4_9"
    if count <= 24:
        return "couplets_10_24"
    if count <= 74:
        return "couplets_25_74"
    return "couplets_75_plus"


def select_pilot_poems(
    poems: Sequence[PoemRecord],
    settings: GenerationSettings,
    task_type: str = TASK_POEM_GENERATION,
) -> tuple[list[PoemRecord], dict[str, list[str]]]:
    """Select the exact deterministic, tail-heavy pilot strata."""
    profile = get_task_workflow(task_type).pilot_profile
    quotas = PILOT_QUOTAS if profile == "tail-heavy" else BOUNDED_TASK_PILOT_QUOTAS
    grouped: dict[str, list[PoemRecord]] = {name: [] for name in quotas}
    for poem in poems:
        name = _group_name(poem, settings)
        if name in grouped:
            grouped[name].append(poem)
    selected: list[PoemRecord] = []
    selected_groups: dict[str, list[str]] = {}
    for name, quota in quotas.items():
        candidates = grouped[name]
        if len(candidates) < quota:
            raise ValueError(
                f"Pilot stratum {name} contains {len(candidates)} poems; "
                f"requires {quota}"
            )
        if name == "oversized":
            largest = max(candidates, key=lambda poem: len(poem.poem_text))
            remainder = sorted(
                (poem for poem in candidates if poem.sample_id != largest.sample_id),
                key=lambda poem: poem.sample_id,
            )[: quota - 1]
            chosen = [largest, *remainder]
        else:
            chosen = sorted(candidates, key=lambda poem: poem.sample_id)[:quota]
        selected.extend(chosen)
        selected_groups[name] = [poem.sample_id for poem in chosen]
    if len({poem.sample_id for poem in selected}) != 300:
        raise ValueError("Pilot selection did not produce 300 unique poems")
    return selected, selected_groups


def _review_records(
    selected_groups: dict[str, list[str]],
    successes: dict[str, dict[str, Any]],
    existing: dict[str, dict[str, Any]],
    task_type: str = TASK_POEM_GENERATION,
) -> list[dict[str, Any]]:
    review_ids: list[str] = []
    if task_type == TASK_POEM_GENERATION:
        review_groups = [
            (selected_groups[name], 5)
            for name in (
                "oversized",
                "couplets_1_3",
                "couplets_4_9",
                "couplets_10_24",
                "couplets_25_74",
                "couplets_75_plus",
            )
        ]
    elif task_type == TASK_MCQ:
        review_groups = [
            (
                [
                    sample_id
                    for sample_id, record in successes.items()
                    if record.get("question_domain") == domain_id
                ],
                6,
            )
            for domain_id, _label in QUESTION_DOMAINS
        ]
    else:
        review_groups = [
            (
                [
                    sample_id
                    for sample_id, record in successes.items()
                    if record.get("corruption_count") == count
                ],
                10,
            )
            for count in (1, 2, 3)
        ]
    for group, quota in review_groups:
        candidates = sorted(
            group,
            key=lambda sample_id: (
                successes.get(sample_id, {}).get("validation_status")
                != "passed_after_repair",
                sample_id,
            ),
        )
        review_ids.extend(candidates[:quota])
    if len(review_ids) < 30:
        remaining = sorted(set(successes) - set(review_ids))
        review_ids.extend(remaining[: 30 - len(review_ids)])
    review_ids = review_ids[:30]
    return [
        {
            "sample_id": sample_id,
            "approved": existing.get(sample_id, {}).get("approved"),
            "notes": existing.get(sample_id, {}).get("notes", ""),
        }
        for sample_id in review_ids
    ]


def _pilot_content_fingerprint(
    *,
    generation: GenerationSettings,
    capacity_fingerprint: str,
    selected_ids: set[str],
    successes: dict[str, dict[str, Any]],
    task_type: str = TASK_POEM_GENERATION,
) -> str:
    value = {
        "task_type": task_type,
        "generation_fingerprint": generation_fingerprint(generation, task_type),
        "capacity_report_fingerprint": capacity_fingerprint,
        "selected_ids": sorted(selected_ids),
        "outputs": [
            {
                "sample_id": sample_id,
                "instruction": successes[sample_id].get("instruction"),
                "response": successes[sample_id].get("response"),
                "validation_status": successes[sample_id].get("validation_status"),
            }
            for sample_id in sorted(successes)
        ],
    }
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def run_pilot(settings: PilotSettings) -> int:
    if not settings.generation.is_multi_endpoint:
        raise ValueError("The production pilot requires indexed three-endpoint settings")
    poems = load_poems(settings.input)
    selected, selected_groups = select_pilot_poems(
        poems, settings.generation, settings.task_type
    )
    source_sha256 = file_sha256(settings.input)
    capacity = load_capacity_report(
        settings.capacity_report,
        settings.generation,
        source_sha256=source_sha256,
        task_type=settings.task_type,
    )
    run_settings = RunSettings(
        input=settings.input,
        output_dir=settings.output_dir,
        concurrency=capacity.total_capacity,
        limit=None,
        trace=settings.trace,
        generation=settings.generation,
        capacity_report=settings.capacity_report,
        per_sample_chunk_cap=settings.per_sample_chunk_cap,
        enforce_pilot_gate=False,
        selected_sample_ids=frozenset(poem.sample_id for poem in selected),
        max_couplets=None,
        task_type=settings.task_type,
    )
    started = time.perf_counter()
    run(run_settings)
    elapsed_seconds = max(0.000001, time.perf_counter() - started)

    successes, failures = load_checkpoint(
        settings.output_dir / "generation_checkpoint.jsonl"
    )
    selected_ids = {poem.sample_id for poem in selected}
    successes = {
        sample_id: record
        for sample_id, record in successes.items()
        if sample_id in selected_ids
        and record.get("task_type", TASK_POEM_GENERATION) == settings.task_type
    }
    failures = {
        sample_id: error
        for sample_id, error in failures.items()
        if sample_id in selected_ids and sample_id not in successes
    }
    repaired = sum(
        record.get("validation_status") == "passed_after_repair"
        for record in successes.values()
    )
    truncations = sum(
        int(record.get("truncated_completions", 0))
        for record in successes.values()
    )
    success_count = len(successes)
    automatic_gates = {
        "success_rate": {
            "value": success_count / 300,
            "minimum": 0.98,
            "passed": success_count >= 294,
        },
        "repaired_samples": {
            "value": repaired,
            "maximum": 75,
            "passed": repaired <= 75,
        },
        "truncated_completions": {
            "value": truncations,
            "maximum": 0,
            "passed": truncations == 0,
        },
    }
    passed = all(gate["passed"] for gate in automatic_gates.values())
    content_fingerprint = _pilot_content_fingerprint(
        generation=settings.generation,
        capacity_fingerprint=capacity.report_fingerprint,
        selected_ids=selected_ids,
        successes=successes,
        task_type=settings.task_type,
    )
    report = {
        "report_version": PILOT_REPORT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "task_type": settings.task_type,
        "task_version": get_task_workflow(settings.task_type).version,
        "source_sha256": source_sha256,
        "generation_fingerprint": generation_fingerprint(
            settings.generation, settings.task_type
        ),
        "capacity_report_fingerprint": capacity.report_fingerprint,
        "content_fingerprint": content_fingerprint,
        "selected_samples": 300,
        "successful_samples": success_count,
        "failed_samples": len(failures),
        "strata": selected_groups,
        "automatic_gates": automatic_gates,
        "endpoint_ids": sorted(
            {
                endpoint_id
                for record in successes.values()
                for endpoint_id in record.get("generation_endpoint_ids", [])
            }
        ),
        "throughput": {
            "elapsed_seconds": elapsed_seconds,
            "validated_samples_per_hour": success_count / elapsed_seconds * 3600,
            "validated_couplets_per_hour": (
                sum(
                    poem.couplet_count
                    for poem in selected
                    if poem.sample_id in successes
                )
                / elapsed_seconds
                * 3600
            ),
            "accepted_response_characters_per_hour": (
                sum(len(str(record.get("response", ""))) for record in successes.values())
                / elapsed_seconds
                * 3600
            ),
        },
    }
    report_bytes = (
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = settings.output_dir / "pilot_report.json"
    report_path.write_bytes(report_bytes)
    report_fingerprint = hashlib.sha256(report_bytes).hexdigest()

    review_path = settings.output_dir / "pilot_review.json"
    existing: dict[str, dict[str, Any]] = {}
    if review_path.exists():
        try:
            previous = json.loads(review_path.read_text("utf-8"))
            if previous.get("pilot_content_fingerprint") == content_fingerprint:
                existing = {
                    item["sample_id"]: item
                    for item in previous.get("reviews", [])
                    if isinstance(item, dict) and "sample_id" in item
                }
        except (OSError, json.JSONDecodeError):
            existing = {}
    review = {
        "pilot_report_fingerprint": report_fingerprint,
        "pilot_content_fingerprint": content_fingerprint,
        "instructions": (
            "Inspect each sample in ashaar_sft.jsonl or ashaar_sft.parquet, "
            "then set approved to true or false and add optional notes."
        ),
        "reviews": _review_records(
            selected_groups, successes, existing, settings.task_type
        ),
    }
    review_path.write_text(
        json.dumps(review, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0 if passed else 1
