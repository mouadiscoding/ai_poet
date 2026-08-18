"""Coordinate resumable, capacity-aware corpus generation runs."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from .capacity import (
    CapacityPlan,
    configured_capacity_plan,
    load_capacity_report,
    validate_pilot_gate,
    workflow_fingerprint,
)
from .checkpoint import (
    CheckpointWriter,
    load_checkpoint,
    load_checkpoint_state,
)
from .client import GemmaClient, GemmaPoolClient
from .config import RunSettings
from .corpus import load_poems
from .errors import GemmaConnectionError, classify_generation_failure
from .generation import generate_one
from .outputs import append_jsonl, write_jsonl, write_outputs
from .runtime_metrics import GenerationProgress, RuntimeMetricsWriter
from .tasks.base import TASK_POEM_GENERATION, get_task_workflow
from .tracing import GenerationTracer, PRINT_LOCK


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def estimated_request_work(work_item: Any, run_settings: RunSettings) -> int:
    """Estimate logical calls so shortest-processing-time scheduling is stable."""
    return get_task_workflow(run_settings.task_type).estimate_work(
        work_item, run_settings
    )


def _normalize_reused_record(record: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    normalized.setdefault("generation_endpoint_ids", ["checkpoint_legacy"])
    normalized.setdefault("endpoint_failover_count", 0)
    normalized.setdefault("network_attempts", 0)
    normalized.setdefault("truncated_completions", 0)
    return normalized


def _validate_output_task(output_dir: Path, task_type: str) -> None:
    """Prevent one task from overwriting another task's durable artifacts."""
    manifest_path = output_dir / "manifest.json"
    existing: str | None = None
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot inspect existing output manifest: {exc}") from exc
        existing = str(manifest.get("task_type", TASK_POEM_GENERATION))
    checkpoint_path = output_dir / "generation_checkpoint.jsonl"
    if existing is None and checkpoint_path.exists():
        try:
            with checkpoint_path.open("r", encoding="utf-8") as handle:
                first = next((line for line in handle if line.strip()), None)
            if first is not None:
                event = json.loads(first)
                existing = str(event.get("task_type", TASK_POEM_GENERATION))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot inspect existing output checkpoint: {exc}") from exc
    if existing is None:
        return
    if existing != task_type:
        raise ValueError(
            f"Output directory belongs to task {existing}; selected task is {task_type}"
        )


def _trace_run_start(
    tracer: GenerationTracer | None,
    run_settings: RunSettings,
    source_fingerprint: str,
    selected_poems: int,
    target_records: int,
    pending: int,
    reused: int,
    capacity: CapacityPlan | None,
    max_couplets: int | None,
    excluded_long_poems: int,
) -> None:
    if tracer is None:
        return
    settings = run_settings.generation
    workflow = get_task_workflow(run_settings.task_type)
    tracer.emit(
        {
            "event": "run_start",
            "task_type": workflow.task_type,
            "task_version": workflow.version,
            "source_path": str(run_settings.input),
            "source_sha256": source_fingerprint,
            "selected_poems": selected_poems,
            "target_records": target_records,
            "checkpoint_reused": reused,
            "pending_generation": pending,
            "max_couplets": max_couplets,
            "excluded_long_poems": excluded_long_poems,
            "model": settings.model,
            "endpoints": [
                {
                    "endpoint_id": endpoint.endpoint_id,
                    "endpoint": endpoint.endpoint,
                    "model": endpoint.model or settings.model,
                    "hard_capacity": (
                        capacity.hard_caps[endpoint.endpoint_id]
                        if capacity is not None
                        else run_settings.concurrency
                    ),
                }
                for endpoint in settings.configured_endpoints
            ],
            "generation_settings": {
                "temperature": settings.temperature,
                "top_p": settings.top_p,
                "max_tokens": settings.max_tokens,
                "min_chars": settings.min_chars,
                "max_source_chars": settings.max_source_chars,
                "chunk_chars": settings.chunk_chars,
                "timeout": settings.timeout,
                "max_network_retries": settings.max_network_retries,
                "max_repairs": settings.max_repairs,
            },
            **workflow.trace_metadata(),
        }
    )


def run(run_settings: RunSettings) -> int:
    """Run generation with partial resume and optional three-endpoint pooling."""
    settings = run_settings.generation
    workflow = get_task_workflow(run_settings.task_type)
    poems = load_poems(run_settings.input)
    if run_settings.selected_sample_ids is not None:
        available_ids = {poem.sample_id for poem in poems}
        missing_ids = run_settings.selected_sample_ids - available_ids
        if missing_ids:
            raise ValueError(
                f"Selected sample IDs are absent from the corpus: {len(missing_ids)}"
            )
        poems = [
            poem for poem in poems if poem.sample_id in run_settings.selected_sample_ids
        ]
    excluded_long_poems = 0
    if run_settings.max_couplets is not None:
        eligible = [
            poem for poem in poems if poem.couplet_count <= run_settings.max_couplets
        ]
        excluded_long_poems = len(poems) - len(eligible)
        poems = eligible
        if excluded_long_poems:
            with PRINT_LOCK:
                print(
                    f"Excluded {excluded_long_poems} poems above "
                    f"--max-couplets={run_settings.max_couplets}."
                )
    if run_settings.limit is not None:
        poems = poems[: run_settings.limit]
    if not poems:
        raise ValueError("No poems remain after applying the selection filters")
    if run_settings.task_type != TASK_POEM_GENERATION:
        oversized = [
            poem
            for poem in poems
            if len(poem.poem_text) > settings.max_source_chars
        ]
        if oversized:
            longest = max(len(poem.poem_text) for poem in oversized)
            raise ValueError(
                f"Task {run_settings.task_type} requires each complete poem, but "
                f"{len(oversized)} selected poems exceed --max-source-chars="
                f"{settings.max_source_chars} (longest: {longest}); lower "
                "--max-couplets or raise --max-source-chars"
            )
    work_items = workflow.expand_work_items(poems)
    if not work_items:
        raise ValueError(
            f"No {run_settings.task_type} work items remain after task eligibility filters"
        )
    source_fingerprint = file_sha256(run_settings.input)
    contract_fingerprint = workflow_fingerprint(
        settings, source_fingerprint, run_settings.task_type
    )

    _validate_output_task(run_settings.output_dir, run_settings.task_type)
    run_settings.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_settings.output_dir / "generation_checkpoint.jsonl"
    # Keep the legacy loader call so downstream integrations that patch or wrap it
    # continue to observe the same public checkpoint entry point.
    successes, previous_failures = load_checkpoint(checkpoint_path)
    partial_state = load_checkpoint_state(checkpoint_path)
    selected_ids = {workflow.work_item_id(item) for item in work_items}
    successes = {
        sample_id: _normalize_reused_record(record)
        for sample_id, record in successes.items()
        if sample_id in selected_ids
        and record.get("task_type", TASK_POEM_GENERATION) == run_settings.task_type
        and record.get("task_version", record.get("template_version"))
        == workflow.version
        and (
            sample_id not in partial_state.success_fingerprints
            or partial_state.success_fingerprints[sample_id]
            == contract_fingerprint
        )
    }
    failures = {
        sample_id: error
        for sample_id, error in previous_failures.items()
        if sample_id in selected_ids and sample_id not in successes
    }
    pending = [
        item
        for item in work_items
        if workflow.work_item_id(item) not in successes
    ]
    pending.sort(
        key=lambda item: (
            estimated_request_work(item, run_settings),
            workflow.work_item_id(item),
        )
    )

    capacity: CapacityPlan | None = None
    pilot_report_fingerprint: str | None = None
    pilot_review_fingerprint: str | None = None
    if settings.is_multi_endpoint:
        if run_settings.capacity_report is not None:
            capacity = load_capacity_report(
                run_settings.capacity_report,
                settings,
                source_sha256=source_fingerprint,
                task_type=run_settings.task_type,
            )
        elif not run_settings.enforce_pilot_gate:
            capacity = configured_capacity_plan(settings)
        else:
            raise ValueError("Multi-endpoint generation requires a capacity report")
        if run_settings.enforce_pilot_gate:
            if run_settings.pilot_report is None or run_settings.pilot_review is None:
                raise ValueError("Multi-endpoint generation requires pilot gate artifacts")
            pilot_report_fingerprint, pilot_review_fingerprint = validate_pilot_gate(
                run_settings.pilot_report,
                run_settings.pilot_review,
                settings=settings,
                source_sha256=source_fingerprint,
                capacity_fingerprint=capacity.report_fingerprint,
                task_type=run_settings.task_type,
            )

    tracer = (
        GenerationTracer(
            run_settings.output_dir / "generation_trace.jsonl",
            secrets=settings.secrets,
        )
        if run_settings.trace
        else None
    )
    _trace_run_start(
        tracer,
        run_settings,
        source_fingerprint,
        len(poems),
        len(work_items),
        len(pending),
        len(work_items) - len(pending),
        capacity,
        run_settings.max_couplets,
        excluded_long_poems,
    )

    if capacity is None:
        client: GemmaClient | GemmaPoolClient = GemmaClient(settings, tracer=tracer)
        worker_count = run_settings.concurrency
        chunk_executor: ThreadPoolExecutor | None = None
        metrics_writer: RuntimeMetricsWriter | None = None
        progress: GenerationProgress | None = None
    else:
        pool = GemmaPoolClient(settings, capacity, tracer=tracer)
        client = pool
        worker_count = capacity.total_capacity
        chunk_executor = ThreadPoolExecutor(
            max_workers=capacity.total_capacity,
            thread_name_prefix="gemma-chunk",
        )
        progress = GenerationProgress(
            pool,
            total_samples=len(work_items),
            initial_completed=len(successes),
            initial_repaired=sum(
                record.get("validation_status") == "passed_after_repair"
                for record in successes.values()
            ),
            initial_response_characters=sum(
                len(str(record.get("response", "")))
                for record in successes.values()
            ),
        )
        metrics_writer = RuntimeMetricsWriter(
            run_settings.output_dir / "generation_metrics.jsonl",
            progress,
        )
        metrics_writer.start()

    sft_jsonl = run_settings.output_dir / "ashaar_sft.jsonl"
    write_jsonl(
        sft_jsonl,
        (
            successes[workflow.work_item_id(item)]
            for item in work_items
            if workflow.work_item_id(item) in successes
        ),
    )
    failures_jsonl = run_settings.output_dir / "failures.jsonl"
    write_jsonl(failures_jsonl, ())
    checkpoint_writer = CheckpointWriter(checkpoint_path)
    completed = len(work_items) - len(pending)
    task_generate_one = (
        generate_one
        if run_settings.task_type == TASK_POEM_GENERATION
        else workflow.generate_one
    )
    try:
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="gemma-sample",
        ) as sample_executor:
            futures = {}
            for item in pending:
                item_id = workflow.work_item_id(item)
                per_sample_parallelism = run_settings.per_sample_chunk_cap
                futures[
                    sample_executor.submit(
                        task_generate_one,
                        item,
                        client,
                        settings,
                        generation_fingerprint=contract_fingerprint,
                        resume_instruction=partial_state.instructions.get(item_id),
                        resume_chunks=partial_state.reasoning_chunks.get(
                            item_id, {}
                        ),
                        resume_stages=partial_state.stages.get(item_id, {}),
                        checkpoint_writer=checkpoint_writer,
                        chunk_executor=chunk_executor,
                        chunk_parallelism=per_sample_parallelism,
                    )
                ] = item

            for future in as_completed(futures):
                item = futures[future]
                item_id = workflow.work_item_id(item)
                completed += 1
                try:
                    record = future.result()
                    successes[item_id] = record
                    failures.pop(item_id, None)
                    event = {
                        "event": "sample_success",
                        "sample_id": item_id,
                        "task_type": workflow.task_type,
                        "task_version": workflow.version,
                        "generation_fingerprint": contract_fingerprint,
                        "record": record,
                    }
                    status = "ok"
                    if progress is not None:
                        progress.record_success(record)
                except GemmaConnectionError:
                    for pending_future in futures:
                        pending_future.cancel()
                    raise
                except Exception as exc:  # keep the corpus run resumable
                    error = str(exc)
                    for secret in settings.secrets:
                        error = error.replace(secret, "[REDACTED]")
                    failures[item_id] = error
                    failure_category = classify_generation_failure(error)
                    event = {
                        "event": "sample_failure",
                        "sample_id": item_id,
                        "task_type": workflow.task_type,
                        "task_version": workflow.version,
                        "generation_fingerprint": contract_fingerprint,
                        "category": failure_category,
                        "error": error,
                    }
                    status = "failed"
                    if progress is not None:
                        progress.record_failure(failure_category)
                    if tracer is not None:
                        tracer.emit(
                            {
                                "event": "sample_failure",
                                "task_type": workflow.task_type,
                                "sample_id": item_id,
                                "error_type": type(exc).__name__,
                                "category": failure_category,
                                "error": error,
                            }
                        )
                checkpoint_writer.append(event)
                if status == "ok":
                    append_jsonl(sft_jsonl, record)
                else:
                    append_jsonl(
                        failures_jsonl,
                        {
                            "sample_id": item_id,
                            "task_type": workflow.task_type,
                            "category": failure_category,
                            "error": error,
                        },
                    )
                with PRINT_LOCK:
                    print(
                        f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] "
                        f"[{completed}/{len(work_items)}] "
                        f"{item_id[:24]} {status}"
                    )
    finally:
        if chunk_executor is not None:
            chunk_executor.shutdown(wait=True, cancel_futures=True)
        if metrics_writer is not None:
            metrics_writer.stop()

    endpoint_metrics = client.snapshot() if isinstance(client, GemmaPoolClient) else None
    write_outputs(
        run_settings.output_dir,
        poems,
        successes,
        failures,
        settings,
        source_fingerprint=source_fingerprint,
        trace_run_id=tracer.run_id if tracer is not None else None,
        capacity_report_fingerprint=(
            capacity.report_fingerprint
            if capacity is not None and capacity.report_fingerprint
            else None
        ),
        pilot_report_fingerprint=pilot_report_fingerprint,
        pilot_review_fingerprint=pilot_review_fingerprint,
        endpoint_metrics=endpoint_metrics,
        max_couplets=run_settings.max_couplets,
        excluded_long_poems=excluded_long_poems,
        task_type=workflow.task_type,
        task_version=workflow.version,
        ordered_work_ids=[workflow.work_item_id(item) for item in work_items],
    )
    unresolved = [
        item
        for item in work_items
        if workflow.work_item_id(item) not in successes
    ]
    if tracer is not None:
        tracer.emit(
            {
                "event": "run_summary",
                "task_type": workflow.task_type,
                "task_version": workflow.version,
                "selected_poems": len(poems),
                "target_records": len(work_items),
                "generated_successes": sum(
                    workflow.work_item_id(item) in successes for item in pending
                ),
                "unresolved_failures": len(unresolved),
                "max_couplets": run_settings.max_couplets,
                "excluded_long_poems": excluded_long_poems,
                "complete": not unresolved,
                "template_distribution": dict(
                    Counter(
                        record["template_id"]
                        for record in successes.values()
                        if "template_id" in record
                    )
                ),
                "metadata_field_distribution": dict(
                    Counter(
                        record["metadata_field"]
                        for record in successes.values()
                        if "metadata_field" in record
                    )
                ),
                "prompt_distribution": dict(
                    Counter(
                        record["prompt_id"]
                        for record in successes.values()
                        if "prompt_id" in record
                    )
                ),
                "corruption_count_distribution": dict(
                    Counter(
                        str(record["corruption_count"])
                        for record in successes.values()
                        if "corruption_count" in record
                    )
                ),
                "validation_status_distribution": dict(
                    Counter(
                        record["validation_status"] for record in successes.values()
                    )
                ),
                "endpoint_metrics": endpoint_metrics,
            }
        )
    return 1 if unresolved else 0
