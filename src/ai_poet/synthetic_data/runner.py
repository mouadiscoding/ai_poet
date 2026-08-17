"""Coordinate resumable, concurrent corpus generation runs."""

from __future__ import annotations

import hashlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .checkpoint import append_checkpoint, load_checkpoint
from .client import GemmaClient
from .config import RunSettings
from .corpus import load_poems
from .errors import GemmaConnectionError
from .generation import TEMPLATE_RATIONALE, generate_one
from .outputs import append_jsonl, write_jsonl, write_outputs
from .prompts.qcm_templates import QCM_PROMPT_TEMPLATES, QCM_TEMPLATE_VERSION
from .tracing import PRINT_LOCK, GenerationTracer


def file_sha256(path: Path) -> str:
    """Calculate the SHA-256 fingerprint of a file without loading it at once.

    The file is streamed in one-megabyte binary blocks, so memory usage remains
    bounded for large Parquet datasets.

    Args:
        path: File whose exact byte content should be hashed.

    Returns:
        The lowercase hexadecimal SHA-256 digest.

    Raises:
        OSError: If the file cannot be opened or read.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def run(run_settings: RunSettings) -> int:
    """Execute a resumable, concurrent SFT dataset generation run.

    Source poems are loaded and optionally limited, then prior checkpoint events
    are replayed and restricted to the current selection. Poems without a saved
    success are submitted to a thread pool. Each finished task is immediately
    checkpointed as a success or sanitized failure, allowing later invocations
    to resume without regenerating completed samples. Individual generation
    exceptions do not abort the corpus, except an exhausted Gemma connection
    failure, which cancels pending work and aborts the run. Successful records
    are flushed to the SFT JSONL file as each task finishes; final ordered
    JSONL, Parquet, failure, and manifest outputs are written after all pending
    tasks finish.

    Args:
        run_settings: Validated source, output, concurrency, and generation
            configuration.

    Returns:
        Zero when every selected poem has a successful record, otherwise one.

    Raises:
        GemmaConnectionError: If Gemma remains unreachable after every
            configured connection retry.
        ValueError: If configuration, the optional limit, or source poem data is
            invalid.
        OSError: If required input or output files cannot be accessed.
    """
    settings = run_settings.generation
    poems = load_poems(run_settings.input)
    if run_settings.limit is not None:
        poems = poems[: run_settings.limit]

    run_settings.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_settings.output_dir / "generation_checkpoint.jsonl"
    successes, previous_failures = load_checkpoint(checkpoint)
    selected_ids = {poem.sample_id for poem in poems}
    successes = {
        sample_id: record
        for sample_id, record in successes.items()
        if sample_id in selected_ids
        and record.get("template_version") == QCM_TEMPLATE_VERSION
    }
    failures = {
        sample_id: error
        for sample_id, error in previous_failures.items()
        if sample_id in selected_ids and sample_id not in successes
    }
    pending = [poem for poem in poems if poem.sample_id not in successes]
    sft_jsonl = run_settings.output_dir / "ashaar_sft.jsonl"
    write_jsonl(
        sft_jsonl,
        (successes[poem.sample_id] for poem in poems if poem.sample_id in successes),
    )
    source_fingerprint = file_sha256(run_settings.input)
    tracer = (
        GenerationTracer(
            run_settings.output_dir / "generation_trace.jsonl",
            secrets=(settings.api_key,),
        )
        if run_settings.trace
        else None
    )
    if tracer is not None:
        tracer.emit(
            {
                "event": "run_start",
                "source_path": str(run_settings.input),
                "source_sha256": source_fingerprint,
                "selected_poems": len(poems),
                "checkpoint_reused": len(poems) - len(pending),
                "pending_generation": len(pending),
                "model": settings.model,
                "endpoint": settings.endpoint,
                "concurrency": run_settings.concurrency,
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
                "template_version": QCM_TEMPLATE_VERSION,
                "template_rationale": TEMPLATE_RATIONALE,
                "prompt_templates": [
                    {
                        "template_id": template.template_id,
                        "prompt": template.prompt,
                        "why_available": (
                            "supports both metered and prose records and requires "
                            "a poem-grounded question, four choices, and a "
                            "demonstrative reasoning"
                        ),
                    }
                    for template in QCM_PROMPT_TEMPLATES
                ],
                "checkpoint_note": (
                    "checkpoint_reused counts existing successful records; all "
                    "per-sample events in this run are newly generated"
                ),
            }
        )
    client = GemmaClient(settings, tracer=tracer)
    completed = len(poems) - len(pending)

    with ThreadPoolExecutor(max_workers=run_settings.concurrency) as executor:
        futures = {
            executor.submit(generate_one, poem, client, settings): poem
            for poem in pending
        }
        for future in as_completed(futures):
            poem = futures[future]
            completed += 1
            try:
                record = future.result()
                successes[poem.sample_id] = record
                failures.pop(poem.sample_id, None)
                event = {
                    "status": "success",
                    "sample_id": poem.sample_id,
                    "record": record,
                }
                status = "ok"
            except GemmaConnectionError:
                for pending_future in futures:
                    pending_future.cancel()
                raise
            except Exception as exc:  # keep the corpus run resumable
                error = str(exc).replace(settings.api_key, "[REDACTED]")
                failures[poem.sample_id] = error
                event = {
                    "status": "failure",
                    "sample_id": poem.sample_id,
                    "error": error,
                }
                status = "failed"
                if tracer is not None:
                    tracer.emit(
                        {
                            "event": "sample_failure",
                            "sample_id": poem.sample_id,
                            "error_type": type(exc).__name__,
                            "error": error,
                            "origin": "generated_in_this_run",
                        }
                    )
            append_checkpoint(checkpoint, event)
            if status == "ok":
                append_jsonl(sft_jsonl, record)
            with PRINT_LOCK:
                print(f"[{completed}/{len(poems)}] {poem.sample_id[:12]} {status}")

    write_outputs(
        run_settings.output_dir,
        poems,
        successes,
        failures,
        settings,
        source_fingerprint=source_fingerprint,
        trace_run_id=tracer.run_id if tracer is not None else None,
    )
    unresolved = [poem for poem in poems if poem.sample_id not in successes]
    if tracer is not None:
        tracer.emit(
            {
                "event": "run_summary",
                "selected_poems": len(poems),
                "checkpoint_reused": len(poems) - len(pending),
                "generated_successes": sum(
                    poem.sample_id in successes for poem in pending
                ),
                "unresolved_failures": len(unresolved),
                "complete": not unresolved,
                "template_distribution": dict(
                    Counter(record["template_id"] for record in successes.values())
                ),
                "validation_status_distribution": dict(
                    Counter(
                        record["validation_status"] for record in successes.values()
                    )
                ),
            }
        )
    return 1 if unresolved else 0
