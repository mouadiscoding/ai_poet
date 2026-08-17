"""Publish live and final synthetic dataset artifacts."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from .config import GenerationSettings
from .errors import classify_generation_failure
from .poems import PoemRecord
from .prompts.templates import TEMPLATE_VERSION
from .tasks.base import TASK_POEM_GENERATION


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """Replace a UTF-8 JSONL file with serialized records.

    Records are consumed lazily in iteration order. Each is written on exactly
    one line with non-ASCII text preserved, making the result suitable for both
    human inspection and streaming dataset readers.

    Args:
        path: Existing or new file to overwrite. Its parent must already exist.
        records: Iterable of JSON-serializable dictionaries.

    Raises:
        OSError: If the destination cannot be opened or written.
        TypeError: If a record is not JSON serializable.
    """
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append and immediately flush one UTF-8 JSONL record."""
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def write_outputs(
    output_dir: Path,
    poems: Sequence[PoemRecord],
    successes: dict[str, dict[str, Any]],
    failures: dict[str, str],
    settings: GenerationSettings,
    source_fingerprint: str | None = None,
    trace_run_id: str | None = None,
    capacity_report_fingerprint: str | None = None,
    pilot_report_fingerprint: str | None = None,
    pilot_review_fingerprint: str | None = None,
    endpoint_metrics: dict[str, Any] | None = None,
    max_couplets: int | None = None,
    excluded_long_poems: int = 0,
    task_type: str = TASK_POEM_GENERATION,
    task_version: int = TEMPLATE_VERSION,
) -> None:
    """Materialize generated records, failures, and a run manifest.

    Successful records are ordered to match the canonical poem sequence and
    written to JSONL; when at least one exists, the same records are also
    written as a PyArrow Parquet table. Failures are sorted by sample ID and
    exclude any sample that also has a success. The manifest reports completion
    state, source and generated counts, model/request settings, optional source
    digest, split and template distributions, and oversize/metadata-conflict
    totals. The API key and retry controls are intentionally not exported.

    Args:
        output_dir: Directory in which dataset and manifest files are created.
        poems: Full selected poem sequence, used for output ordering and
            completeness calculations.
        successes: Generated SFT records keyed by sample ID.
        failures: Latest error text keyed by sample ID.
        settings: Generation configuration to describe in the manifest.
        source_fingerprint: Optional SHA-256 digest of the source dataset.
        trace_run_id: Optional audit run identifier recorded in the manifest.
        max_couplets: Optional poem-size selection cap used for this run.
        excluded_long_poems: Number of poems removed by that cap.

    Raises:
        OSError: If the output directory or any output file cannot be written.
        TypeError: If records or manifest values cannot be serialized.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered = [successes[poem.sample_id] for poem in poems if poem.sample_id in successes]
    write_jsonl(output_dir / "ashaar_sft.jsonl", ordered)
    if ordered:
        pq.write_table(pa.Table.from_pylist(ordered), output_dir / "ashaar_sft.parquet")
    failure_records = [
        {
            "sample_id": sample_id,
            "task_type": task_type,
            "category": classify_generation_failure(error),
            "error": error,
        }
        for sample_id, error in sorted(failures.items())
        if sample_id not in successes
    ]
    write_jsonl(output_dir / "failures.jsonl", failure_records)
    manifest = {
        "task_type": task_type,
        "task_version": task_version,
        "complete": len(ordered) == len(poems) and not failure_records,
        "source_poems": len(poems),
        "generated_poems": len(ordered),
        "unresolved_failures": len(failure_records),
        "failure_categories": dict(
            Counter(record["category"] for record in failure_records)
        ),
        "selection": {
            "max_couplets": max_couplets,
            "excluded_long_poems": excluded_long_poems,
        },
        "model": settings.model,
        "endpoint": settings.endpoint,
        "endpoints": [
            {
                "endpoint_id": endpoint.endpoint_id,
                "endpoint": endpoint.endpoint,
                "model": endpoint.model or settings.model,
            }
            for endpoint in settings.configured_endpoints
        ],
        "source_sha256": source_fingerprint,
        "template_version": (
            TEMPLATE_VERSION if task_type == TASK_POEM_GENERATION else None
        ),
        "generation": {
            "temperature": settings.temperature,
            "top_p": settings.top_p,
            "max_tokens": settings.max_tokens,
            "min_chars": settings.min_chars,
            "max_source_chars": settings.max_source_chars,
            "chunk_chars": settings.chunk_chars,
        },
        "trace": {
            "enabled": trace_run_id is not None,
            "run_id": trace_run_id,
            "file": "generation_trace.jsonl" if trace_run_id is not None else None,
        },
        "capacity_report_fingerprint": capacity_report_fingerprint,
        "pilot_report_fingerprint": pilot_report_fingerprint,
        "pilot_review_fingerprint": pilot_review_fingerprint,
        "endpoint_metrics": endpoint_metrics,
        "repaired_samples": sum(
            record.get("validation_status") == "passed_after_repair"
            for record in ordered
        ),
        "truncated_completions": sum(
            int(record.get("truncated_completions", 0)) for record in ordered
        ),
        "splits": dict(Counter(record["sft_split"] for record in ordered)),
        "templates": dict(
            Counter(
                record["template_id"]
                for record in ordered
                if "template_id" in record
            )
        ),
        "question_domains": dict(
            Counter(
                record["question_domain"]
                for record in ordered
                if "question_domain" in record
            )
        ),
        "corruption_counts": dict(
            Counter(
                str(record["corruption_count"])
                for record in ordered
                if "corruption_count" in record
            )
        ),
        "oversized_for_sft": sum(
            bool(record.get("oversized_for_sft", False)) for record in ordered
        ),
        "metadata_conflicts": sum(
            bool(record.get("metadata_conflict", False)) for record in ordered
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
