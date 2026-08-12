"""Capacity-report fingerprints and production gate validation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from .config import GenerationSettings
from .prompts.templates import TEMPLATE_VERSION


REPORT_VERSION = 1
PILOT_REPORT_VERSION = 1


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def generation_contract(settings: GenerationSettings) -> dict[str, Any]:
    """Return the secret-free settings that determine request compatibility."""
    return {
        "model": settings.model,
        "endpoints": [
            {
                "endpoint_id": endpoint.endpoint_id,
                "endpoint": endpoint.endpoint,
                "model": endpoint.model or settings.model,
            }
            for endpoint in settings.configured_endpoints
        ],
        "template_version": TEMPLATE_VERSION,
        "temperature": settings.temperature,
        "top_p": settings.top_p,
        "max_tokens": settings.max_tokens,
        "min_chars": settings.min_chars,
        "max_source_chars": settings.max_source_chars,
        "chunk_chars": settings.chunk_chars,
    }


def generation_fingerprint(settings: GenerationSettings) -> str:
    return _canonical_digest(generation_contract(settings))


def workflow_fingerprint(
    settings: GenerationSettings,
    source_sha256: str,
) -> str:
    """Fingerprint accepted stages against settings and exact source bytes."""
    return _canonical_digest(
        {
            "generation_fingerprint": generation_fingerprint(settings),
            "source_sha256": source_sha256,
        }
    )


@dataclass(frozen=True)
class CapacityPlan:
    """Validated endpoint ceilings and latency baselines for one run."""

    report_path: Path
    report_fingerprint: str
    hard_caps: dict[str, int]
    latency_baselines: dict[str, dict[str, float]]
    raw: dict[str, Any]

    @property
    def total_capacity(self) -> int:
        return sum(self.hard_caps.values())


def configured_capacity_plan(settings: GenerationSettings) -> CapacityPlan:
    """Build an uncertified plan from configured endpoint ceilings."""
    return CapacityPlan(
        report_path=Path("<configured-endpoint-limits>"),
        report_fingerprint="",
        hard_caps={
            endpoint.endpoint_id: endpoint.max_concurrency
            for endpoint in settings.configured_endpoints
        },
        latency_baselines={},
        raw={"certified": False, "source": "configured_endpoint_limits"},
    )


def load_capacity_report(
    path: Path,
    settings: GenerationSettings,
    *,
    source_sha256: str | None = None,
) -> CapacityPlan:
    """Load and strictly validate a certified endpoint-capacity report."""
    try:
        raw_bytes = path.read_bytes()
        report = json.loads(raw_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read capacity report {path}: {exc}") from exc
    if report.get("report_version") != REPORT_VERSION:
        raise ValueError("Capacity report version is unsupported")
    if report.get("certified") is not True:
        raise ValueError("Capacity report is not certified")
    expected_fingerprint = generation_fingerprint(settings)
    if report.get("generation_fingerprint") != expected_fingerprint:
        raise ValueError("Capacity report does not match current generation settings")
    if source_sha256 is not None and report.get("source_sha256") != source_sha256:
        raise ValueError("Capacity report does not match the input dataset")

    endpoint_reports = report.get("endpoints")
    if not isinstance(endpoint_reports, list):
        raise ValueError("Capacity report endpoints must be a list")
    configured = {
        endpoint.endpoint_id: endpoint for endpoint in settings.configured_endpoints
    }
    hard_caps: dict[str, int] = {}
    baselines: dict[str, dict[str, float]] = {}
    for item in endpoint_reports:
        if not isinstance(item, dict):
            raise ValueError("Capacity report contains an invalid endpoint record")
        endpoint_id = str(item.get("endpoint_id", ""))
        endpoint = configured.get(endpoint_id)
        if endpoint is None or item.get("endpoint") != endpoint.endpoint:
            raise ValueError(f"Capacity report endpoint mismatch for {endpoint_id}")
        if item.get("model") != (endpoint.model or settings.model):
            raise ValueError(f"Capacity report model mismatch for {endpoint_id}")
        selected = item.get("selected_concurrency")
        if not isinstance(selected, int) or not 1 <= selected <= endpoint.max_concurrency:
            raise ValueError(f"Invalid selected concurrency for {endpoint_id}")
        raw_baselines = item.get("latency_baselines", {})
        if not isinstance(raw_baselines, dict):
            raise ValueError(f"Invalid latency baselines for {endpoint_id}")
        hard_caps[endpoint_id] = selected
        baselines[endpoint_id] = {
            str(kind): float(value)
            for kind, value in raw_baselines.items()
            if isinstance(value, (int, float)) and value > 0
        }
    if set(hard_caps) != set(configured):
        raise ValueError("Capacity report must contain every configured endpoint once")
    return CapacityPlan(
        report_path=path,
        report_fingerprint=hashlib.sha256(raw_bytes).hexdigest(),
        hard_caps=hard_caps,
        latency_baselines=baselines,
        raw=report,
    )


def validate_pilot_gate(
    report_path: Path,
    review_path: Path,
    *,
    settings: GenerationSettings,
    source_sha256: str,
    capacity_fingerprint: str,
) -> tuple[str, str]:
    """Require a compatible passing pilot and thirty explicit approvals."""
    try:
        report_bytes = report_path.read_bytes()
        report = json.loads(report_bytes.decode("utf-8"))
        review_bytes = review_path.read_bytes()
        review = json.loads(review_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read pilot gate artifacts: {exc}") from exc

    if report.get("report_version") != PILOT_REPORT_VERSION:
        raise ValueError("Pilot report version is unsupported")
    if report.get("passed") is not True:
        raise ValueError("Pilot automatic gate did not pass")
    if report.get("source_sha256") != source_sha256:
        raise ValueError("Pilot report does not match the input dataset")
    if report.get("generation_fingerprint") != generation_fingerprint(settings):
        raise ValueError("Pilot report does not match current generation settings")
    if report.get("capacity_report_fingerprint") != capacity_fingerprint:
        raise ValueError("Pilot report does not match the selected capacity report")

    report_fingerprint = hashlib.sha256(report_bytes).hexdigest()
    if review.get("pilot_report_fingerprint") != report_fingerprint:
        raise ValueError("Pilot review does not match the pilot report")
    content_fingerprint = report.get("content_fingerprint")
    if not isinstance(content_fingerprint, str) or not content_fingerprint:
        raise ValueError("Pilot report lacks a content fingerprint")
    if review.get("pilot_content_fingerprint") != content_fingerprint:
        raise ValueError("Pilot review content fingerprint is stale")
    reviews = review.get("reviews")
    if not isinstance(reviews, list) or len(reviews) != 30:
        raise ValueError("Pilot review must contain exactly 30 records")
    sample_ids = [item.get("sample_id") for item in reviews if isinstance(item, dict)]
    if len(sample_ids) != 30 or len(set(sample_ids)) != 30:
        raise ValueError("Pilot review sample IDs must be unique")
    if any(item.get("approved") is not True for item in reviews):
        raise ValueError("Every pilot review record must have approved=true")
    return report_fingerprint, hashlib.sha256(review_bytes).hexdigest()
