"""Configuration models and environment-backed settings loading."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


DEFAULT_MAX_COUPLETS = 24


@dataclass(frozen=True)
class EndpointSettings:
    """Connection and administrative limits for one serving endpoint."""

    endpoint_id: str
    endpoint: str
    api_key: str
    max_concurrency: int
    model: str | None = None
    metrics_url: str | None = None


@dataclass(frozen=True)
class GenerationSettings:
    endpoint: str
    model: str
    api_key: str
    insecure: bool
    timeout: float
    max_network_retries: int
    max_repairs: int
    temperature: float
    top_p: float
    max_tokens: int
    min_chars: int
    max_source_chars: int
    chunk_chars: int
    endpoints: tuple[EndpointSettings, ...] = ()

    @property
    def configured_endpoints(self) -> tuple[EndpointSettings, ...]:
        """Return endpoint records, including a compatible legacy fallback."""
        if self.endpoints:
            return self.endpoints
        return (
            EndpointSettings(
                endpoint_id="legacy",
                endpoint=self.endpoint,
                api_key=self.api_key,
                max_concurrency=1,
                model=self.model,
            ),
        )

    @property
    def is_multi_endpoint(self) -> bool:
        return len(self.configured_endpoints) > 1

    @property
    def secrets(self) -> tuple[str, ...]:
        return tuple(endpoint.api_key for endpoint in self.configured_endpoints)


@dataclass(frozen=True)
class RunSettings:
    """Validated corpus-run settings, independent of the CLI parser."""

    input: Path
    output_dir: Path
    concurrency: int
    limit: int | None
    trace: bool
    generation: GenerationSettings
    capacity_report: Path | None = None
    pilot_report: Path | None = None
    pilot_review: Path | None = None
    per_sample_chunk_cap: int = 4
    enforce_pilot_gate: bool = True
    selected_sample_ids: frozenset[str] | None = None
    max_couplets: int | None = DEFAULT_MAX_COUPLETS


def _load_endpoints(global_model: str | None) -> tuple[EndpointSettings, ...]:
    """Load either exactly three indexed endpoints or one legacy endpoint."""
    indexed_names = tuple(
        name
        for index in range(1, 4)
        for name in (
            f"GEMMA_ENDPOINT_{index}",
            f"GEMMA_MODEL_{index}",
            f"GEMMA_API_KEY_{index}",
            f"GEMMA_MAX_CONCURRENCY_{index}",
            f"GEMMA_METRICS_URL_{index}",
        )
    )
    indexed_present = any(os.environ.get(name) is not None for name in indexed_names)
    legacy_present = bool(
        os.environ.get("GEMMA_ENDPOINT") or os.environ.get("GEMMA_API_KEY")
    )
    if indexed_present:
        if legacy_present:
            raise ValueError(
                "Do not mix GEMMA_ENDPOINT/GEMMA_API_KEY with indexed endpoint settings"
            )
        endpoints: list[EndpointSettings] = []
        indexed_models_present = any(
            os.environ.get(f"GEMMA_MODEL_{index}") is not None
            for index in range(1, 4)
        )
        if indexed_models_present and global_model:
            raise ValueError(
                "Do not mix GEMMA_MODEL with indexed GEMMA_MODEL_1..3 settings"
            )
        if not indexed_models_present and not global_model:
            raise ValueError(
                "Set GEMMA_MODEL_1..3 or one shared GEMMA_MODEL in multi-endpoint mode"
            )
        for index in range(1, 4):
            endpoint_name = f"GEMMA_ENDPOINT_{index}"
            key_name = f"GEMMA_API_KEY_{index}"
            endpoint = os.environ.get(endpoint_name)
            api_key = os.environ.get(key_name)
            model_name = f"GEMMA_MODEL_{index}"
            model = os.environ.get(model_name) if indexed_models_present else global_model
            if not endpoint:
                raise ValueError(f"{endpoint_name} must be set for multi-endpoint mode")
            if not api_key:
                raise ValueError(f"{key_name} must be set for multi-endpoint mode")
            if not model:
                raise ValueError(f"{model_name} must be set for multi-endpoint mode")
            concurrency_name = f"GEMMA_MAX_CONCURRENCY_{index}"
            raw_concurrency = os.environ.get(concurrency_name, "32")
            try:
                max_concurrency = int(raw_concurrency)
            except ValueError as exc:
                raise ValueError(f"{concurrency_name} must be an integer") from exc
            if max_concurrency < 1:
                raise ValueError(f"{concurrency_name} must be at least 1")
            endpoints.append(
                EndpointSettings(
                    endpoint_id=f"endpoint_{index}",
                    endpoint=endpoint,
                    api_key=api_key,
                    max_concurrency=max_concurrency,
                    model=model,
                    metrics_url=os.environ.get(f"GEMMA_METRICS_URL_{index}") or None,
                )
            )
        return tuple(endpoints)

    endpoint = os.environ.get("GEMMA_ENDPOINT")
    api_key = os.environ.get("GEMMA_API_KEY")
    if not endpoint:
        raise ValueError(
            "GEMMA_ENDPOINT must be set in .env or the process environment"
        )
    if not api_key:
        raise ValueError(
            "GEMMA_API_KEY must be set in .env or the process environment"
        )
    if not global_model:
        raise ValueError(
            "GEMMA_MODEL must be set in .env or the process environment"
        )
    return (
        EndpointSettings(
            endpoint_id="legacy",
            endpoint=endpoint,
            api_key=api_key,
            max_concurrency=1,
            model=global_model,
        ),
    )


def load_generation_settings(args: Namespace) -> GenerationSettings:
    """Convert parsed CLI arguments into validated generation settings.

    The endpoint, model, and API key are loaded from ``.env`` so they do not
    need to appear in source code or on the command line. Existing
    process-environment values take precedence over values in the file.
    Remaining CLI values are copied into an immutable
    :class:`GenerationSettings` instance.

    Args:
        args: Namespace produced by :func:`build_parser`.

    Returns:
        Immutable settings for the client and generation pipeline.

    Raises:
        ValueError: If the selected legacy or indexed endpoint configuration is
            incomplete or mixed, or if the retry count is negative.
        AttributeError: If the namespace lacks an expected CLI attribute.
    """
    load_dotenv()
    global_model = os.environ.get("GEMMA_MODEL")
    endpoints = _load_endpoints(global_model)
    endpoint = endpoints[0].endpoint
    model = endpoints[0].model
    if model is None:  # Enforced by _load_endpoints; narrows the type here.
        raise AssertionError("configured endpoint has no model")
    api_key = endpoints[0].api_key
    if args.max_network_retries < 0:
        raise ValueError("--max-network-retries cannot be negative")
    return GenerationSettings(
        endpoint=endpoint,
        model=model,
        api_key=api_key,
        insecure=args.insecure,
        timeout=args.timeout,
        max_network_retries=args.max_network_retries,
        max_repairs=args.max_repairs,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        min_chars=args.min_chars,
        max_source_chars=args.max_source_chars,
        chunk_chars=args.chunk_chars,
        endpoints=endpoints,
    )


def load_run_settings(args: Namespace) -> RunSettings:
    """Convert a CLI namespace into validated runner configuration."""
    generation = load_generation_settings(args)
    requested_concurrency = getattr(args, "concurrency", None)
    if requested_concurrency is not None and requested_concurrency < 1:
        raise ValueError("--concurrency must be at least 1")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1")
    max_couplets = getattr(args, "max_couplets", DEFAULT_MAX_COUPLETS)
    if max_couplets < 1:
        raise ValueError("--max-couplets must be at least 1")
    per_sample_chunk_cap = getattr(args, "per_sample_chunk_cap", 4)
    if per_sample_chunk_cap < 1:
        raise ValueError("--per-sample-chunk-cap must be at least 1")
    capacity_report = getattr(args, "capacity_report", None)
    pilot_report = getattr(args, "pilot_report", None)
    pilot_review = getattr(args, "pilot_review", None)
    enforce_pilot_gate = getattr(args, "enforce_pilot_gate", True)
    if generation.is_multi_endpoint:
        if enforce_pilot_gate and capacity_report is None:
            raise ValueError("--capacity-report is required in multi-endpoint mode")
        if enforce_pilot_gate and (pilot_report is None or pilot_review is None):
            raise ValueError(
                "--pilot-report and --pilot-review are required in multi-endpoint mode"
            )
    return RunSettings(
        input=args.input,
        output_dir=args.output_dir,
        concurrency=requested_concurrency or 4,
        limit=args.limit,
        trace=args.trace,
        generation=generation,
        capacity_report=capacity_report,
        pilot_report=pilot_report,
        pilot_review=pilot_review,
        per_sample_chunk_cap=per_sample_chunk_cap,
        enforce_pilot_gate=enforce_pilot_gate,
        max_couplets=max_couplets,
    )
