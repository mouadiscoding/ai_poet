"""Configuration models and environment-backed settings loading."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


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


@dataclass(frozen=True)
class RunSettings:
    """Validated corpus-run settings, independent of the CLI parser."""

    input: Path
    output_dir: Path
    concurrency: int
    limit: int | None
    trace: bool
    generation: GenerationSettings


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
        ValueError: If ``GEMMA_ENDPOINT``, ``GEMMA_MODEL``, or
            ``GEMMA_API_KEY`` is empty or missing, or if the retry count is
            negative.
        AttributeError: If the namespace lacks an expected CLI attribute.
    """
    load_dotenv()
    endpoint = os.environ.get("GEMMA_ENDPOINT")
    if not endpoint:
        raise ValueError(
            "GEMMA_ENDPOINT must be set in .env or the process environment"
        )
    model = os.environ.get("GEMMA_MODEL")
    if not model:
        raise ValueError(
            "GEMMA_MODEL must be set in .env or the process environment"
        )
    api_key = os.environ.get("GEMMA_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMMA_API_KEY must be set in .env or the process environment"
        )
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
    )


def load_run_settings(args: Namespace) -> RunSettings:
    """Convert a CLI namespace into validated runner configuration."""
    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1")
    return RunSettings(
        input=args.input,
        output_dir=args.output_dir,
        concurrency=args.concurrency,
        limit=args.limit,
        trace=args.trace,
        generation=load_generation_settings(args),
    )
