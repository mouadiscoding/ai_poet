"""Command-line interface for synthetic Arabic poetry SFT generation."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import sys

from .config import load_run_settings
from .errors import GemmaConnectionError
from .runner import run


def build_parser() -> ArgumentParser:
    """Construct the command-line interface for dataset generation.

    The parser exposes source/output paths, TLS verification opt-out, worker
    concurrency, network and repair limits, sampling settings, prompt size
    thresholds, an optional poem limit, and full audit tracing. Defaults
    describe the standard local Ashaar generation run, but no arguments are
    parsed by this function.

    Returns:
        A configured :class:`argparse.ArgumentParser` ready to parse CLI input.
    """
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/ashaar_classic_moroccan.parquet"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/ashaar_sft"))
    parser.add_argument("--insecure", action="store_true")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-network-retries", type=int, default=3)
    parser.add_argument("--max-repairs", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--min-chars", type=int, default=1500)
    parser.add_argument("--max-source-chars", type=int, default=24000)
    parser.add_argument("--chunk-chars", type=int, default=12000)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--trace",
        action="store_true",
        help=(
            "print full generation audit events and append them to "
            "OUTPUT_DIR/generation_trace.jsonl"
        ),
    )
    return parser



def main() -> None:
    """Parse CLI arguments, run generation, and terminate with a shell status.

    Normal completion exits with the status returned by :func:`run`: zero for a
    complete corpus and one for unresolved per-poem failures. Top-level
    Gemma connection failures, ``OSError``, and ``ValueError`` exceptions are
    rendered as concise messages on standard error and converted to exit status
    two. Other unexpected exceptions are allowed to propagate with their
    traceback.

    Raises:
        SystemExit: Always, with status zero, one, or two as described above.
    """
    try:
        code = run(load_run_settings(build_parser().parse_args()))
    except (GemmaConnectionError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    raise SystemExit(code)
