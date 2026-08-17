"""Command-line interface for synthetic Arabic poetry SFT generation."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import sys

from .config import DEFAULT_MAX_COUPLETS, load_run_settings
from .errors import GemmaConnectionError
from .runner import run
from .tasks.base import TASK_POEM_GENERATION, TASK_TYPES


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
    parser.add_argument(
        "--task",
        choices=TASK_TYPES,
        default=TASK_POEM_GENERATION,
        help="SFT workflow to generate (default: poem-generation)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="task-specific default is used when omitted",
    )
    parser.add_argument("--insecure", action="store_true")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-network-retries", type=int, default=3)
    parser.add_argument("--max-repairs", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--min-chars", type=int, default=1500)
    parser.add_argument("--max-source-chars", type=int, default=24000)
    parser.add_argument("--chunk-chars", type=int, default=12000)
    parser.add_argument(
        "--max-couplets",
        type=int,
        default=DEFAULT_MAX_COUPLETS,
        help=(
            "exclude poems longer than this many couplets before --limit is "
            f"applied (default: {DEFAULT_MAX_COUPLETS})"
        ),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--capacity-report",
        type=Path,
        help="Certified endpoint_capacity.json required for multi-endpoint runs",
    )
    parser.add_argument(
        "--pilot-report",
        type=Path,
        help="Passing pilot_report.json required for multi-endpoint full runs",
    )
    parser.add_argument(
        "--pilot-review",
        type=Path,
        help="Completed pilot_review.json required for multi-endpoint full runs",
    )
    parser.add_argument(
        "--skip-pilot-review",
        action="store_false",
        dest="enforce_pilot_gate",
        help="bypass both pilot artifacts (unsafe; emits a warning)",
    )
    parser.add_argument(
        "--per-sample-chunk-cap",
        type=int,
        default=4,
        help="Administrative maximum for concurrent chunks belonging to one poem",
    )
    parser.set_defaults(enforce_pilot_gate=True)
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
        args = build_parser().parse_args()
        if not args.enforce_pilot_gate:
            print(
                "\033[33mWARNING: capacity certification, pilot report, and human "
                "review checks are being skipped. Full generation will use the "
                "configured endpoint concurrency limits.\033[0m",
                file=sys.stderr,
            )
        code = run(load_run_settings(args))
    except (GemmaConnectionError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    raise SystemExit(code)
