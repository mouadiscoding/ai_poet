"""CLI for the strict, checkpoint-reusable three-endpoint SFT pilot."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import sys

from .config import load_generation_settings
from .errors import GemmaConnectionError
from .pilot import PilotSettings, run_pilot
from .tasks.base import (
    TASK_POEM_GENERATION,
    TASK_TYPES,
    default_output_dir,
)


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/ashaar_classic_moroccan.parquet"),
    )
    parser.add_argument(
        "--task", choices=TASK_TYPES, default=TASK_POEM_GENERATION
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--capacity-report", type=Path, required=True)
    parser.add_argument("--per-sample-chunk-cap", type=int, default=4)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--insecure", action="store_true")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-network-retries", type=int, default=3)
    parser.add_argument("--max-repairs", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--min-chars", type=int, default=1500)
    parser.add_argument("--max-source-chars", type=int, default=24000)
    parser.add_argument("--chunk-chars", type=int, default=12000)
    return parser


def main() -> None:
    try:
        args = build_parser().parse_args()
        if args.per_sample_chunk_cap < 1:
            raise ValueError("--per-sample-chunk-cap must be at least 1")
        code = run_pilot(
            PilotSettings(
                input=args.input,
                output_dir=args.output_dir or default_output_dir(args.task),
                capacity_report=args.capacity_report,
                generation=load_generation_settings(args),
                trace=args.trace,
                per_sample_chunk_cap=args.per_sample_chunk_cap,
                task_type=args.task,
            )
        )
    except (GemmaConnectionError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    raise SystemExit(code)


if __name__ == "__main__":
    main()
