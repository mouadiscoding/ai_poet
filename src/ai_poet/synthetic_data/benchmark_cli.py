"""CLI for measuring and certifying Gemma endpoint concurrency."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import sys

from .benchmark import BenchmarkSettings, DEFAULT_LEVELS, run_benchmark
from .config import load_generation_settings
from .tasks.base import TASK_POEM_GENERATION, TASK_TYPES


def _levels(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(dict.fromkeys(int(item) for item in value.split(",")))
    except ValueError as exc:
        raise ValueError("--concurrency-levels must be comma-separated integers") from exc
    if not parsed or any(level < 1 for level in parsed):
        raise ValueError("--concurrency-levels values must be positive")
    return parsed


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
    parser.add_argument(
        "--output-dir", type=Path
    )
    parser.add_argument("--duration-per-level", type=float, default=300.0)
    parser.add_argument("--warmup-seconds", type=float, default=30.0)
    parser.add_argument(
        "--concurrency-levels",
        type=_levels,
        default=DEFAULT_LEVELS,
    )
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
        settings = BenchmarkSettings(
            input=args.input,
            output_dir=(
                args.output_dir
                or (
                    Path("data/gemma_capacity")
                    if args.task == TASK_POEM_GENERATION
                    else Path(f"data/gemma_capacity_{args.task.replace('-', '_')}")
                )
            ),
            generation=load_generation_settings(args),
            task_type=args.task,
            duration_per_level=args.duration_per_level,
            warmup_seconds=args.warmup_seconds,
            concurrency_levels=args.concurrency_levels,
        )
        code = run_benchmark(settings)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    raise SystemExit(code)


if __name__ == "__main__":
    main()
