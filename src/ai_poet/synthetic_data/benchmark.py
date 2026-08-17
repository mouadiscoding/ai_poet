"""Measure and certify sustainable concurrency for configured Gemma endpoints."""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import ssl
import threading
import time
from typing import Any, Sequence
from urllib.request import Request, urlopen

from .capacity import REPORT_VERSION, generation_fingerprint
from .client import EndpointClient, EndpointRequestError
from .config import EndpointSettings, GenerationSettings
from .corpus import load_poems
from .generation import _repair_messages
from .poems import PoemRecord, split_poem_chunks
from .prompts.builder import (
    build_instruction_validation_messages,
    build_messages,
    build_reasoning_messages,
    build_reasoning_validation_messages,
)
from .prompts.examples import METERED_FEW_SHOTS, METERED_REASONING_FEW_SHOT
from .prompts.templates import PROMPT_TEMPLATES
from .runner import file_sha256
from .tasks.base import (
    TASK_MCQ,
    TASK_POEM_GENERATION,
    get_task_workflow,
)
from .tasks.mcq import (
    MCQWorkItem,
    build_generation_messages as build_mcq_messages,
    build_validation_messages as build_mcq_validation_messages,
    build_work_items as build_mcq_work_items,
)
from .tasks.reconstruction import (
    build_generation_messages as build_reconstruction_messages,
    build_validation_messages as build_reconstruction_validation_messages,
    corruption_count,
)


DEFAULT_LEVELS = (1, 2, 4, 8, 16, 24, 32)


@dataclass(frozen=True)
class BenchmarkSettings:
    input: Path
    output_dir: Path
    generation: GenerationSettings
    task_type: str = TASK_POEM_GENERATION
    duration_per_level: float = 300.0
    warmup_seconds: float = 30.0
    concurrency_levels: tuple[int, ...] = DEFAULT_LEVELS


@dataclass(frozen=True)
class BenchmarkFixture:
    request_kind: str
    messages: tuple[dict[str, str], ...]
    max_tokens: int | None
    temperature: float | None
    seed: int


def _progress(message: str) -> None:
    print(f"[benchmark] {message}", flush=True)


def _result_summary(result: dict[str, Any]) -> str:
    errors = result["retryable_errors"] + result["nonretryable_errors"]
    return (
        f"{result['successes']}/{result['requests']} successful, "
        f"{errors} errors, {result['requests_per_second']:.2f} req/s, "
        f"p95 {result['p95_latency_seconds']:.2f}s"
    )


def _stable_poem_choices(poems: Sequence[PoemRecord]) -> list[PoemRecord]:
    buckets: list[list[PoemRecord]] = [[], [], [], [], []]
    for poem in poems:
        count = poem.couplet_count
        index = 0 if count <= 3 else 1 if count <= 9 else 2 if count <= 24 else 3 if count <= 74 else 4
        buckets[index].append(poem)
    selected: list[PoemRecord] = []
    for bucket in buckets:
        selected.extend(sorted(bucket, key=lambda poem: poem.sample_id)[:4])
    if not selected:
        raise ValueError("Benchmark input contains no poems")
    return selected


def _build_poem_generation_fixture_bank(
    poems: Sequence[PoemRecord],
    settings: GenerationSettings,
) -> tuple[list[BenchmarkFixture], list[BenchmarkFixture]]:
    """Build an 80-request production-shaped mix plus oversized probes."""
    selected = _stable_poem_choices(poems)
    example = METERED_FEW_SHOTS[0]
    example_reasoning = METERED_REASONING_FEW_SHOT
    fixtures: list[BenchmarkFixture] = []

    for index in range(8):
        poem = selected[index % len(selected)]
        messages = build_messages(
            template=PROMPT_TEMPLATES[index % len(PROMPT_TEMPLATES)],
            meter_name=poem.meter_name,
            couplet_count=poem.couplet_count,
            poem=poem.poem_text,
            minimum_chars=settings.min_chars,
        )
        if index < 2:
            messages = _repair_messages(
                messages,
                '{"instruction":"incomplete"}',
                ["benchmark repair-shaped request"],
            )
        fixtures.append(
            BenchmarkFixture(
                "instruction_generation" if index >= 2 else "instruction_repair",
                tuple(messages),
                settings.max_tokens,
                settings.temperature,
                index + 1,
            )
        )

    for index in range(8):
        poem = selected[index % len(selected)]
        fixtures.append(
            BenchmarkFixture(
                "instruction_validation",
                tuple(
                    build_instruction_validation_messages(
                        instruction=example.instruction,
                        meter_name=poem.meter_name,
                        couplet_count=poem.couplet_count,
                        poem=poem.poem_text,
                        minimum_chars=settings.min_chars,
                    )
                ),
                1200,
                0.0,
                10_000 + index,
            )
        )

    for index in range(32):
        poem = selected[index % len(selected)]
        couplets = poem.poem_text.splitlines()
        start_offset = (index * 3) % len(couplets)
        chunk = couplets[start_offset : start_offset + 3]
        messages = build_reasoning_messages(
            instruction=example.instruction,
            meter_name=poem.meter_name,
            total_couplet_count=poem.couplet_count,
            start_index=start_offset + 1,
            couplets=chunk,
            previous_couplet=couplets[start_offset - 1] if start_offset else None,
            next_couplet=(
                couplets[start_offset + len(chunk)]
                if start_offset + len(chunk) < len(couplets)
                else None
            ),
            include_overview=start_offset == 0,
        )
        repaired = index < 8
        if repaired:
            messages = _repair_messages(
                messages,
                '{"verse_reasoning":[]}',
                ["benchmark repair-shaped request"],
            )
        fixtures.append(
            BenchmarkFixture(
                "reasoning_repair" if repaired else "reasoning_generation",
                tuple(messages),
                settings.max_tokens,
                settings.temperature,
                20_000 + index,
            )
        )

    validation_targets = [
        block["revised_draft"]
        for block in example_reasoning["response"]["verse_reasoning"]
    ]
    validation_messages = tuple(
        build_reasoning_validation_messages(
            instruction=example_reasoning["instruction"],
            meter_name=example_reasoning["meter_name"],
            expected_couplets=validation_targets,
            candidate=example_reasoning["response"],
        )
    )
    fixtures.extend(
        BenchmarkFixture(
            "reasoning_validation",
            validation_messages,
            1200,
            0.0,
            30_000 + index,
        )
        for index in range(32)
    )

    oversized = sorted(poems, key=lambda poem: (-len(poem.poem_text), poem.sample_id))
    probes: list[BenchmarkFixture] = []
    for index, poem in enumerate(oversized[:4], start=1):
        chunks = split_poem_chunks(poem.verses, settings.chunk_chars)
        messages = (
            {
                "role": "system",
                "content": "Summarize the supplied Arabic poem segment as compact JSON.",
            },
            {
                "role": "user",
                "content": f"meter={poem.meter_name}\nsegment={chunks[0]}",
            },
        )
        probes.append(
            BenchmarkFixture("chunk_analysis", messages, 800, 0.2, 40_000 + index)
        )
    return fixtures, probes


def _example_mcq_candidate(item: MCQWorkItem) -> dict[str, Any]:
    answers = [item.ground_truth, "إجابة بديلة أولى", "إجابة بديلة ثانية", "إجابة بديلة ثالثة"]
    return {
        "question": item.prompt.question,
        "correct_answer": answers[0],
        "distractors": answers[1:],
        "reasoning": {
            "approach": "أربط صورة الليل بما يليها من صبر وضياء لأحدد وظيفتها الدلالية.",
            "evidence": ["اقتران الليل بالضياء اللاحق"],
            "answer_assessments": [
                {
                    "answer": answer,
                    "assessment": "هذا تقدير مفصل لعلاقة الخيار بالمعنى الظاهر في سياق القصيدة."
                }
                for answer in answers
            ],
            "conclusion": "يؤيد تتابع الشدة والانفراج الجواب الأول دون بقية البدائل."
        },
    }


def _example_reconstruction_candidate(poem: PoemRecord) -> dict[str, Any]:
    return {
        "corrupted_poem": poem.poem_text,
        "repairs": [
            {
                "couplet_index": 1,
                "corrupted_fragment": "لفظ محرّف",
                "corrected_fragment": "لفظ سليم",
                "diagnosis": "اللفظ المحرف يقطع المعنى الذي يمهد له سياق البيت.",
                "context_evidence": "تدل الألفاظ المجاورة على معنى مختلف متماسك.",
                "repair_reason": "يعيد اللفظ السليم الصلة الدلالية والجرس المناسب."
            }
        ],
    }


def _build_simple_task_fixture_bank(
    poems: Sequence[PoemRecord],
    settings: GenerationSettings,
    task_type: str,
) -> tuple[list[BenchmarkFixture], list[BenchmarkFixture]]:
    selected = _stable_poem_choices(poems)
    mcq_items = build_mcq_work_items(selected) if task_type == TASK_MCQ else []
    fixtures: list[BenchmarkFixture] = []
    for index in range(40):
        poem = selected[index % len(selected)]
        if task_type == TASK_MCQ:
            item = mcq_items[index % len(mcq_items)]
            messages = build_mcq_messages(item)
            kind = "mcq_generation"
        else:
            messages = build_reconstruction_messages(poem, corruption_count(poem))
            kind = "reconstruction_generation"
        if index < 10:
            messages = _repair_messages(
                messages,
                "{}",
                ["benchmark repair-shaped request"],
            )
            kind = kind.replace("_generation", "_repair")
        fixtures.append(
            BenchmarkFixture(
                kind,
                tuple(messages),
                settings.max_tokens,
                settings.temperature,
                index + 1,
            )
        )

    for index in range(40):
        poem = selected[index % len(selected)]
        if task_type == TASK_MCQ:
            item = mcq_items[index % len(mcq_items)]
            messages = build_mcq_validation_messages(
                item,
                _example_mcq_candidate(item),
            )
            kind = "mcq_validation"
        else:
            messages = build_reconstruction_validation_messages(
                poem,
                _example_reconstruction_candidate(poem),
            )
            kind = "reconstruction_validation"
        fixtures.append(
            BenchmarkFixture(kind, tuple(messages), 1200, 0.0, 10_000 + index)
        )
    return fixtures, []


def build_fixture_bank(
    poems: Sequence[PoemRecord],
    settings: GenerationSettings,
    task_type: str = TASK_POEM_GENERATION,
) -> tuple[list[BenchmarkFixture], list[BenchmarkFixture]]:
    profile = get_task_workflow(task_type).benchmark_profile
    if profile == "poem-generation":
        return _build_poem_generation_fixture_bank(poems, settings)
    if profile == "single-generation-validation":
        return _build_simple_task_fixture_bank(poems, settings, task_type)
    raise ValueError(f"Unsupported benchmark profile: {profile}")


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def run_workload(
    endpoint: EndpointSettings,
    settings: GenerationSettings,
    fixtures: Sequence[BenchmarkFixture],
    *,
    concurrency: int,
    duration_seconds: float,
    warmup_seconds: float,
) -> dict[str, Any]:
    """Run one fixed fixture cycle at a bounded client concurrency."""
    client = EndpointClient(endpoint, settings)
    index = 0
    index_lock = threading.Lock()
    result_lock = threading.Lock()
    results: list[dict[str, Any]] = []

    def next_fixture() -> BenchmarkFixture:
        nonlocal index
        with index_lock:
            fixture = fixtures[index % len(fixtures)]
            index += 1
            return fixture

    def worker(deadline: float, *, measured: bool) -> None:
        while time.perf_counter() < deadline:
            fixture = next_fixture()
            started = time.perf_counter()
            try:
                response = client.chat_once(
                    fixture.messages,
                    max_tokens=fixture.max_tokens,
                    temperature=fixture.temperature,
                    seed=fixture.seed,
                )
            except EndpointRequestError as exc:
                item = {
                    "kind": fixture.request_kind,
                    "success": False,
                    "retryable": exc.retryable,
                    "status": exc.status,
                    "latency": time.perf_counter() - started,
                }
            else:
                item = {
                    "kind": fixture.request_kind,
                    "success": True,
                    "retryable": False,
                    "status": None,
                    "latency": response.elapsed_seconds,
                    "prompt_tokens": response.usage.get("prompt_tokens", 0),
                    "completion_tokens": response.usage.get("completion_tokens", 0),
                    "finish_reason": response.finish_reason,
                    "response_model": response.payload.get("model"),
                }
            if measured:
                with result_lock:
                    results.append(item)

    if warmup_seconds > 0:
        deadline = time.perf_counter() + warmup_seconds
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(worker, deadline, measured=False) for _ in range(concurrency)]
            for future in futures:
                future.result()
    measured_started = time.perf_counter()
    deadline = measured_started + duration_seconds
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(worker, deadline, measured=True) for _ in range(concurrency)]
        for future in futures:
            future.result()
    measured_seconds = max(0.000001, time.perf_counter() - measured_started)

    successes = [item for item in results if item["success"]]
    failures = [item for item in results if not item["success"]]
    retryable = sum(bool(item["retryable"]) for item in failures)
    nonretryable = len(failures) - retryable
    by_kind: dict[str, list[float]] = defaultdict(list)
    for item in successes:
        by_kind[item["kind"]].append(item["latency"])
    latency_baselines = {
        kind: round(_percentile(values, 0.95), 6)
        for kind, values in by_kind.items()
    }
    total = len(results)
    retryable_rate = retryable / total if total else 1.0
    p95 = _percentile([item["latency"] for item in results], 0.95)
    return {
        "endpoint_id": endpoint.endpoint_id,
        "concurrency": concurrency,
        "duration_seconds": measured_seconds,
        "requests": total,
        "successes": len(successes),
        "retryable_errors": retryable,
        "nonretryable_errors": nonretryable,
        "retryable_error_rate": retryable_rate,
        "http_429": sum(item.get("status") == 429 for item in failures),
        "truncations": sum(
            item.get("finish_reason") == "length" for item in successes
        ),
        "requests_per_second": len(successes) / measured_seconds,
        "prompt_tokens": sum(item.get("prompt_tokens", 0) for item in successes),
        "completion_tokens": sum(
            item.get("completion_tokens", 0) for item in successes
        ),
        "p50_latency_seconds": _percentile(
            [item["latency"] for item in results], 0.5
        ),
        "p95_latency_seconds": p95,
        "latency_baselines": latency_baselines,
        "response_models": sorted(
            {
                str(item["response_model"])
                for item in successes
                if item.get("response_model") is not None
            }
        ),
        "safe": (
            nonretryable == 0
            and retryable_rate <= 0.005
            and p95 < settings.timeout / 2
        ),
    }


def select_capacity(results: Sequence[dict[str, Any]]) -> tuple[int, bool]:
    safe = [result for result in results if result.get("safe")]
    if not safe:
        raise ValueError("Endpoint has no safe benchmark concurrency")
    maximum = max(result["requests_per_second"] for result in safe)
    selected = next(
        result
        for result in sorted(safe, key=lambda item: item["concurrency"])
        if result["requests_per_second"] >= maximum * 0.95
    )
    ordered = sorted(results, key=lambda item: item["concurrency"])
    nonconverged = False
    if len(ordered) >= 2 and ordered[-1].get("safe"):
        previous = ordered[-2]["requests_per_second"]
        gain = (
            (ordered[-1]["requests_per_second"] - previous) / previous
            if previous > 0
            else 1.0
        )
        nonconverged = gain >= 0.05
    return int(selected["concurrency"]), nonconverged


def _benchmark_digest(settings: BenchmarkSettings, source_sha256: str) -> str:
    value = {
        "task_type": settings.task_type,
        "generation_fingerprint": generation_fingerprint(
            settings.generation, settings.task_type
        ),
        "source_sha256": source_sha256,
        "duration_per_level": settings.duration_per_level,
        "warmup_seconds": settings.warmup_seconds,
        "concurrency_levels": settings.concurrency_levels,
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_completed(
    path: Path,
    benchmark_fingerprint: str,
) -> dict[tuple[str, str, int], dict[str, Any]]:
    completed: dict[tuple[str, str, int], dict[str, Any]] = {}
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("benchmark_fingerprint") == benchmark_fingerprint:
                result = event["result"]
                completed[
                    (
                        str(event.get("phase", "isolated")),
                        result["endpoint_id"],
                        result["concurrency"],
                    )
                ] = result
    return completed


def _append_result(
    path: Path,
    fingerprint: str,
    result: dict[str, Any],
    *,
    phase: str,
) -> None:
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "benchmark_fingerprint": fingerprint,
        "phase": phase,
        "result": result,
    }
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        handle.flush()


def _fetch_server_metrics(
    endpoint: EndpointSettings,
    settings: GenerationSettings,
) -> dict[str, float] | None:
    if endpoint.metrics_url is None:
        return None
    request = Request(
        endpoint.metrics_url,
        method="GET",
        headers={"Authorization": f"Bearer {endpoint.api_key}"},
    )
    context = (
        ssl._create_unverified_context()  # noqa: SLF001 - explicit CLI opt-in
        if settings.insecure
        else ssl.create_default_context()
    )
    try:
        with urlopen(request, timeout=10, context=context) as response:  # noqa: S310
            lines = response.read().decode("utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return None
    wanted = (
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:gpu_cache_usage_perc",
        "vllm:prompt_tokens_total",
        "vllm:generation_tokens_total",
    )
    values: dict[str, float] = defaultdict(float)
    for line in lines:
        if not line or line.startswith("#") or " " not in line:
            continue
        name, raw = line.rsplit(" ", 1)
        base = name.split("{", 1)[0]
        if base in wanted:
            try:
                values[base] += float(raw)
            except ValueError:
                continue
    return dict(values)


def run_benchmark(settings: BenchmarkSettings) -> int:
    """Run resumable isolated curves and certify a combined endpoint plan."""
    if not settings.generation.is_multi_endpoint:
        raise ValueError("Endpoint benchmark requires indexed three-endpoint settings")
    if settings.duration_per_level <= 0 or settings.warmup_seconds < 0:
        raise ValueError("Benchmark durations must be positive")
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    poems = load_poems(settings.input)
    source_sha256 = file_sha256(settings.input)
    fixtures, oversized_probes = build_fixture_bank(
        poems, settings.generation, settings.task_type
    )
    fingerprint = _benchmark_digest(settings, source_sha256)
    checkpoint = settings.output_dir / "endpoint_benchmark.jsonl"
    completed = _load_completed(checkpoint, fingerprint)
    endpoint_reports: list[dict[str, Any]] = []
    _progress(
        f"loaded {len(fixtures)} fixtures for "
        f"{len(settings.generation.configured_endpoints)} endpoints; "
        f"checkpoint has {len(completed)} reusable results"
    )

    preflight: dict[str, dict[str, Any]] = {}
    prompt_token_counts: list[int] = []
    for endpoint in settings.generation.configured_endpoints:
        _progress(f"preflight {endpoint.endpoint_id}: contacting {endpoint.endpoint}")
        try:
            result = EndpointClient(endpoint, settings.generation).chat_once(
                fixtures[0].messages,
                max_tokens=fixtures[0].max_tokens,
                temperature=fixtures[0].temperature,
                seed=fixtures[0].seed,
            )
        except EndpointRequestError as exc:
            raise ValueError(
                f"Preflight failed for {endpoint.endpoint_id}: {exc}"
            ) from exc
        response_model = result.payload.get("model")
        expected_model = endpoint.model or settings.generation.model
        prompt_tokens = result.usage.get("prompt_tokens", 0)
        if prompt_tokens:
            prompt_token_counts.append(prompt_tokens)
        preflight[endpoint.endpoint_id] = {
            "expected_model": expected_model,
            "response_model": response_model,
            "model_matches": response_model == expected_model,
            "prompt_tokens": prompt_tokens or None,
            "finish_reason": result.finish_reason,
        }
        _progress(
            f"preflight {endpoint.endpoint_id}: ready "
            f"(model={response_model or 'unknown'}, prompt_tokens={prompt_tokens or 'unknown'})"
        )
    tokenizer_usage_matches: bool | None = (
        len(set(prompt_token_counts)) == 1
        if len(prompt_token_counts) == len(settings.generation.configured_endpoints)
        else None
    )

    for endpoint in settings.generation.configured_endpoints:
        levels = tuple(
            level
            for level in settings.concurrency_levels
            if level <= endpoint.max_concurrency
        )
        if not levels or levels[-1] != endpoint.max_concurrency:
            levels = (*levels, endpoint.max_concurrency)
        results: list[dict[str, Any]] = []
        _progress(
            f"starting isolated tests for {endpoint.endpoint_id} at levels "
            + ",".join(str(level) for level in dict.fromkeys(levels))
        )
        for level in dict.fromkeys(levels):
            result = completed.get(("isolated", endpoint.endpoint_id, level))
            if result is None:
                _progress(
                    f"{endpoint.endpoint_id} concurrency {level}: warming for "
                    f"{settings.warmup_seconds:g}s, then measuring for "
                    f"{settings.duration_per_level:g}s"
                )
                result = run_workload(
                    endpoint,
                    settings.generation,
                    fixtures,
                    concurrency=level,
                    duration_seconds=settings.duration_per_level,
                    warmup_seconds=settings.warmup_seconds,
                )
                _append_result(
                    checkpoint, fingerprint, result, phase="isolated"
                )
                _progress(
                    f"{endpoint.endpoint_id} concurrency {level}: "
                    + _result_summary(result)
                )
            else:
                _progress(
                    f"{endpoint.endpoint_id} concurrency {level}: reused checkpoint "
                    f"({_result_summary(result)})"
                )
            results.append(result)

        _progress(f"{endpoint.endpoint_id}: running oversized request probes")
        probe_results = [
            run_workload(
                endpoint,
                settings.generation,
                [probe],
                concurrency=1,
                duration_seconds=min(1.0, settings.duration_per_level),
                warmup_seconds=0,
            )
            for probe in oversized_probes
        ]
        selected, nonconverged = select_capacity(results)
        selected_result = next(
            result for result in results if result["concurrency"] == selected
        )
        model_mismatch = not preflight[endpoint.endpoint_id]["model_matches"]
        probes_ok = all(
            result["successes"] > 0
            and result["retryable_errors"] == 0
            and result["nonretryable_errors"] == 0
            for result in probe_results
        )
        _progress(
            f"{endpoint.endpoint_id}: selected concurrency {selected}; "
            f"probes {'passed' if probes_ok else 'failed'}; "
            f"curve {'not converged' if nonconverged else 'converged'}"
        )
        endpoint_reports.append(
            {
                "endpoint_id": endpoint.endpoint_id,
                "endpoint": endpoint.endpoint,
                "model": endpoint.model or settings.generation.model,
                "configured_ceiling": endpoint.max_concurrency,
                "selected_concurrency": selected,
                "certified": (
                    not nonconverged
                    and not model_mismatch
                    and probes_ok
                    and tokenizer_usage_matches is not False
                ),
                "nonconverged": nonconverged,
                "model_mismatch": model_mismatch,
                "preflight": preflight[endpoint.endpoint_id],
                "oversized_probes_passed": probes_ok,
                "latency_baselines": selected_result["latency_baselines"],
                "levels": results,
                "oversized_probes": probe_results,
                "server_metrics": _fetch_server_metrics(
                    endpoint, settings.generation
                ),
            }
        )

    isolated_selected = {
        item["endpoint_id"]: next(
            result
            for result in item["levels"]
            if result["concurrency"] == item["selected_concurrency"]
        )
        for item in endpoint_reports
    }
    combined_results: dict[str, dict[str, Any]] = {}
    missing_combined: list[EndpointSettings] = []
    for endpoint in settings.generation.configured_endpoints:
        selected = next(
            item["selected_concurrency"]
            for item in endpoint_reports
            if item["endpoint_id"] == endpoint.endpoint_id
        )
        existing = completed.get(("combined", endpoint.endpoint_id, selected))
        if existing is not None:
            combined_results[endpoint.endpoint_id] = existing
        else:
            missing_combined.append(endpoint)
    if missing_combined:
        # Run all endpoints together even if one prior partial combined result exists;
        # only a complete simultaneous phase is considered certifying.
        combined_results = {}
        _progress(
            "starting simultaneous combined test for all configured endpoints"
        )
        with ThreadPoolExecutor(max_workers=3) as executor:
            combined_futures = {
                endpoint.endpoint_id: executor.submit(
                    run_workload,
                    endpoint,
                    settings.generation,
                    fixtures,
                    concurrency=next(
                        item["selected_concurrency"]
                        for item in endpoint_reports
                        if item["endpoint_id"] == endpoint.endpoint_id
                    ),
                    duration_seconds=settings.duration_per_level,
                    warmup_seconds=settings.warmup_seconds,
                )
                for endpoint in settings.generation.configured_endpoints
            }
            for endpoint_id, future in combined_futures.items():
                result = future.result()
                combined_results[endpoint_id] = result
                _append_result(
                    checkpoint, fingerprint, result, phase="combined"
                )
                _progress(f"combined {endpoint_id}: {_result_summary(result)}")
    else:
        _progress("reused all combined-test results from checkpoint")
    predicted = sum(
        result["requests_per_second"] for result in isolated_selected.values()
    )
    observed = sum(
        result["requests_per_second"] for result in combined_results.values()
    )
    best_single = max(
        result["requests_per_second"] for result in isolated_selected.values()
    )
    combined_error_count = sum(
        result["retryable_errors"] + result["nonretryable_errors"]
        for result in combined_results.values()
    )
    combined_requests = sum(result["requests"] for result in combined_results.values())
    combined = {
        "results": combined_results,
        "predicted_requests_per_second": predicted,
        "observed_requests_per_second": observed,
        "efficiency_ratio": observed / predicted if predicted else 0.0,
        "speedup_over_best_single": observed / best_single if best_single else 0.0,
        "error_rate": combined_error_count / combined_requests if combined_requests else 1.0,
    }
    combined["certified"] = (
        combined["efficiency_ratio"] >= 0.9
        and combined["speedup_over_best_single"] >= 2.5
        and combined["error_rate"] <= 0.005
    )
    certified = all(item["certified"] for item in endpoint_reports) and combined["certified"]
    report = {
        "report_version": REPORT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "certified": certified,
        "task_type": settings.task_type,
        "task_version": get_task_workflow(settings.task_type).version,
        "benchmark_fingerprint": fingerprint,
        "generation_fingerprint": generation_fingerprint(
            settings.generation, settings.task_type
        ),
        "source_sha256": source_sha256,
        "fixture_mix": {
            "total": len(fixtures),
            "request_kinds": dict(
                sorted(Counter(fixture.request_kind for fixture in fixtures).items())
            ),
            "repair_shaped_generation": sum(
                "repair" in fixture.request_kind for fixture in fixtures
            ),
        },
        "preflight": {
            "endpoints": preflight,
            "matching_prompt_token_counts": tokenizer_usage_matches,
        },
        "duration_per_level": settings.duration_per_level,
        "warmup_seconds": settings.warmup_seconds,
        "endpoints": endpoint_reports,
        "combined": combined,
    }
    (settings.output_dir / "endpoint_capacity.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _progress(
        f"finished: {'certified' if certified else 'not certified'}; report written to "
        f"{settings.output_dir / 'endpoint_capacity.json'}"
    )
    return 0 if certified else 1
