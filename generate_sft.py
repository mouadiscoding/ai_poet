"""Generate a few-shot Arabic poetry SFT dataset with an OpenAI-compatible API."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import re
import ssl
import sys
import threading
import time
from typing import Any, Iterable, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pyarrow as pa
import pyarrow.parquet as pq

from sft_templates import (
    METER_NAMES,
    TemplateFamily,
    build_messages,
    eligible_families,
)


DEFAULT_ENDPOINT = (
    "https://vllm-gemma4-31b-mtrna-ns1.apps.olympus.atlasxai.ma/"
    "v1/chat/completions"
)
ARABIC_RE = re.compile(r"[\u0600-\u06ff]")
LATIN_RE = re.compile(r"[A-Za-z]")
DIACRITICS_RE = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06edـ]")
CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)
PRINT_LOCK = threading.Lock()


@dataclass(frozen=True)
class PoemRecord:
    sample_id: str
    source_row_indices: tuple[int, ...]
    source_urls: tuple[str, ...]
    poet_name: str
    poem_title: str | None
    meter_id: int
    meter_name: str
    verses: tuple[str, ...]
    metadata_conflict: bool

    @property
    def couplet_count(self) -> int:
        return len(self.verses) // 2

    @property
    def poem_text(self) -> str:
        return format_poem(self.verses)


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


class GenerationError(RuntimeError):
    pass


class GemmaClient:
    def __init__(self, settings: GenerationSettings) -> None:
        self.settings = settings
        self.ssl_context = (
            ssl._create_unverified_context()  # noqa: SLF001 - explicit CLI opt-in
            if settings.insecure
            else ssl.create_default_context()
        )

    def chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> str:
        body: dict[str, Any] = {
            "model": self.settings.model,
            "messages": list(messages),
            "temperature": (
                self.settings.temperature if temperature is None else temperature
            ),
            "top_p": self.settings.top_p,
            "max_tokens": max_tokens or self.settings.max_tokens,
        }
        if seed is not None:
            body["seed"] = seed

        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = Request(
            self.settings.endpoint,
            data=data,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.settings.api_key}",
                "Content-Type": "application/json",
            },
        )

        for attempt in range(self.settings.max_network_retries):
            try:
                with urlopen(  # noqa: S310 - user-configured HTTPS endpoint
                    request,
                    timeout=self.settings.timeout,
                    context=self.ssl_context,
                ) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                return str(payload["choices"][0]["message"]["content"])
            except HTTPError as exc:
                retryable = exc.code == 429 or exc.code >= 500
                if not retryable or attempt + 1 == self.settings.max_network_retries:
                    raise GenerationError(f"API returned HTTP {exc.code}") from exc
            except (URLError, TimeoutError, OSError, KeyError, ValueError) as exc:
                if attempt + 1 == self.settings.max_network_retries:
                    safe_error = str(exc).replace(
                        self.settings.api_key, "[REDACTED]"
                    )
                    raise GenerationError(f"API request failed: {safe_error}") from exc
            delay = min(2**attempt, 16) + random.random()
            time.sleep(delay)
        raise AssertionError("network retry loop terminated unexpectedly")


def poem_hash(verses: Sequence[str]) -> str:
    joined = "\u241e".join(verses)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def format_poem(verses: Sequence[str]) -> str:
    if not verses or len(verses) % 2:
        raise ValueError("poem_verses must contain a non-empty even number of items")
    return "\n".join(
        f"{verses[index]} = {verses[index + 1]}"
        for index in range(0, len(verses), 2)
    )


def meter_name(meter_id: int) -> str:
    if not isinstance(meter_id, int) or not 0 <= meter_id < len(METER_NAMES):
        raise ValueError(f"Unknown meter ID: {meter_id}")
    return METER_NAMES[meter_id]


def _metadata_score(row: dict[str, Any], selected_meter: int) -> tuple[int, int]:
    return (
        int(row["poem_meter"] == selected_meter),
        sum(bool(row.get(key)) for key in ("poem_title", "poet_name", "poem_url")),
    )


def load_poems(path: Path) -> list[PoemRecord]:
    columns = [
        "poem_title",
        "poem_meter",
        "poem_verses",
        "poem_url",
        "poet_name",
    ]
    rows = pq.read_table(path, columns=columns).to_pylist()
    grouped: dict[tuple[str, ...], list[tuple[int, dict[str, Any]]]] = {}
    for index, row in enumerate(rows):
        verses = tuple(row["poem_verses"] or ())
        if not verses or len(verses) % 2:
            raise ValueError(f"Source row {index} has an invalid poem_verses list")
        grouped.setdefault(verses, []).append((index, row))

    poems: list[PoemRecord] = []
    for verses, group in grouped.items():
        meter_counts = Counter(int(row["poem_meter"]) for _, row in group)
        top_count = max(meter_counts.values())
        top_meters = sorted(
            meter for meter, count in meter_counts.items() if count == top_count
        )
        if len(top_meters) != 1:
            indices = [index for index, _ in group]
            raise ValueError(f"Tied meter metadata for source rows {indices}")
        selected_meter = top_meters[0]
        canonical_index, canonical = max(
            group,
            key=lambda item: _metadata_score(item[1], selected_meter),
        )
        del canonical_index
        urls = tuple(
            dict.fromkeys(
                str(row["poem_url"])
                for _, row in group
                if row.get("poem_url")
            )
        )
        poems.append(
            PoemRecord(
                sample_id=poem_hash(verses),
                source_row_indices=tuple(index for index, _ in group),
                source_urls=urls,
                poet_name=str(canonical.get("poet_name") or ""),
                poem_title=canonical.get("poem_title"),
                meter_id=selected_meter,
                meter_name=meter_name(selected_meter),
                verses=verses,
                metadata_conflict=len(meter_counts) > 1,
            )
        )
    poems.sort(key=lambda poem: poem.source_row_indices[0])
    return poems


def sft_split(sample_id: str) -> str:
    bucket = int(sample_id[:8], 16) % 100
    if bucket < 98:
        return "train"
    if bucket == 98:
        return "validation"
    return "test"


def choose_family(poem: PoemRecord) -> TemplateFamily:
    families = eligible_families(poem.meter_name)
    offset = int(poem.sample_id[8:16], 16) % len(families)
    return families[offset]


def split_poem_chunks(verses: Sequence[str], max_chars: int) -> list[str]:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    chunks: list[str] = []
    current: list[str] = []
    current_chars = 0
    for index in range(0, len(verses), 2):
        line = f"{verses[index]} = {verses[index + 1]}"
        added = len(line) + int(bool(current))
        if current and current_chars + added > max_chars:
            chunks.append("\n".join(current))
            current = []
            current_chars = 0
        current.append(line)
        current_chars += len(line) + int(len(current) > 1)
    if current:
        chunks.append("\n".join(current))
    return chunks


def extract_json_object(raw: str) -> dict[str, Any]:
    cleaned = CODE_FENCE_RE.sub("", raw.strip()).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("response does not contain a JSON object")
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("response JSON must be an object")
    return value


def _normalise_arabic(text: str) -> str:
    text = DIACRITICS_RE.sub("", text)
    return re.sub(r"[^\u0600-\u06ff]", "", text)


def validate_generation(
    value: dict[str, Any], poem: PoemRecord, min_chars: int
) -> list[str]:
    errors: list[str] = []
    if set(value) != {"instruction", "reasoning"}:
        errors.append("JSON must contain only instruction and reasoning")
    instruction = value.get("instruction")
    reasoning = value.get("reasoning")
    if not isinstance(instruction, str) or not isinstance(reasoning, str):
        return errors + ["instruction and reasoning must be strings"]
    if len(instruction.strip()) < min_chars:
        errors.append(f"instruction must contain at least {min_chars} characters")
    if len(reasoning.strip()) < min_chars:
        errors.append(f"reasoning must contain at least {min_chars} characters")

    combined = instruction + reasoning
    arabic = len(ARABIC_RE.findall(combined))
    latin = len(LATIN_RE.findall(combined))
    if arabic < 100 or arabic < 4 * latin:
        errors.append("content must be predominantly Arabic")
    if str(poem.couplet_count) not in instruction:
        errors.append("instruction must state the exact couplet count in digits")
    if poem.meter_name == "النثر":
        if "النثر" not in instruction and "منثور" not in instruction:
            errors.append("prose instruction must explicitly request prose poetry")
    elif poem.meter_name not in instruction:
        errors.append(f"instruction must name بحر {poem.meter_name}")

    for other_meter in METER_NAMES:
        if other_meter != poem.meter_name and f"بحر {other_meter}" in instruction:
            errors.append(f"instruction contradicts the source meter with {other_meter}")
            break

    normalized_instruction = _normalise_arabic(instruction)
    for verse in poem.verses:
        normalized_verse = _normalise_arabic(verse)
        if len(normalized_verse) >= 20 and normalized_verse in normalized_instruction:
            errors.append("instruction copies a complete source hemistich")
            break

    required_groups = (
        ("معنى", "المعاني", "دلالة"),
        ("صورة", "بلاغ", "استعار", "تشبيه"),
        ("صياغ", "تعديل", "تحرير", "محاولة"),
    )
    for group in required_groups:
        if not any(term in reasoning for term in group):
            errors.append(f"reasoning is missing required discussion: {group[0]}")
    if poem.meter_name != "النثر" and not any(
        term in reasoning for term in ("وزن", "عروض", "إيقاع", "بحر")
    ):
        errors.append("metered reasoning must discuss prosody")
    if "النتيجة النهائية" not in reasoning:
        errors.append("reasoning must end with the final-result transition")
    return errors


def compose_response(reasoning: str, poem: PoemRecord) -> str:
    """Place editorial reasoning first and the exact source poem last."""
    marker = "النتيجة النهائية:"
    reasoning = reasoning.replace(poem.poem_text, "")

    source_lines = {
        line.strip() for line in poem.poem_text.splitlines() if line.strip()
    }
    source_lines.update(verse.strip() for verse in poem.verses if verse.strip())
    standalone_poem_headers = {
        "القصيدة:",
        "القصيدة النهائية:",
        "النص النهائي:",
    }
    retained_lines = []
    for line in reasoning.splitlines():
        stripped = line.strip()
        if stripped == marker or stripped in source_lines:
            continue
        if stripped in standalone_poem_headers:
            continue
        retained_lines.append(line.rstrip())

    editorial_reasoning = "\n".join(retained_lines).strip()
    editorial_reasoning = re.sub(r"\n{3,}", "\n\n", editorial_reasoning)
    return (
        f"{editorial_reasoning}\n\n{marker}\n\n{poem.poem_text}"
    )


def _chunk_analysis(client: GemmaClient, poem: PoemRecord, max_chars: int) -> str:
    summaries: list[str] = []
    chunks = split_poem_chunks(poem.verses, max_chars)
    for index, chunk in enumerate(chunks, start=1):
        messages = [
            {
                "role": "system",
                "content": (
                    "حلل مقطعًا من قصيدة عربية في 300 إلى 600 محرف. اذكر المعاني "
                    "والصور والنبرة والقافية الظاهرة فقط، ولا تنشئ تعليمات ولا قصيدة جديدة."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"البحر الأساس: {poem.meter_name}\n"
                    f"المقطع {index} من {len(chunks)}:\n{chunk}"
                ),
            },
        ]
        summaries.append(
            client.chat(
                messages,
                max_tokens=800,
                temperature=0.2,
                seed=int(poem.sample_id[:8], 16) + index,
            ).strip()
        )
    return "\n\n".join(
        f"ملخص المقطع {index}: {summary}"
        for index, summary in enumerate(summaries, start=1)
    )


def generate_one(
    poem: PoemRecord,
    client: GemmaClient,
    settings: GenerationSettings,
) -> dict[str, Any]:
    family = choose_family(poem)
    oversized = len(poem.poem_text) > settings.max_source_chars
    notes = (
        _chunk_analysis(client, poem, settings.chunk_chars) if oversized else None
    )
    messages = build_messages(
        family=family,
        meter_name=poem.meter_name,
        couplet_count=poem.couplet_count,
        poem_text=None if oversized else poem.poem_text,
        minimum_chars=settings.min_chars,
        analysis_notes=notes,
    )

    raw = ""
    errors: list[str] = []
    attempts = 0
    seed = int(poem.sample_id[:8], 16)
    for repair in range(settings.max_repairs + 1):
        attempts += 1
        if repair:
            repair_messages = list(messages)
            repair_messages.extend(
                [
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": (
                            "الجواب السابق غير صالح للأسباب الآتية:\n- "
                            + "\n- ".join(errors)
                            + "\nأعد كائن JSON مصححًا كاملًا فقط."
                        ),
                    },
                ]
            )
        else:
            repair_messages = messages
        raw = client.chat(repair_messages, seed=seed + repair)
        try:
            value = extract_json_object(raw)
            errors = validate_generation(value, poem, settings.min_chars)
        except (ValueError, json.JSONDecodeError) as exc:
            value = {}
            errors = [str(exc)]
        if not errors:
            instruction = str(value["instruction"]).strip()
            reasoning = str(value["reasoning"]).strip()
            response = compose_response(reasoning, poem)
            return {
                "sample_id": poem.sample_id,
                "source_row_indices": list(poem.source_row_indices),
                "source_urls": list(poem.source_urls),
                "poet_name": poem.poet_name,
                "poem_title": poem.poem_title,
                "meter_id": poem.meter_id,
                "meter_name": poem.meter_name,
                "couplet_count": poem.couplet_count,
                "template_id": family.template_id,
                "instruction": instruction,
                "response": response,
                "messages": [
                    {"role": "user", "content": instruction},
                    {
                        "role": "assistant",
                        "content": response,
                    },
                ],
                "sft_split": sft_split(poem.sample_id),
                "oversized_for_sft": oversized,
                "metadata_conflict": poem.metadata_conflict,
                "generation_attempts": attempts,
                "validation_status": (
                    "passed" if attempts == 1 else "passed_after_repair"
                ),
            }
    raise GenerationError("validation failed after repairs: " + "; ".join(errors))


def load_checkpoint(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    successes: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    if not path.exists():
        return successes, failures
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid checkpoint JSON on line {line_number}"
                ) from exc
            sample_id = event["sample_id"]
            if event["status"] == "success":
                successes[sample_id] = event["record"]
                failures.pop(sample_id, None)
            else:
                failures[sample_id] = event.get("error", "unknown error")
    return successes, failures


def append_checkpoint(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        handle.flush()


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_outputs(
    output_dir: Path,
    poems: Sequence[PoemRecord],
    successes: dict[str, dict[str, Any]],
    failures: dict[str, str],
    settings: GenerationSettings,
    source_fingerprint: str | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered = [successes[poem.sample_id] for poem in poems if poem.sample_id in successes]
    write_jsonl(output_dir / "ashaar_sft.jsonl", ordered)
    if ordered:
        pq.write_table(pa.Table.from_pylist(ordered), output_dir / "ashaar_sft.parquet")
    failure_records = [
        {"sample_id": sample_id, "error": error}
        for sample_id, error in sorted(failures.items())
        if sample_id not in successes
    ]
    write_jsonl(output_dir / "failures.jsonl", failure_records)
    manifest = {
        "complete": len(ordered) == len(poems) and not failure_records,
        "source_poems": len(poems),
        "generated_poems": len(ordered),
        "unresolved_failures": len(failure_records),
        "model": settings.model,
        "endpoint": settings.endpoint,
        "source_sha256": source_fingerprint,
        "template_version": 1,
        "generation": {
            "temperature": settings.temperature,
            "top_p": settings.top_p,
            "max_tokens": settings.max_tokens,
            "min_chars": settings.min_chars,
            "max_source_chars": settings.max_source_chars,
            "chunk_chars": settings.chunk_chars,
        },
        "splits": dict(Counter(record["sft_split"] for record in ordered)),
        "templates": dict(Counter(record["template_id"] for record in ordered)),
        "oversized_for_sft": sum(
            bool(record["oversized_for_sft"]) for record in ordered
        ),
        "metadata_conflicts": sum(
            bool(record["metadata_conflict"]) for record in ordered
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/ashaar_classic_moroccan.parquet"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/ashaar_sft"))
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default="gemma-4-31B")
    parser.add_argument("--api-key-env", default="GEMMA_API_KEY")
    parser.add_argument("--insecure", action="store_true")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-network-retries", type=int, default=5)
    parser.add_argument("--max-repairs", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--min-chars", type=int, default=1500)
    parser.add_argument("--max-source-chars", type=int, default=24000)
    parser.add_argument("--chunk-chars", type=int, default=12000)
    parser.add_argument("--limit", type=int)
    return parser


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _settings(args: Namespace) -> GenerationSettings:
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise ValueError(
            f"Environment variable {args.api_key_env} must contain the API key"
        )
    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1")
    return GenerationSettings(
        endpoint=args.endpoint,
        model=args.model,
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


def run(args: Namespace) -> int:
    settings = _settings(args)
    poems = load_poems(args.input)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1")
        poems = poems[: args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "generation_checkpoint.jsonl"
    successes, previous_failures = load_checkpoint(checkpoint)
    selected_ids = {poem.sample_id for poem in poems}
    successes = {
        sample_id: record
        for sample_id, record in successes.items()
        if sample_id in selected_ids
    }
    failures = {
        sample_id: error
        for sample_id, error in previous_failures.items()
        if sample_id in selected_ids and sample_id not in successes
    }
    pending = [poem for poem in poems if poem.sample_id not in successes]
    client = GemmaClient(settings)
    completed = len(poems) - len(pending)

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
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
            except Exception as exc:  # keep the corpus run resumable
                error = str(exc).replace(settings.api_key, "[REDACTED]")
                failures[poem.sample_id] = error
                event = {
                    "status": "failure",
                    "sample_id": poem.sample_id,
                    "error": error,
                }
                status = "failed"
            append_checkpoint(checkpoint, event)
            with PRINT_LOCK:
                print(f"[{completed}/{len(poems)}] {poem.sample_id[:12]} {status}")

    write_outputs(
        args.output_dir,
        poems,
        successes,
        failures,
        settings,
        source_fingerprint=file_sha256(args.input),
    )
    unresolved = [poem for poem in poems if poem.sample_id not in successes]
    return 1 if unresolved else 0


def main() -> None:
    try:
        code = run(build_parser().parse_args())
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    raise SystemExit(code)


if __name__ == "__main__":
    main()
