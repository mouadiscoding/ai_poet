"""Generate a few-shot Arabic poetry SFT dataset with an OpenAI-compatible API."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
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
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv

from sft_templates import (
    METER_NAMES,
    TEMPLATE_FAMILIES,
    TemplateFamily,
    build_messages,
    eligible_families,
)


ARABIC_RE = re.compile(r"[\u0600-\u06ff]")
LATIN_RE = re.compile(r"[A-Za-z]")
DIACRITICS_RE = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06edـ]")
CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)
PRINT_LOCK = threading.Lock()
TEMPLATE_RATIONALE = (
    "Meta-template families diversify the reverse-generated instructions while "
    "keeping the shared poetry constraints fixed. Selection is deterministic "
    "from the poem hash so it is reproducible and approximately balanced."
)


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
        """Return the number of complete verse pairs stored in the record.

        ``verses`` stores each hemistich as a separate item, with consecutive
        items forming one couplet. Source validation guarantees an even number
        of items, so the count is exactly half the sequence length.
        """
        return len(self.verses) // 2

    @property
    def poem_text(self) -> str:
        """Return the poem in the line-oriented format used by SFT prompts.

        Each pair of hemistichs is joined with ``" = "``, and the resulting
        couplets are separated by newlines. Formatting is delegated to
        :func:`format_poem`, which also enforces the non-empty/even invariant.
        """
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


def _redact(value: Any, secrets: Sequence[str]) -> Any:
    """Recursively replace configured secrets in a JSON-compatible value."""
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[REDACTED]")
        return value
    if isinstance(value, dict):
        return {key: _redact(item, secrets) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(item, secrets) for item in value]
    return value


class GenerationTracer:
    """Write full, secret-scrubbed generation events to JSONL and stdout."""

    def __init__(self, path: Path, *, secrets: Sequence[str] = ()) -> None:
        self.path = path
        self.run_id = uuid4().hex
        self.secrets = tuple(secret for secret in secrets if secret)

    def emit(self, event: dict[str, Any]) -> None:
        record = _redact(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "run_id": self.run_id,
                **event,
            },
            self.secrets,
        )
        serialized = json.dumps(record, ensure_ascii=False)
        rendered = json.dumps(record, ensure_ascii=False, indent=2)
        label = str(record.get("event", "event"))
        sample_id = record.get("sample_id")
        if sample_id:
            label += f" sample={str(sample_id)[:12]}"
        with PRINT_LOCK:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(serialized + "\n")
            print(f"\n=== SFT TRACE {label} ===")
            print(rendered)
            print("=== END SFT TRACE ===", flush=True)


class GenerationError(RuntimeError):
    pass


class GemmaClient:
    def __init__(
        self,
        settings: GenerationSettings,
        tracer: GenerationTracer | None = None,
    ) -> None:
        """Create an API client from immutable generation settings.

        The client retains all request defaults and builds one reusable TLS
        context. Certificate verification is disabled only when the caller has
        explicitly enabled ``settings.insecure``; otherwise the platform's
        default trust configuration is used.

        Args:
            settings: Endpoint, authentication, sampling, timeout, retry, and
                validation configuration for subsequent requests.
            tracer: Optional audit writer for complete request/response events.
        """
        self.settings = settings
        self.tracer = tracer
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
        trace_context: dict[str, Any] | None = None,
    ) -> str:
        """Send a chat-completion request and return its generated text.

        The method serializes an OpenAI-compatible request containing the
        configured model and sampling values. Per-call token, temperature, and
        seed values can override or augment those defaults. HTTP 429 and 5xx
        responses, along with transport, timeout, decoding, and malformed
        response errors, are retried with capped exponential backoff and
        jitter. Other HTTP failures are reported immediately. The expected
        response shape is ``choices[0].message.content``.

        Args:
            messages: Ordered chat messages. Each mapping must provide the
                OpenAI-compatible ``role`` and ``content`` string fields.
            max_tokens: Optional completion-token limit for this request. When
                omitted, the configured default is used.
            temperature: Optional sampling temperature for this request. When
                omitted, the configured default is used.
            seed: Optional deterministic sampling seed sent to endpoints that
                support it.
            trace_context: Optional sample and attempt fields to merge into the
                audit event. Request headers are never included.

        Returns:
            The first completion choice's message content, coerced to a string.

        Raises:
            GenerationError: If a non-retryable HTTP response is received or
                every configured retry fails. Error text is sanitized so the
                configured API key is not exposed.
        """
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

        started = time.perf_counter()
        retry_errors: list[dict[str, Any]] = []
        for attempt in range(self.settings.max_network_retries):
            try:
                with urlopen(  # noqa: S310 - user-configured HTTPS endpoint
                    request,
                    timeout=self.settings.timeout,
                    context=self.ssl_context,
                ) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                content = str(payload["choices"][0]["message"]["content"])
            except HTTPError as exc:
                retryable = exc.code == 429 or exc.code >= 500
                retry_errors.append(
                    {
                        "network_attempt": attempt + 1,
                        "error_type": type(exc).__name__,
                        "http_status": exc.code,
                        "retryable": retryable,
                    }
                )
                if not retryable or attempt + 1 == self.settings.max_network_retries:
                    self._trace_api_failure(
                        body, trace_context, attempt + 1, retry_errors, started
                    )
                    raise GenerationError(f"API returned HTTP {exc.code}") from exc
            except (URLError, TimeoutError, OSError, KeyError, ValueError) as exc:
                safe_error = str(exc).replace(self.settings.api_key, "[REDACTED]")
                retry_errors.append(
                    {
                        "network_attempt": attempt + 1,
                        "error_type": type(exc).__name__,
                        "message": safe_error,
                        "retryable": True,
                    }
                )
                if attempt + 1 == self.settings.max_network_retries:
                    self._trace_api_failure(
                        body, trace_context, attempt + 1, retry_errors, started
                    )
                    raise GenerationError(f"API request failed: {safe_error}") from exc
            else:
                if self.tracer is not None:
                    self.tracer.emit(
                        {
                            "event": "api_exchange",
                            **(trace_context or {}),
                            "endpoint": self.settings.endpoint,
                            "request": body,
                            "response": payload,
                            "network_attempts": attempt + 1,
                            "retry_errors": retry_errors,
                            "elapsed_seconds": round(time.perf_counter() - started, 3),
                        }
                    )
                return content
            delay = min(2**attempt, 16) + random.random()
            time.sleep(delay)
        raise AssertionError("network retry loop terminated unexpectedly")

    def _trace_api_failure(
        self,
        body: dict[str, Any],
        trace_context: dict[str, Any] | None,
        network_attempts: int,
        retry_errors: list[dict[str, Any]],
        started: float,
    ) -> None:
        if self.tracer is not None:
            self.tracer.emit(
                {
                    "event": "api_failure",
                    **(trace_context or {}),
                    "endpoint": self.settings.endpoint,
                    "request": body,
                    "network_attempts": network_attempts,
                    "retry_errors": retry_errors,
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                }
            )


def poem_hash(verses: Sequence[str]) -> str:
    """Compute a stable content identifier for an ordered verse sequence.

    Hemistichs are joined with the Unicode record-separator symbol ``U+241E``
    before hashing, which preserves item boundaries that ordinary string
    concatenation would lose. The digest depends only on verse content and
    order, not on metadata such as poet, title, URL, or meter.

    Args:
        verses: Hemistich strings in their original order.

    Returns:
        The lowercase hexadecimal SHA-256 digest of the joined UTF-8 text.
    """
    joined = "\u241e".join(verses)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def format_poem(verses: Sequence[str]) -> str:
    """Format alternating hemistichs as newline-separated couplets.

    Items at positions ``0`` and ``1`` form the first couplet, positions ``2``
    and ``3`` form the second, and so on. The two sides are separated by
    ``" = "`` to match the representation used in prompts and output records.

    Args:
        verses: A non-empty, even-length sequence of hemistich strings.

    Returns:
        A single string containing one formatted couplet per line.

    Raises:
        ValueError: If ``verses`` is empty or contains an odd number of items.
    """
    if not verses or len(verses) % 2:
        raise ValueError("poem_verses must contain a non-empty even number of items")
    return "\n".join(
        f"{verses[index]} = {verses[index + 1]}"
        for index in range(0, len(verses), 2)
    )


def meter_name(meter_id: int) -> str:
    """Resolve a numeric meter identifier to its canonical Arabic name.

    Meter identifiers are zero-based indices into :data:`METER_NAMES`. Boolean
    values technically satisfy Python's ``int`` check and therefore resolve as
    indices in the same way as integers.

    Args:
        meter_id: Zero-based index of a supported poetic meter.

    Returns:
        The Arabic meter name at the requested index.

    Raises:
        ValueError: If the identifier is not an integer or lies outside the
            configured meter-name table.
    """
    if not isinstance(meter_id, int) or not 0 <= meter_id < len(METER_NAMES):
        raise ValueError(f"Unknown meter ID: {meter_id}")
    return METER_NAMES[meter_id]


def _metadata_score(row: dict[str, Any], selected_meter: int) -> tuple[int, int]:
    """Score a duplicate source row for selection as canonical metadata.

    Rows using the majority meter receive highest priority. Within that group,
    rows are ranked by how many of title, poet name, and poem URL are populated.
    Returning a tuple lets Python compare these criteria lexicographically.

    Args:
        row: A source row containing poem metadata.
        selected_meter: The meter chosen by majority vote for the poem group.

    Returns:
        ``(meter_matches, populated_field_count)``, where the first element is
        either zero or one and the second ranges from zero to three.
    """
    return (
        int(row["poem_meter"] == selected_meter),
        sum(bool(row.get(key)) for key in ("poem_title", "poet_name", "poem_url")),
    )


def load_poems(path: Path) -> list[PoemRecord]:
    """Load, validate, deduplicate, and canonicalize poems from Parquet.

    Only the columns required for generation are read. Rows with identical
    verse tuples are grouped as the same poem. Within each group, the meter is
    selected by an unambiguous majority vote, and canonical descriptive
    metadata comes from the best-populated row that uses that meter. All source
    row indices and distinct non-empty URLs are retained for provenance. The
    resulting sample identifier is based solely on the verses, and records are
    sorted by their first source-row position to preserve source order.

    Args:
        path: Parquet dataset containing ``poem_title``, ``poem_meter``,
            ``poem_verses``, ``poem_url``, and ``poet_name`` columns.

    Returns:
        One immutable :class:`PoemRecord` per distinct verse sequence.

    Raises:
        ValueError: If a row has no verses, has an odd number of hemistichs, a
            duplicate group has a tied meter vote, or the selected meter ID is
            unsupported.
        OSError: If the source file cannot be read.
    """
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
    """Assign a sample deterministically to the train, validation, or test set.

    The first eight hexadecimal digits of the content-derived sample ID are
    mapped into one hundred buckets. Buckets 0--97 are training data, bucket 98
    is validation data, and bucket 99 is test data, yielding a stable 98/1/1
    allocation that does not depend on input ordering.

    Args:
        sample_id: Hexadecimal sample identifier, normally from
            :func:`poem_hash`.

    Returns:
        ``"train"``, ``"validation"``, or ``"test"``.

    Raises:
        ValueError: If the first eight characters are not valid hexadecimal.
    """
    bucket = int(sample_id[:8], 16) % 100
    if bucket < 98:
        return "train"
    if bucket == 98:
        return "validation"
    return "test"


def choose_family(poem: PoemRecord) -> TemplateFamily:
    """Choose a deterministic prompt-template family for a poem.

    Eligibility is determined by the poem's meter; notably, prose poems cannot
    use the prosody-and-rhyme family. The next eight hexadecimal digits of the
    sample ID select uniformly by modulo from the eligible tuple, so duplicate
    content always receives the same template across runs.

    Args:
        poem: Canonical poem whose meter and sample ID drive selection.

    Returns:
        The selected :class:`~sft_templates.TemplateFamily`.

    Raises:
        ValueError: If the relevant sample-ID characters are not hexadecimal.
    """
    families = eligible_families(poem.meter_name)
    offset = int(poem.sample_id[8:16], 16) % len(families)
    return families[offset]


def _emit_client_trace(client: Any, event: dict[str, Any]) -> None:
    tracer = getattr(client, "tracer", None)
    if tracer is not None:
        tracer.emit(event)


def split_poem_chunks(verses: Sequence[str], max_chars: int) -> list[str]:
    """Partition a poem into size-limited chunks without splitting couplets.

    Each adjacent hemistich pair is first formatted as ``left = right``. Lines
    are accumulated until adding the next complete couplet, including its
    separating newline, would exceed ``max_chars``. A single couplet longer
    than the limit remains intact in its own oversized chunk because preserving
    semantic and metrical pairs takes precedence over the character target.

    Args:
        verses: Alternating left and right hemistichs. Callers are expected to
            supply a complete, even-length sequence.
        max_chars: Maximum preferred character count per chunk.

    Returns:
        Formatted poem chunks in source order. An empty sequence produces an
        empty list.

    Raises:
        ValueError: If ``max_chars`` is zero or negative.
        IndexError: If ``verses`` contains an unmatched final hemistich.
    """
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
    """Extract a top-level JSON object from a model response.

    Leading or trailing Markdown JSON fences are stripped first. The cleaned
    response is parsed as-is; if that fails, the substring spanning its first
    opening brace through its last closing brace is parsed to tolerate prose or
    other wrapper text around the object. Arrays and scalar JSON values are
    rejected even when syntactically valid.

    Args:
        raw: Untrusted text returned by the generation endpoint.

    Returns:
        The decoded JSON object as a dictionary.

    Raises:
        ValueError: If no plausible object is present or the decoded top-level
            value is not an object.
        json.JSONDecodeError: If the selected object substring is invalid JSON.
    """
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
    """Reduce Arabic text to comparable base letters.

    Arabic combining marks, Quranic annotation marks, and tatweel are removed,
    followed by every character outside the Arabic Unicode block. The result is
    intentionally aggressive: spaces and punctuation disappear so copied
    hemistichs can be detected despite superficial formatting or diacritics.

    Args:
        text: Arbitrary text to normalize for containment checks.

    Returns:
        A compact string containing only normalized Arabic-block characters.
    """
    text = DIACRITICS_RE.sub("", text)
    return re.sub(r"[^\u0600-\u06ff]", "", text)


def validate_generation(
    value: dict[str, Any], poem: PoemRecord, min_chars: int
) -> list[str]:
    """Validate a generated instruction/reasoning object for SFT use.

    Validation covers the exact two-key schema, string types, minimum lengths,
    dominance of Arabic over Latin characters, explicit numeric couplet count,
    and correct meter or prose terminology. It rejects mention of conflicting
    meters and instructions that reproduce a complete sufficiently long source
    hemistich after Arabic normalization. The reasoning must discuss semantics,
    imagery or rhetoric, and revision; metered poems must also discuss prosody,
    and every response must contain the final-result transition.

    The function accumulates independent failures instead of raising, allowing
    the caller to give the model a complete repair prompt. A schema type error
    returns early because content checks require string fields.

    Args:
        value: Decoded model JSON to validate.
        poem: Source poem providing the expected meter, count, and verse text.
        min_chars: Minimum stripped length required independently for the
            ``instruction`` and ``reasoning`` fields.

    Returns:
        Human-readable validation errors. An empty list means the generation
        passed every rule.
    """
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
    """Combine cleaned editorial reasoning with the exact source poem.

    Any occurrence of the fully formatted poem is removed from the generated
    reasoning. The cleanup also drops lines that exactly duplicate a source
    couplet or hemistich, repeated final-result markers, and standalone headers
    that would introduce a model-generated poem. Excess blank lines are
    collapsed before one canonical Arabic final-result marker and the source
    poem are appended verbatim. This guarantees that training targets end with
    trusted source text rather than a potentially altered model reproduction.

    Args:
        reasoning: Model-generated editorial analysis, possibly containing
            redundant poem text or result headings.
        poem: Canonical source poem to append.

    Returns:
        The cleaned reasoning followed by ``النتيجة النهائية:`` and the
        formatted source poem.
    """
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
    """Summarize a long poem chunk by chunk through the generation API.

    The poem is divided on couplet boundaries, then each chunk is sent in its
    own low-temperature Arabic analysis request. Requests identify the source
    meter and the chunk's position, constrain the answer to meaning, imagery,
    tone, and observable rhyme, and use a deterministic seed derived from the
    poem ID plus the one-based chunk index. The ordered summaries provide a
    compact substitute for source text that would exceed the main prompt limit.

    Args:
        client: Configured chat client used for every analysis request.
        poem: Canonical poem to split and analyze.
        max_chars: Preferred maximum size passed to :func:`split_poem_chunks`.

    Returns:
        The stripped summaries in source order, each prefixed with an Arabic
        one-based chunk label and separated by a blank line.

    Raises:
        ValueError: If ``max_chars`` is not positive.
        GenerationError: If any chunk request cannot be completed.
    """
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
                trace_context={
                    "sample_id": poem.sample_id,
                    "request_kind": "chunk_analysis",
                    "chunk_index": index,
                    "chunk_count": len(chunks),
                },
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
    """Generate and validate one complete supervised fine-tuning record.

    A prompt family is selected deterministically from the poem ID. Poems over
    ``max_source_chars`` are summarized in chunks before prompt construction;
    shorter poems are included verbatim. The model is asked for JSON containing
    an instruction and editorial reasoning. Invalid JSON or content triggers up
    to ``max_repairs`` follow-up attempts that include the previous answer and
    all validation errors. Seeds are stable across runs and incremented for
    each repair.

    On success, the reasoning is cleaned and combined with the exact source
    poem. The returned dictionary contains provenance, meter and template
    metadata, a two-message SFT conversation, deterministic dataset split,
    oversize/conflict flags, attempt count, and validation status.

    Args:
        poem: Canonical poem that supplies source text and metadata.
        client: Chat-completion client used for analysis and generation calls.
        settings: Prompt limits, repair count, and other generation settings.

    Returns:
        A JSON-serializable SFT record ready for checkpointing and export.

    Raises:
        GenerationError: If network generation fails or the response remains
            invalid after the configured number of repair attempts.
        ValueError: If chunk settings or poem formatting invariants are invalid.
    """
    family = choose_family(poem)
    families = eligible_families(poem.meter_name)
    selector_hex = poem.sample_id[8:16]
    selected_index = int(selector_hex, 16) % len(families)
    _emit_client_trace(
        client,
        {
            "event": "template_selection",
            "sample_id": poem.sample_id,
            "source_row_indices": list(poem.source_row_indices),
            "poet_name": poem.poet_name,
            "poem_title": poem.poem_title,
            "meter_name": poem.meter_name,
            "couplet_count": poem.couplet_count,
            "source_characters": len(poem.poem_text),
            "eligible_template_ids": [item.template_id for item in families],
            "selected_template_id": family.template_id,
            "selected_template_focus": family.focus,
            "why_used": {
                "purpose": TEMPLATE_RATIONALE,
                "eligibility": (
                    "prosody_rhyme is excluded because this is prose poetry"
                    if poem.meter_name == "النثر"
                    else "all template families are eligible for a metered poem"
                ),
                "selection": (
                    f"int(sample_id[8:16], 16) % {len(families)} = "
                    f"int('{selector_hex}', 16) % {len(families)} = {selected_index}"
                ),
            },
        },
    )
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
        raw = client.chat(
            repair_messages,
            seed=seed + repair,
            trace_context={
                "sample_id": poem.sample_id,
                "request_kind": "initial_generation" if repair == 0 else "repair",
                "generation_attempt": attempts,
                "repair_index": repair,
                "template_id": family.template_id,
            },
        )
        try:
            value = extract_json_object(raw)
            errors = validate_generation(value, poem, settings.min_chars)
        except (ValueError, json.JSONDecodeError) as exc:
            value = {}
            errors = [str(exc)]
        _emit_client_trace(
            client,
            {
                "event": "validation_result",
                "sample_id": poem.sample_id,
                "generation_attempt": attempts,
                "raw_model_content": raw,
                "parsed_output": value,
                "passed": not errors,
                "validation_errors": errors,
            },
        )
        if not errors:
            instruction = str(value["instruction"]).strip()
            reasoning = str(value["reasoning"]).strip()
            response = compose_response(reasoning, poem)
            source_lines = {
                line.strip() for line in poem.poem_text.splitlines() if line.strip()
            }
            source_lines.update(verse.strip() for verse in poem.verses if verse.strip())
            marker = "النتيجة النهائية:"
            postprocessing = {
                "full_poem_occurrences_removed": reasoning.count(poem.poem_text),
                "source_lines_removed": sum(
                    line.strip() in source_lines for line in reasoning.splitlines()
                ),
                "result_marker_lines_removed": sum(
                    line.strip().startswith(marker) for line in reasoning.splitlines()
                ),
                "canonical_result_marker_added": True,
                "exact_source_poem_appended": True,
            }
            record = {
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
            _emit_client_trace(
                client,
                {
                    "event": "final_output",
                    "sample_id": poem.sample_id,
                    "template_id": family.template_id,
                    "parsed_instruction": instruction,
                    "gemma_editorial_reasoning": reasoning,
                    "final_assistant_response": response,
                    "postprocessing": postprocessing,
                    "generation_attempts": attempts,
                    "validation_status": record["validation_status"],
                    "origin": "generated_in_this_run",
                },
            )
            return record
    raise GenerationError("validation failed after repairs: " + "; ".join(errors))


def load_checkpoint(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Replay an append-only generation checkpoint into current sample state.

    Each non-blank JSONL event is processed in file order, so later events for
    a sample replace earlier ones. A success stores its record and clears any
    prior failure for that sample. A failure stores its message but deliberately
    does not remove an already recorded success; callers ultimately give the
    success mapping precedence when determining completed work.

    Args:
        path: Checkpoint JSONL path. A missing file is treated as an empty
            checkpoint.

    Returns:
        A pair ``(successes, failures)`` keyed by sample ID. Successful values
        are full SFT records; failure values are error strings.

    Raises:
        ValueError: If any non-blank line contains malformed JSON.
        KeyError: If a decoded event lacks required ``sample_id``, ``status``,
            or successful ``record`` fields.
        OSError: If an existing checkpoint cannot be read.
    """
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
    """Append one durable JSON event to the generation checkpoint.

    Parent directories are created as needed. The event is encoded as a single
    UTF-8 JSON line with Arabic characters preserved, then the Python stream is
    flushed so progress is visible to subsequent resume attempts even while a
    larger corpus run is still active.

    Args:
        path: Destination JSONL checkpoint path.
        event: JSON-serializable success or failure event.

    Raises:
        OSError: If directories or the checkpoint file cannot be written.
        TypeError: If ``event`` contains values unsupported by ``json.dumps``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        handle.flush()


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


def write_outputs(
    output_dir: Path,
    poems: Sequence[PoemRecord],
    successes: dict[str, dict[str, Any]],
    failures: dict[str, str],
    settings: GenerationSettings,
    source_fingerprint: str | None = None,
    trace_run_id: str | None = None,
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
        "trace": {
            "enabled": trace_run_id is not None,
            "run_id": trace_run_id,
            "file": "generation_trace.jsonl" if trace_run_id is not None else None,
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
    parser.add_argument("--max-network-retries", type=int, default=5)
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


def file_sha256(path: Path) -> str:
    """Calculate the SHA-256 fingerprint of a file without loading it at once.

    The file is streamed in one-megabyte binary blocks, so memory usage remains
    bounded for large Parquet datasets.

    Args:
        path: File whose exact byte content should be hashed.

    Returns:
        The lowercase hexadecimal SHA-256 digest.

    Raises:
        OSError: If the file cannot be opened or read.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _settings(args: Namespace) -> GenerationSettings:
    """Convert parsed CLI arguments into validated generation settings.

    The endpoint, model, and API key are loaded from ``.env`` so they do not
    need to appear in source code or on the command line. Existing
    process-environment values take precedence over values in the file.
    Concurrency is validated here even though it belongs to run orchestration
    rather than the returned settings; the remaining values are copied into an
    immutable :class:`GenerationSettings` instance.

    Args:
        args: Namespace produced by :func:`build_parser`.

    Returns:
        Immutable settings for the client and generation pipeline.

    Raises:
        ValueError: If ``GEMMA_ENDPOINT``, ``GEMMA_MODEL``, or
            ``GEMMA_API_KEY`` is empty or missing, or if concurrency is less
            than one.
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
    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1")
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


def run(args: Namespace) -> int:
    """Execute a resumable, concurrent SFT dataset generation run.

    Source poems are loaded and optionally limited, then prior checkpoint events
    are replayed and restricted to the current selection. Poems without a saved
    success are submitted to a thread pool. Each finished task is immediately
    checkpointed as a success or sanitized failure, allowing later invocations
    to resume without regenerating completed samples. Individual generation
    exceptions do not abort the corpus; final JSONL, Parquet, failure, and
    manifest outputs are written after all pending tasks finish.

    Args:
        args: Parsed CLI namespace containing every option registered by
            :func:`build_parser`.

    Returns:
        Zero when every selected poem has a successful record, otherwise one.

    Raises:
        ValueError: If configuration, the optional limit, or source poem data is
            invalid.
        OSError: If required input or output files cannot be accessed.
    """
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
    source_fingerprint = file_sha256(args.input)
    tracer = (
        GenerationTracer(
            args.output_dir / "generation_trace.jsonl",
            secrets=(settings.api_key,),
        )
        if args.trace
        else None
    )
    if tracer is not None:
        tracer.emit(
            {
                "event": "run_start",
                "source_path": str(args.input),
                "source_sha256": source_fingerprint,
                "selected_poems": len(poems),
                "checkpoint_reused": len(poems) - len(pending),
                "pending_generation": len(pending),
                "model": settings.model,
                "endpoint": settings.endpoint,
                "concurrency": args.concurrency,
                "generation_settings": {
                    "temperature": settings.temperature,
                    "top_p": settings.top_p,
                    "max_tokens": settings.max_tokens,
                    "min_chars": settings.min_chars,
                    "max_source_chars": settings.max_source_chars,
                    "chunk_chars": settings.chunk_chars,
                    "timeout": settings.timeout,
                    "max_network_retries": settings.max_network_retries,
                    "max_repairs": settings.max_repairs,
                },
                "meta_template_rationale": TEMPLATE_RATIONALE,
                "meta_templates": [
                    {
                        "template_id": family.template_id,
                        "focus": family.focus,
                        "why_available": (
                            "foregrounds this poetic dimension while retaining "
                            "the shared instruction and reasoning contract"
                        ),
                    }
                    for family in TEMPLATE_FAMILIES
                ],
                "checkpoint_note": (
                    "checkpoint_reused counts existing successful records; all "
                    "per-sample events in this run are newly generated"
                ),
            }
        )
    client = GemmaClient(settings, tracer=tracer)
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
                if tracer is not None:
                    tracer.emit(
                        {
                            "event": "sample_failure",
                            "sample_id": poem.sample_id,
                            "error_type": type(exc).__name__,
                            "error": error,
                            "origin": "generated_in_this_run",
                        }
                    )
            append_checkpoint(checkpoint, event)
            with PRINT_LOCK:
                print(f"[{completed}/{len(poems)}] {poem.sample_id[:12]} {status}")

    write_outputs(
        args.output_dir,
        poems,
        successes,
        failures,
        settings,
        source_fingerprint=source_fingerprint,
        trace_run_id=tracer.run_id if tracer is not None else None,
    )
    unresolved = [poem for poem in poems if poem.sample_id not in successes]
    if tracer is not None:
        tracer.emit(
            {
                "event": "run_summary",
                "selected_poems": len(poems),
                "checkpoint_reused": len(poems) - len(pending),
                "generated_successes": sum(
                    poem.sample_id in successes for poem in pending
                ),
                "unresolved_failures": len(unresolved),
                "complete": not unresolved,
                "template_distribution": dict(
                    Counter(record["template_id"] for record in successes.values())
                ),
                "validation_status_distribution": dict(
                    Counter(
                        record["validation_status"] for record in successes.values()
                    )
                ),
            }
        )
    return 1 if unresolved else 0


def main() -> None:
    """Parse CLI arguments, run generation, and terminate with a shell status.

    Normal completion exits with the status returned by :func:`run`: zero for a
    complete corpus and one for unresolved per-poem failures. Top-level
    ``OSError`` and ``ValueError`` exceptions are rendered as concise messages
    on standard error and converted to exit status two. Other unexpected
    exceptions are allowed to propagate with their traceback.

    Raises:
        SystemExit: Always, with status zero, one, or two as described above.
    """
    try:
        code = run(build_parser().parse_args())
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    raise SystemExit(code)


if __name__ == "__main__":
    main()
