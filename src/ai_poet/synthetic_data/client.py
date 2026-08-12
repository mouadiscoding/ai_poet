"""OpenAI-compatible single-endpoint transport and adaptive endpoint pool."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import math
import random
import ssl
import threading
import time
from typing import Any, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .capacity import CapacityPlan
from .config import EndpointSettings, GenerationSettings
from .errors import GemmaConnectionError, GenerationError
from .tracing import GenerationTracer


@dataclass(frozen=True)
class ChatResult:
    """One decoded API exchange with transport and usage metadata."""

    content: str
    payload: dict[str, Any]
    endpoint_id: str
    finish_reason: str | None
    usage: dict[str, int]
    elapsed_seconds: float


class EndpointRequestError(RuntimeError):
    """Classified failure from exactly one endpoint attempt."""

    def __init__(
        self,
        message: str,
        *,
        kind: str,
        retryable: bool,
        transport: bool = False,
        status: int | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.retryable = retryable
        self.transport = transport
        self.status = status
        self.retry_after = retry_after

    @property
    def authentication_failure(self) -> bool:
        return self.status in {401, 403}


def _retry_after_seconds(exc: HTTPError) -> float | None:
    value = exc.headers.get("Retry-After") if exc.headers is not None else None
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            parsed = parsedate_to_datetime(value)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return max(0.0, (parsed - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


class EndpointClient:
    """Perform one HTTP attempt against one configured endpoint."""

    def __init__(
        self,
        endpoint: EndpointSettings,
        settings: GenerationSettings,
    ) -> None:
        self.endpoint = endpoint
        self.settings = settings
        self.ssl_context = (
            ssl._create_unverified_context()  # noqa: SLF001 - explicit CLI opt-in
            if settings.insecure
            else ssl.create_default_context()
        )

    def chat_once(
        self,
        messages: Sequence[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> ChatResult:
        body: dict[str, Any] = {
            "model": self.endpoint.model or self.settings.model,
            "messages": list(messages),
            "temperature": (
                self.settings.temperature if temperature is None else temperature
            ),
            "top_p": self.settings.top_p,
            "max_tokens": (
                self.settings.max_tokens if max_tokens is None else max_tokens
            ),
        }
        if seed is not None:
            body["seed"] = seed
        request = Request(
            self.endpoint.endpoint,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {self.endpoint.api_key}",
                "Content-Type": "application/json",
            },
        )
        started = time.perf_counter()
        try:
            with urlopen(  # noqa: S310 - user-configured HTTPS endpoint
                request,
                timeout=self.settings.timeout,
                context=self.ssl_context,
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
            choice = payload["choices"][0]
            content = choice["message"]["content"]
            if not isinstance(content, str):
                raise ValueError("choices[0].message.content must be a string")
            raw_usage = payload.get("usage") or {}
            usage = {
                key: int(raw_usage.get(key, 0) or 0)
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            }
            return ChatResult(
                content=content,
                payload=payload,
                endpoint_id=self.endpoint.endpoint_id,
                finish_reason=(
                    str(choice["finish_reason"])
                    if choice.get("finish_reason") is not None
                    else None
                ),
                usage=usage,
                elapsed_seconds=time.perf_counter() - started,
            )
        except HTTPError as exc:
            retryable = exc.code in {408, 429} or exc.code >= 500
            raise EndpointRequestError(
                f"API returned HTTP {exc.code}",
                kind="http",
                retryable=retryable,
                status=exc.code,
                retry_after=_retry_after_seconds(exc),
            ) from exc
        except (TimeoutError,) as exc:
            raise EndpointRequestError(
                str(exc),
                kind="timeout",
                retryable=True,
                transport=True,
            ) from exc
        except (URLError, OSError) as exc:
            raise EndpointRequestError(
                str(exc),
                kind="transport",
                retryable=True,
                transport=True,
            ) from exc
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise EndpointRequestError(
                str(exc),
                kind="malformed",
                retryable=True,
            ) from exc


class GemmaClient:
    """Backward-compatible single-endpoint client with local retry/backoff."""

    def __init__(
        self,
        settings: GenerationSettings,
        tracer: GenerationTracer | None = None,
    ) -> None:
        self.settings = settings
        self.tracer = tracer
        endpoint = EndpointSettings(
            endpoint_id="legacy",
            endpoint=settings.endpoint,
            api_key=settings.api_key,
            max_concurrency=1,
            model=settings.model,
        )
        self._client = EndpointClient(endpoint, settings)

    def chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        trace_context: dict[str, Any] | None = None,
    ) -> str:
        started = time.perf_counter()
        errors: list[dict[str, Any]] = []
        for attempt in range(self.settings.max_network_retries + 1):
            try:
                result = self._client.chat_once(
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    seed=seed,
                )
            except EndpointRequestError as exc:
                safe_error = str(exc).replace(self.settings.api_key, "[REDACTED]")
                errors.append(
                    {
                        "network_attempt": attempt + 1,
                        "error_type": exc.kind,
                        "http_status": exc.status,
                        "message": safe_error,
                        "retryable": exc.retryable,
                    }
                )
                if not exc.retryable or attempt == self.settings.max_network_retries:
                    self._trace_failure(
                        messages,
                        max_tokens,
                        temperature,
                        seed,
                        trace_context,
                        attempt + 1,
                        errors,
                        started,
                    )
                    if exc.transport:
                        raise GemmaConnectionError(
                            "Gemma REST API connection failed after "
                            f"{self.settings.max_network_retries} retries: {safe_error}"
                        ) from exc
                    raise GenerationError(safe_error) from exc
                delay = min(2**attempt, 16) + random.random()
                time.sleep(delay)
                continue
            self._trace_success(
                result,
                messages,
                max_tokens,
                temperature,
                seed,
                trace_context,
                attempt + 1,
                errors,
            )
            return result.content
        raise AssertionError("network retry loop terminated unexpectedly")

    def _body(
        self,
        messages: Sequence[dict[str, str]],
        max_tokens: int | None,
        temperature: float | None,
        seed: int | None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.settings.model,
            "messages": list(messages),
            "temperature": self.settings.temperature if temperature is None else temperature,
            "top_p": self.settings.top_p,
            "max_tokens": self.settings.max_tokens if max_tokens is None else max_tokens,
        }
        if seed is not None:
            body["seed"] = seed
        return body

    def _trace_success(
        self,
        result: ChatResult,
        messages: Sequence[dict[str, str]],
        max_tokens: int | None,
        temperature: float | None,
        seed: int | None,
        context: dict[str, Any] | None,
        attempts: int,
        errors: list[dict[str, Any]],
    ) -> None:
        if self.tracer is not None:
            self.tracer.emit(
                {
                    "event": "api_exchange",
                    **(context or {}),
                    "endpoint": self.settings.endpoint,
                    "endpoint_id": result.endpoint_id,
                    "request": self._body(messages, max_tokens, temperature, seed),
                    "response": result.payload,
                    "network_attempts": attempts,
                    "retry_errors": errors,
                    "elapsed_seconds": round(result.elapsed_seconds, 3),
                }
            )

    def _trace_failure(
        self,
        messages: Sequence[dict[str, str]],
        max_tokens: int | None,
        temperature: float | None,
        seed: int | None,
        context: dict[str, Any] | None,
        attempts: int,
        errors: list[dict[str, Any]],
        started: float,
    ) -> None:
        if self.tracer is not None:
            self.tracer.emit(
                {
                    "event": "api_failure",
                    **(context or {}),
                    "endpoint": self.settings.endpoint,
                    "endpoint_id": "legacy",
                    "request": self._body(messages, max_tokens, temperature, seed),
                    "network_attempts": attempts,
                    "retry_errors": errors,
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                }
            )


@dataclass
class _EndpointState:
    endpoint: EndpointSettings
    client: EndpointClient
    hard_capacity: int
    effective_capacity: int
    latency_baselines: dict[str, float]
    in_flight: int = 0
    disabled: bool = False
    circuit_until: float = 0.0
    half_open: bool = False
    consecutive_failures: int = 0
    successes_since_change: int = 0
    last_capacity_change: float = field(default_factory=time.monotonic)
    last_latency_check: float = field(default_factory=time.monotonic)
    latency_breaches: dict[str, int] = field(default_factory=dict)
    recent_latencies: dict[str, deque[tuple[float, float]]] = field(
        default_factory=lambda: defaultdict(deque)
    )
    requests: int = 0
    successes: int = 0
    retryable_failures: int = 0
    nonretryable_failures: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    truncations: int = 0
    capacity_changes: list[dict[str, Any]] = field(default_factory=list)


class GemmaPoolClient:
    """Thread-safe request-level router with endpoint-local backpressure."""

    def __init__(
        self,
        settings: GenerationSettings,
        capacity: CapacityPlan,
        tracer: GenerationTracer | None = None,
    ) -> None:
        self.settings = settings
        self.tracer = tracer
        self._condition = threading.Condition()
        self._round_robin = 0
        self._states = {
            endpoint.endpoint_id: _EndpointState(
                endpoint=endpoint,
                client=EndpointClient(endpoint, settings),
                hard_capacity=capacity.hard_caps[endpoint.endpoint_id],
                effective_capacity=capacity.hard_caps[endpoint.endpoint_id],
                latency_baselines=capacity.latency_baselines.get(
                    endpoint.endpoint_id, {}
                ),
            )
            for endpoint in settings.configured_endpoints
        }
        self._sample_stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "endpoint_ids": set(),
                "network_attempts": 0,
                "endpoint_failovers": 0,
                "truncations": 0,
            }
        )

    @property
    def total_hard_capacity(self) -> int:
        return sum(state.hard_capacity for state in self._states.values())

    @property
    def total_effective_capacity(self) -> int:
        with self._condition:
            return sum(
                state.effective_capacity
                for state in self._states.values()
                if not state.disabled
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
        started = time.perf_counter()
        errors: list[dict[str, Any]] = []
        attempted_endpoint_ids: list[str] = []
        excluded: set[str] = set()
        transport_failures = 0
        for attempt in range(self.settings.max_network_retries + 1):
            state = self._acquire(excluded)
            attempted_endpoint_ids.append(state.endpoint.endpoint_id)
            try:
                result = state.client.chat_once(
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    seed=seed,
                )
            except EndpointRequestError as exc:
                if exc.transport:
                    transport_failures += 1
                self._release_failure(state, exc, attempt)
                safe_error = str(exc)
                for secret in self.settings.secrets:
                    safe_error = safe_error.replace(secret, "[REDACTED]")
                errors.append(
                    {
                        "network_attempt": attempt + 1,
                        "endpoint_id": state.endpoint.endpoint_id,
                        "error_type": exc.kind,
                        "http_status": exc.status,
                        "message": safe_error,
                        "retryable": exc.retryable,
                    }
                )
                if exc.authentication_failure and attempt < self.settings.max_network_retries:
                    excluded = {state.endpoint.endpoint_id}
                    continue
                if not exc.retryable:
                    self._trace_pool_failure(
                        messages,
                        max_tokens,
                        temperature,
                        seed,
                        state.endpoint.model,
                        trace_context,
                        errors,
                        started,
                    )
                    raise GenerationError(safe_error) from exc
                if attempt == self.settings.max_network_retries:
                    self._trace_pool_failure(
                        messages,
                        max_tokens,
                        temperature,
                        seed,
                        state.endpoint.model,
                        trace_context,
                        errors,
                        started,
                    )
                    if transport_failures == len(errors):
                        raise GemmaConnectionError(
                            "All Gemma endpoints failed transport attempts: "
                            + safe_error
                        ) from exc
                    raise GenerationError(
                        "Gemma request exhausted cross-endpoint retries: " + safe_error
                    ) from exc
                excluded = {state.endpoint.endpoint_id}
                continue

            self._release_success(state, result, trace_context)
            self._record_sample_stats(
                trace_context,
                attempted_endpoint_ids,
                result.finish_reason,
            )
            if self.tracer is not None:
                self.tracer.emit(
                    {
                        "event": "api_exchange",
                        **(trace_context or {}),
                        "endpoint": state.endpoint.endpoint,
                        "endpoint_id": state.endpoint.endpoint_id,
                        "request": self._body(
                            messages,
                            max_tokens,
                            temperature,
                            seed,
                            state.endpoint.model,
                        ),
                        "response": result.payload,
                        "network_attempts": attempt + 1,
                        "retry_errors": errors,
                        "elapsed_seconds": round(
                            time.perf_counter() - started, 3
                        ),
                    }
                )
            return result.content
        raise AssertionError("network retry loop terminated unexpectedly")

    def _acquire(self, excluded: set[str]) -> _EndpointState:
        with self._condition:
            while True:
                now = time.monotonic()
                enabled = [state for state in self._states.values() if not state.disabled]
                if not enabled:
                    raise GemmaConnectionError("No usable Gemma endpoints remain")
                allowed = [
                    state
                    for state in enabled
                    if state.endpoint.endpoint_id not in excluded
                ]
                if not allowed:
                    allowed = enabled
                candidates: list[_EndpointState] = []
                for state in allowed:
                    if state.circuit_until and now >= state.circuit_until:
                        state.circuit_until = 0.0
                        state.half_open = True
                    limit = 1 if state.half_open else state.effective_capacity
                    if state.circuit_until <= now and state.in_flight < limit:
                        candidates.append(state)
                if candidates:
                    ordered = sorted(
                        candidates,
                        key=lambda item: (
                            item.in_flight / max(1, item.effective_capacity),
                            (
                                list(self._states).index(item.endpoint.endpoint_id)
                                - self._round_robin
                            )
                            % len(self._states),
                        ),
                    )
                    chosen = ordered[0]
                    chosen.in_flight += 1
                    chosen.requests += 1
                    self._round_robin = (
                        list(self._states).index(chosen.endpoint.endpoint_id) + 1
                    ) % len(self._states)
                    return chosen
                circuit_times = [
                    state.circuit_until
                    for state in allowed
                    if state.circuit_until > now
                ]
                wait_for = min(circuit_times) - now if circuit_times else 0.25
                self._condition.wait(timeout=max(0.01, min(wait_for, 1.0)))

    def _release_success(
        self,
        state: _EndpointState,
        result: ChatResult,
        context: dict[str, Any] | None,
    ) -> None:
        now = time.monotonic()
        kind = str((context or {}).get("request_kind", "unknown"))
        with self._condition:
            state.in_flight -= 1
            state.successes += 1
            state.consecutive_failures = 0
            state.half_open = False
            state.successes_since_change += 1
            state.prompt_tokens += result.usage.get("prompt_tokens", 0)
            state.completion_tokens += result.usage.get("completion_tokens", 0)
            if result.finish_reason == "length":
                state.truncations += 1
            latencies = state.recent_latencies[kind]
            latencies.append((now, result.elapsed_seconds))
            while latencies and latencies[0][0] < now - 120:
                latencies.popleft()
            self._evaluate_latency(state, kind, now)
            if (
                state.effective_capacity < state.hard_capacity
                and state.successes_since_change >= 20
                and now - state.last_capacity_change >= 60
            ):
                self._change_capacity(state, state.effective_capacity + 1, "recovery")
            self._condition.notify_all()

    def _release_failure(
        self,
        state: _EndpointState,
        error: EndpointRequestError,
        attempt: int,
    ) -> None:
        now = time.monotonic()
        with self._condition:
            state.in_flight -= 1
            state.consecutive_failures += 1
            state.successes_since_change = 0
            if error.retryable:
                state.retryable_failures += 1
            else:
                state.nonretryable_failures += 1
            if error.authentication_failure:
                state.disabled = True
                self._change_capacity(state, 0, "authentication_failure")
            elif error.status == 429:
                self._change_capacity(
                    state,
                    max(1, math.floor(state.effective_capacity / 2)),
                    "http_429",
                )
                state.circuit_until = now + (
                    error.retry_after
                    if error.retry_after is not None
                    else min(2**attempt, 16)
                )
            elif error.kind == "timeout":
                self._change_capacity(
                    state,
                    max(1, math.floor(state.effective_capacity / 2)),
                    "timeout",
                )
            if (
                error.retryable
                and error.kind in {"transport", "timeout", "malformed", "http"}
                and state.consecutive_failures >= 3
            ):
                state.circuit_until = max(state.circuit_until, now + 30)
                state.half_open = False
            self._condition.notify_all()

    def _evaluate_latency(
        self,
        state: _EndpointState,
        kind: str,
        now: float,
    ) -> None:
        baseline = state.latency_baselines.get(kind)
        if baseline is None or now - state.last_latency_check < 60:
            return
        recent = [
            latency
            for timestamp, latency in state.recent_latencies[kind]
            if timestamp >= now - 60
        ]
        state.last_latency_check = now
        if len(recent) < 5:
            return
        recent.sort()
        p95 = recent[min(len(recent) - 1, math.ceil(0.95 * len(recent)) - 1)]
        if p95 > baseline * 1.5:
            state.latency_breaches[kind] = state.latency_breaches.get(kind, 0) + 1
        else:
            state.latency_breaches[kind] = 0
        if state.latency_breaches[kind] >= 2:
            self._change_capacity(
                state,
                max(1, state.effective_capacity - 1),
                f"latency:{kind}",
            )
            state.latency_breaches[kind] = 0

    def _change_capacity(
        self,
        state: _EndpointState,
        new_capacity: int,
        reason: str,
    ) -> None:
        new_capacity = max(0, min(new_capacity, state.hard_capacity))
        if new_capacity == state.effective_capacity:
            return
        state.effective_capacity = new_capacity
        state.last_capacity_change = time.monotonic()
        state.successes_since_change = 0
        state.capacity_changes.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "effective_capacity": new_capacity,
                "reason": reason,
            }
        )

    def _record_sample_stats(
        self,
        context: dict[str, Any] | None,
        endpoint_ids: list[str],
        finish_reason: str | None,
    ) -> None:
        sample_id = (context or {}).get("sample_id")
        if not sample_id:
            return
        with self._condition:
            stats = self._sample_stats[str(sample_id)]
            stats["endpoint_ids"].update(endpoint_ids)
            stats["network_attempts"] += len(endpoint_ids)
            stats["endpoint_failovers"] += max(0, len(set(endpoint_ids)) - 1)
            if finish_reason == "length":
                stats["truncations"] += 1

    def sample_stats(self, sample_id: str) -> dict[str, Any]:
        with self._condition:
            stats = self._sample_stats.get(sample_id)
            if stats is None:
                return {
                    "generation_endpoint_ids": [],
                    "network_attempts": 0,
                    "endpoint_failover_count": 0,
                    "truncated_completions": 0,
                }
            return {
                "generation_endpoint_ids": sorted(stats["endpoint_ids"]),
                "network_attempts": stats["network_attempts"],
                "endpoint_failover_count": stats["endpoint_failovers"],
                "truncated_completions": stats["truncations"],
            }

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_hard_capacity": self.total_hard_capacity,
                "total_effective_capacity": sum(
                    state.effective_capacity
                    for state in self._states.values()
                    if not state.disabled
                ),
                "endpoints": [
                    {
                        "endpoint_id": state.endpoint.endpoint_id,
                        "endpoint": state.endpoint.endpoint,
                        "model": state.endpoint.model or self.settings.model,
                        "hard_capacity": state.hard_capacity,
                        "effective_capacity": state.effective_capacity,
                        "in_flight": state.in_flight,
                        "disabled": state.disabled,
                        "circuit_open": state.circuit_until > time.monotonic(),
                        "requests": state.requests,
                        "successes": state.successes,
                        "retryable_failures": state.retryable_failures,
                        "nonretryable_failures": state.nonretryable_failures,
                        "prompt_tokens": state.prompt_tokens,
                        "completion_tokens": state.completion_tokens,
                        "truncations": state.truncations,
                        "capacity_changes": list(state.capacity_changes),
                    }
                    for state in self._states.values()
                ],
            }

    def _body(
        self,
        messages: Sequence[dict[str, str]],
        max_tokens: int | None,
        temperature: float | None,
        seed: int | None,
        model: str | None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model or self.settings.model,
            "messages": list(messages),
            "temperature": self.settings.temperature if temperature is None else temperature,
            "top_p": self.settings.top_p,
            "max_tokens": self.settings.max_tokens if max_tokens is None else max_tokens,
        }
        if seed is not None:
            body["seed"] = seed
        return body

    def _trace_pool_failure(
        self,
        messages: Sequence[dict[str, str]],
        max_tokens: int | None,
        temperature: float | None,
        seed: int | None,
        model: str | None,
        context: dict[str, Any] | None,
        errors: list[dict[str, Any]],
        started: float,
    ) -> None:
        if self.tracer is not None:
            self.tracer.emit(
                {
                    "event": "api_failure",
                    **(context or {}),
                    "request": self._body(
                        messages, max_tokens, temperature, seed, model
                    ),
                    "network_attempts": len(errors),
                    "retry_errors": errors,
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                }
            )
