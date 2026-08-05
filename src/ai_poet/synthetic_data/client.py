"""OpenAI-compatible HTTP client for the Gemma generation endpoint."""

from __future__ import annotations

import json
import random
import ssl
import time
from typing import Any, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import GenerationSettings
from .errors import GemmaConnectionError, GenerationError
from .tracing import GenerationTracer


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
        jitter. Transport failures that remain after all retries raise
        :class:`GemmaConnectionError`; other HTTP failures are reported
        immediately. The expected response shape is
        ``choices[0].message.content``.

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
            GemmaConnectionError: If Gemma remains unreachable after every
                configured connection retry.
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
        for attempt in range(self.settings.max_network_retries + 1):
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
                if not retryable or attempt == self.settings.max_network_retries:
                    self._trace_api_failure(
                        body, trace_context, attempt + 1, retry_errors, started
                    )
                    raise GenerationError(f"API returned HTTP {exc.code}") from exc
            except (URLError, TimeoutError, OSError) as exc:
                safe_error = str(exc).replace(self.settings.api_key, "[REDACTED]")
                retry_errors.append(
                    {
                        "network_attempt": attempt + 1,
                        "error_type": type(exc).__name__,
                        "message": safe_error,
                        "retryable": True,
                    }
                )
                if attempt == self.settings.max_network_retries:
                    self._trace_api_failure(
                        body, trace_context, attempt + 1, retry_errors, started
                    )
                    raise GemmaConnectionError(
                        "Gemma REST API connection failed after "
                        f"{self.settings.max_network_retries} retries: {safe_error}"
                    ) from exc
            except (KeyError, ValueError) as exc:
                safe_error = str(exc).replace(self.settings.api_key, "[REDACTED]")
                retry_errors.append(
                    {
                        "network_attempt": attempt + 1,
                        "error_type": type(exc).__name__,
                        "message": safe_error,
                        "retryable": True,
                    }
                )
                if attempt == self.settings.max_network_retries:
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
