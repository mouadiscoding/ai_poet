from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time
import unittest
from unittest.mock import patch

from ai_poet.synthetic_data.capacity import CapacityPlan
from ai_poet.synthetic_data.client import (
    ChatResult,
    EndpointClient,
    EndpointRequestError,
    GemmaPoolClient,
)
from ai_poet.synthetic_data.config import EndpointSettings
from ai_poet.synthetic_data.errors import GemmaConnectionError
from tests.synthetic_data_helpers import settings


def endpoint_settings() -> tuple[EndpointSettings, ...]:
    return tuple(
        EndpointSettings(
            endpoint_id=f"endpoint_{index}",
            endpoint=f"https://endpoint-{index}.test/v1/chat/completions",
            api_key=f"secret-{index}",
            max_concurrency=4,
            model=f"gemma-alias-{index}",
        )
        for index in range(1, 4)
    )


def capacity(*, cap: int = 2) -> CapacityPlan:
    endpoints = endpoint_settings()
    return CapacityPlan(
        report_path=Path("capacity.json"),
        report_fingerprint="capacity-fingerprint",
        hard_caps={endpoint.endpoint_id: cap for endpoint in endpoints},
        latency_baselines={endpoint.endpoint_id: {} for endpoint in endpoints},
        raw={},
    )


def response(endpoint_id: str) -> ChatResult:
    return ChatResult(
        content="ok",
        payload={
            "model": "gemma-test",
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1},
        },
        endpoint_id=endpoint_id,
        finish_reason="stop",
        usage={"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        elapsed_seconds=0.01,
    )


class EndpointPoolTests(unittest.TestCase):
    def make_pool(self, **setting_overrides) -> GemmaPoolClient:
        configured = endpoint_settings()
        generation = settings(endpoints=configured, **setting_overrides)
        return GemmaPoolClient(generation, capacity())

    def test_endpoint_attempt_uses_its_served_model_alias(self) -> None:
        endpoint = endpoint_settings()[1]
        generation = settings(endpoints=endpoint_settings())
        captured: dict[str, object] = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self) -> bytes:
                return json.dumps(
                    {
                        "model": endpoint.model,
                        "choices": [
                            {
                                "message": {"content": "ok"},
                                "finish_reason": "stop",
                            }
                        ],
                    }
                ).encode()

        def fake_urlopen(request, **_kwargs):
            captured.update(json.loads(request.data.decode()))
            return FakeResponse()

        with patch("ai_poet.synthetic_data.client.urlopen", side_effect=fake_urlopen):
            EndpointClient(endpoint, generation).chat_once(
                [{"role": "user", "content": "hello"}]
            )

        self.assertEqual(captured["model"], "gemma-alias-2")

    def test_concurrent_requests_respect_caps_and_use_all_endpoints(self) -> None:
        pool = self.make_pool()
        lock = threading.Lock()
        active = {endpoint.endpoint_id: 0 for endpoint in endpoint_settings()}
        maximum = dict(active)

        def fake_chat_once(client, *_args, **_kwargs):
            endpoint_id = client.endpoint.endpoint_id
            with lock:
                active[endpoint_id] += 1
                maximum[endpoint_id] = max(maximum[endpoint_id], active[endpoint_id])
            time.sleep(0.003 * int(endpoint_id.rsplit("_", 1)[1]))
            with lock:
                active[endpoint_id] -= 1
            return response(endpoint_id)

        with (
            patch(
                "ai_poet.synthetic_data.client.EndpointClient.chat_once",
                autospec=True,
                side_effect=fake_chat_once,
            ),
            ThreadPoolExecutor(max_workers=18) as executor,
        ):
            results = list(
                executor.map(
                    lambda index: pool.chat(
                        [{"role": "user", "content": str(index)}],
                        trace_context={"sample_id": str(index), "request_kind": "test"},
                    ),
                    range(36),
                )
            )

        self.assertEqual(results, ["ok"] * 36)
        self.assertTrue(all(0 < value <= 2 for value in maximum.values()))
        self.assertTrue(all(item["successes"] for item in pool.snapshot()["endpoints"]))

    def test_retry_fails_over_and_halves_429_endpoint(self) -> None:
        pool = self.make_pool(max_network_retries=2)
        calls: list[str] = []

        def fake_chat_once(client, *_args, **_kwargs):
            endpoint_id = client.endpoint.endpoint_id
            calls.append(endpoint_id)
            if len(calls) == 1:
                raise EndpointRequestError(
                    "busy",
                    kind="http",
                    retryable=True,
                    status=429,
                    retry_after=0.01,
                )
            return response(endpoint_id)

        with patch(
            "ai_poet.synthetic_data.client.EndpointClient.chat_once",
            autospec=True,
            side_effect=fake_chat_once,
        ):
            result = pool.chat(
                [{"role": "user", "content": "test"}],
                trace_context={"sample_id": "sample", "request_kind": "test"},
            )

        self.assertEqual(result, "ok")
        self.assertNotEqual(calls[0], calls[1])
        first = next(
            item
            for item in pool.snapshot()["endpoints"]
            if item["endpoint_id"] == calls[0]
        )
        self.assertEqual(first["effective_capacity"], 1)
        stats = pool.sample_stats("sample")
        self.assertEqual(stats["network_attempts"], 2)
        self.assertEqual(stats["endpoint_failover_count"], 1)

    def test_authentication_failure_disables_only_one_endpoint(self) -> None:
        pool = self.make_pool(max_network_retries=2)
        calls: list[str] = []

        def fake_chat_once(client, *_args, **_kwargs):
            endpoint_id = client.endpoint.endpoint_id
            calls.append(endpoint_id)
            if endpoint_id == "endpoint_1":
                raise EndpointRequestError(
                    "unauthorized",
                    kind="http",
                    retryable=False,
                    status=401,
                )
            return response(endpoint_id)

        with patch(
            "ai_poet.synthetic_data.client.EndpointClient.chat_once",
            autospec=True,
            side_effect=fake_chat_once,
        ):
            self.assertEqual(
                pool.chat([{"role": "user", "content": "test"}]), "ok"
            )

        first = pool.snapshot()["endpoints"][0]
        self.assertTrue(first["disabled"])
        self.assertEqual(first["effective_capacity"], 0)
        self.assertGreaterEqual(len(calls), 2)

    def test_all_transport_failures_raise_global_connection_error(self) -> None:
        pool = self.make_pool(max_network_retries=2)
        with (
            patch(
                "ai_poet.synthetic_data.client.EndpointClient.chat_once",
                side_effect=EndpointRequestError(
                    "offline",
                    kind="transport",
                    retryable=True,
                    transport=True,
                ),
            ),
            self.assertRaises(GemmaConnectionError),
        ):
            pool.chat([{"role": "user", "content": "test"}])
