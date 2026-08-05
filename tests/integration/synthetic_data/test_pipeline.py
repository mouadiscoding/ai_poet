from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import unittest
from unittest.mock import patch

import pyarrow.parquet as pq

from ai_poet.synthetic_data.checkpoint import load_checkpoint
from ai_poet.synthetic_data.client import GemmaClient
from ai_poet.synthetic_data.config import RunSettings
from ai_poet.synthetic_data.generation import generate_one
from ai_poet.synthetic_data.outputs import append_jsonl, write_outputs
from ai_poet.synthetic_data.runner import run
from ai_poet.synthetic_data.tracing import GenerationTracer
from tests.synthetic_data_helpers import TEST_TMP, QueueClient, make_poem, remove_test_files, settings, valid_value

class PipelineTests(unittest.TestCase):
    def test_api_trace_contains_full_exchange_and_redacts_secret(self) -> None:
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps(
                    {
                        "id": "completion-1",
                        "model": "gemma-test",
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": "raw answer",
                                    "reasoning_content": "server reasoning",
                                },
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 4},
                    }
                ).encode("utf-8")

        TEST_TMP.mkdir(exist_ok=True)
        path = TEST_TMP / "api_trace.jsonl"
        remove_test_files(path.name)
        self.addCleanup(remove_test_files, path.name)
        tracer = GenerationTracer(path, secrets=("secret",))
        client = GemmaClient(settings(), tracer=tracer)
        stdout = io.StringIO()
        with patch(
            "ai_poet.synthetic_data.client.urlopen", return_value=FakeResponse()
        ):
            with redirect_stdout(stdout):
                result = client.chat(
                    [{"role": "user", "content": "prompt containing secret"}],
                    seed=42,
                    trace_context={"sample_id": "abc", "request_kind": "initial"},
                )

        self.assertEqual(result, "raw answer")
        event = json.loads(path.read_text("utf-8"))
        self.assertEqual(event["request"]["messages"][0]["content"],
                         "prompt containing [REDACTED]")
        self.assertEqual(
            event["response"]["choices"][0]["message"]["reasoning_content"],
            "server reasoning",
        )
        self.assertEqual(event["response"]["usage"]["completion_tokens"], 4)
        self.assertNotIn("secret", path.read_text("utf-8"))
        self.assertNotIn("secret", stdout.getvalue())

    def test_generation_trace_explains_template_and_records_final_output(self) -> None:
        poem = make_poem()
        value = valid_value(poem)
        TEST_TMP.mkdir(exist_ok=True)
        path = TEST_TMP / "generation_trace.jsonl"
        remove_test_files(path.name)
        self.addCleanup(remove_test_files, path.name)
        tracer = GenerationTracer(path, secrets=("secret",))
        client = QueueClient([json.dumps(value, ensure_ascii=False)], tracer=tracer)

        with redirect_stdout(io.StringIO()):
            record = generate_one(poem, client, settings())

        events = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
        by_type = {event["event"]: event for event in events}
        self.assertIn("why_used", by_type["template_selection"])
        self.assertEqual(
            by_type["validation_result"]["raw_model_content"],
            json.dumps(value, ensure_ascii=False),
        )
        self.assertEqual(
            by_type["final_output"]["final_assistant_response"], record["response"]
        )
        self.assertNotIn("raw_model_content", record)
        self.assertNotIn("generation_trace", record)

    def test_generate_one_repairs_invalid_json(self) -> None:
        poem = make_poem()
        valid = json.dumps(valid_value(poem), ensure_ascii=False)
        client = QueueClient(["{}", valid])
        record = generate_one(poem, client, settings())
        self.assertEqual(record["generation_attempts"], 2)
        self.assertEqual(record["validation_status"], "passed_after_repair")
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(record["messages"][1]["content"], record["response"])

    def test_oversized_poem_is_summarized_then_generated(self) -> None:
        poem = make_poem(verses=("أ" * 30, "ب" * 30, "ج" * 30, "د" * 30))
        valid = json.dumps(valid_value(poem), ensure_ascii=False)
        client = QueueClient(["ملخص عربي واضح", "ملخص عربي ثان", valid])
        record = generate_one(
            poem,
            client,
            settings(max_source_chars=20, chunk_chars=70),
        )
        self.assertTrue(record["oversized_for_sft"])
        self.assertEqual(len(client.calls), 3)
        self.assertIn("ملخص المقطع", client.calls[-1][-1]["content"])

    def test_checkpoint_and_exports_round_trip(self) -> None:
        poem = make_poem()
        value = valid_value(poem)
        client = QueueClient([json.dumps(value, ensure_ascii=False)])
        record = generate_one(poem, client, settings())
        generated_files = (
            "checkpoint.jsonl",
            "ashaar_sft.jsonl",
            "ashaar_sft.parquet",
            "failures.jsonl",
            "manifest.json",
        )
        remove_test_files(*generated_files)
        self.addCleanup(remove_test_files, *generated_files)
        root = TEST_TMP
        try:
            checkpoint = root / "checkpoint.jsonl"
            checkpoint.write_text(
                json.dumps(
                    {
                        "status": "success",
                        "sample_id": poem.sample_id,
                        "record": record,
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            successes, failures = load_checkpoint(checkpoint)
            self.assertIn(poem.sample_id, successes)
            self.assertFalse(failures)
            write_outputs(root, [poem], successes, failures, settings())
            self.assertEqual(pq.read_table(root / "ashaar_sft.parquet").num_rows, 1)
            manifest = json.loads((root / "manifest.json").read_text("utf-8"))
            self.assertTrue(manifest["complete"])
            self.assertFalse(manifest["trace"]["enabled"])
        finally:
            remove_test_files(*generated_files)

    def test_append_jsonl_makes_each_record_visible(self) -> None:
        path = TEST_TMP / "live_sft.jsonl"
        remove_test_files(path.name)
        self.addCleanup(remove_test_files, path.name)

        append_jsonl(path, {"sample_id": "first"})
        self.assertEqual(
            [json.loads(line) for line in path.read_text("utf-8").splitlines()],
            [{"sample_id": "first"}],
        )

        append_jsonl(path, {"sample_id": "second"})
        self.assertEqual(
            [json.loads(line) for line in path.read_text("utf-8").splitlines()],
            [{"sample_id": "first"}, {"sample_id": "second"}],
        )

    def test_run_publishes_sft_record_before_final_export(self) -> None:
        poem = make_poem()
        record = generate_one(
            poem,
            QueueClient([json.dumps(valid_value(poem), ensure_ascii=False)]),
            settings(),
        )
        generated_files = ("generation_checkpoint.jsonl", "ashaar_sft.jsonl")
        remove_test_files(*generated_files)
        self.addCleanup(remove_test_files, *generated_files)
        run_settings = RunSettings(
            input=TEST_TMP / "source.parquet",
            output_dir=TEST_TMP,
            concurrency=4,
            limit=None,
            trace=False,
            generation=settings(),
        )

        def assert_live_jsonl(*_args, **_kwargs) -> None:
            lines = (TEST_TMP / "ashaar_sft.jsonl").read_text("utf-8").splitlines()
            self.assertEqual([json.loads(line) for line in lines], [record])

        with (
            patch("ai_poet.synthetic_data.runner.load_poems", return_value=[poem]),
            patch("ai_poet.synthetic_data.runner.file_sha256", return_value="source-digest"),
            patch("ai_poet.synthetic_data.runner.generate_one", return_value=record),
            patch(
                "ai_poet.synthetic_data.runner.write_outputs",
                side_effect=assert_live_jsonl,
            ),
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(run(run_settings), 0)
