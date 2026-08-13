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
from ai_poet.synthetic_data.errors import GenerationError
from ai_poet.synthetic_data.generation import generate_one
from ai_poet.synthetic_data.outputs import append_jsonl, write_outputs
from ai_poet.synthetic_data.runner import run
from ai_poet.synthetic_data.tracing import GenerationTracer
from ai_poet.synthetic_data.prompts.templates import PROMPT_TEMPLATES, TEMPLATE_VERSION
from tests.synthetic_data_helpers import (
    TEST_TMP,
    QueueClient,
    make_poem,
    remove_test_files,
    settings,
    valid_instruction_value,
    valid_pipeline_outputs,
    valid_reasoning_value,
    valid_verdict,
)

class PipelineTests(unittest.TestCase):
    def tearDown(self) -> None:
        remove_test_files("failures.jsonl")

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
        instruction_value = valid_instruction_value(poem)
        reasoning_value = valid_reasoning_value(poem)
        TEST_TMP.mkdir(exist_ok=True)
        path = TEST_TMP / "generation_trace.jsonl"
        remove_test_files(path.name)
        self.addCleanup(remove_test_files, path.name)
        tracer = GenerationTracer(path, secrets=("secret",))
        client = QueueClient(valid_pipeline_outputs(poem), tracer=tracer)

        with redirect_stdout(io.StringIO()):
            record = generate_one(poem, client, settings())

        events = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
        by_type = {event["event"]: event for event in events}
        self.assertIn("why_used", by_type["template_selection"])
        self.assertEqual(
            by_type["instruction_generation_result"]["raw_model_content"],
            json.dumps(instruction_value, ensure_ascii=False),
        )
        self.assertEqual(
            by_type["reasoning_chunk_generation_result"]["parsed_output"],
            reasoning_value,
        )
        self.assertTrue(by_type["instruction_validation_result"]["passed"])
        self.assertTrue(by_type["reasoning_chunk_validation_result"]["passed"])
        self.assertEqual(
            by_type["final_output"]["final_assistant_response"], record["response"]
        )
        self.assertNotIn("raw_model_content", record)
        self.assertNotIn("generation_trace", record)

    def test_generate_one_repairs_invalid_json(self) -> None:
        poem = make_poem()
        valid = json.dumps(valid_instruction_value(poem), ensure_ascii=False)
        client = QueueClient(
            [
                "{}",
                valid,
                json.dumps(valid_verdict(), ensure_ascii=False),
                json.dumps(valid_reasoning_value(poem), ensure_ascii=False),
                json.dumps(valid_verdict(), ensure_ascii=False),
            ]
        )
        record = generate_one(poem, client, settings())
        self.assertEqual(record["generation_attempts"], 3)
        self.assertEqual(record["instruction_generation_attempts"], 2)
        self.assertEqual(record["reasoning_generation_attempts"], 1)
        self.assertEqual(record["validation_status"], "passed_after_repair")
        self.assertEqual(len(client.calls), 5)
        self.assertEqual(record["messages"][1]["content"], record["response"])

    def test_gemma_instruction_rejection_repairs_instruction_before_reasoning(self) -> None:
        poem = make_poem()
        candidate = json.dumps(valid_instruction_value(poem), ensure_ascii=False)
        rejection = json.dumps(
            {"passed": False, "errors": ["لم يذكر المخاطَب بوضوح"]},
            ensure_ascii=False,
        )
        client = QueueClient(
            [
                candidate,
                rejection,
                candidate,
                json.dumps(valid_verdict(), ensure_ascii=False),
                json.dumps(valid_reasoning_value(poem), ensure_ascii=False),
                json.dumps(valid_verdict(), ensure_ascii=False),
            ]
        )
        with patch(
            "ai_poet.synthetic_data.generation.random.choice",
            return_value=PROMPT_TEMPLATES[2],
        ) as choice:
            record = generate_one(poem, client, settings())

        self.assertEqual(record["generation_attempts"], 3)
        self.assertEqual(record["template_id"], "imagery_rhetoric")
        self.assertIn("لم يذكر المخاطَب بوضوح", client.calls[2][-1]["content"])
        choice.assert_called_once_with(PROMPT_TEMPLATES)

    def test_gemma_reasoning_rejection_repairs_only_that_chunk(self) -> None:
        poem = make_poem()
        reasoning = json.dumps(valid_reasoning_value(poem), ensure_ascii=False)
        rejection = json.dumps(
            {"passed": False, "errors": ["سبب مراجعة المسودة عام وغير ملموس"]},
            ensure_ascii=False,
        )
        client = QueueClient(
            [
                json.dumps(valid_instruction_value(poem), ensure_ascii=False),
                json.dumps(valid_verdict(), ensure_ascii=False),
                reasoning,
                rejection,
                reasoning,
                json.dumps(valid_verdict(), ensure_ascii=False),
            ]
        )
        record = generate_one(poem, client, settings())
        self.assertEqual(record["instruction_generation_attempts"], 1)
        self.assertEqual(record["reasoning_generation_attempts"], 2)
        self.assertIn("سبب مراجعة المسودة عام", client.calls[4][-1]["content"])

    def test_malformed_gemma_verdict_is_never_exported(self) -> None:
        poem = make_poem()
        candidate = json.dumps(valid_instruction_value(poem), ensure_ascii=False)
        client = QueueClient([candidate, "{}", "{}", "{}"])
        with self.assertRaises(GenerationError):
            generate_one(poem, client, settings())

    def test_malformed_gemma_verdict_is_retried_independently(self) -> None:
        poem = make_poem()
        outputs = valid_pipeline_outputs(poem)
        outputs[1:2] = ["not json", json.dumps(valid_verdict(), ensure_ascii=False)]
        client = QueueClient(outputs)

        record = generate_one(poem, client, settings())

        self.assertEqual(record["instruction_generation_attempts"], 1)
        self.assertEqual(
            client.call_kwargs[2]["trace_context"]["validator_format_attempt"],
            2,
        )

    def test_oversized_poem_is_summarized_then_generated(self) -> None:
        poem = make_poem(verses=("أ" * 30, "ب" * 30, "ج" * 30, "د" * 30))
        client = QueueClient(
            [
                "ملخص عربي واضح",
                "ملخص عربي ثان",
                json.dumps(valid_instruction_value(poem), ensure_ascii=False),
                json.dumps(valid_verdict(), ensure_ascii=False),
                json.dumps(valid_reasoning_value(poem), ensure_ascii=False),
                json.dumps(valid_verdict(), ensure_ascii=False),
            ]
        )
        record = generate_one(
            poem,
            client,
            settings(max_source_chars=20, chunk_chars=70),
        )
        self.assertTrue(record["oversized_for_sft"])
        self.assertEqual(len(client.calls), 6)
        self.assertIn("ملخص المقطع", client.calls[2][-1]["content"])
        self.assertIn("ملخص المقطع", client.calls[3][-1]["content"])
        for couplet in poem.poem_text.splitlines():
            self.assertIn(couplet, client.calls[4][-1]["content"])

    def test_long_reasoning_is_generated_in_ordered_three_couplet_chunks(self) -> None:
        poem = make_poem(
            verses=tuple(
                part
                for index in range(1, 11)
                for part in (f"صدر البيت {index}", f"عجز البيت {index}")
            )
        )
        outputs = [
            json.dumps(valid_instruction_value(poem), ensure_ascii=False),
            json.dumps(valid_verdict(), ensure_ascii=False),
        ]
        for start_offset, chunk_size in ((0, 3), (3, 3), (6, 3), (9, 1)):
            outputs.extend(
                [
                    json.dumps(
                        valid_reasoning_value(
                            poem,
                            start_offset=start_offset,
                            chunk_size=chunk_size,
                        ),
                        ensure_ascii=False,
                    ),
                    json.dumps(valid_verdict(), ensure_ascii=False),
                ]
            )

        client = QueueClient(outputs)
        record = generate_one(poem, client, settings())

        self.assertEqual(record["reasoning_chunk_count"], 4)
        self.assertEqual(record["reasoning_generation_attempts"], 4)
        self.assertEqual(record["generation_attempts"], 5)
        for index in range(1, 11):
            self.assertEqual(record["response"].count(f"البيت {index}:"), 1)
        self.assertTrue(record["response"].endswith(poem.poem_text))

    def test_checkpoint_and_exports_round_trip(self) -> None:
        poem = make_poem()
        client = QueueClient(valid_pipeline_outputs(poem))
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
            self.assertEqual(manifest["template_version"], TEMPLATE_VERSION)
            self.assertEqual(record["template_version"], TEMPLATE_VERSION)
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

    def test_failure_exports_include_categories_and_selection_counts(self) -> None:
        poem = make_poem()
        generated_files = ("ashaar_sft.jsonl", "failures.jsonl", "manifest.json")
        remove_test_files(*generated_files)
        self.addCleanup(remove_test_files, *generated_files)

        write_outputs(
            TEST_TMP,
            [poem],
            {},
            {poem.sample_id: "Gemma rejected reasoning: contradiction"},
            settings(),
            max_couplets=24,
            excluded_long_poems=3,
        )

        failure = json.loads((TEST_TMP / "failures.jsonl").read_text("utf-8"))
        manifest = json.loads((TEST_TMP / "manifest.json").read_text("utf-8"))
        self.assertEqual(failure["category"], "semantic_rejection")
        self.assertEqual(manifest["failure_categories"], {"semantic_rejection": 1})
        self.assertEqual(manifest["selection"]["excluded_long_poems"], 3)

    def test_run_publishes_sft_record_before_final_export(self) -> None:
        poem = make_poem()
        record = generate_one(
            poem,
            QueueClient(valid_pipeline_outputs(poem)),
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

    def test_run_regenerates_legacy_checkpoint_success(self) -> None:
        poem = make_poem()
        legacy_record = {"sample_id": poem.sample_id, "template_version": 1}
        current_record = {"sample_id": poem.sample_id, "template_version": TEMPLATE_VERSION}
        generated_files = ("generation_checkpoint.jsonl", "ashaar_sft.jsonl")
        remove_test_files(*generated_files)
        self.addCleanup(remove_test_files, *generated_files)
        run_settings = RunSettings(
            input=TEST_TMP / "source.parquet",
            output_dir=TEST_TMP,
            concurrency=1,
            limit=None,
            trace=False,
            generation=settings(),
        )

        with (
            patch("ai_poet.synthetic_data.runner.load_poems", return_value=[poem]),
            patch(
                "ai_poet.synthetic_data.runner.load_checkpoint",
                return_value=({poem.sample_id: legacy_record}, {}),
            ),
            patch(
                "ai_poet.synthetic_data.runner.file_sha256",
                return_value="source-digest",
            ),
            patch(
                "ai_poet.synthetic_data.runner.generate_one",
                return_value=current_record,
            ) as generate,
            patch("ai_poet.synthetic_data.runner.write_outputs"),
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(run(run_settings), 0)

        generate.assert_called_once()

    def test_run_excludes_long_poems_and_schedules_shortest_first(self) -> None:
        def poem_with_count(count: int):
            return make_poem(
                verses=tuple(
                    part
                    for index in range(count)
                    for part in (
                        f"صدر فريد {count}-{index}",
                        f"عجز فريد {count}-{index}",
                    )
                )
            )

        medium = poem_with_count(4)
        excluded = poem_with_count(25)
        short = poem_with_count(2)
        generated_files = (
            "generation_checkpoint.jsonl",
            "ashaar_sft.jsonl",
            "failures.jsonl",
        )
        remove_test_files(*generated_files)
        self.addCleanup(remove_test_files, *generated_files)
        run_settings = RunSettings(
            input=TEST_TMP / "source.parquet",
            output_dir=TEST_TMP,
            concurrency=1,
            limit=None,
            trace=False,
            generation=settings(),
            max_couplets=24,
        )
        call_order: list[str] = []

        def generate(poem, *_args, **_kwargs):
            call_order.append(poem.sample_id)
            return {"sample_id": poem.sample_id, "template_version": TEMPLATE_VERSION}

        with (
            patch(
                "ai_poet.synthetic_data.runner.load_poems",
                return_value=[medium, excluded, short],
            ),
            patch(
                "ai_poet.synthetic_data.runner.file_sha256",
                return_value="source-digest",
            ),
            patch("ai_poet.synthetic_data.runner.generate_one", side_effect=generate),
            patch("ai_poet.synthetic_data.runner.write_outputs") as write_outputs,
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(run(run_settings), 0)

        self.assertEqual(call_order, [short.sample_id, medium.sample_id])
        self.assertEqual(write_outputs.call_args.kwargs["excluded_long_poems"], 1)
        self.assertEqual(write_outputs.call_args.kwargs["max_couplets"], 24)
