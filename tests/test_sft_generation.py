from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from generate_sft import (
    GemmaClient,
    GenerationSettings,
    GenerationTracer,
    PoemRecord,
    append_jsonl,
    choose_family,
    compose_response,
    extract_json_object,
    format_poem,
    generate_one,
    load_checkpoint,
    load_poems,
    meter_name,
    poem_hash,
    run,
    sft_split,
    split_poem_chunks,
    validate_generation,
    write_outputs,
    _settings,
    build_parser,
)
from sft_templates import TEMPLATE_FAMILIES, build_messages, eligible_families


TEST_TMP = Path(__file__).parent / "_tmp"


def remove_test_files(*names: str) -> None:
    for name in names:
        (TEST_TMP / name).unlink(missing_ok=True)


def make_poem(
    *,
    meter_id: int = 1,
    verses: tuple[str, ...] = (
        "يا قلب صبرا على الأيام",
        "فالليل يعقبه الضياء",
        "واجعل رجاءك خير زاد",
        "حتى يزول بك العناء",
    ),
) -> PoemRecord:
    return PoemRecord(
        sample_id=poem_hash(verses),
        source_row_indices=(7,),
        source_urls=("https://example.test/poem/7",),
        poet_name="شاعر الاختبار",
        poem_title="عنوان الاختبار",
        meter_id=meter_id,
        meter_name=meter_name(meter_id),
        verses=verses,
        metadata_conflict=False,
    )


def valid_value(poem: PoemRecord, minimum: int = 80) -> dict[str, str]:
    instruction_seed = (
        f"أنت شاعر عربي فصيح. اكتب {poem.couplet_count} من الأبيات على بحر "
        f"{poem.meter_name}. فصّل الموضوع والمعاني والصور والبلاغة والقافية "
        "والجو العاطفي، وراع سلامة اللغة والنطق العروضي عند بناء النص. "
    )
    reasoning_seed = (
        "مرحلة التفكير والتحرير: أحدد المعنى ثم أختار صورة بلاغية واستعارة مناسبة. "
        "أجرب صياغة أولى، ثم أجري تعديلًا وتحريرًا بعد فحص الوزن والإيقاع والقافية. "
        "أحافظ على وحدة المعاني وأشرح سبب كل محاولة وتعديل. النتيجة النهائية:"
    )
    repetitions = max(2, minimum // min(len(instruction_seed), len(reasoning_seed)) + 2)
    return {
        "instruction": instruction_seed * repetitions,
        "reasoning": reasoning_seed * repetitions,
    }


class QueueClient:
    def __init__(
        self,
        outputs: list[str],
        tracer: GenerationTracer | None = None,
    ) -> None:
        self.outputs = list(outputs)
        self.calls: list[list[dict[str, str]]] = []
        self.tracer = tracer

    def chat(self, messages, **kwargs):
        self.calls.append(list(messages))
        if not self.outputs:
            raise AssertionError("unexpected client call")
        return self.outputs.pop(0)


class ConfigurationTests(unittest.TestCase):
    def test_settings_load_required_values_from_dotenv(self) -> None:
        args = build_parser().parse_args([])
        environment = {
            "GEMMA_ENDPOINT": "https://env.example/v1/chat/completions",
            "GEMMA_MODEL": "env-model",
            "GEMMA_API_KEY": "env-secret",
        }

        with (
            patch("generate_sft.load_dotenv") as load_dotenv,
            patch.dict("os.environ", environment, clear=True),
        ):
            result = _settings(args)

        load_dotenv.assert_called_once_with()
        self.assertEqual(result.endpoint, environment["GEMMA_ENDPOINT"])
        self.assertEqual(result.model, environment["GEMMA_MODEL"])
        self.assertEqual(result.api_key, environment["GEMMA_API_KEY"])

    def test_settings_require_every_dotenv_value(self) -> None:
        args = build_parser().parse_args([])
        environment = {
            "GEMMA_ENDPOINT": "https://env.example/v1/chat/completions",
            "GEMMA_MODEL": "env-model",
            "GEMMA_API_KEY": "env-secret",
        }

        for missing_name in environment:
            with self.subTest(missing_name=missing_name):
                incomplete = environment | {missing_name: ""}
                with (
                    patch("generate_sft.load_dotenv"),
                    patch.dict("os.environ", incomplete, clear=True),
                    self.assertRaisesRegex(ValueError, missing_name),
                ):
                    _settings(args)

    def test_parser_has_no_endpoint_or_model_override(self) -> None:
        help_text = build_parser().format_help()
        self.assertNotIn("--endpoint", help_text)
        self.assertNotIn("--model", help_text)


def settings(**overrides) -> GenerationSettings:
    values = {
        "endpoint": "https://example.test/v1/chat/completions",
        "model": "gemma-test",
        "api_key": "secret",
        "insecure": False,
        "timeout": 1,
        "max_network_retries": 1,
        "max_repairs": 2,
        "temperature": 0.4,
        "top_p": 0.9,
        "max_tokens": 4096,
        "min_chars": 80,
        "max_source_chars": 24000,
        "chunk_chars": 12000,
    }
    values.update(overrides)
    return GenerationSettings(**values)


class TemplateTests(unittest.TestCase):
    def test_every_family_builds_two_few_shot_pairs(self) -> None:
        poem = make_poem()
        family_specific_examples = set()
        for family in TEMPLATE_FAMILIES:
            messages = build_messages(
                family=family,
                meter_name=poem.meter_name,
                couplet_count=poem.couplet_count,
                poem_text=poem.poem_text,
                minimum_chars=1500,
            )
            self.assertEqual([m["role"] for m in messages], [
                "system", "user", "assistant", "user", "assistant", "user"
            ])
            for assistant in (messages[2], messages[4]):
                self.assertEqual(
                    set(json.loads(assistant["content"])),
                    {"instruction", "reasoning"},
                )
            self.assertIn(family.focus, messages[0]["content"])
            self.assertIn(family.focus, messages[3]["content"])
            family_specific_examples.add(messages[4]["content"])
        self.assertEqual(len(family_specific_examples), len(TEMPLATE_FAMILIES))

    def test_prose_uses_prose_examples_and_excludes_prosody_family(self) -> None:
        families = eligible_families("النثر")
        self.assertNotIn("prosody_rhyme", {f.template_id for f in families})
        messages = build_messages(
            family=families[0],
            meter_name="النثر",
            couplet_count=1,
            poem_text="صورة أولى = صورة ثانية",
            minimum_chars=100,
        )
        prose_demonstration = json.loads(messages[2]["content"])
        self.assertIn("الشعر المنثور", prose_demonstration["instruction"])


class CorpusTests(unittest.TestCase):
    def test_meter_mapping_and_poem_format(self) -> None:
        self.assertEqual(meter_name(0), "البسيط")
        self.assertEqual(meter_name(14), "النثر")
        with self.assertRaises(ValueError):
            meter_name(17)
        with self.assertRaises(ValueError):
            meter_name(-1)
        self.assertEqual(format_poem(("صدر", "عجز")), "صدر = عجز")
        with self.assertRaises(ValueError):
            format_poem(("شطر وحيد",))

    def test_hash_split_and_family_are_stable(self) -> None:
        poem = make_poem()
        self.assertEqual(poem_hash(poem.verses), poem.sample_id)
        self.assertEqual(sft_split(poem.sample_id), sft_split(poem.sample_id))
        self.assertEqual(
            choose_family(poem).template_id,
            choose_family(poem).template_id,
        )

    def test_chunks_preserve_couplet_boundaries(self) -> None:
        verses = ("أ" * 10, "ب" * 10, "ج" * 10, "د" * 10)
        chunks = split_poem_chunks(verses, 25)
        self.assertEqual(chunks, ["أ" * 10 + " = " + "ب" * 10, "ج" * 10 + " = " + "د" * 10])

    def test_load_poems_deduplicates_and_uses_meter_majority(self) -> None:
        verses = ["صدر البيت", "عجز البيت"]
        rows = [
            {
                "poem_title": None,
                "poem_meter": 0,
                "poem_verses": verses,
                "poem_url": "https://example.test/a",
                "poet_name": "شاعر",
            },
            {
                "poem_title": "عنوان",
                "poem_meter": 10,
                "poem_verses": verses,
                "poem_url": "https://example.test/b",
                "poet_name": "شاعر",
            },
            {
                "poem_title": "عنوان",
                "poem_meter": 10,
                "poem_verses": verses,
                "poem_url": "https://example.test/c",
                "poet_name": "شاعر",
            },
        ]
        path = TEST_TMP / "source.parquet"
        remove_test_files(path.name)
        self.addCleanup(remove_test_files, path.name)
        try:
            pq.write_table(pa.Table.from_pylist(rows), path)
            poems = load_poems(path)
        finally:
            remove_test_files(path.name)
        self.assertEqual(len(poems), 1)
        self.assertEqual(poems[0].meter_name, "المديد")
        self.assertTrue(poems[0].metadata_conflict)
        self.assertEqual(len(poems[0].source_urls), 3)


class ValidationTests(unittest.TestCase):
    def test_extracts_plain_and_fenced_json(self) -> None:
        value = {"instruction": "تعليمات", "reasoning": "تحرير"}
        raw = json.dumps(value, ensure_ascii=False)
        self.assertEqual(extract_json_object(raw), value)
        self.assertEqual(extract_json_object(f"```json\n{raw}\n```"), value)

    def test_validation_accepts_grounded_result(self) -> None:
        poem = make_poem()
        self.assertEqual(validate_generation(valid_value(poem), poem, 80), [])

    def test_validation_catches_wrong_count_and_short_content(self) -> None:
        poem = make_poem()
        errors = validate_generation(
            {"instruction": "بحر الخفيف", "reasoning": "قصير"}, poem, 80
        )
        self.assertTrue(any("couplet count" in error for error in errors))
        self.assertTrue(any("at least" in error for error in errors))

    def test_response_ends_with_exact_source_poem(self) -> None:
        poem = make_poem()
        response = compose_response("تحرير\n\nالنتيجة النهائية:", poem)
        self.assertTrue(response.startswith("تحرير"))
        self.assertTrue(response.endswith(poem.poem_text))
        self.assertEqual(response.count(poem.poem_text), 1)

    def test_response_removes_poem_echoed_before_reasoning(self) -> None:
        poem = make_poem()
        model_reasoning = (
            f"القصيدة النهائية:\n{poem.poem_text}\n\n"
            "مرحلة التفكير والتحرير:\nأشرح المعنى والصورة والتعديل.\n"
            "النتيجة النهائية:"
        )
        response = compose_response(model_reasoning, poem)
        self.assertTrue(response.startswith("مرحلة التفكير والتحرير:"))
        self.assertTrue(response.endswith(poem.poem_text))
        self.assertEqual(response.count(poem.poem_text), 1)
        self.assertEqual(response.count("النتيجة النهائية:"), 1)

    def test_response_removes_individually_echoed_hemistichs(self) -> None:
        poem = make_poem()
        model_reasoning = (
            "\n".join(poem.verses)
            + "\nمرحلة التفكير والتحرير:\nأشرح المعنى والصورة والتعديل."
        )
        response = compose_response(model_reasoning, poem)
        self.assertTrue(response.startswith("مرحلة التفكير والتحرير:"))
        self.assertEqual(response.count(poem.poem_text), 1)
        marker_position = response.index("النتيجة النهائية:")
        poem_position = response.index(poem.poem_text)
        self.assertLess(marker_position, poem_position)


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
        with patch("generate_sft.urlopen", return_value=FakeResponse()):
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
        args = build_parser().parse_args(
            ["--input", str(TEST_TMP / "source.parquet"), "--output-dir", str(TEST_TMP)]
        )

        def assert_live_jsonl(*_args, **_kwargs) -> None:
            lines = (TEST_TMP / "ashaar_sft.jsonl").read_text("utf-8").splitlines()
            self.assertEqual([json.loads(line) for line in lines], [record])

        with (
            patch("generate_sft._settings", return_value=settings()),
            patch("generate_sft.load_poems", return_value=[poem]),
            patch("generate_sft.file_sha256", return_value="source-digest"),
            patch("generate_sft.generate_one", return_value=record),
            patch("generate_sft.write_outputs", side_effect=assert_live_jsonl),
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(run(args), 0)


if __name__ == "__main__":
    unittest.main()
