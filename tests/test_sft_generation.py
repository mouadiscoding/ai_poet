from __future__ import annotations

import json
from pathlib import Path
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from generate_sft import (
    GenerationSettings,
    PoemRecord,
    choose_family,
    compose_response,
    extract_json_object,
    format_poem,
    generate_one,
    load_checkpoint,
    load_poems,
    meter_name,
    poem_hash,
    sft_split,
    split_poem_chunks,
    validate_generation,
    write_outputs,
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
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages, **kwargs):
        self.calls.append(list(messages))
        if not self.outputs:
            raise AssertionError("unexpected client call")
        return self.outputs.pop(0)


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
        finally:
            remove_test_files(*generated_files)


if __name__ == "__main__":
    unittest.main()
