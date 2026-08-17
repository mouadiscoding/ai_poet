from __future__ import annotations

import json
from pathlib import Path
import unittest
from dataclasses import replace
from unittest.mock import patch

from ai_poet.synthetic_data.checkpoint import CheckpointWriter, load_checkpoint_state
from ai_poet.synthetic_data.config import RunSettings
from ai_poet.synthetic_data.runner import _validate_output_task, run
from ai_poet.synthetic_data.tasks.base import (
    TASK_MCQ,
    TASK_POEM_GENERATION,
    TASK_POEM_RECONSTRUCTION,
    default_output_dir,
    get_task_workflow,
)
from ai_poet.synthetic_data.tasks.mcq import (
    extract_candidate as extract_mcq_candidate,
    generate_one as generate_mcq,
    order_choices,
    question_domain,
)
from ai_poet.synthetic_data.tasks.reconstruction import (
    build_generation_messages as build_reconstruction_messages,
    corruption_count,
    extract_candidate as extract_reconstruction_candidate,
    generate_one as generate_reconstruction,
)
from tests.synthetic_data_helpers import (
    QueueClient,
    TEST_TMP,
    make_poem,
    remove_test_files,
    settings,
    valid_verdict,
)


def valid_mcq_candidate() -> dict[str, object]:
    answers = [
        "التدرج من الشدة إلى الأمل",
        "الاستسلام الكامل لليأس",
        "وصف رحلة في البحر",
        "الفخر بالنسب والقبيلة",
    ]
    return {
        "question": "ما المسار الدلالي الأوضح الذي تنتظم فيه أبيات القصيدة؟",
        "correct_answer": answers[0],
        "distractors": answers[1:],
        "reasoning": {
            "approach": "أتتبع انتقال المعنى بين صورة الليل والضياء ثم أثر الرجاء في الخاتمة.",
            "evidence": ["فالليل يعقبه الضياء", "واجعل رجاءك خير زاد"],
            "answer_assessments": [
                {
                    "answer": answer,
                    "assessment": (
                        "أقارن هذا الجواب بصور الليل والضياء والرجاء لأحدد مدى "
                        "اتصاله بالمسار الدلالي الظاهر."
                    ),
                }
                for answer in answers
            ],
            "conclusion": (
                "تؤكد الشواهد انتقال القصيدة من احتمال الشدة إلى انتظار الانفراج."
            ),
        },
    }


def valid_reconstruction_candidate():
    poem = make_poem()
    count = corruption_count(poem)
    original_lines = poem.poem_text.splitlines()
    replacements = [("يا", "أيها"), ("واجعل", "واترك")]
    corrupted_lines = list(original_lines)
    repairs = []
    for index in range(count):
        correct, corrupted = replacements[index]
        corrupted_lines[index] = corrupted_lines[index].replace(
            correct, corrupted, 1
        )
        repairs.append(
            {
                "couplet_index": index + 1,
                "corrupted_fragment": corrupted,
                "corrected_fragment": correct,
                "diagnosis": (
                    "يفسد اللفظ المحرف اتجاه الخطاب ويضعف الصلة بالسياق الذي يحيط به."
                ),
                "context_evidence": (
                    "تدل صورة الصبر والضياء والرجاء في الأبيات المجاورة على اللفظ الأصلي."
                ),
                "repair_reason": (
                    "يعيد التصحيح المعنى المتدرج والجرس المتسق مع بقية ألفاظ القصيدة."
                ),
            }
        )
    return poem, {
        "corrupted_poem": "\n".join(corrupted_lines),
        "repairs": repairs,
    }


class TaskRegistryTests(unittest.TestCase):
    def test_registry_and_default_directories(self) -> None:
        self.assertEqual(
            get_task_workflow(TASK_POEM_GENERATION).task_type,
            TASK_POEM_GENERATION,
        )
        self.assertEqual(get_task_workflow(TASK_MCQ).version, 1)
        self.assertEqual(get_task_workflow(TASK_POEM_RECONSTRUCTION).version, 2)
        self.assertEqual(default_output_dir(TASK_MCQ), Path("data/ashaar_mcq_sft"))

    def test_output_directory_rejects_another_task(self) -> None:
        TEST_TMP.mkdir(parents=True, exist_ok=True)
        manifest = TEST_TMP / "manifest.json"
        manifest.unlink(missing_ok=True)
        manifest.write_text(
            json.dumps({"task_type": TASK_MCQ}), encoding="utf-8"
        )
        self.addCleanup(manifest.unlink, missing_ok=True)
        with self.assertRaisesRegex(ValueError, "belongs to task mcq"):
            _validate_output_task(TEST_TMP, TASK_POEM_RECONSTRUCTION)

    def test_v3_generic_stage_maps_poem_generation_resume(self) -> None:
        name = "generic_stage_checkpoint.jsonl"
        remove_test_files(name)
        self.addCleanup(remove_test_files, name)
        TEST_TMP.mkdir(exist_ok=True)
        path = TEST_TMP / name
        CheckpointWriter(path).append(
            {
                "event": "stage_success",
                "sample_id": "sample",
                "task_type": TASK_POEM_GENERATION,
                "task_version": 8,
                "stage_name": "instruction",
                "stage_key": "instruction",
                "workflow_fingerprint": "fingerprint",
                "payload": {
                    "instruction": "تعليمات",
                    "instruction_fingerprint": "instruction-fingerprint",
                    "template_id": "semantic_arc",
                },
            }
        )
        state = load_checkpoint_state(path)
        self.assertEqual(state.instructions["sample"]["instruction"], "تعليمات")
        self.assertEqual(
            state.stages["sample"]["instruction"]["generation_fingerprint"],
            "fingerprint",
        )

    def test_runner_dispatches_selected_task_and_writes_task_manifest(self) -> None:
        poem = make_poem()
        real_workflow = get_task_workflow(TASK_MCQ)

        def fake_generate(source_poem, _client, _settings, **_kwargs):
            return {
                "sample_id": source_poem.sample_id,
                "record_id": f"mcq:{source_poem.sample_id}",
                "task_type": TASK_MCQ,
                "task_version": 1,
                "instruction": "تعليمات السؤال",
                "response": "تحليل\n\nالإجابة الصحيحة: أ) جواب",
                "messages": [],
                "sft_split": "train",
                "metadata_conflict": False,
                "validation_status": "passed",
                "generation_endpoint_ids": ["legacy"],
                "endpoint_failover_count": 0,
                "network_attempts": 2,
                "truncated_completions": 0,
            }

        workflow = replace(real_workflow, generate_one=fake_generate)
        TEST_TMP.mkdir(exist_ok=True)
        generated_files = (
            "generation_checkpoint.jsonl",
            "ashaar_sft.jsonl",
            "ashaar_sft.parquet",
            "failures.jsonl",
            "manifest.json",
        )
        remove_test_files(*generated_files)
        self.addCleanup(remove_test_files, *generated_files)
        output_dir = TEST_TMP
        run_settings = RunSettings(
            input=Path("unused.parquet"),
            output_dir=output_dir,
            concurrency=1,
            limit=None,
            trace=False,
            generation=settings(),
            task_type=TASK_MCQ,
        )
        with (
            patch("ai_poet.synthetic_data.runner.load_poems", return_value=[poem]),
            patch("ai_poet.synthetic_data.runner.file_sha256", return_value="source"),
            patch(
                "ai_poet.synthetic_data.runner.get_task_workflow",
                return_value=workflow,
            ),
            patch("ai_poet.synthetic_data.runner.GemmaClient", return_value=object()),
        ):
            self.assertEqual(run(run_settings), 0)
        manifest = json.loads(
            (output_dir / "manifest.json").read_text("utf-8")
        )
        self.assertEqual(manifest["task_type"], TASK_MCQ)
        records = (output_dir / "ashaar_sft.jsonl").read_text("utf-8")
        self.assertIn(f'"task_type": "{TASK_MCQ}"', records)


class McqWorkflowTests(unittest.TestCase):
    def test_domain_and_choice_order_are_deterministic(self) -> None:
        poem = make_poem()
        candidate = extract_mcq_candidate(valid_mcq_candidate())
        self.assertEqual(question_domain(poem.sample_id), question_domain(poem.sample_id))
        self.assertEqual(
            order_choices(poem.sample_id, candidate),
            order_choices(poem.sample_id, candidate),
        )
        self.assertEqual(len(order_choices(poem.sample_id, candidate)), 4)

    def test_duplicate_answers_are_rejected(self) -> None:
        candidate = valid_mcq_candidate()
        candidate["distractors"][0] = candidate["correct_answer"]
        with self.assertRaisesRegex(ValueError, "distinct"):
            extract_mcq_candidate(candidate)

    def test_generation_renders_reasoning_and_exact_final_answer(self) -> None:
        poem = make_poem()
        outputs = [
            json.dumps(valid_mcq_candidate(), ensure_ascii=False),
            json.dumps(valid_verdict(), ensure_ascii=False),
        ]
        record = generate_mcq(poem, QueueClient(outputs), settings())
        self.assertEqual(record["task_type"], TASK_MCQ)
        self.assertEqual(len(record["choices"]), 4)
        self.assertIn(poem.poem_text, record["instruction"])
        self.assertTrue(record["response"].startswith("التحليل والاستدلال:"))
        correct = next(
            choice
            for choice in record["choices"]
            if choice["label"] == record["correct_choice_label"]
        )
        self.assertTrue(
            record["response"].endswith(
                f"الإجابة الصحيحة: {correct['label']}) {correct['text']}"
            )
        )

    def test_invalid_mcq_is_repaired_before_export(self) -> None:
        poem = make_poem()
        outputs = [
            "{}",
            json.dumps(valid_mcq_candidate(), ensure_ascii=False),
            json.dumps(valid_verdict(), ensure_ascii=False),
        ]
        client = QueueClient(outputs)
        record = generate_mcq(poem, client, settings())
        self.assertEqual(record["generation_attempts"], 2)
        self.assertEqual(record["validation_status"], "passed_after_repair")
        self.assertIn("الجواب السابق غير صالح", client.calls[1][-1]["content"])


class ReconstructionWorkflowTests(unittest.TestCase):
    def test_corruption_count_is_stable_and_bounded(self) -> None:
        poem = make_poem()
        self.assertEqual(corruption_count(poem), corruption_count(poem))
        self.assertIn(corruption_count(poem), (1, 2))

    def test_structural_and_local_diff_contract(self) -> None:
        poem, candidate = valid_reconstruction_candidate()
        parsed = extract_reconstruction_candidate(
            candidate,
            poem=poem,
            expected_count=corruption_count(poem),
        )
        self.assertEqual(len(parsed["repairs"]), corruption_count(poem))

        invalid = dict(candidate)
        invalid["corrupted_poem"] = candidate["corrupted_poem"].splitlines()[0]
        with self.assertRaisesRegex(ValueError, "couplet count"):
            extract_reconstruction_candidate(
                invalid,
                poem=poem,
                expected_count=corruption_count(poem),
            )

        leaked = json.loads(json.dumps(candidate, ensure_ascii=False))
        leaked["repairs"][0]["diagnosis"] = poem.poem_text
        with self.assertRaisesRegex(ValueError, "complete original poem"):
            extract_reconstruction_candidate(
                leaked,
                poem=poem,
                expected_count=corruption_count(poem),
            )

    def test_prompt_defines_one_based_couplet_indices(self) -> None:
        poem = make_poem()
        messages = build_reconstruction_messages(poem, corruption_count(poem))
        self.assertIn("couplet_index يبدأ من 1", messages[0]["content"])
        self.assertIn("20 حرفًا على الأقل", messages[0]["content"])

    def test_zero_based_couplet_indices_are_canonicalized(self) -> None:
        poem, candidate = valid_reconstruction_candidate()
        for repair in candidate["repairs"]:
            repair["couplet_index"] -= 1

        parsed = extract_reconstruction_candidate(
            candidate,
            poem=poem,
            expected_count=corruption_count(poem),
        )

        self.assertEqual(
            [repair["couplet_index"] for repair in parsed["repairs"]],
            list(range(1, corruption_count(poem) + 1)),
        )

    def test_diacritic_only_fragment_difference_uses_exact_source_text(self) -> None:
        poem = make_poem(
            verses=("جاء الرَبيع بالبشائر", "فأضحى الروض مبتسما"),
        )
        candidate = {
            "corrupted_poem": poem.poem_text.replace("الرَبيع", "الشتاء"),
            "repairs": [
                {
                    "couplet_index": 1,
                    "corrupted_fragment": "الشتاء",
                    "corrected_fragment": "الربيع",
                    "diagnosis": "يخالف هذا الفصل صورة البشائر والنماء التي يبنيها البيت.",
                    "context_evidence": "يربط السياق مجيء الفصل بالبشائر وابتسام الروض بعده.",
                    "repair_reason": "يعيد اللفظ الأصلي الترابط بين الفصل والنماء في صورة البيت.",
                }
            ],
        }

        parsed = extract_reconstruction_candidate(
            candidate,
            poem=poem,
            expected_count=1,
        )

        self.assertEqual(parsed["repairs"][0]["corrected_fragment"], "الرَبيع")

    def test_index_mismatch_reports_actual_and_one_based_indices(self) -> None:
        poem, candidate = valid_reconstruction_candidate()
        for repair in candidate["repairs"]:
            repair["couplet_index"] += 10

        with self.assertRaisesRegex(
            ValueError,
            r"reported repair couplet indices .* changed couplets .* one-based",
        ):
            extract_reconstruction_candidate(
                candidate,
                poem=poem,
                expected_count=corruption_count(poem),
            )

    def test_short_repair_detail_reports_required_length(self) -> None:
        poem, candidate = valid_reconstruction_candidate()
        candidate["repairs"][0]["context_evidence"] = "شاهد قصير"

        with self.assertRaisesRegex(
            ValueError,
            r"context_evidence must contain at least 20 characters",
        ):
            extract_reconstruction_candidate(
                candidate,
                poem=poem,
                expected_count=corruption_count(poem),
            )

    def test_generation_appends_exact_original_without_model_copy(self) -> None:
        poem, candidate = valid_reconstruction_candidate()
        outputs = [
            json.dumps(candidate, ensure_ascii=False),
            json.dumps(valid_verdict(), ensure_ascii=False),
        ]
        record = generate_reconstruction(poem, QueueClient(outputs), settings())
        self.assertEqual(record["task_type"], TASK_POEM_RECONSTRUCTION)
        self.assertIn(record["corrupted_poem"], record["instruction"])
        self.assertTrue(record["response"].startswith("التحليل والاستدلال:"))
        self.assertTrue(record["response"].endswith(poem.poem_text))
        prefix = record["response"].rsplit(poem.poem_text, 1)[0]
        self.assertNotIn(poem.poem_text, prefix)


if __name__ == "__main__":
    unittest.main()
