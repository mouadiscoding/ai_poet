from __future__ import annotations

import json
import unittest

from ai_poet.synthetic_data.responses import compose_response, render_reasoning
from ai_poet.synthetic_data.validation import (
    extract_instruction,
    extract_json_object,
    extract_reasoning_chunk,
    instruction_contract_errors,
    parse_field_verdict,
)
from tests.synthetic_data_helpers import (
    make_poem,
    valid_instruction_value,
    valid_reasoning_value,
)


class ValidationTests(unittest.TestCase):
    def test_extracts_plain_and_fenced_json(self) -> None:
        value = {"instruction": "تعليمات"}
        raw = json.dumps(value, ensure_ascii=False)
        self.assertEqual(extract_json_object(raw), value)
        self.assertEqual(extract_json_object(f"```json\n{raw}\n```"), value)

    def test_instruction_extraction_requires_one_string_field(self) -> None:
        self.assertEqual(
            extract_instruction({"instruction": " تعليمات "}), "تعليمات"
        )
        with self.assertRaises(ValueError):
            extract_instruction({"instruction": "تعليمات", "reasoning": "زائد"})
        with self.assertRaises(ValueError):
            extract_instruction({"instruction": []})

    def test_instruction_contract_checks_deterministic_requirements(self) -> None:
        poem = make_poem()
        instruction = valid_instruction_value(poem)["instruction"]
        self.assertEqual(
            instruction_contract_errors(
                instruction,
                meter_name=poem.meter_name,
                couplet_count=poem.couplet_count,
                minimum_chars=80,
                source_hemistichs=poem.verses,
            ),
            [],
        )
        self.assertTrue(
            instruction_contract_errors(
                "الموضوع العام:\nقصير",
                meter_name=poem.meter_name,
                couplet_count=poem.couplet_count,
                minimum_chars=80,
                source_hemistichs=poem.verses,
            )
        )

    def test_instruction_contract_names_the_exact_missing_heading(self) -> None:
        poem = make_poem()
        instruction = valid_instruction_value(poem)["instruction"].replace(
            "شرح البحر المطلوب:", "شرح البحر الرمل:"
        )
        errors = instruction_contract_errors(
            instruction,
            meter_name=poem.meter_name,
            couplet_count=poem.couplet_count,
            minimum_chars=80,
            source_hemistichs=poem.verses,
        )
        self.assertIn(
            "instruction is missing required heading: شرح البحر المطلوب:",
            errors,
        )

    def test_extracts_complete_source_linked_reasoning_chunk(self) -> None:
        poem = make_poem()
        value = valid_reasoning_value(poem)
        overview, blocks = extract_reasoning_chunk(
            value,
            expected_indices=[1, 2],
            expected_couplets=poem.poem_text.splitlines(),
            include_overview=True,
        )
        self.assertTrue(overview)
        self.assertEqual([block["verse_index"] for block in blocks], [1, 2])
        self.assertEqual(
            [block["revised_draft"] for block in blocks],
            poem.poem_text.splitlines(),
        )

    def test_reasoning_chunk_rejects_missing_verse_and_wrong_revision(self) -> None:
        poem = make_poem()
        missing = valid_reasoning_value(poem)
        missing["verse_reasoning"] = missing["verse_reasoning"][:1]
        with self.assertRaisesRegex(ValueError, "expected 2"):
            extract_reasoning_chunk(
                missing,
                expected_indices=[1, 2],
                expected_couplets=poem.poem_text.splitlines(),
                include_overview=True,
            )

        wrong = valid_reasoning_value(poem)
        wrong["verse_reasoning"][0]["revised_draft"] = "بيت مختلف = تمامًا"
        with self.assertRaisesRegex(ValueError, "exactly match"):
            extract_reasoning_chunk(
                wrong,
                expected_indices=[1, 2],
                expected_couplets=poem.poem_text.splitlines(),
                include_overview=True,
            )

    def test_reasoning_chunk_rejects_generation_metatext(self) -> None:
        poem = make_poem()
        value = valid_reasoning_value(poem)
        value["verse_reasoning"][0]["intended_meaning"] = (
            "وضعتُ في الموضوع العام وصف القصيدة"
        )
        with self.assertRaisesRegex(ValueError, "metatext"):
            extract_reasoning_chunk(
                value,
                expected_indices=[1, 2],
                expected_couplets=poem.poem_text.splitlines(),
                include_overview=True,
            )

    def test_reasoning_chunk_rejects_retrospective_claim_before_first_draft(self) -> None:
        poem = make_poem()
        value = valid_reasoning_value(poem)
        value["verse_reasoning"][0]["imagery_and_diction"] = (
            "استخدمت لفظ الضياء في صياغة البيت لتقوية صورة الرجاء بعد الشدة."
        )
        with self.assertRaisesRegex(ValueError, "must describe a plan before first_draft"):
            extract_reasoning_chunk(
                value,
                expected_indices=[1, 2],
                expected_couplets=poem.poem_text.splitlines(),
                include_overview=True,
            )

    def test_parses_strict_field_verdict(self) -> None:
        raw = json.dumps({"passed": False, "errors": ["الوزن عام"]}, ensure_ascii=False)
        self.assertEqual(
            parse_field_verdict(raw),
            {"passed": False, "errors": ["الوزن عام"]},
        )
        malformed = json.dumps({"passed": True, "errors": ["تناقض"]}, ensure_ascii=False)
        with self.assertRaises(ValueError):
            parse_field_verdict(malformed)

    def test_rendered_response_has_one_block_per_verse_and_exact_final_poem(self) -> None:
        poem = make_poem()
        value = valid_reasoning_value(poem)
        reasoning = render_reasoning(
            value["overview"], value["verse_reasoning"], is_prose=False
        )
        response = compose_response(reasoning, poem)
        self.assertIn("البيت 1:", response)
        self.assertIn("البيت 2:", response)
        self.assertIn("خطة الصورة والمعجم:", response)
        self.assertIn("صياغة أولى:", response)
        self.assertIn("قرار المراجعة:", response)
        self.assertLess(
            response.index("خطة الصورة والمعجم:"),
            response.index("صياغة أولى:"),
        )
        self.assertLess(
            response.index("صياغة أولى:"),
            response.index("قرار المراجعة:"),
        )
        self.assertTrue(response.endswith(poem.poem_text))
        self.assertEqual(response.count("النتيجة النهائية:"), 1)

    def test_response_discards_accidental_final_section(self) -> None:
        poem = make_poem()
        model_reasoning = (
            "مرحلة التفكير والتحرير:\nأشرح قرارًا تحريريًا.\n"
            f"النتيجة النهائية:\n{poem.poem_text}\nنص زائد"
        )
        response = compose_response(model_reasoning, poem)
        self.assertNotIn("نص زائد", response)
        self.assertTrue(response.endswith(poem.poem_text))
        self.assertEqual(response.count("النتيجة النهائية:"), 1)

    def test_response_preserves_verse_quotations_inside_reasoning(self) -> None:
        poem = make_poem()
        first_couplet = poem.poem_text.splitlines()[0]
        response = compose_response(
            f"مرحلة التفكير والتحرير:\nالصياغة المنقحة:\n{first_couplet}",
            poem,
        )
        marker_position = response.index("النتيجة النهائية:")
        self.assertIn(first_couplet, response[:marker_position])

    def test_single_couplet_revision_is_not_mistaken_for_a_full_poem_dump(self) -> None:
        poem = make_poem(verses=("صدر وحيد", "عجز وحيد"))
        response = compose_response(
            f"مرحلة التفكير والتحرير:\nالصياغة المنقحة:\n{poem.poem_text}",
            poem,
        )
        marker_position = response.index("النتيجة النهائية:")
        self.assertIn(poem.poem_text, response[:marker_position])
        self.assertTrue(response.endswith(poem.poem_text))


if __name__ == "__main__":
    unittest.main()
