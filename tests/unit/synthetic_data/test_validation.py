from __future__ import annotations

import json
import unittest

from ai_poet.synthetic_data.responses import compose_response
from ai_poet.synthetic_data.validation import (
    extract_generated_pair,
    extract_json_object,
    parse_validation_verdict,
    verdict_errors,
)
from tests.synthetic_data_helpers import make_poem

class ValidationTests(unittest.TestCase):
    def test_extracts_plain_and_fenced_json(self) -> None:
        value = {"instruction": "تعليمات", "reasoning": "تحرير"}
        raw = json.dumps(value, ensure_ascii=False)
        self.assertEqual(extract_json_object(raw), value)
        self.assertEqual(extract_json_object(f"```json\n{raw}\n```"), value)

    def test_pair_extraction_enforces_only_required_string_fields(self) -> None:
        value = {
            "instruction": "تعليمات قصيرة بعدد أو بحر خاطئ",
            "reasoning": "تعليل قصير",
            "extra": "يبقى لمدقق Gemma أن يرفضه",
        }
        self.assertEqual(
            extract_generated_pair(value),
            (value["instruction"], value["reasoning"]),
        )

    def test_pair_extraction_rejects_missing_or_non_string_fields(self) -> None:
        with self.assertRaises(ValueError):
            extract_generated_pair({"instruction": "تعليمات"})
        with self.assertRaises(ValueError):
            extract_generated_pair({"instruction": "تعليمات", "reasoning": []})

    def test_parses_strict_gemma_verdict_and_flattens_rejections(self) -> None:
        raw = json.dumps(
            {
                "instruction": {"passed": False, "errors": ["العدد خاطئ"]},
                "reasoning": {"passed": True, "errors": []},
            },
            ensure_ascii=False,
        )
        verdict = parse_validation_verdict(raw)
        self.assertEqual(
            verdict_errors(verdict),
            ["Gemma rejected instruction: العدد خاطئ"],
        )

    def test_rejects_malformed_or_inconsistent_gemma_verdict(self) -> None:
        malformed = json.dumps(
            {
                "instruction": {"passed": True, "errors": ["تناقض"]},
                "reasoning": {"passed": True, "errors": []},
            },
            ensure_ascii=False,
        )
        with self.assertRaises(ValueError):
            parse_validation_verdict(malformed)

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
