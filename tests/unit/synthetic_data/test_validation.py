from __future__ import annotations

import json
import unittest

from ai_poet.synthetic_data.responses import compose_response
from ai_poet.synthetic_data.validation import extract_json_object, validate_generation
from tests.synthetic_data_helpers import make_poem, valid_value

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
