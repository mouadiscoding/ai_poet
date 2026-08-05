from __future__ import annotations

import json
import unittest

from ai_poet.synthetic_data.prompts.builder import (
    build_messages,
    build_validation_messages,
)
from ai_poet.synthetic_data.prompts.templates import (
    ALL_FOCUS_REQUIREMENTS,
    PROMPT_TEMPLATES,
)
from tests.synthetic_data_helpers import make_poem, valid_value


FOCUS_TERM_GROUPS = (
    ("وزن", "إيقاع"),
    ("معنى", "دلال"),
    ("صورة", "استعار", "بلاغ"),
    ("نبرة", "عاطف", "صوت"),
    ("مقام", "مخاطب", "المخاطَب"),
    ("تحرير", "صياغ", "بديل", "معجم"),
)


class TemplateTests(unittest.TestCase):
    def test_every_concrete_template_renders_all_focuses_and_poem(self) -> None:
        poem = make_poem()
        self.assertEqual(len(PROMPT_TEMPLATES), 6)
        for template in PROMPT_TEMPLATES:
            self.assertIn("{poem}", template.prompt)
            messages = build_messages(
                template=template,
                meter_name=poem.meter_name,
                couplet_count=poem.couplet_count,
                poem=poem.poem_text,
                minimum_chars=1500,
            )
            self.assertEqual(
                [message["role"] for message in messages],
                ["system", "user", "assistant", "user", "assistant", "user"],
            )
            self.assertIn(ALL_FOCUS_REQUIREMENTS, messages[0]["content"])
            self.assertIn(ALL_FOCUS_REQUIREMENTS, messages[-1]["content"])
            self.assertIn(poem.poem_text, messages[-1]["content"])
            self.assertNotIn("{poem}", messages[-1]["content"])

    def test_every_few_shot_field_demonstrates_all_focuses(self) -> None:
        poem = make_poem()
        for meter_name in (poem.meter_name, "النثر"):
            messages = build_messages(
                template=PROMPT_TEMPLATES[0],
                meter_name=meter_name,
                couplet_count=poem.couplet_count,
                poem=poem.poem_text,
                minimum_chars=100,
            )
            for assistant in (messages[2], messages[4]):
                example = json.loads(assistant["content"])
                self.assertEqual(set(example), {"instruction", "reasoning"})
                for field in ("instruction", "reasoning"):
                    for terms in FOCUS_TERM_GROUPS:
                        self.assertTrue(
                            any(term in example[field] for term in terms),
                            f"{meter_name} {field} misses {terms}",
                        )

    def test_prose_can_use_every_template_and_adapts_prosody(self) -> None:
        for template in PROMPT_TEMPLATES:
            messages = build_messages(
                template=template,
                meter_name="النثر",
                couplet_count=1,
                poem="صورة أولى = صورة ثانية",
                minimum_chars=100,
            )
            self.assertIn("بلا وزن خليلي", messages[-1]["content"])
            self.assertIn("الإيقاع الداخلي", messages[-1]["content"])

    def test_validation_prompt_contains_pair_reference_and_all_criteria(self) -> None:
        poem = make_poem()
        candidate = valid_value(poem)
        messages = build_validation_messages(
            candidate=candidate,
            meter_name=poem.meter_name,
            couplet_count=poem.couplet_count,
            poem=poem.poem_text,
            minimum_chars=80,
        )
        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertIn(poem.poem_text, messages[1]["content"])
        self.assertIn(candidate["instruction"], messages[1]["content"])
        self.assertIn(ALL_FOCUS_REQUIREMENTS, messages[1]["content"])
        self.assertIn("instruction", messages[0]["content"])
        self.assertIn("reasoning", messages[0]["content"])
