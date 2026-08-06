from __future__ import annotations

import json
from string import Formatter
import unittest

from ai_poet.synthetic_data.prompts.builder import (
    build_messages,
    build_validation_messages,
)
from ai_poet.synthetic_data.meters import METER_NAMES
from ai_poet.synthetic_data.prompts.templates import (
    ALL_FOCUS_REQUIREMENTS,
    METER_DEFINITIONS,
    PROMPT_TEMPLATES,
    meter_explanation,
    plan_heading,
)
from tests.synthetic_data_helpers import make_poem, valid_value


FOCUS_TERM_GROUPS = (
    ("وزن", "إيقاع"),
    ("معنى", "معاني", "دلال"),
    ("صورة", "استعار", "بلاغ"),
    ("نبرة", "عاطف", "صوت"),
    ("مقام", "مخاطب", "المخاطَب"),
    ("تحرير", "صياغ", "بديل", "معجم"),
)

EXPECTED_TEMPLATE_FIELDS = {
    "meter_name",
    "couplet_count",
    "minimum_chars",
    "form_guidance",
    "meter_explanation",
    "plan_heading",
    "poem",
}

REQUIRED_INSTRUCTION_HEADINGS = (
    "الموضوع العام:",
    "الجو العاطفي المطلوب:",
    "ألفاظ وصور يُستحسن استعمالها أو الدوران حولها:",
    "القافية:",
    "شرح البحر المطلوب:",
    "الصورة الصوتية التقريبية:",
)


class TemplateTests(unittest.TestCase):
    def test_meter_definitions_cover_every_supported_meter(self) -> None:
        self.assertEqual(set(METER_DEFINITIONS), set(METER_NAMES))
        for meter_name, details in METER_DEFINITIONS.items():
            self.assertEqual(
                set(details),
                {"definition", "full_verse_pattern", "sound_pattern"},
            )
            self.assertTrue(all(details.values()))
            rendered = meter_explanation(meter_name)
            self.assertIn(details["definition"], rendered)
            self.assertIn(details["full_verse_pattern"], rendered)
            self.assertIn(details["sound_pattern"], rendered)
            self.assertIn("وزنه في كل بيت كامل:", rendered)

    def test_every_template_is_complete_and_uses_the_same_fields(self) -> None:
        poem = make_poem()
        self.assertEqual(len(PROMPT_TEMPLATES), 6)
        self.assertEqual(len({template.prompt for template in PROMPT_TEMPLATES}), 6)
        for template in PROMPT_TEMPLATES:
            fields = {
                field_name
                for _, field_name, _, _ in Formatter().parse(template.prompt)
                if field_name is not None
            }
            self.assertEqual(fields, EXPECTED_TEMPLATE_FIELDS)
            self.assertIn("JSON", template.prompt)
            self.assertIn("instruction", template.prompt)
            self.assertIn("reasoning", template.prompt)
            positions = [
                template.prompt.index(heading)
                for heading in REQUIRED_INSTRUCTION_HEADINGS
            ]
            positions.append(template.prompt.index("{plan_heading}"))
            self.assertEqual(positions, sorted(positions))
            for terms in FOCUS_TERM_GROUPS:
                self.assertTrue(
                    any(term in template.prompt for term in terms),
                    f"{template.template_id} misses {terms}",
                )
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
            self.assertNotIn(ALL_FOCUS_REQUIREMENTS, messages[0]["content"])
            self.assertNotIn(ALL_FOCUS_REQUIREMENTS, messages[-1]["content"])
            self.assertIn(poem.poem_text, messages[-1]["content"])
            self.assertIn(meter_explanation(poem.meter_name), messages[-1]["content"])
            self.assertIn(plan_heading(poem.meter_name), messages[-1]["content"])
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
                self.assertTrue(example["instruction"].startswith("الموضوع العام:"))
                positions = [
                    example["instruction"].index(heading)
                    for heading in REQUIRED_INSTRUCTION_HEADINGS
                ]
                positions.append(
                    example["instruction"].index("خطة عملية لصناعة")
                )
                self.assertEqual(positions, sorted(positions))
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
            self.assertIn(
                "خطة عملية لصناعة نص شعري منثور:", messages[-1]["content"]
            )

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
        self.assertIn(meter_explanation(poem.meter_name), messages[1]["content"])
        for heading in REQUIRED_INSTRUCTION_HEADINGS:
            self.assertIn(heading, messages[1]["content"])
        self.assertIn("instruction", messages[0]["content"])
        self.assertIn("reasoning", messages[0]["content"])
