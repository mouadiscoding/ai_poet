from __future__ import annotations

import json
import unittest
from string import Formatter

from ai_poet.synthetic_data.meters import METER_NAMES
from ai_poet.synthetic_data.prompts.builder import (
    build_instruction_validation_messages,
    build_messages,
    build_qcm_messages,
    build_qcm_validation_messages,
    build_reasoning_messages,
    build_reasoning_validation_messages,
)
from ai_poet.synthetic_data.prompts.qcm_templates import (
    QCM_PROMPT_TEMPLATES,
)
from ai_poet.synthetic_data.prompts.templates import (
    ALL_FOCUS_REQUIREMENTS,
    METER_DEFINITIONS,
    PROMPT_TEMPLATES,
    meter_explanation,
    plan_heading,
)

from tests.synthetic_data_helpers import (
    make_poem,
    valid_instruction_value,
    valid_qcm_value,
    valid_reasoning_value,
)

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
            self.assertNotIn("reasoning", template.prompt)
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
                self.assertEqual(set(example), {"instruction"})
                self.assertTrue(example["instruction"].startswith("الموضوع العام:"))
                positions = [
                    example["instruction"].index(heading)
                    for heading in REQUIRED_INSTRUCTION_HEADINGS
                ]
                positions.append(example["instruction"].index("خطة عملية لصناعة"))
                self.assertEqual(positions, sorted(positions))
                for terms in FOCUS_TERM_GROUPS:
                    self.assertTrue(
                        any(term in example["instruction"] for term in terms),
                        f"{meter_name} instruction misses {terms}",
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
            self.assertIn("خطة عملية لصناعة نص شعري منثور:", messages[-1]["content"])

    def test_reasoning_prompt_requires_one_structured_block_per_target(self) -> None:
        poem = make_poem()
        instruction = valid_instruction_value(poem)["instruction"]
        messages = build_reasoning_messages(
            instruction=instruction,
            meter_name=poem.meter_name,
            total_couplet_count=poem.couplet_count,
            start_index=1,
            couplets=poem.poem_text.splitlines(),
            previous_couplet=None,
            next_couplet=None,
            include_overview=True,
        )
        self.assertEqual(
            [message["role"] for message in messages],
            ["system", "user", "assistant", "user"],
        )
        prompt = messages[-1]["content"]
        self.assertIn("verse_reasoning", prompt)
        self.assertIn("first_draft", prompt)
        self.assertIn("problem_with_first_draft", prompt)
        self.assertIn("revised_draft", prompt)
        self.assertIn("خطة سابقة للمسودة", prompt)
        self.assertIn("لفظًا موجودًا فعلًا في first_draft", prompt)
        self.assertIn("تضم 2 أبيات", prompt)
        for couplet in poem.poem_text.splitlines():
            self.assertIn(couplet, prompt)
        example = json.loads(messages[2]["content"])
        self.assertGreaterEqual(len(example["verse_reasoning"]), 2)

    def test_qcm_templates_are_complete_and_poem_grounded(self) -> None:
        poem = make_poem()
        self.assertEqual(len(QCM_PROMPT_TEMPLATES), 4)
        self.assertEqual(len({template.prompt for template in QCM_PROMPT_TEMPLATES}), 4)
        for template in QCM_PROMPT_TEMPLATES:
            self.assertIn("{meter_name}", template.prompt)
            self.assertIn("{couplet_count}", template.prompt)
            self.assertIn("{poem}", template.prompt)
            self.assertIn("QCM_QUESTION_CATEGORIES", template.prompt)
            self.assertIn("QCM_REASONING_PROCESS", template.prompt)
            self.assertIn("QCM_OUTPUT_CONTRACT", template.prompt)
            messages = build_qcm_messages(
                template=template,
                meter_name=poem.meter_name,
                couplet_count=poem.couplet_count,
                poem=poem.poem_text,
            )
            self.assertEqual(
                [message["role"] for message in messages],
                [
                    "system",
                    "user",
                    "assistant",
                    "user",
                    "assistant",
                    "user",
                    "assistant",
                    "user",
                ],
            )
            self.assertIn(poem.poem_text, messages[-1]["content"])
            self.assertNotIn("{poem}", messages[-1]["content"])

    def test_qcm_validation_prompt_checks_reasoning_quality(self) -> None:
        poem = make_poem()
        qcm = valid_qcm_value(poem)
        messages = build_qcm_validation_messages(
            poem=poem.poem_text,
            candidate=qcm,
        )
        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertIn(poem.poem_text, messages[1]["content"])
        self.assertIn(qcm["question"], messages[1]["content"])
        self.assertIn("الاستدلال ليس عبارة عامة", messages[0]["content"])

    def test_validation_prompts_are_phase_specific(self) -> None:
        poem = make_poem()
        instruction = valid_instruction_value(poem)["instruction"]
        messages = build_instruction_validation_messages(
            instruction=instruction,
            meter_name=poem.meter_name,
            couplet_count=poem.couplet_count,
            poem=poem.poem_text,
            minimum_chars=80,
        )
        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertIn(poem.poem_text, messages[1]["content"])
        self.assertIn(instruction, messages[1]["content"])
        self.assertIn(ALL_FOCUS_REQUIREMENTS, messages[1]["content"])
        self.assertIn("لا تطلب عناوين إضافية", messages[1]["content"])
        self.assertNotIn("مسودة وعيب", messages[0]["content"])

        candidate = valid_reasoning_value(poem)
        reasoning_messages = build_reasoning_validation_messages(
            instruction=instruction,
            meter_name=poem.meter_name,
            expected_couplets=poem.poem_text.splitlines(),
            candidate=candidate,
        )
        self.assertIn("مسودة شعرية حقيقية", reasoning_messages[1]["content"])
        self.assertIn("تسلسلًا زمنيًا صادقًا", reasoning_messages[1]["content"])
        self.assertIn("لا تدقق صحة التقطيع", reasoning_messages[1]["content"])
        self.assertIn(poem.poem_text, reasoning_messages[1]["content"])
        self.assertNotIn(
            candidate["verse_reasoning"][0]["first_hemistich_scansion"],
            reasoning_messages[1]["content"],
        )
        self.assertNotIn(
            candidate["verse_reasoning"][0]["second_hemistich_scansion"],
            reasoning_messages[1]["content"],
        )
        self.assertIn(
            candidate["verse_reasoning"][0]["intended_meaning"],
            reasoning_messages[1]["content"],
        )
