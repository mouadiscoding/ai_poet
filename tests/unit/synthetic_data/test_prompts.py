from __future__ import annotations

import json
import unittest

from ai_poet.synthetic_data.prompts.builder import build_messages
from ai_poet.synthetic_data.prompts.families import TEMPLATE_FAMILIES, eligible_families
from tests.synthetic_data_helpers import make_poem

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
