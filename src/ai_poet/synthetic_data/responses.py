"""Render structured editorial work and compose trusted SFT responses."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from .poems import PoemRecord


def render_reasoning(
    overview: str,
    verse_reasoning: Sequence[dict[str, Any]],
    *,
    is_prose: bool,
) -> str:
    """Render validated verse-work blocks as readable Arabic editorial prose."""
    unit_name = "الوحدة" if is_prose else "البيت"
    first_sound_label = "فحص إيقاع الجزء الأول" if is_prose else "فحص الصدر عروضيًا"
    second_sound_label = "فحص إيقاع الجزء الثاني" if is_prose else "فحص العجز عروضيًا"
    rhyme_label = "فحص الجرس" if is_prose else "فحص القافية"

    sections = ["مرحلة التفكير والتحرير:", "", overview.strip()]
    for block in verse_reasoning:
        sections.extend(
            [
                "",
                f"{unit_name} {block['verse_index']}:",
                "",
                "المعنى المقصود:",
                block["intended_meaning"],
                "",
                "صلته بما قبله:",
                block["connection_to_previous"],
                "",
                "خطة الصورة والمعجم:",
                block["imagery_and_diction"],
                "",
                "صياغة أولى:",
                block["first_draft"],
                "",
                "قرار المراجعة:",
                block["problem_with_first_draft"],
                "",
                "الصياغة المنقحة:",
                block["revised_draft"],
                "",
                f"{first_sound_label}:",
                block["first_hemistich_scansion"],
                "",
                f"{second_sound_label}:",
                block["second_hemistich_scansion"],
                "",
                f"{rhyme_label}:",
                block["rhyme_check"],
            ]
        )
    return "\n".join(sections).strip()


def compose_qcm_response(qcm: dict[str, Any], poem: PoemRecord) -> str:
    """Compose the assistant response for a QCM record.

    The response contains the exact source poem, the question, the four
    choices, the reasoning, and the correct answer with its text.
    """
    choices = qcm["choices"]
    correct = qcm["correct_answer"]
    lines = [
        poem.poem_text,
        "",
        "السؤال:",
        qcm["question"],
        "",
        "الخيارات:",
        f"ا. {choices['ا']}",
        f"ب. {choices['ب']}",
        f"ج. {choices['ج']}",
        f"د. {choices['د']}",
        "",
        "الاستدلال:",
        qcm["reasoning"],
        "",
        f"الإجابة الصحيحة: {correct}",
        choices[correct],
    ]
    return "\n".join(lines)


def compose_response(reasoning: str, poem: PoemRecord) -> str:
    """Append one canonical result marker and the exact source poem.

    Verse and hemistich quotations inside the editorial work are preserved.
    Only a leading or explicitly labeled full-poem dump and anything after an
    accidental result marker are discarded before the trusted poem is appended.
    """
    marker = "النتيجة النهائية:"
    standalone_poem_headers = {
        "القصيدة:",
        "القصيدة النهائية:",
        "النص النهائي:",
    }
    editorial_reasoning = reasoning.split(marker, 1)[0]
    for header in standalone_poem_headers:
        editorial_reasoning = editorial_reasoning.replace(
            f"{header}\n{poem.poem_text}", ""
        )
    if editorial_reasoning.lstrip().startswith(poem.poem_text):
        leading_space = len(editorial_reasoning) - len(editorial_reasoning.lstrip())
        editorial_reasoning = editorial_reasoning[leading_space + len(poem.poem_text) :]

    retained_lines = [
        line.rstrip()
        for line in editorial_reasoning.splitlines()
        if line.strip() not in standalone_poem_headers
    ]
    editorial_reasoning = "\n".join(retained_lines).strip()
    editorial_reasoning = re.sub(r"\n{3,}", "\n\n", editorial_reasoning)
    return f"{editorial_reasoning}\n\n{marker}\n\n{poem.poem_text}"
