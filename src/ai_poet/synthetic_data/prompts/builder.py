"""Build few-shot chat conversations for synthetic SFT generation."""

from __future__ import annotations

import json

from .examples import FewShot, METERED_FEW_SHOTS, PROSE_FEW_SHOTS
from .families import TemplateFamily


SYSTEM_PROMPT = """أنت تبني بيانات تدريب إشرافي لشاعر عربي. مهمتك عكسية: تتلقى قصيدة مرجعية، ثم تكتب تعليمات مفصلة كان يمكن أن تؤدي إليها، وتعليلًا تحريريًا طويلًا يشرح صناعة النص. التزم بالحقائق الظاهرة في المرجع ولا تنسب النص إلى شاعره ولا تذكر عنوانه أو رابطه في التعليمات. لا تنسخ شطرًا كاملًا داخل التعليمات، ويجوز ذكر ألفاظ مفردة تخدم القافية أو الصورة.

اسم البحر المعطى هو البحر الأساس فقط. لا تدّع أن الصورة تامة أو مجزوءة أو مشطورة، ولا تسرد تفعيلات مخصوصة إن لم تكن الصورة مثبتة. عند البحر «النثر» صرّح بعدم طلب الوزن الخليلي واستبدله بالإيقاع الداخلي.

أعد كائن JSON صحيحًا فقط، بلا سياج Markdown، وبمفتاحين نصيين لا غير: "instruction" و"reasoning". يجب أن يكون كل حقل طويلًا ومفصلًا ولا يقل عن {minimum_chars} محرفًا في المهمة الفعلية. الأمثلة التالية مختصرة نسبيًا لاقتصاد السياق. يجب أن يتضمن التعليل مرحلة تفكير وتحرير بصيغة المتكلم، والمعاني، والصور، ومحاولات أولية، وفحص الوزن والقافية عند وجود بحر، وأسباب التعديل. اختم التعليل بعبارة «النتيجة النهائية:» من غير أن تورد القصيدة النهائية مجتمعة؛ سيضيفها البرنامج حرفيًا.

تركيز هذا القالب: {focus}"""


def _example_user(example: FewShot, family: TemplateFamily | None = None) -> str:
    """Render the user half of a few-shot demonstration."""
    prompt = (
        "ابنِ زوج تعليمات وتعليل من المرجع الآتي.\n"
        f"البحر الأساس: {example.meter_name}\n"
        f"عدد الأبيات أو الوحدات: {example.couplet_count}\n"
        "النص المرجعي:\n"
        f"{example.poem}"
    )
    if family is not None:
        prompt += f"\nمحور هذا المثال: {family.focus}"
    return prompt


def _example_assistant(
    example: FewShot, family: TemplateFamily | None = None
) -> str:
    """Serialize the assistant half of a few-shot demonstration."""
    instruction = example.instruction
    reasoning = example.reasoning
    if family is not None:
        instruction += f" {family.instruction_addition}"
        marker = "النتيجة النهائية:"
        marker_position = reasoning.rfind(marker)
        if marker_position >= 0:
            reasoning = (
                reasoning[:marker_position].rstrip()
                + f"\n\n{family.reasoning_addition}\n\n{marker}"
            )
        else:
            reasoning += f"\n\n{family.reasoning_addition}"
    return json.dumps(
        {"instruction": instruction, "reasoning": reasoning},
        ensure_ascii=False,
    )


def build_messages(
    *,
    family: TemplateFamily,
    meter_name: str,
    couplet_count: int,
    poem_text: str | None,
    minimum_chars: int,
    analysis_notes: str | None = None,
) -> list[dict[str, str]]:
    """Build the complete few-shot conversation for one source poem."""
    examples = PROSE_FEW_SHOTS if meter_name == "النثر" else METERED_FEW_SHOTS
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT.format(
                minimum_chars=minimum_chars,
                focus=family.focus,
            ),
        }
    ]
    for index, example in enumerate(examples):
        example_family = family if index == 1 else None
        messages.extend(
            [
                {"role": "user", "content": _example_user(example, example_family)},
                {
                    "role": "assistant",
                    "content": _example_assistant(example, example_family),
                },
            ]
        )

    if analysis_notes is not None:
        reference = (
            "القصيدة طويلة، وهذه ملخصات تحليلية لمقاطعها مرتبة. ابنِ منها زوجًا واحدًا "
            "يغطي القصيدة كلها ولا تفترض معنى غير مذكور:\n"
            f"{analysis_notes}"
        )
    else:
        reference = f"النص المرجعي:\n{poem_text}"

    messages.append(
        {
            "role": "user",
            "content": (
                "الآن أنجز المهمة الفعلية وفق القالب والأمثلة.\n"
                f"البحر الأساس: {meter_name}\n"
                f"عدد الأبيات أو الوحدات المطلوب ذكره صراحة بالأرقام: {couplet_count}\n"
                f"{reference}\n\n"
                "أعد JSON فقط. لا تضع القصيدة النهائية مجتمعة داخل reasoning."
            ),
        }
    )
    return messages
