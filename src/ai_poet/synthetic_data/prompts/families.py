"""Prompt-family definitions and eligibility policy."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TemplateFamily:
    """A stable prompt family and its demonstration specialization."""

    template_id: str
    focus: str
    instruction_addition: str
    reasoning_addition: str


TEMPLATE_FAMILIES = (
    TemplateFamily(
        "prosody_rhyme",
        "قدّم الوزن والقافية والنطق العروضي، ثم اربطهما بالمعنى من غير ادعاء صورة عروضية لا تثبتها البيانات.",
        "اجعل وصف القافية والنطق العروضي جزءًا بارزًا من التعليمات، واربط الأثر الصوتي بالمعنى.",
        "أفرد للجرس والروي وقفة أوضح، فأراجع موضع الوقف والوصل وأتأكد أن الاختيار الصوتي يعزز التعظيم ولا يتحول إلى حشو تقني.",
    ),
    TemplateFamily(
        "semantic_arc",
        "قدّم تدرج المعاني ووحدة القصيدة، وبيّن وظيفة كل بيت في الانتقال من المطلع إلى الخاتمة.",
        "اشرح وظيفة الصدر والعجز في تدرج الدلالة، واجعل الوحدة المعنوية مقدمة على الزينة اللفظية.",
        "أراجع مسار الدلالة خاصة: يبدأ النص باستحالة البلوغ ثم يقدم الصورة الكونية برهانًا شعريًا، ولذلك لا أقدم العجز على الصدر.",
    ),
    TemplateFamily(
        "imagery_rhetoric",
        "قدّم الصور الحسية والاستعارات والتشبيهات والطباق وسائر العلاقات البلاغية التي يبني عليها النص أثره.",
        "وسع توجيه الصورة والاستعارة، وبيّن العلاقة البلاغية بين العلو الحسي ورفعة المقام.",
        "أركز في التحرير البلاغي على نقل السماء من مشبه به عابر إلى استعارة يخاطب بها الممدوح، لأن هذا التحول هو مركز أثر البيت.",
    ),
    TemplateFamily(
        "emotion_voice",
        "قدّم المقام العاطفي ودرجة الانفعال وصوت المتكلم وتحول النبرة بين الأبيات.",
        "حدد درجة الانفعال وصوت المتكلم، واجعل التعظيم ممتزجًا بالدهشة والصفاء لا بالمبالغة الصاخبة.",
        "أختبر النبرة بعد الصياغة: الاستفهام يحمل الدهشة، والنداء يحمل القرب، واجتماعهما يمنح الصوت محبة مهيبة من غير صخب.",
    ),
    TemplateFamily(
        "occasion_addressee",
        "قدّم المناسبة والمخاطَب ومقصد القول، واضبط ما يلائم المقام من تعظيم أو عتاب أو حنين أو حكمة.",
        "وضح مقام الخطاب وصفة المخاطَب وغرض الاستفهام والنداء، بحيث تلائم الألفاظ مناسبة التعظيم.",
        "أتعامل مع البيت بوصفه خطابًا مباشرًا في مقام المدح؛ لذلك أبقي ضمير المخاطب ظاهرًا وأجعل السؤال تعظيميًا لا طلبًا لمعلومة.",
    ),
    TemplateFamily(
        "diction_revision",
        "قدّم المعجم والتراكيب وبدائل التحرير، وفسّر لماذا تخدم الصياغة النهائية الوزن والمعنى والنبرة.",
        "اقترح بدائل لفظية محددة، ثم فضّل بينها بحسب الدقة والجزالة والخفة على الإيقاع.",
        "أقارن في التحرير بين «بلغتها» و«طاولتها»: الأولى تؤدي الوصول وحده، أما الثانية فتجمع المنافسة والارتفاع، ولذلك أختارها للنسخة النهائية.",
    ),
)


def family_by_id(template_id: str) -> TemplateFamily:
    """Return the family with the given stable identifier."""
    for family in TEMPLATE_FAMILIES:
        if family.template_id == template_id:
            return family
    raise KeyError(f"Unknown template family: {template_id}")


def eligible_families(meter_name: str) -> tuple[TemplateFamily, ...]:
    """Return families eligible for the given meter or prose marker."""
    if meter_name == "النثر":
        return tuple(
            family
            for family in TEMPLATE_FAMILIES
            if family.template_id != "prosody_rhyme"
        )
    return TEMPLATE_FAMILIES
