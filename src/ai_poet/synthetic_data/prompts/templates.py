"""Concrete prompt templates for reverse-generating Arabic poetry SFT data."""

from __future__ import annotations

from dataclasses import dataclass


TEMPLATE_VERSION = 2

ALL_FOCUS_REQUIREMENTS = """عالج المحاور الستة الآتية كلها من غير اختلاق ما لا يثبته المرجع:
1. الوزن أو الإيقاع والقافية والجرس والنطق عند انطباقه.
2. تدرج المعاني ووحدة النص ووظيفة أجزائه.
3. الصور الحسية والاستعارات والتشبيهات والعلاقات البلاغية.
4. المقام العاطفي وصوت المتكلم وتحول النبرة.
5. مناسبة القول والمخاطَب ومقصد الخطاب، مع التصريح بعدم التعيين إذا لم يدل المرجع عليه.
6. المعجم والتراكيب وبدائل الصياغة وأسباب المراجعة والتحرير."""


@dataclass(frozen=True)
class PromptTemplate:
    """A renderable, comprehensive prompt with a stable output identifier."""

    template_id: str
    prompt: str


def _concrete_prompt(approach: str) -> str:
    """Build one full prompt while preserving ``str.format`` placeholders."""
    return f"""أنجز المهمة الفعلية باستعمال هذا المسار التحريري: {approach}

البحر الأساس: {{meter_name}}
عدد الأبيات أو الوحدات المطلوب ذكره صراحة بالأرقام: {{couplet_count}}
الحد الأدنى لكل من instruction وreasoning: {{minimum_chars}} محرفًا.
ضابط الشكل: {{form_guidance}}

{ALL_FOCUS_REQUIREMENTS}

يجب أن تتضمن instruction قيودًا عملية مستنبطة من المرجع في المحاور الستة، وأن يشرح reasoning بصيغة المتكلم كيف راجعت هذه المحاور ووازنت بينها. لا تنسب النص إلى شاعر، ولا تذكر عنوانًا أو رابطًا، ولا تنسخ شطرًا كاملًا في instruction، ولا تخترع مناسبة تاريخية أو مخاطَبًا لا يدل عليه النص.

النص المرجعي:
{{poem}}

أعد كائن JSON فقط بالمفتاحين instruction وreasoning. لا تضع القصيدة النهائية مجتمعة داخل reasoning."""


PROMPT_TEMPLATES = (
    PromptTemplate(
        "prosody_rhyme",
        _concrete_prompt(
            "ابدأ بفحص البنية الصوتية، ثم اربط الوزن أو الإيقاع والقافية بسائر المعنى والصورة والصوت والمقام والاختيارات اللفظية"
        ),
    ),
    PromptTemplate(
        "semantic_arc",
        _concrete_prompt(
            "ابدأ برسم مسار الدلالة ووظيفة أجزاء النص، ثم اختبر كيف تخدمه الموسيقى والصور والنبرة والمقام والتحرير اللفظي"
        ),
    ),
    PromptTemplate(
        "imagery_rhetoric",
        _concrete_prompt(
            "ابدأ بمركز الصورة والعلاقات البلاغية، ثم صلها بوحدة المعنى والصوت والعاطفة والمخاطَب وقرارات الصياغة"
        ),
    ),
    PromptTemplate(
        "emotion_voice",
        _concrete_prompt(
            "ابدأ بصوت المتكلم ومنحنى الانفعال، ثم اضبط المعنى والصورة والموسيقى ومقام الخطاب والمعجم بما يحفظ هذا الصوت"
        ),
    ),
    PromptTemplate(
        "occasion_addressee",
        _concrete_prompt(
            "ابدأ بمقصد القول والمخاطَب والمناسبة التي يثبتها النص، ثم راجع اتساقها مع الدلالة والصورة والنبرة والموسيقى والتحرير"
        ),
    ),
    PromptTemplate(
        "diction_revision",
        _concrete_prompt(
            "ابدأ بمقارنة الألفاظ والتراكيب والبدائل، ثم برر الاختيار النهائي بآثاره في المعنى والصورة والصوت والعاطفة ومقام الخطاب"
        ),
    ),
)
