"""Arabic few-shot prompts for reverse-constructing poetry SFT examples."""

from __future__ import annotations

from dataclasses import dataclass
import json


METER_NAMES = (
    "البسيط",
    "الخفيف",
    "الرجز",
    "الرمل",
    "السريع",
    "الطويل",
    "الكامل",
    "المتدارك",
    "المتقارب",
    "المجتث",
    "المديد",
    "المضارع",
    "المقتضب",
    "المنسرح",
    "النثر",
    "الهزج",
    "الوافر",
)


@dataclass(frozen=True)
class TemplateFamily:
    template_id: str
    focus: str


@dataclass(frozen=True)
class FewShot:
    meter_name: str
    couplet_count: int
    poem: str
    instruction: str
    reasoning: str


TEMPLATE_FAMILIES = (
    TemplateFamily(
        "prosody_rhyme",
        "قدّم الوزن والقافية والنطق العروضي، ثم اربطهما بالمعنى من غير ادعاء صورة عروضية لا تثبتها البيانات.",
    ),
    TemplateFamily(
        "semantic_arc",
        "قدّم تدرج المعاني ووحدة القصيدة، وبيّن وظيفة كل بيت في الانتقال من المطلع إلى الخاتمة.",
    ),
    TemplateFamily(
        "imagery_rhetoric",
        "قدّم الصور الحسية والاستعارات والتشبيهات والطباق وسائر العلاقات البلاغية التي يبني عليها النص أثره.",
    ),
    TemplateFamily(
        "emotion_voice",
        "قدّم المقام العاطفي ودرجة الانفعال وصوت المتكلم وتحول النبرة بين الأبيات.",
    ),
    TemplateFamily(
        "occasion_addressee",
        "قدّم المناسبة والمخاطَب ومقصد القول، واضبط ما يلائم المقام من تعظيم أو عتاب أو حنين أو حكمة.",
    ),
    TemplateFamily(
        "diction_revision",
        "قدّم المعجم والتراكيب وبدائل التحرير، وفسّر لماذا تخدم الصياغة النهائية الوزن والمعنى والنبرة.",
    ),
)


FAMILY_DEMO_ADDITIONS = {
    "prosody_rhyme": (
        "اجعل وصف القافية والنطق العروضي جزءًا بارزًا من التعليمات، واربط الأثر الصوتي بالمعنى.",
        "أفرد للجرس والروي وقفة أوضح، فأراجع موضع الوقف والوصل وأتأكد أن الاختيار الصوتي يعزز التعظيم ولا يتحول إلى حشو تقني.",
    ),
    "semantic_arc": (
        "اشرح وظيفة الصدر والعجز في تدرج الدلالة، واجعل الوحدة المعنوية مقدمة على الزينة اللفظية.",
        "أراجع مسار الدلالة خاصة: يبدأ النص باستحالة البلوغ ثم يقدم الصورة الكونية برهانًا شعريًا، ولذلك لا أقدم العجز على الصدر.",
    ),
    "imagery_rhetoric": (
        "وسع توجيه الصورة والاستعارة، وبيّن العلاقة البلاغية بين العلو الحسي ورفعة المقام.",
        "أركز في التحرير البلاغي على نقل السماء من مشبه به عابر إلى استعارة يخاطب بها الممدوح، لأن هذا التحول هو مركز أثر البيت.",
    ),
    "emotion_voice": (
        "حدد درجة الانفعال وصوت المتكلم، واجعل التعظيم ممتزجًا بالدهشة والصفاء لا بالمبالغة الصاخبة.",
        "أختبر النبرة بعد الصياغة: الاستفهام يحمل الدهشة، والنداء يحمل القرب، واجتماعهما يمنح الصوت محبة مهيبة من غير صخب.",
    ),
    "occasion_addressee": (
        "وضح مقام الخطاب وصفة المخاطَب وغرض الاستفهام والنداء، بحيث تلائم الألفاظ مناسبة التعظيم.",
        "أتعامل مع البيت بوصفه خطابًا مباشرًا في مقام المدح؛ لذلك أبقي ضمير المخاطب ظاهرًا وأجعل السؤال تعظيميًا لا طلبًا لمعلومة.",
    ),
    "diction_revision": (
        "اقترح بدائل لفظية محددة، ثم فضّل بينها بحسب الدقة والجزالة والخفة على الإيقاع.",
        "أقارن في التحرير بين «بلغتها» و«طاولتها»: الأولى تؤدي الوصول وحده، أما الثانية فتجمع المنافسة والارتفاع، ولذلك أختارها للنسخة النهائية.",
    ),
}


METERED_FEW_SHOTS = (
    FewShot(
        meter_name="الطويل",
        couplet_count=1,
        poem="أَلا كُلُّ شَيءٍ ما خَلا اللَهَ باطِلُ = وَكُلُّ نَعيمٍ لا مَحالَةَ زائِلُ",
        instruction=(
            "أنت شاعر عربي فصيح متمكن من الحكمة وعلم العروض. انظم بيتًا واحدًا (1) فقط على بحر الطويل، "
            "يجمع في بناء محكم بين تقرير فناء الموجودات وبقاء الله تعالى، ثم يعطف على ذلك زوال كل نعيم "
            "دنيوي. اجعل النبرة يقينية هادئة لا وعظًا خطابيًا مباشرًا، وابدأ بأداة تنبيه تهيئ السامع للحكمة. "
            "استخدم مقابلة بين البقاء والبطلان، وبين لذة النعيم ومصيره إلى الزوال، واختم الشطرين بروي اللام "
            "المضمومة مع المحافظة على جزالة اللفظ. راع النطق العروضي، وفك الشدة، وإثبات المنطوق من التنوين، "
            "ولا تجعل حدود الكلمات مساوية لحدود التفعيلات. اعرض قبل النتيجة مرحلة تحرير تذكر المعنى والصورة "
            "ومحاولة أولية وفحصًا موجزًا للوزن، ثم اختم بالبيت وحده."
        ),
        reasoning=(
            "مرحلة التفكير والتحرير:\n\nأحتاج إلى بيت حكمي واحد يؤدي قضيتين متلازمتين: الأولى أن كل موجود "
            "سوى الله مآله البطلان، والثانية أن النعيم مهما صفا لا ينجو من الزوال. الأنسب أن تأتي القضية الأولى "
            "في الصدر أصلًا كليًا، ثم يأتي العجز نتيجة محسوسة تمس ما يتعلق به الإنسان. اخترت أداة التنبيه «ألا» "
            "لأنها تمنح المطلع جرس الحكمة الملقاة على جمع من السامعين.\n\nالصورة ليست تشبيهًا مزخرفًا، بل "
            "مقابلة بين ظاهر الثبات وحقيقة الفناء. صياغة أولى مثل «كل الوجود إلى فناء» تؤدي المعنى، لكنها تقريرية "
            "وتفقد الاستثناء العقدي المركزي. لذلك تصبح صيغة «كل شيء ما خلا الله» أدق وأقوى. وفي العجز فكرت في "
            "«وكل سرور آخر الأمر راحل»، غير أن «نعيم» أوسع دلالة، و«زائل» تقيم روي اللام وترد على «باطل» برد "
            "صوتي ودلالي.\n\nعند الفحص لا أزن الرسم الإملائي منفصلًا، بل أصل الكلمات وأنطق المشدد حرفين وأدع التفعيلة "
            "تعبر حدود الكلمات. ترتيب الألفاظ مستقر على إيقاع الطويل الموروث، كما أن اللام في ختام الشطرين توحد "
            "القافية الداخلية للبيت. لا أحتاج إلى زيادة صفة أو جار ومجرور، لأن الإيجاز جزء من قوة الحكمة.\n\n"
            "النتيجة النهائية:"
        ),
    ),
    FewShot(
        meter_name="الخفيف",
        couplet_count=1,
        poem="كَيْفَ تَرْقَى رُقِيَّكَ الأَنْبِيَاءُ = يَا سَمَاءً مَا طَاوَلَتْهَا سَمَاءُ",
        instruction=(
            "أنت شاعر عربي كلاسيكي متقن لعلم العروض. اكتب بيتًا واحدًا (1) فقط على بحر الخفيف في مقام التعظيم، "
            "واجعل معناه أن منزلة الممدوح أرفع من أن تبلغها منازل ذوي القدر. ابن الصدر على استفهام تعظيمي لا "
            "ينتظر جوابًا، واستثمر اشتقاقًا يجمع فعل العلو ومصدره. وفي العجز انتقل إلى استعارة السماء التي تعجز "
            "سماء أخرى عن مطاولتها. اجعل الجو مزيجًا من الهيبة والمحبة والصفاء، واستعمل فصحى رفيعة بعيدة عن "
            "الحشو. لتكن القافية همزة ممدودة كما في «سماء» و«سناء». راع سقوط همزة الوصل عند الوصل، وفك الشدة، "
            "وامتداد حركة الروي عند الإنشاد، واقبل زحافات الخفيف الموروثة من غير كسر. قبل البيت اعرض تحريرًا "
            "يوضح المعنى والصورة وتجربة الصياغة وفحص الإيقاع، ثم ضع البيت النهائي وحده."
        ),
        reasoning=(
            "مرحلة التفكير والتحرير:\n\nالمعنى المركزي هو العلو الذي لا تدركه المقامات الأخرى. أحتاج في الصدر إلى "
            "صيغة لا تتحول إلى خبر بارد، ولذلك أختار الاستفهام التعظيمي. عبارة أولى مثل «لا يبلغ القوم منك العلا» "
            "واضحة، لكنها تغلق المعنى ولا تمنحه دهشة. أما الجمع بين «ترقى» و«رقيك» فيولد اشتقاقًا يضاعف إحساس "
            "الصعود ويجعل السؤال نفسه شاهدًا على التعذر.\n\nفي العجز أبحث عن صورة كونية. قول «أنت أعلى من السماء» مباشر، "
            "بينما مخاطبة الممدوح بوصفه سماء ثم نفي مطاولة أي سماء لها تصنع استعارة ممتدة وتتيح القافية المطلوبة. "
            "كلمة «طاولتها» تجمع المنافسة والارتفاع، ولذلك هي أدق من «بلغتها».\n\nأفحص الصدر والعجز بالنطق الموصول: "
            "لا أثبت همزة الوصل كما لو كانت في الابتداء، وأفك الحرف المشدد، وأسمح للتفعيلة بأن تمتد بين كلمتين. "
            "النبرة الصوتية تتدرج من السؤال إلى النداء، ثم تستقر على الهمزة الممدودة، فيتحد المعنى والإيقاع. حذفت "
            "أي صفة إضافية بعد «سماء» لأنها ستضعف الإطلاق وقد تثقل الوزن.\n\nالنتيجة النهائية:"
        ),
    ),
)


PROSE_FEW_SHOTS = (
    FewShot(
        meter_name="النثر",
        couplet_count=2,
        poem="يجيء المساء وفي كفّه بقايا الضوء = فأجمع من صمته نافذةً للرجاء\nيمرّ الغياب على القلب ثقيلًا = لكنّ ذكرى الأحبة تعلّمه أن يضيء",
        instruction=(
            "اكتب مقطعين (2) من الشعر المنثور بالفصحى عن مساء يوقظ الحزن ثم يحوله إلى رجاء. اجعل الضوء والصمت "
            "والنافذة والغياب مفاتيح للصورة، وانتقل في المقطع الأول من المشهد الخارجي إلى الداخل، ثم اجعل المقطع "
            "الثاني يوازن ثقل الفقد بقدرة الذكرى على الإضاءة. لا تلتزم بحرًا خليليًا ولا تدّع وزنًا، لكن حافظ على "
            "إيقاع داخلي ينشأ من التوازي وقصر الجمل. قبل النص اعرض تحريرًا يشرح مسار الصورة والبدائل اللفظية."
        ),
        reasoning=(
            "مرحلة التفكير والتحرير:\n\nأبني المقطعين على حركة من العتمة إلى الضوء. في البداية لا أجعل المساء مجرد "
            "زمن، بل شخصية تحمل بقايا النور، ثم أنقل أثر صمته إلى المتكلم. فكرت في «أفتح من صمته بابًا»، لكن "
            "النافذة ألطف لأنها لا تلغي العتمة بل تسمح بمرور قدر من الرجاء.\n\nفي المقطع الثاني يصبح الغياب عابرًا "
            "ثقيل الخطو، وتأتي الذكرى قوة مقابلة لا تمحو الفقد بل تعلم القلب الإضاءة. التوازي بين «يجيء المساء» "
            "و«يمر الغياب» يمنح النص إيقاعًا داخليًا من غير ادعاء بحر خليلي. أحذف الزخارف الزائدة حتى تبقى الصور "
            "الأربع مترابطة، وأختم بفعل «يضيء» لأنه يحسم التحول العاطفي.\n\nالنتيجة النهائية:"
        ),
    ),
    FewShot(
        meter_name="النثر",
        couplet_count=1,
        poem="على حافة الطريق شجرة وحيدة = تحفظ للعابرين أسماء الذين عادوا ولم يجدوا بيوتهم",
        instruction=(
            "اكتب مقطعًا واحدًا (1) من الشعر المنثور يصور شجرة وحيدة بوصفها ذاكرة للمكان والعابرين. اجعل الطريق "
            "والعودة والبيت عناصر متوترة بين الألفة والاقتلاع، واستعمل تشخيصًا هادئًا يجعل الشجرة حافظة للأسماء. "
            "تجنب الوزن الخليلي والقافية المصطنعة، واصنع الموسيقى من توازن الجملة وتكرار الأصوات. قدم قبل النتيجة "
            "تعليلًا تحريريًا يشرح اختيار مركز الصورة وسبب حذف التفاصيل غير الضرورية."
        ),
        reasoning=(
            "مرحلة التفكير والتحرير:\n\nالمشهد يحتاج إلى مركز بصري واحد، لذلك أجعل الشجرة وحيدة على حافة الطريق. "
            "هي ليست زينة طبيعية، بل ذاكرة بديلة حين تضيع البيوت. صياغة أولى مثل «تذكر الشجرة كل المسافرين» "
            "عامة، أما حفظ أسماء الذين عادوا ولم يجدوا بيوتهم فيمنح الصورة تاريخًا وألمًا مكتومًا.\n\nأحافظ على "
            "الجملة طويلة نسبيًا لتشبه امتداد الطريق، وأوازنها بوقفة عند علامة المساواة. لا أضيف وصف لون الشجرة أو "
            "الفصل، لأن ذلك يصرف الانتباه عن علاقة الذاكرة بالاقتلاع. الموسيقى هنا داخلية، مصدرها تردد الراء "
            "والعين وتوازن «عادوا» مع «لم يجدوا»، لا بحرًا أو قافية خارجية.\n\nالنتيجة النهائية:"
        ),
    ),
)


SYSTEM_PROMPT = """أنت تبني بيانات تدريب إشرافي لشاعر عربي. مهمتك عكسية: تتلقى قصيدة مرجعية، ثم تكتب تعليمات مفصلة كان يمكن أن تؤدي إليها، وتعليلًا تحريريًا طويلًا يشرح صناعة النص. التزم بالحقائق الظاهرة في المرجع ولا تنسب النص إلى شاعره ولا تذكر عنوانه أو رابطه في التعليمات. لا تنسخ شطرًا كاملًا داخل التعليمات، ويجوز ذكر ألفاظ مفردة تخدم القافية أو الصورة.

اسم البحر المعطى هو البحر الأساس فقط. لا تدّع أن الصورة تامة أو مجزوءة أو مشطورة، ولا تسرد تفعيلات مخصوصة إن لم تكن الصورة مثبتة. عند البحر «النثر» صرّح بعدم طلب الوزن الخليلي واستبدله بالإيقاع الداخلي.

أعد كائن JSON صحيحًا فقط، بلا سياج Markdown، وبمفتاحين نصيين لا غير: "instruction" و"reasoning". يجب أن يكون كل حقل طويلًا ومفصلًا ولا يقل عن {minimum_chars} محرفًا في المهمة الفعلية. الأمثلة التالية مختصرة نسبيًا لاقتصاد السياق. يجب أن يتضمن التعليل مرحلة تفكير وتحرير بصيغة المتكلم، والمعاني، والصور، ومحاولات أولية، وفحص الوزن والقافية عند وجود بحر، وأسباب التعديل. اختم التعليل بعبارة «النتيجة النهائية:» من غير أن تورد القصيدة النهائية مجتمعة؛ سيضيفها البرنامج حرفيًا.

تركيز هذا القالب: {focus}"""


def family_by_id(template_id: str) -> TemplateFamily:
    """Look up a prompt-template family by its stable identifier.

    Families are searched in declaration order and the stored singleton object
    is returned, preserving the canonical Arabic focus text associated with the
    identifier.

    Args:
        template_id: Exact identifier from a :class:`TemplateFamily`, such as
            ``"semantic_arc"``.

    Returns:
        The matching template-family definition.

    Raises:
        KeyError: If no configured family has ``template_id``.
    """
    for family in TEMPLATE_FAMILIES:
        if family.template_id == template_id:
            return family
    raise KeyError(f"Unknown template family: {template_id}")


def eligible_families(meter_name: str) -> tuple[TemplateFamily, ...]:
    """Return the prompt families that may be used with a poetic meter.

    Prose poetry (``النثر``) excludes the ``prosody_rhyme`` family because that
    template assumes a classical meter. Every other meter receives the complete
    family tuple. Unknown names are treated like metered poetry; validation of
    meter vocabulary belongs to the dataset-loading layer.

    Args:
        meter_name: Canonical Arabic meter name or the prose marker ``النثر``.

    Returns:
        An ordered tuple of eligible families. For metered input this is the
        shared :data:`TEMPLATE_FAMILIES` tuple; for prose it is a filtered tuple.
    """
    if meter_name == "النثر":
        return tuple(
            family
            for family in TEMPLATE_FAMILIES
            if family.template_id != "prosody_rhyme"
        )
    return TEMPLATE_FAMILIES


def _example_user(
    example: FewShot, family: TemplateFamily | None = None
) -> str:
    """Render the user half of one few-shot demonstration.

    The prompt identifies the example's base meter and number of couplets or
    prose units, then includes its reference poem verbatim. When a family is
    supplied, its focus is appended as an additional demonstration constraint;
    otherwise the example remains family-neutral.

    Args:
        example: Demonstration data containing meter, count, and source poem.
        family: Optional template family whose Arabic focus should specialize
            this example.

    Returns:
        An Arabic user message suitable for an OpenAI-compatible chat payload.
    """
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
    """Serialize the assistant half of one few-shot demonstration as JSON.

    The example's instruction and reasoning are copied before modification. If
    a family is provided, the corresponding demonstration additions are
    appended to the instruction and inserted into the reasoning immediately
    before its last ``النتيجة النهائية:`` marker. If the marker is absent, the
    reasoning addition is appended at the end. JSON serialization preserves
    Arabic characters and emits exactly the ``instruction`` and ``reasoning``
    keys expected from the model.

    Args:
        example: Base few-shot answer to serialize.
        family: Optional family used to specialize both generated fields.

    Returns:
        A JSON object encoded as a string for use as an assistant chat message.

    Raises:
        KeyError: If ``family.template_id`` has no entry in
            :data:`FAMILY_DEMO_ADDITIONS`.
    """
    instruction = example.instruction
    reasoning = example.reasoning
    if family is not None:
        instruction_addition, reasoning_addition = FAMILY_DEMO_ADDITIONS[
            family.template_id
        ]
        instruction += f" {instruction_addition}"
        marker = "النتيجة النهائية:"
        marker_position = reasoning.rfind(marker)
        if marker_position >= 0:
            reasoning = (
                reasoning[:marker_position].rstrip()
                + f"\n\n{reasoning_addition}\n\n{marker}"
            )
        else:
            reasoning += f"\n\n{reasoning_addition}"
    return json.dumps(
        {
            "instruction": instruction,
            "reasoning": reasoning,
        },
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
    """Build the complete few-shot chat conversation for one source poem.

    The conversation begins with the Arabic system policy formatted with the
    requested minimum field length and selected family focus. It then adds two
    paired demonstrations chosen from the prose or metered example bank. The
    second demonstration is specialized to the requested family so the model
    sees both a general pattern and a focus-specific pattern. A final user
    message supplies the actual meter, explicit numeric couplet/unit count, and
    source material, then reiterates the JSON-only response contract.

    For ordinary poems, ``poem_text`` is embedded verbatim. For long poems,
    non-``None`` ``analysis_notes`` take precedence and are presented as ordered
    summaries from which the model must infer a single poem-wide instruction
    and reasoning pair. The function performs prompt assembly only; it does not
    call the model or validate the eventual response.

    Args:
        family: Template family controlling the system focus and specialized
            second demonstration.
        meter_name: Canonical Arabic source meter, or ``النثر`` for prose.
        couplet_count: Exact number the generated instruction must state.
        poem_text: Formatted source poem for the normal prompt path. It may be
            ``None`` when ``analysis_notes`` is provided.
        minimum_chars: Minimum length interpolated into the system requirements
            for each generated JSON field.
        analysis_notes: Optional ordered summaries used instead of full source
            text for an oversized poem.

    Returns:
        Ordered OpenAI-compatible message dictionaries, beginning with one
        system message and ending with the real user task.

    Raises:
        KeyError: If the selected family lacks demonstration additions.
    """
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
                {
                    "role": "user",
                    "content": _example_user(example, example_family),
                },
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
