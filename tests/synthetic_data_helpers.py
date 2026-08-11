"""Shared offline fixtures for synthetic data tests."""

from __future__ import annotations

import json
from pathlib import Path

from ai_poet.synthetic_data.config import GenerationSettings
from ai_poet.synthetic_data.meters import meter_name
from ai_poet.synthetic_data.poems import PoemRecord, poem_hash
from ai_poet.synthetic_data.tracing import GenerationTracer


TEST_TMP = Path(__file__).parent / "_tmp"


def remove_test_files(*names: str) -> None:
    for name in names:
        (TEST_TMP / name).unlink(missing_ok=True)


def make_poem(
    *,
    meter_id: int = 1,
    verses: tuple[str, ...] = (
        "يا قلب صبرا على الأيام",
        "فالليل يعقبه الضياء",
        "واجعل رجاءك خير زاد",
        "حتى يزول بك العناء",
    ),
) -> PoemRecord:
    return PoemRecord(
        sample_id=poem_hash(verses),
        source_row_indices=(7,),
        source_urls=("https://example.test/poem/7",),
        poet_name="شاعر الاختبار",
        poem_title="عنوان الاختبار",
        meter_id=meter_id,
        meter_name=meter_name(meter_id),
        verses=verses,
        metadata_conflict=False,
    )


def valid_instruction_value(
    poem: PoemRecord, minimum: int = 80
) -> dict[str, str]:
    instruction = (
        f"الموضوع العام:\nأنت شاعر عربي فصيح. اكتب {poem.couplet_count} من "
        f"الأبيات على بحر {poem.meter_name} في معنى الصبر والرجاء، ورتب المعاني "
        "من الشدة إلى الانفراج.\n\n"
        "الجو العاطفي المطلوب:\nصوت متكلم هادئ ونبرة واثقة تخاطب سامعًا عامًا.\n\n"
        "ألفاظ وصور يُستحسن استعمالها أو الدوران حولها:\nالقلب والليل والضياء "
        "والرجاء، مع مقابلة العتمة بالنور وتجريب بدائل معجمية.\n\n"
        "القافية:\nوحّد الروي بما يلائم الجرس الهادئ والمعنى المتدرج.\n\n"
        f"شرح البحر المطلوب:\nبحر {poem.meter_name} بحر عربي موزون.\n"
        "وزنه في كل بيت كامل:\nتفعيلات البحر الأصلية في الصدر والعجز.\n\n"
        "الصورة الصوتية التقريبية:\nراجع المقاطع الطويلة والقصيرة وزن المنطوق، "
        "وفك الشدة واقبل الزحافات الصحيحة.\n\n"
        f"خطة عملية لصناعة بيت على بحر {poem.meter_name}:\nحدد المعنى والصورة، "
        "ثم اكتب مسودة وانطقها وقطّعها وعدّل اللفظ وافحص كل شطر. "
    )
    if len(instruction) < minimum:
        instruction += "راجع وحدة المعنى والصورة والقافية. " * (
            minimum // 35 + 1
        )
    return {"instruction": instruction}


def valid_reasoning_value(
    poem: PoemRecord,
    *,
    start_offset: int = 0,
    chunk_size: int | None = None,
) -> dict[str, object]:
    couplets = poem.poem_text.splitlines()
    selected = couplets[
        start_offset : None if chunk_size is None else start_offset + chunk_size
    ]
    blocks = []
    for offset, couplet in enumerate(selected, start=start_offset + 1):
        blocks.append(
            {
                "verse_index": offset,
                "intended_meaning": "أحدد لهذا البيت معنى متدرجًا يخدم انتقال القصيدة نحو الرجاء.",
                "connection_to_previous": "أصله بما قبله، أو أجعله مطلعًا ممهدًا إن كان أول بيت.",
                "imagery_and_diction": "أختار صورة الضوء وأوازن بين اللفظ المباشر والصورة البلاغية.",
                "first_draft": f"مسودة مختلفة لصدر البيت {offset} = ومسودة مختلفة لعجزه",
                "problem_with_first_draft": "عبارة مسودة مختلفة تقريرية وفي إيقاعها ثقل؛ لذلك أستبدلها بصورة الضوء وأخفف ترتيب العجز.",
                "revised_draft": couplet,
                "first_hemistich_scansion": "أنطق الصدر موصولًا وأقسمه إلى تفعيلات البحر مع فك الشدة.",
                "second_hemistich_scansion": "أنطق العجز وأراجع حدوده الصوتية وموضع الضرب في آخره.",
                "rhyme_check": "أفحص حرف الروي وحركته وأربط جرْسه بخاتمة المعنى.",
            }
        )
    value: dict[str, object] = {"verse_reasoning": blocks}
    if start_offset == 0:
        value["overview"] = (
            "أبني الأبيات في مسار واحد يبدأ بالصبر وينتهي بالضياء، وأجعل كل بيت "
            "يمهد للذي يليه في المعنى والصورة."
        )
    return value


def valid_verdict() -> dict[str, object]:
    return {"passed": True, "errors": []}


def valid_pipeline_outputs(poem: PoemRecord) -> list[str]:
    """Return queued outputs for one successful single-chunk generation."""
    return [
        json.dumps(valid_instruction_value(poem), ensure_ascii=False),
        json.dumps(valid_verdict(), ensure_ascii=False),
        json.dumps(valid_reasoning_value(poem), ensure_ascii=False),
        json.dumps(valid_verdict(), ensure_ascii=False),
    ]


class QueueClient:
    def __init__(
        self,
        outputs: list[str],
        tracer: GenerationTracer | None = None,
    ) -> None:
        self.outputs = list(outputs)
        self.calls: list[list[dict[str, str]]] = []
        self.call_kwargs: list[dict[str, object]] = []
        self.tracer = tracer

    def chat(self, messages, **kwargs):
        self.calls.append(list(messages))
        self.call_kwargs.append(dict(kwargs))
        if not self.outputs:
            raise AssertionError("unexpected client call")
        return self.outputs.pop(0)


def settings(**overrides) -> GenerationSettings:
    values = {
        "endpoint": "https://example.test/v1/chat/completions",
        "model": "gemma-test",
        "api_key": "secret",
        "insecure": False,
        "timeout": 1,
        "max_network_retries": 1,
        "max_repairs": 2,
        "temperature": 0.4,
        "top_p": 0.9,
        "max_tokens": 4096,
        "min_chars": 80,
        "max_source_chars": 24000,
        "chunk_chars": 12000,
    }
    values.update(overrides)
    return GenerationSettings(**values)
