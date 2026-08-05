"""Shared offline fixtures for synthetic data tests."""

from __future__ import annotations

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


def valid_value(poem: PoemRecord, minimum: int = 80) -> dict[str, str]:
    instruction_seed = (
        f"أنت شاعر عربي فصيح. اكتب {poem.couplet_count} من الأبيات على بحر "
        f"{poem.meter_name}. فصّل الموضوع والمعاني والصور والبلاغة والقافية "
        "والجو العاطفي، وراع سلامة اللغة والنطق العروضي عند بناء النص. "
    )
    reasoning_seed = (
        "مرحلة التفكير والتحرير: أحدد المعنى ثم أختار صورة بلاغية واستعارة مناسبة. "
        "أجرب صياغة أولى، ثم أجري تعديلًا وتحريرًا بعد فحص الوزن والإيقاع والقافية. "
        "أحافظ على وحدة المعاني وأشرح سبب كل محاولة وتعديل. النتيجة النهائية:"
    )
    repetitions = max(2, minimum // min(len(instruction_seed), len(reasoning_seed)) + 2)
    return {
        "instruction": instruction_seed * repetitions,
        "reasoning": reasoning_seed * repetitions,
    }


class QueueClient:
    def __init__(
        self,
        outputs: list[str],
        tracer: GenerationTracer | None = None,
    ) -> None:
        self.outputs = list(outputs)
        self.calls: list[list[dict[str, str]]] = []
        self.tracer = tracer

    def chat(self, messages, **kwargs):
        self.calls.append(list(messages))
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
