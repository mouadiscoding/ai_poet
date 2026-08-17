"""Poem-grounded multiple-choice SFT workflow."""

from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Sequence

from ..assignment import sft_split
from ..errors import GenerationError
from ..poems import PoemRecord
from ..validation import extract_json_object
from ..workflow import (
    client_provenance,
    emit_client_trace,
    repair_messages,
    request_verdict,
)
from .base import TASK_MCQ, TaskWorkflow


TASK_VERSION = 3
CHOICE_LABELS = ("أ", "ب", "ج", "د")


@dataclass(frozen=True)
class MCQPrompt:
    prompt_id: str
    question: str


@dataclass(frozen=True)
class MCQTemplate:
    template_id: str
    metadata_field: str
    prompts: tuple[MCQPrompt, ...]


@dataclass(frozen=True)
class MCQWorkItem:
    poem: PoemRecord
    template: MCQTemplate
    prompt: MCQPrompt
    ground_truth: str

    @property
    def sample_id(self) -> str:
        """Expose the source ID for task-neutral integrations and test doubles."""
        return self.poem.sample_id

    @property
    def work_id(self) -> str:
        return f"{TASK_MCQ}:{self.poem.sample_id}:{self.template.template_id}"


MCQ_TEMPLATES = (
    MCQTemplate(
        template_id="poem_meter",
        metadata_field="poem_meter",
        prompts=(
            MCQPrompt("meter_01", "ما البحر الشعري الذي نُظمت عليه هذه القصيدة؟"),
            MCQPrompt("meter_02", "إلى أي بحر شعري تنتمي هذه القصيدة؟"),
            MCQPrompt("meter_03", "أي بحر من بحور الشعر يمثل وزن هذه القصيدة؟"),
            MCQPrompt("meter_04", "ما اسم البحر العروضي لهذه القصيدة؟"),
            MCQPrompt("meter_05", "اختر البحر الشعري الصحيح الذي تنتمي إليه القصيدة."),
        ),
    ),
    MCQTemplate(
        template_id="poem_theme",
        metadata_field="poem_theme",
        prompts=(
            MCQPrompt("theme_01", "ما الموضوع المسجل لهذه القصيدة؟"),
            MCQPrompt("theme_02", "أي موضوع تصف به البيانات الوصفية هذه القصيدة؟"),
            MCQPrompt("theme_03", "ما الثيمة الأساسية المسجلة للقصيدة؟"),
            MCQPrompt("theme_04", "تحت أي موضوع صُنفت هذه القصيدة؟"),
            MCQPrompt("theme_05", "اختر الموضوع الصحيح المسجل للقصيدة."),
        ),
    ),
    MCQTemplate(
        template_id="poem_title",
        metadata_field="poem_title",
        prompts=(
            MCQPrompt("title_01", "ما العنوان المسجل لهذه القصيدة؟"),
            MCQPrompt("title_02", "بأي عنوان وردت هذه القصيدة؟"),
            MCQPrompt("title_03", "ما الاسم الذي تحمله هذه القصيدة في السجل؟"),
            MCQPrompt("title_04", "أي عنوان من الآتي هو عنوان القصيدة؟"),
            MCQPrompt("title_05", "اختر العنوان الصحيح المسجل للقصيدة."),
        ),
    ),
)

SYSTEM_PROMPT = """أنت تنشئ بيانات سؤال اختيار من متعدد بالعربية من قصيدة وبيانات وصفية موثوقة. السؤال والإجابة الصحيحة مقدمان لك بوصفهما حقيقة مرجعية؛ لا تستبدلهما ولا تستنتج إجابة أخرى. أنشئ ثلاثة مشتتات معقولة لكنها مختلفة وخاطئة.

أعد كائن JSON فقط بالمفاتيح question وcorrect_answer وdistractors وreasoning. يجب أن تكون distractors قائمة من ثلاثة نصوص. ويجب أن يحتوي reasoning المفاتيح approach وevidence وanswer_assessments وconclusion؛ evidence قائمة شواهد نصية قصيرة، وanswer_assessments قائمة من أربعة كائنات بالمفتاحين answer وassessment تغطي الإجابة الصحيحة والمشتتات مرة واحدة. لا تستخدم حروف الخيارات لأن البرنامج سيرتب الإجابات لاحقًا."""

VALIDATION_SYSTEM_PROMPT = """أنت مدقق سؤال اختيار من متعدد مبني على قصيدة عربية وبيانات وصفية موثوقة. تحقق أن السؤال يطابق القالب المقدم، وأن الإجابة المحددة تطابق الحقيقة المرجعية حرفيًا، وأن المشتتات الثلاثة معقولة ومختلفة عنها، وأن الاستدلال يقيّم الخيارات الأربعة. لا تصلح المرشح.

أعد JSON فقط بهذه البنية: {"passed":true,"errors":[]}."""


def _ground_truth(poem: PoemRecord, template: MCQTemplate) -> str | None:
    if template.metadata_field == "poem_meter":
        value: Any = poem.meter_name
    else:
        value = getattr(poem, template.metadata_field)
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def build_work_items(poems: Sequence[PoemRecord]) -> list[MCQWorkItem]:
    """Apply every eligible template with one uniformly selected prompt."""
    items: list[MCQWorkItem] = []
    for poem in poems:
        for template in MCQ_TEMPLATES:
            ground_truth = _ground_truth(poem, template)
            if ground_truth is not None:
                seed = int(
                    hashlib.sha256(
                        f"mcq-prompt:{poem.sample_id}:{template.template_id}".encode()
                    ).hexdigest()[:16],
                    16,
                )
                prompt = random.Random(seed).choice(template.prompts)
                items.append(MCQWorkItem(poem, template, prompt, ground_truth))
    return items


def work_item_id(item: MCQWorkItem) -> str:
    return item.work_id


def _normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


def _answer_has_label(value: str) -> bool:
    return bool(re.match(r"^\s*[\(\[]?[أبجد][\)\]\-.:：]", value))


def extract_candidate(
    value: dict[str, Any],
    *,
    expected_question: str | None = None,
    ground_truth: str | None = None,
) -> dict[str, Any]:
    required = {"question", "correct_answer", "distractors", "reasoning"}
    if set(value) != required:
        raise ValueError("MCQ JSON must contain exactly the required fields")
    question = value["question"]
    correct = value["correct_answer"]
    distractors = value["distractors"]
    reasoning = value["reasoning"]
    if not isinstance(question, str) or len(question.strip()) < 12:
        raise ValueError("question must be a substantive string")
    if not isinstance(correct, str) or not correct.strip():
        raise ValueError("correct_answer must be a non-empty string")
    if expected_question is not None and question.strip() != expected_question:
        raise ValueError("question must exactly match the metadata template")
    if ground_truth is not None and _normalize(correct) != _normalize(ground_truth):
        raise ValueError("correct_answer must match the provided ground truth")
    if not isinstance(distractors, list) or len(distractors) != 3:
        raise ValueError("distractors must contain exactly three answers")
    if not all(isinstance(item, str) and item.strip() for item in distractors):
        raise ValueError("every distractor must be a non-empty string")
    answers = [correct.strip(), *(item.strip() for item in distractors)]
    if len({_normalize(answer) for answer in answers}) != 4:
        raise ValueError("all four answers must be distinct")
    if any(_answer_has_label(answer) for answer in answers):
        raise ValueError("answers must not contain choice labels")

    if not isinstance(reasoning, dict) or set(reasoning) != {
        "approach",
        "evidence",
        "answer_assessments",
        "conclusion",
    }:
        raise ValueError("reasoning must contain exactly the required fields")
    for field in ("approach", "conclusion"):
        if not isinstance(reasoning[field], str) or len(reasoning[field].strip()) < 20:
            raise ValueError(f"reasoning.{field} must be detailed")
    evidence = reasoning["evidence"]
    if not isinstance(evidence, list) or not evidence or not all(
        isinstance(item, str) and item.strip() for item in evidence
    ):
        raise ValueError("reasoning.evidence must contain non-empty textual evidence")
    assessments = reasoning["answer_assessments"]
    if not isinstance(assessments, list) or len(assessments) != 4:
        raise ValueError("reasoning.answer_assessments must contain four items")
    assessment_answers: list[str] = []
    cleaned_assessments = []
    for assessment in assessments:
        if not isinstance(assessment, dict) or set(assessment) != {
            "answer",
            "assessment",
        }:
            raise ValueError("each answer assessment must contain answer and assessment")
        answer = assessment["answer"]
        detail = assessment["assessment"]
        if not isinstance(answer, str) or not isinstance(detail, str) or len(detail.strip()) < 20:
            raise ValueError("each answer assessment must be detailed")
        assessment_answers.append(_normalize(answer))
        cleaned_assessments.append(
            {"answer": answer.strip(), "assessment": detail.strip()}
        )
    if sorted(assessment_answers) != sorted(_normalize(answer) for answer in answers):
        raise ValueError("answer assessments must cover each answer exactly once")
    return {
        "question": expected_question or question.strip(),
        "correct_answer": ground_truth or correct.strip(),
        "distractors": [item.strip() for item in distractors],
        "reasoning": {
            "approach": reasoning["approach"].strip(),
            "evidence": [item.strip() for item in evidence],
            "answer_assessments": cleaned_assessments,
            "conclusion": reasoning["conclusion"].strip(),
        },
    }


def build_generation_messages(item: MCQWorkItem) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"حقل البيانات الوصفية: {item.template.metadata_field}\n"
                f"معرّف صياغة السؤال: {item.prompt.prompt_id}\n"
                f"السؤال المطلوب حرفيًا: {item.prompt.question}\n"
                f"الإجابة الصحيحة المرجعية حرفيًا: {item.ground_truth}\n"
                "اجعل كل خيار مستقلًا وموجزًا، وفسر في reasoning سبب قبول أو رفض كل جواب.\n"
                f"<poem>\n{item.poem.poem_text}\n</poem>"
            ),
        },
    ]


def build_validation_messages(
    item: MCQWorkItem,
    candidate: dict[str, Any],
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": VALIDATION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"حقل البيانات الوصفية: {item.template.metadata_field}\n"
                f"معرّف صياغة السؤال: {item.prompt.prompt_id}\n"
                f"السؤال المطلوب حرفيًا: {item.prompt.question}\n"
                f"الإجابة الصحيحة المرجعية حرفيًا: {item.ground_truth}\n"
                f"<poem>\n{item.poem.poem_text}\n</poem>\n"
                f"<candidate>\n{json.dumps(candidate, ensure_ascii=False)}\n</candidate>"
            ),
        },
    ]


def order_choices(sample_id: str, candidate: dict[str, Any]) -> list[dict[str, Any]]:
    answers = [
        {"text": candidate["correct_answer"], "correct": True},
        *({"text": text, "correct": False} for text in candidate["distractors"]),
    ]
    seed = int(
        hashlib.sha256(f"mcq-order:{sample_id}".encode()).hexdigest()[:16], 16
    )
    random.Random(seed).shuffle(answers)
    return [
        {"label": label, **answer}
        for label, answer in zip(CHOICE_LABELS, answers, strict=True)
    ]


def render_instruction(poem: PoemRecord, question: str, choices: Sequence[dict[str, Any]]) -> str:
    rendered_choices = "\n".join(
        f"{choice['label']}) {choice['text']}" for choice in choices
    )
    return (
        "اقرأ القصيدة الآتية، ثم أجب عن سؤال الاختيار من متعدد. "
        "حلل السؤال والخيارات بالتفصيل قبل ذكر الإجابة الصحيحة.\n\n"
        f"القصيدة:\n{poem.poem_text}\n\nالسؤال:\n{question}\n\n"
        f"الخيارات:\n{rendered_choices}"
    )


def render_response(candidate: dict[str, Any], choices: Sequence[dict[str, Any]]) -> str:
    assessments = {
        _normalize(item["answer"]): item["assessment"]
        for item in candidate["reasoning"]["answer_assessments"]
    }
    sections = [
        "التحليل والاستدلال:",
        "",
        candidate["reasoning"]["approach"],
        "",
        "الشواهد المعتمدة:",
        *(f"- {item}" for item in candidate["reasoning"]["evidence"]),
        "",
        "فحص الخيارات:",
        *(
            f"- {choice['label']}) {choice['text']}: "
            f"{assessments[_normalize(choice['text'])]}"
            for choice in choices
        ),
        "",
        candidate["reasoning"]["conclusion"],
    ]
    correct = next(choice for choice in choices if choice["correct"])
    sections.extend(
        ["", f"الإجابة الصحيحة: {correct['label']}) {correct['text']}"]
    )
    return "\n".join(sections)


def generate_one(
    item: MCQWorkItem,
    client: Any,
    settings: Any,
    *,
    generation_fingerprint: str = "legacy",
    **_kwargs: Any,
) -> dict[str, Any]:
    poem = item.poem
    base_messages = build_generation_messages(item)
    seed = int(hashlib.sha256(item.work_id.encode()).hexdigest()[:8], 16)
    raw = ""
    errors: list[str] = []
    candidate: dict[str, Any] | None = None
    attempts = 0
    for repair in range(settings.max_repairs + 1):
        attempts = repair + 1
        messages = base_messages if repair == 0 else repair_messages(
            base_messages, raw, errors
        )
        raw = client.chat(
            messages,
            seed=seed + repair,
            trace_context={
                "sample_id": item.work_id,
                "task_type": TASK_MCQ,
                "request_kind": "mcq_generation" if repair == 0 else "mcq_repair",
                "generation_attempt": attempts,
            },
        )
        try:
            candidate = extract_candidate(
                extract_json_object(raw),
                expected_question=item.prompt.question,
                ground_truth=item.ground_truth,
            )
            errors = []
        except (ValueError, json.JSONDecodeError) as exc:
            candidate = None
            errors = [str(exc)]
        emit_client_trace(
            client,
            {
                "event": "mcq_generation_result",
                "task_type": TASK_MCQ,
                "sample_id": item.work_id,
                "poem_sample_id": poem.sample_id,
                "metadata_field": item.template.metadata_field,
                "prompt_id": item.prompt.prompt_id,
                "generation_attempt": attempts,
                "raw_model_content": raw,
                "parsed_output": candidate,
                "passed": not errors,
                "deterministic_errors": errors,
            },
        )
        if errors or candidate is None:
            continue
        validation_raw, verdict, validator_attempts = request_verdict(
            client,
            build_validation_messages(item, candidate),
            max_tokens=1200,
            seed=seed + 100_000 + repair,
            trace_context={
                "sample_id": item.work_id,
                "task_type": TASK_MCQ,
                "request_kind": "mcq_validation",
                "generation_attempt": attempts,
            },
        )
        errors = [f"Gemma rejected MCQ: {error}" for error in verdict["errors"]]
        emit_client_trace(
            client,
            {
                "event": "mcq_validation_result",
                "task_type": TASK_MCQ,
                "sample_id": item.work_id,
                "poem_sample_id": poem.sample_id,
                "metadata_field": item.template.metadata_field,
                "prompt_id": item.prompt.prompt_id,
                "raw_validator_content": validation_raw,
                "parsed_verdict": verdict,
                "validator_format_attempts": validator_attempts,
                "passed": not errors,
            },
        )
        if not errors:
            break
    else:
        raise GenerationError("MCQ remained invalid after repairs: " + "; ".join(errors))

    if candidate is None:
        raise GenerationError("MCQ generation produced no candidate")
    choices = order_choices(item.work_id, candidate)
    instruction = render_instruction(poem, candidate["question"], choices)
    response = render_response(candidate, choices)
    provenance = client_provenance(client, item.work_id)
    if not hasattr(client, "sample_stats"):
        provenance["network_attempts"] = attempts * 2
    public_choices = [
        {"label": choice["label"], "text": choice["text"]} for choice in choices
    ]
    correct_label = next(choice["label"] for choice in choices if choice["correct"])
    record = {
        "sample_id": poem.sample_id,
        "record_id": item.work_id,
        "task_type": TASK_MCQ,
        "task_version": TASK_VERSION,
        "source_row_indices": list(poem.source_row_indices),
        "source_urls": list(poem.source_urls),
        "poet_name": poem.poet_name,
        "poem_title": poem.poem_title,
        "poem_theme": poem.poem_theme,
        "meter_id": poem.meter_id,
        "meter_name": poem.meter_name,
        "couplet_count": poem.couplet_count,
        "template_id": item.template.template_id,
        "prompt_id": item.prompt.prompt_id,
        "metadata_field": item.template.metadata_field,
        "ground_truth_answer": item.ground_truth,
        "question": candidate["question"],
        "choices": public_choices,
        "correct_choice_label": correct_label,
        "instruction": instruction,
        "response": response,
        "messages": [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ],
        "sft_split": sft_split(poem.sample_id),
        "metadata_conflict": poem.metadata_conflict,
        "generation_attempts": attempts,
        "validation_status": "passed_after_repair" if attempts > 1 else "passed",
        **provenance,
    }
    emit_client_trace(
        client,
        {
            "event": "final_output",
            "task_type": TASK_MCQ,
            "sample_id": item.work_id,
            "poem_sample_id": poem.sample_id,
            "metadata_field": item.template.metadata_field,
            "prompt_id": item.prompt.prompt_id,
            "parsed_candidate": candidate,
            "final_assistant_response": response,
            "generation_fingerprint": generation_fingerprint,
            **provenance,
        },
    )
    return record


def estimate_work(_item: MCQWorkItem, _run_settings: Any) -> int:
    return 2


def contract_settings(settings: Any) -> dict[str, Any]:
    return {
        "temperature": settings.temperature,
        "top_p": settings.top_p,
        "max_tokens": settings.max_tokens,
        "max_source_chars": settings.max_source_chars,
    }


def trace_metadata() -> dict[str, Any]:
    return {
        "mcq_templates": [
            {
                "template_id": template.template_id,
                "metadata_field": template.metadata_field,
                "prompts": [
                    {
                        "prompt_id": prompt.prompt_id,
                        "question": prompt.question,
                    }
                    for prompt in template.prompts
                ],
            }
            for template in MCQ_TEMPLATES
        ],
        "generation_system_prompt": SYSTEM_PROMPT,
        "validation_system_prompt": VALIDATION_SYSTEM_PROMPT,
    }


WORKFLOW = TaskWorkflow(
    task_type=TASK_MCQ,
    version=TASK_VERSION,
    generate_one=generate_one,
    estimate_work=estimate_work,
    contract_settings=contract_settings,
    trace_metadata=trace_metadata,
    checkpoint_stages=(),
    benchmark_profile="single-generation-validation",
    pilot_profile="bounded-metadata-template",
    expand_work_items=build_work_items,
    work_item_id=work_item_id,
)
