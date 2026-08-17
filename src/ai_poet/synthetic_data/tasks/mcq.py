"""Poem-grounded multiple-choice SFT workflow."""

from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
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


TASK_VERSION = 1
CHOICE_LABELS = ("أ", "ب", "ج", "د")
QUESTION_DOMAINS = (
    ("meaning_inference", "المعنى والاستنتاج"),
    ("imagery_rhetoric", "الصورة والبلاغة"),
    ("contextual_wording", "دلالة اللفظ في السياق"),
    ("semantic_structure", "تدرج المعنى وبنية القصيدة"),
    ("rhyme_sound", "القافية والجرس الظاهران في النص"),
)

SYSTEM_PROMPT = """أنت تنشئ سؤال اختيار من متعدد بالعربية من قصيدة معطاة. يجب أن يكون السؤال قابلًا للإجابة من القصيدة وحدها، وأن تكون له إجابة صحيحة واحدة فقط وثلاثة مشتتات معقولة لكنها خاطئة. لا تعتمد على اسم الشاعر أو عنوان القصيدة أو معلومات تاريخية أو عروضية غير ظاهرة في النص.

أعد كائن JSON فقط بالمفاتيح question وcorrect_answer وdistractors وreasoning. يجب أن تكون distractors قائمة من ثلاثة نصوص. ويجب أن يحتوي reasoning المفاتيح approach وevidence وanswer_assessments وconclusion؛ evidence قائمة شواهد نصية قصيرة، وanswer_assessments قائمة من أربعة كائنات بالمفتاحين answer وassessment تغطي الإجابة الصحيحة والمشتتات مرة واحدة. لا تستخدم حروف الخيارات لأن البرنامج سيرتب الإجابات لاحقًا."""

VALIDATION_SYSTEM_PROMPT = """أنت مدقق سؤال اختيار من متعدد مبني على قصيدة عربية. تحقق أن السؤال يجيب عنه النص وحده، وأن الإجابة المحددة هي الوحيدة الصحيحة، وأن المشتتات الثلاثة معقولة لكنها غير صحيحة، وأن الاستدلال مفصل ومؤيد بشواهد ولا يعتمد على معلومات خارجية. لا تصلح المرشح.

أعد JSON فقط بهذه البنية: {"passed":true,"errors":[]}."""


def question_domain(sample_id: str) -> tuple[str, str]:
    return QUESTION_DOMAINS[int(sample_id[:8], 16) % len(QUESTION_DOMAINS)]


def _normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


def _answer_has_label(value: str) -> bool:
    return bool(re.match(r"^\s*[\(\[]?[أبجد][\)\]\-.:：]", value))


def extract_candidate(value: dict[str, Any]) -> dict[str, Any]:
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
        "question": question.strip(),
        "correct_answer": correct.strip(),
        "distractors": [item.strip() for item in distractors],
        "reasoning": {
            "approach": reasoning["approach"].strip(),
            "evidence": [item.strip() for item in evidence],
            "answer_assessments": cleaned_assessments,
            "conclusion": reasoning["conclusion"].strip(),
        },
    }


def build_generation_messages(poem: PoemRecord, domain_label: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"أنشئ سؤالًا في مجال: {domain_label}.\n"
                "اجعل كل خيار مستقلًا وموجزًا، وفسر في reasoning سبب قبول أو رفض كل جواب.\n"
                f"<poem>\n{poem.poem_text}\n</poem>"
            ),
        },
    ]


def build_validation_messages(
    poem: PoemRecord,
    domain_label: str,
    candidate: dict[str, Any],
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": VALIDATION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"المجال المطلوب: {domain_label}\n<poem>\n{poem.poem_text}\n</poem>\n"
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
        "اقرأ القصيدة الآتية، ثم أجب عن سؤال الاختيار من متعدد اعتمادًا على "
        "النص وحده. حلل السؤال والخيارات بالتفصيل قبل ذكر الإجابة الصحيحة.\n\n"
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
        "الشواهد من القصيدة:",
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
    poem: PoemRecord,
    client: Any,
    settings: Any,
    *,
    generation_fingerprint: str = "legacy",
    **_kwargs: Any,
) -> dict[str, Any]:
    domain_id, domain_label = question_domain(poem.sample_id)
    base_messages = build_generation_messages(poem, domain_label)
    seed = int(poem.sample_id[:8], 16)
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
                "sample_id": poem.sample_id,
                "task_type": TASK_MCQ,
                "request_kind": "mcq_generation" if repair == 0 else "mcq_repair",
                "generation_attempt": attempts,
            },
        )
        try:
            candidate = extract_candidate(extract_json_object(raw))
            errors = []
        except (ValueError, json.JSONDecodeError) as exc:
            candidate = None
            errors = [str(exc)]
        emit_client_trace(
            client,
            {
                "event": "mcq_generation_result",
                "task_type": TASK_MCQ,
                "sample_id": poem.sample_id,
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
            build_validation_messages(poem, domain_label, candidate),
            max_tokens=1200,
            seed=seed + 100_000 + repair,
            trace_context={
                "sample_id": poem.sample_id,
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
                "sample_id": poem.sample_id,
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
    choices = order_choices(poem.sample_id, candidate)
    instruction = render_instruction(poem, candidate["question"], choices)
    response = render_response(candidate, choices)
    provenance = client_provenance(client, poem.sample_id)
    if not hasattr(client, "sample_stats"):
        provenance["network_attempts"] = attempts * 2
    public_choices = [
        {"label": choice["label"], "text": choice["text"]} for choice in choices
    ]
    correct_label = next(choice["label"] for choice in choices if choice["correct"])
    record = {
        "sample_id": poem.sample_id,
        "record_id": f"{TASK_MCQ}:{poem.sample_id}",
        "task_type": TASK_MCQ,
        "task_version": TASK_VERSION,
        "source_row_indices": list(poem.source_row_indices),
        "source_urls": list(poem.source_urls),
        "poet_name": poem.poet_name,
        "poem_title": poem.poem_title,
        "meter_id": poem.meter_id,
        "meter_name": poem.meter_name,
        "couplet_count": poem.couplet_count,
        "question_domain": domain_id,
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
            "sample_id": poem.sample_id,
            "question_domain": domain_id,
            "parsed_candidate": candidate,
            "final_assistant_response": response,
            "generation_fingerprint": generation_fingerprint,
            **provenance,
        },
    )
    return record


def estimate_work(_poem: PoemRecord, _run_settings: Any) -> int:
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
        "question_domains": [
            {"domain_id": domain_id, "label": label}
            for domain_id, label in QUESTION_DOMAINS
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
    pilot_profile="bounded-question-domain",
)
