# How QCM templates become question-answer pairs

The pipeline starts from an existing poem and produces a two-message SFT pair.
The QCM generation is grounded in the poem itself, and the final assistant
message always includes the exact source poem so that the model cannot
hallucinate or rewrite the reference text.

```text
Source poem
   |
   v
Choose one QCM template
   |
   v
Gemma generates {question, choices, reasoning, correct_answer}
   |
   +--> deterministic QCM checks
   +--> Gemma QCM-quality review
   |
   v
Compose assistant response: poem + question + choices + reasoning + answer
   |
   v
Store {user: poem, assistant: response}
```

## 1. Source preparation

[`load_poems()`](../src/ai_poet/synthetic_data/corpus.py) reads alternating
hemistichs from the source dataset. Each adjacent pair becomes one canonical
line in the form `صدر = عجز`. Identical verse sequences are deduplicated while
their provenance is retained.

## 2. QCM generation

[`prompts/qcm_templates.py`](../src/ai_poet/synthetic_data/prompts/qcm_templates.py)
contains four independently worded QCM templates. Every template injects:

- A shared list of question categories that are relevant only when the poem
  actually supports them.
- A shared reasoning process that walks from the question through textual
  evidence, comparison of the four choices, and justification of the correct
  answer.
- A strict JSON output contract with exactly the keys `question`, `choices`,
  `reasoning`, and `correct_answer`.

[`build_qcm_messages()`](../src/ai_poet/synthetic_data/prompts/builder.py)
combines the chosen template with three few-shot demonstrations: a metered
main-idea example, a metered contrast example, and a prose image example. The
model must produce one question whose answer is determinable from the poem, four
plausible choices with exactly one correct answer, and a demonstrative reasoning
that explains why the correct choice is correct and why the others are less
suited.

## 3. Deterministic and semantic validation

[`validation.py`](../src/ai_poet/synthetic_data/validation.py) rejects a QCM
unless:

- It has exactly the keys `question`, `choices`, `reasoning`,
  `correct_answer`.
- `choices` contains exactly four non-empty strings under A, B, C, D.
- `correct_answer` is one of A, B, C, D.
- The question is at least 20 characters.
- The reasoning is at least 150 characters and references at least one choice
  letter.
- No other choice duplicates the correct answer's text.
- The reasoning is not a generic one-liner such as «الإجابة B صحيحة لأن
  النص يتحدث عن ذلك».
- The reasoning contains no generation metatext.

Gemma then judges whether the question is grounded in the poem, the choices are
plausible with exactly one correct answer, and the reasoning is natural,
logical, poem-based, and specific to the question and the choices. A rejected
QCM is repaired with phase-specific feedback until it passes or the repair
budget is exhausted. A malformed validator response fails safely.

## 4. Composing the assistant response

[`compose_qcm_response()`](../src/ai_poet/synthetic_data/responses.py) produces
the exact assistant message:

```text
<exact source poem>

السؤال:
<question>

الخيارات:
A. <choice A>
B. <choice B>
C. <choice C>
D. <choice D>

الاستدلال:
<reasoning>

الإجابة الصحيحة: <letter>
<correct answer text>
```

The exported record keeps the trainer-facing contract:

```json
{
  "poem": "<exact source poem>",
  "question": "<generated question>",
  "choices": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "reasoning": "<demonstrative reasoning>",
  "correct_answer": "B",
  "correct_answer_text": "<text of correct answer>",
  "response": "<composed assistant response>",
  "messages": [
    {"role": "user", "content": "<exact source poem>"},
    {"role": "assistant", "content": "<composed assistant response>"}
  ]
}
```

The structured QCM object and raw model exchanges remain generation-time or
trace data; they are not added to the SFT message schema beyond the fields
listed above.

## 5. Long poems and checkpoint compatibility

For source poems above the configured direct-source limit,
[`_chunk_analysis()`](../src/ai_poet/synthetic_data/generation.py) creates
ordered summaries of every chunk, and those summaries are combined so that the
QCM question is built from the poem as a whole rather than from a single chunk.
The final record still includes the complete source poem.

QCM template version `8` invalidates older checkpoint successes so they are
regenerated under the QCM contract. Successful records include the total QCM
generation-attempt count and the validation status.