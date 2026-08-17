# Building the Ashaar QCM SFT dataset

## Purpose

This document describes the pipeline implemented by the
[`ai_poet.synthetic_data`](../src/ai_poet/synthetic_data) package. The pipeline
converts the poems in
`data/ashaar_classic_moroccan.parquet` into supervised fine-tuning examples
containing:

1. The exact source poem.
2. A poem-grounded multiple-choice question (QCM).
3. Four plausible choices with exactly one correct answer.
4. A demonstrative reasoning that explains why the correct answer is correct.
5. The correct answer letter and its text.

The generator uses the configured chat-completions endpoint and model to
produce a question that is answerable from the poem itself, that covers an
aspect actually present in the text, and that offers four plausible choices
with exactly one correct answer. The Python code assembles the final assistant
message so that the exact source poem is always preserved.

## Source corpus findings

The current input Parquet contains 50,199 rows. Inspection of the actual file
found:

| Property | Value |
| --- | ---: |
| Source rows | 50,199 |
| Unique poem texts | 50,193 |
| Duplicate source rows beyond the unique texts | 6 |
| Distinct poem-text groups with conflicting meter metadata | 1 |
| Poems longer than the default 24,000-character direct-input limit | 26 |
| Rows with an even, non-empty hemistich list | 50,199 |

`poem_verses` is a list of hemistich strings. Items zero and one form the first
couplet, items two and three form the second, and so on. The maximum source
poem contains 11,608 hemistichs, or 5,804 complete couplets, so sending every
poem through one request is not possible reliably.

Most semantic metadata is sparse:

- `poem_title` is absent for 49,478 source rows.
- `poem_theme` is absent for 49,699 source rows.
- `poem_language_type` is `فصيح` for 49,552 rows and absent for the rest.

Consequently, the model must infer most semantic constraints from the poem
itself rather than relying on titles or theme labels.

## Meter mapping

The saved Parquet stores the base meter as a numeric class ID. The authoritative
class-label order was recovered from the locally cached Hugging Face dataset
metadata:

| ID | Meter | ID | Meter |
| ---: | --- | ---: | --- |
| 0 | البسيط | 9 | المجتث |
| 1 | الخفيف | 10 | المديد |
| 2 | الرجز | 11 | المضارع |
| 3 | الرمل | 12 | المقتضب |
| 4 | السريع | 13 | المنسرح |
| 5 | الطويل | 14 | النثر |
| 6 | الكامل | 15 | الهزج |
| 7 | المتدارك | 16 | الوافر |
| 8 | المتقارب |  |  |

These labels identify only the base meter. They do not establish whether an
individual poem uses a complete, catalectic, truncated, or other form. The
prompts therefore name the base meter but do not force a meter question when
the poem does not make the meter relevant. Records labeled `النثر` use a
separate prompt route where the question must be answerable from the prose
poem's internal cadence, parallel syntax, and imagery without claiming a
classical Arabic meter.

## Deduplication and provenance

The stable sample identifier is the SHA-256 digest of all exact hemistichs
joined with a dedicated separator. Exact poem texts are grouped before any API
requests are made, preventing duplicated targets from consuming generation
capacity or crossing data splits.

For each group the output retains:

- Every original row index.
- Every non-empty source URL.
- A canonical title and poet name.
- A `metadata_conflict` flag.

The current corpus has one conflicting group. Three source rows contain the
same poem, with two identifying its meter as `المديد` and one as `البسيط`.
The pipeline chooses the unique majority label and marks the result as a
metadata conflict. A tied conflict causes input loading to fail instead of
silently choosing a label.

## QCM template architecture

### Why concrete templates are used

A single fixed surface prompt would make the SFT corpus formulaic. The
implementation therefore provides four independently authored, complete Arabic
QCM templates. Each uses the same placeholders for meter, couplet count, and
source poem, and each injects the shared question-category list, reasoning
process, and output contract. The templates vary the framing and the analytical
entry point (comprehension, analysis, inference, formal/discursive) while every
template requires the same core contract:

- A single question whose answer is determinable from the poem.
- Exactly four choices under A, B, C, D with exactly one correct answer.
- A reasoning that follows the full conceptual path from the question to the
  answer and that explains why each other choice is less suited.

One template is selected uniformly without a seed at the start of each poem
generation call and is retained for all repairs in that call. A later rerun of
an unresolved poem may choose another template. Every template is eligible for
prose; the QCM question is always grounded in the poem's actual content, so no
forced formal question is imposed when the poem does not support it.

### Question-category guidance

The shared category list is injected into every template:

- The main idea or general theme of the poem.
- The meaning of a passage, a verse, or a group of verses.
- The poet's intention or the poem's aim.
- The tone, emotional atmosphere, or attitude toward the addressee.
- The emotions expressed.
- The addressee or the person described, and relations among people mentioned.
- A progression, sequence, opposition, contradiction, or contrast within the poem.
- A poetic image, metaphor, comparison, or rhetorical device.
- A lexical field, a choice of specific words, or the meaning of an expression
  in context.
- The occasion or context of the poem when it is identifiable in the text.
- The structure or organisation of the discourse.
- The rhyme or formal elements when they are genuinely relevant.
- The meter when the information can be deduced or verified from the available
  data.
- The relation between different passages of the poem.
- A reasonable inference the reader can make from the text.

The model is explicitly told to pick the most pertinent category for the
concrete poem and never to force a category that the text does not support.
This is what makes the resulting questions varied: the diversity comes from the
poem's actual content rather than from a fixed question type.

### Reasoning process

Every template injects the same reasoning process to guarantee a demonstrative,
poem-specific explanation rather than a generic one-liner:

```
Question
↓
Understand what the question asks
↓
Analyze the poem only from the angle relevant to that question
↓
Identify the elements of the poem that allow an answer
↓
Compare those elements to choices A/B/C/D
↓
Determine the choice that actually matches the text
↓
Explain why that choice is correct
↓
When relevant, explain why the other choices are incorrect or less suited
```

The reasoning must therefore be natural, logical, based on the poem, specific
to the question, specific to the proposed choices, and sufficiently
explanatory to demonstrate why the correct answer is correct.

### Few-shot demonstrations

The QCM build uses a chat sequence of `system` plus three `user`/`assistant`
demonstrations followed by the selected concrete template. The three examples
show different types of question to encourage variety:

- A metered (`الخفيف`) example with a main-idea question.
- A metered (`الرمل`) example with a contrast/contradiction question.
- A prose (`النثر`) example with an image-meaning question.

Each example's reasoning follows the full conceptual path and references
specific words, verses, and choices.

## Output contract

Every QCM generation must produce a JSON object with exactly these keys:

```json
{
  "question": "...",
  "choices": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "reasoning": "...",
  "correct_answer": "B"
}
```

- `question`: the question text.
- `choices`: exactly four non-empty strings under A, B, C, D.
- `reasoning`: the demonstrative reasoning described above.
- `correct_answer`: exactly one of A, B, C, D.

The model must not mention that it is analyzing a reference text, and it must
not talk about the template, fields, or instructions.

## Sequence of generation

For a normal (non-oversized) poem, `generate_one`:

1. Selects one of the four QCM templates uniformly at random.
2. Builds the `system` + three few-shot pairs + final user message containing
   the poem.
3. Sends the conversation to the endpoint and parses the QCM JSON.
4. Runs deterministic validation (`extract_qcm` + `qcm_contract_errors`).
5. Sends the QCM to a Gemma semantic judge at temperature zero, expecting
   `{"passed": true, "errors": []}`.
6. Repairs with phase-specific feedback until the QCM passes or the repair
   budget is exhausted.
7. Assembles the final record with the exact source poem, the question, the
   choices, the reasoning, the correct answer, and the OpenAI-style `messages`.

The repair loop always reuses the original base messages and appends the
rejected assistant content plus actionable repair feedback.

## Oversized-poem processing

The direct source limit defaults to 24,000 characters. A longer poem is split
at complete-couplet boundaries into chunks of at most 12,000 characters.

For each chunk, the endpoint receives a compact-analysis request. The ordered
summaries from all chunks are then combined, and the QCM is built from the
poem as a whole, not from a single chunk. The question therefore reflects the
entire poem, which matches the user requirement that a global QCM be built from
the full text when the poem is too long to send directly.

The final SFT record still includes the complete source poem and marks such
rows with:

```text
oversized_for_sft = true
```

This flag is important: retaining every poem in the master dataset does not
mean every row fits the context length of a downstream training model.

### Authentication and TLS

The endpoint, model, and API key are loaded from `.env` as `GEMMA_ENDPOINT`,
`GEMMA_MODEL`, and `GEMMA_API_KEY`. All three values are required, and there are
no endpoint or model defaults in the source code. Existing process-environment
values take precedence over the file. The key is never written to checkpoints,
output records, manifests, or logs. Errors are scrubbed if the key somehow
appears in their text.

TLS certificate verification is enabled by default. `--insecure` explicitly
creates an unverified TLS context and is equivalent to `curl -k`. It should be
used only for a trusted internal endpoint.

### Full generation tracing

Pass `--trace` to print complete, non-interleaved audit blocks to the terminal
and append the same events as UTF-8 JSON lines to
`generation_trace.jsonl`. The option is intended primarily for smoke samples,
debugging, and audits because logging every few-shot prompt and completion for
the full corpus consumes substantial terminal and disk space.

The append-only trace assigns a new `run_id` to each invocation and records:

- Run settings, source fingerprint, checkpoint reuse count, template version,
  the four QCM concrete prompts, and the shared reasoning process.
- Per-poem provenance, all eligible template IDs, the fresh random selection,
  and why the selected template is retained through repairs.
- Every request kind: oversized-poem analysis, QCM generation and repair, and
  the QCM quality review.
- The full OpenAI-compatible request body, including every system, few-shot,
  and final user message, seed, and decoding settings.
- The complete decoded API response payload. This retains endpoint-provided
  fields such as `message.reasoning_content`, `finish_reason`, and `usage` when
  the server supplies them.
- Network-attempt counts, retry errors, and elapsed time.
- Raw generation content, parsed QCM structures, deterministic contract errors,
  raw and parsed Gemma verdicts, and whether a phase-specific repair is needed.
- The parsed QCM, the final assistant response (including the exact source
  poem), and the final success/failure and validation-status counts.

The request headers are never included. Before an event is printed or written,
the configured API key is recursively replaced with `[REDACTED]` anywhere it
might appear in request, response, or error text.

## Response validation and repair

### Deterministic validation

Python validates the QCM structurally before the semantic Gemma judge is
consulted:

- `extract_qcm` requires exactly the keys `question`, `choices`, `reasoning`,
  and `correct_answer`; non-empty strings; `choices` with exactly A/B/C/D keys;
  and `correct_answer` in {A, B, C, D}.
- `qcm_contract_errors` additionally checks:
  - a minimum question length (20 characters);
  - a minimum reasoning length (150 characters);
  - a minimum choice length (15 characters per choice);
  - that no other choice duplicates the correct answer's text;
  - that the reasoning references at least one choice letter (أ/ب/ج/د or
    A/B/C/D);
  - that the reasoning is not a generic one-liner such as «الإجابة B صحيحة
    لأن النص يتحدث عن ذلك»;
  - that the reasoning does not expose generation metatext.

### Semantic validation

Gemma then reviews the QCM at temperature zero. The judge verifies that:

- the question is related to and grounded in the poem, and the answer is
  determinable from the text;
- the four choices are plausible on the surface but only one is correct or most
  accurate;
- the reasoning is natural, logical, based on the poem, specific to the
  question and to the choices, and demonstrates why the correct answer is
  correct and why the others are less suited;
- the reasoning is not a generic sentence.

The judge returns exactly:

```json
{"passed": true, "errors": []}
```

QCM failures repair only the QCM. Malformed verdicts fail safely. Network
retries remain separate from content repairs.

## Checkpoint and resume semantics

`generation_checkpoint.jsonl` is an append-only event log. Every completed
future writes either:

```json
{"status": "success", "sample_id": "...", "record": {}}
```

or:

```json
{"status": "failure", "sample_id": "...", "error": "..."}
```

On restart, a successful record is reused only when its `template_version` is
8 (`QCM_TEMPLATE_VERSION`). Legacy successes without that version are
regenerated and remain intact in the append-only log. Failed or missing IDs are
also submitted again. A later success supersedes an earlier failure.

Checkpoint events are written by the main thread after each concurrent task
finishes, so individual JSON lines are not interleaved. Each successful SFT
record is then appended and flushed to `ashaar_sft.jsonl`, making it visible
while generation continues. During a run, newly generated records appear in
completion order; when the run finishes, the file is rewritten in source order.

When `--limit` is used, checkpoint entries outside the selected prefix are
ignored for that run. The command returns exit code 1 while any selected sample
is unresolved, exit code 2 for invalid configuration or input, and exit code 0
only when all selected samples have valid records.

## Dataset splits

The first eight hexadecimal digits of the poem hash are mapped to one of 100
buckets:

- Buckets 0–97: `train`
- Bucket 98: `validation`
- Bucket 99: `test`

This creates stable approximate 98/1/1 splits and guarantees that exact poem
duplicates cannot cross splits. For the current 50,193 unique poems the actual
counts are:

| Split | Rows |
| --- | ---: |
| Train | 49,191 |
| Validation | 494 |
| Test | 508 |

## Output schema

The pipeline writes the same logical records to `ashaar_sft.jsonl` and
`ashaar_sft.parquet`.

| Field | Meaning |
| --- | --- |
| `sample_id` | SHA-256 of the exact source hemistich sequence |
| `source_row_indices` | All original rows represented by the record |
| `source_urls` | All retained source URLs |
| `poet_name` | Canonical source poet name for provenance |
| `poem_title` | Canonical title when available |
| `meter_id` | Original numeric base-meter class |
| `meter_name` | Decoded Arabic base-meter name |
| `couplet_count` | Number of hemistich pairs |
| `poem` | The exact source poem text |
| `template_id` | Selected concrete QCM prompt template |
| `template_version` | QCM prompt and validation contract version; currently `8` |
| `question` | Generated Arabic QCM question |
| `choices` | Four-choice object keyed by A, B, C, D |
| `reasoning` | Demonstrative reasoning tied to the question and choices |
| `correct_answer` | Letter (A/B/C/D) of the correct choice |
| `correct_answer_text` | Text of the correct choice |
| `response` | The composed assistant response |
| `messages` | Two-message chat SFT representation |
| `sft_split` | Stable train, validation, or test assignment |
| `oversized_for_sft` | Whether chunk analysis was required |
| `metadata_conflict` | Whether duplicate sources disagreed on meter |
| `generation_attempts` | Total QCM generation attempts including repairs |
| `validation_status` | Whether the QCM passed directly or after repair |

`messages` contains:

```json
[
  {"role": "user", "content": "<poem>"},
  {"role": "assistant", "content": "<response>"}
]
```

The composed assistant response has this exact structure:

```text
<القصيدة الأصلية>

السؤال:
<السؤال>

الخيارات:
A. <الخيار أ>
B. <الخيار ب>
C. <الخيار ج>
D. <الخيار د>

الاستدلال:
<الاستدلال>

الإجابة الصحيحة: <الحرف>
<نص الإجابة الصحيحة>
```

The output directory also contains:

- `generation_checkpoint.jsonl`
- `generation_trace.jsonl` when `--trace` is enabled
- `failures.jsonl`
- `manifest.json`

The manifest records completion status, source SHA-256, template version, model
and endpoint, decoding and length settings, output counts, split distribution,
template distribution, oversized count, and metadata-conflict count.

The trace file is intentionally a sidecar rather than part of the training
schema. Complete prompts repeat the few-shot demonstrations, rejected outputs
are not training targets, and both substantially increase storage. Therefore
`ashaar_sft.jsonl` and `ashaar_sft.parquet` remain identical and
trainer-focused; audit events are joined by `sample_id` when an investigation
requires them. The manifest's `trace` object reports whether tracing was
enabled, its `run_id`, and the sidecar filename.

## Running the pipeline

Create a local `.env` from the committed example, then set the endpoint, model,
and a new API token in that file:

```powershell
Copy-Item .env_example .env
```

```dotenv
GEMMA_ENDPOINT=https://your-host.example/v1/chat/completions
GEMMA_MODEL=your-model-name
GEMMA_API_KEY=replace-with-a-new-token
```

Run a ten-poem inspection sample first:

```powershell
uv run ai-poet-generate-sft `
  --input data/ashaar_classic_moroccan.parquet `
  --output-dir data/ashaar_sft_smoke `
  --limit 10 `
  --trace `
  --insecure
```

After manually reviewing the sample, run the corpus:

```powershell
uv run ai-poet-generate-sft `
  --input data/ashaar_classic_moroccan.parquet `
  --output-dir data/ashaar_sft `
  --concurrency 4 `
  --insecure
```

Re-running the same command resumes from the checkpoint. See every available
override with:

```powershell
uv run ai-poet-generate-sft --help
```

## Verification

The automated suite is offline and uses fake API clients:

```powershell
uv run python -m unittest discover -s tests -v
```

It covers:

- Rendering and fresh random selection of all four QCM templates.
- Comprehensive metered and prose QCM few-shot examples.
- Meter decoding and invalid IDs.
- Exact hemistich pairing.
- Stable hashes and splits.
- Deduplication and majority conflict resolution.
- JSON and fenced-JSON extraction.
- Strict Gemma verdict parsing, rejection feedback, and QCM repair.
- Deterministic QCM structural validation and contract checks.
- Exact source-poem preservation in the composed response.
- Oversized-poem summaries and global-QCM synthesis.
- QCM-template-version-aware checkpoint reuse.
- JSONL and Parquet round trips.

Before a full production run is accepted, verify that:

1. A live smoke sample has been inspected manually.
2. The manifest reports `complete: true`.
3. `generated_poems` equals 50,193 for the current full source.
4. `unresolved_failures` is zero.
5. The JSONL and Parquet files both load successfully.
6. Oversized records are filtered or retained according to the downstream
   trainer's real context limit.

## Known limitations

- The base-meter label does not prove a specific metrical form.
- Question quality and the semantic quality verdict come from the same Gemma
  model, so correlated mistakes remain possible.
- Chunk summaries can lose global detail for QCM questions on extremely long
  poems, although the summaries cover all chunks so that the question still
  reflects the poem as a whole.
- The source dataset's title and theme metadata are too sparse to serve as the
  primary semantic supervision.
- Since the QCM is generated by the same model that validates it, a correlated
  misunderstanding of the poem could pass both stages. Manual review of a
  stratified sample remains important, especially across rare meters, prose,
  repaired generations, metadata conflicts, and oversized poems.

## Licensing and distribution

The upstream Ashaar dataset is published for research and development under a
fair-use, non-commercial restriction. The generated dataset retains source URLs
for provenance, but anyone distributing or training on it must review the
upstream dataset card and applicable rights. The pipeline does not change the
licensing status of the source poems.