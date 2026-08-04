# Building the Ashaar instruction-following SFT dataset

## Purpose

This document describes the pipeline implemented by
[`generate_sft.py`](../generate_sft.py) and
[`sft_templates.py`](../sft_templates.py). The pipeline converts the poems in
`data/ashaar_classic_moroccan.parquet` into supervised fine-tuning examples
containing:

1. A long Arabic instruction describing the poem to be written.
2. A long Arabic editorial reasoning section.
3. The exact source poem as the final answer.

The generator uses the configured chat-completions endpoint and model to infer
the subject, meaning progression, imagery, emotional atmosphere, diction, rhyme,
and other constraints from each source poem. It does not ask Gemma to reproduce
the final target. The Python code appends the original poem itself so that model
copying errors cannot corrupt the SFT answer.

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
prompts therefore name the base meter but explicitly forbid unsupported claims
such as `تام` or `مجزوء`. They also avoid asserting a particular sequence of
feet when that form is not verified.

Records labeled `النثر` use a separate prompt route. Their instructions request
internal cadence, parallel syntax, and sound patterning without claiming a
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

## Few-shot template architecture

### Why meta-templates are used

A small collection of fixed surface instructions would make the SFT corpus
formulaic. Instead, the implementation uses six *meta-template families*. Each
family tells Gemma how to study the reference and which aspect to foreground,
while still requiring every generated instruction to cover all essential
poetic constraints.

The families are:

| Template ID | Primary emphasis |
| --- | --- |
| `prosody_rhyme` | Meter, rhyme, recitation, and sound |
| `semantic_arc` | Meaning progression and poem-level unity |
| `imagery_rhetoric` | Imagery, metaphor, comparison, and rhetorical relations |
| `emotion_voice` | Emotional register, speaker, and changes in tone |
| `occasion_addressee` | Occasion, addressee, and communicative purpose |
| `diction_revision` | Lexicon, syntax, draft alternatives, and revision choices |

The template is selected deterministically from the poem hash. The distribution
over the current corpus is close to balanced. `prosody_rhyme` is not eligible
for prose records.

### True multi-message few-shot prompting

Every final generation request uses this chat sequence:

```text
system:    generation rules, JSON contract, and family emphasis
user:      demonstration source poem 1
assistant: demonstration JSON 1
user:      demonstration source poem 2
assistant: demonstration JSON 2
user:      the actual source poem or the oversized-poem analysis notes
```

This is few-shot prompting at the chat-message level, rather than embedding an
unstructured example paragraph in one user message. The assistant
demonstrations are valid JSON objects with the same fields expected from the
real request:

```json
{
  "instruction": "A detailed Arabic instruction",
  "reasoning": "A detailed Arabic editorial analysis"
}
```

Metered templates use two compact metered examples, one on `الطويل` and one on
`الخفيف`. Prose templates use two separate prose examples. The first example
teaches the shared contract. The second example is augmented dynamically with
an instruction and reasoning passage specific to the selected family, so the
few-shot output itself—not only the system message—demonstrates the desired
emphasis. The examples teach:

- The reverse-construction task.
- Detailed thematic and rhetorical requirements.
- Explicit numeric length requirements.
- Appropriate prosodic language.
- Drafting and revision discussion.
- The exact JSON response contract.
- Ending the reasoning with `النتيجة النهائية:` while omitting the consolidated
  final poem.

The examples are intentionally compact. Repeating a very large demonstration
in all 50,193 calls would add substantial context cost and could distract Gemma
from the actual poem.

### Actual instruction requirements

The system and final user messages tell Gemma that every generated instruction
must cover:

- The exact number of couplets, written in digits.
- The base meter or an explicit prose-poetry requirement.
- The main subject and semantic progression.
- Secondary meanings that should appear.
- Emotional atmosphere and the speaker's voice.
- Imagery and rhetorical devices supported by the source.
- Classical Arabic diction and syntactic expectations.
- Rhyme behavior that can be inferred from the source.
- General recitation and prosodic guidance where applicable.
- The expected response structure.

The instruction may mention isolated words useful for a rhyme or image, but it
must not copy a complete source hemistich. It must not name the original poet,
title, or URL.

## Editorial reasoning and exact final targets

Gemma generates a synthetic editorial rationale in Arabic. The prompt asks it
to discuss:

- The principal meanings.
- The relation between the couplets.
- Selected images and rhetorical techniques.
- Initial wording alternatives.
- Meter and rhyme observations for metered poems.
- Reasons for changing or retaining expressions.

This is intended as an auditable, chain-of-thought-style editorial explanation;
it should not be represented as access to a model's private hidden reasoning.

Gemma is instructed to finish with:

```text
النتيجة النهائية:
```

Before composing the final response, the program removes any complete source
poem, couplet, or standalone hemistich that Gemma may have echoed inside its
reasoning. It also removes misplaced result markers and standalone final-poem
headers. It then reconstructs the response in one invariant order: editorial
reasoning, exactly one result marker, and the exact source poem at the end.
Every pair of hemistichs is formatted as:

```text
صدر البيت = عجز البيت
```

This construction guarantees that the final poem is sourced from the input and
is not rewritten, normalized, re-diacritized, or hallucinated by the endpoint.

## Oversized-poem processing

The direct source limit defaults to 24,000 characters. A longer poem is split
at complete-couplet boundaries into chunks of at most 12,000 characters.

For each chunk, the endpoint receives a separate compact-analysis request. It
is asked to produce 300–600 Arabic characters covering only meanings, images,
tone, and visible rhyme. The ordered summaries are then supplied to the normal
few-shot generation conversation in place of the full source text.

The result remains one instruction/answer pair for the complete poem, and the
complete original poem is still appended to the response. Such rows receive:

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

Any credential pasted into chat, source code, shell history, or logs must be
revoked and replaced before generation.

### Full generation tracing

Pass `--trace` to print complete, non-interleaved audit blocks to the terminal
and append the same events as UTF-8 JSON lines to
`generation_trace.jsonl`. The option is intended primarily for smoke samples,
debugging, and audits because logging every few-shot prompt and completion for
the full corpus consumes substantial terminal and disk space.

The append-only trace assigns a new `run_id` to each invocation and records:

- Run settings, source fingerprint, checkpoint reuse count, all six
  meta-template definitions, and why meta-templates are used.
- Per-poem provenance, eligible families, selected family and focus, and the
  exact `sample_id[8:16]` modulo calculation behind the selection.
- Every request kind: oversized-poem chunk analysis, initial generation, and
  semantic repair.
- The full OpenAI-compatible request body, including every system, few-shot,
  and final user message, seed, and decoding settings.
- The complete decoded API response payload. This retains endpoint-provided
  fields such as `message.reasoning_content`, `finish_reason`, and `usage` when
  the server supplies them.
- Network-attempt counts, retry errors, and elapsed time.
- Raw response content, parsed JSON, validation errors, and whether a repair is
  required.
- Parsed instruction, Gemma's editorial `reasoning` value, post-processing
  counts, and the final assistant response after the exact source poem is
  appended.
- Final success, failure, template, and validation-status counts.

The request body is the complete prompt representation available to this
client. Any later conversion of those chat messages into Gemma's tokenized chat
template happens inside the serving endpoint and cannot be observed here.

The requested `reasoning` field is a synthetic editorial explanation emitted
in Gemma's ordinary response content. It should not be described as private or
otherwise inaccessible model reasoning. Gemma also does not generate the
consolidated final poem: the program removes any poem echoes from the editorial
reasoning and appends the trusted source poem. Consequently the trace labels
the raw model content and final pipeline-composed response separately.

The request headers are never included. Before an event is printed or written,
the configured API key is recursively replaced with `[REDACTED]` anywhere it
might appear in request, response, or error text.

## Response validation and repair

The response parser first attempts to read the entire model response as JSON.
It also handles a JSON Markdown fence or explanatory text surrounding one JSON
object. A valid generation must satisfy all of the following:

- The object has exactly `instruction` and `reasoning`.
- Both values are strings.
- Both meet the configured minimum character count.
- The combined content is predominantly Arabic.
- The instruction includes the exact numeric couplet count.
- The instruction names the correct base meter, or requests prose poetry.
- It does not explicitly name a contradictory meter.
- It does not copy a complete source hemistich of meaningful length.
- The reasoning discusses meaning, imagery, and revision.
- Metered reasoning discusses meter, prosody, or rhythm.
- The reasoning includes the final-result transition.

If validation fails, the pipeline sends the original few-shot conversation,
the invalid response, and a list of concrete validation errors back to Gemma.
It permits two corrected responses after the initial generation. Network-level
retries are separate and apply to timeouts, HTTP 429, and HTTP 5xx failures.

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

On restart, the latest successful record for each sample ID is loaded and
skipped. Failed or missing IDs are submitted again. A later success supersedes
an earlier failure.

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
| `template_id` | Selected meta-template family |
| `instruction` | Generated Arabic user instruction |
| `response` | Generated reasoning plus exact source poem |
| `messages` | Two-message chat SFT representation |
| `sft_split` | Stable train, validation, or test assignment |
| `oversized_for_sft` | Whether chunk analysis was required |
| `metadata_conflict` | Whether duplicate sources disagreed on meter |
| `generation_attempts` | Initial generation plus semantic repairs |
| `validation_status` | `passed` or `passed_after_repair` |

`messages` contains:

```json
[
  {"role": "user", "content": "<instruction>"},
  {"role": "assistant", "content": "<reasoning and exact poem>"}
]
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
uv run python generate_sft.py `
  --input data/ashaar_classic_moroccan.parquet `
  --output-dir data/ashaar_sft_smoke `
  --limit 10 `
  --trace `
  --insecure
```

After manually reviewing the sample, run the corpus:

```powershell
uv run python generate_sft.py `
  --input data/ashaar_classic_moroccan.parquet `
  --output-dir data/ashaar_sft `
  --concurrency 4 `
  --insecure
```

Re-running the same command resumes from the checkpoint. See every available
override with:

```powershell
uv run python generate_sft.py --help
```

## Verification

The automated suite is offline and uses fake API clients:

```powershell
uv run python -m unittest discover -s tests -v
```

It covers:

- All six few-shot message layouts.
- The separate prose examples.
- Meter decoding and invalid IDs.
- Exact hemistich pairing.
- Stable hashes, template selection, and splits.
- Deduplication and majority conflict resolution.
- JSON and fenced-JSON extraction.
- Semantic validation and repair.
- Exact source-poem preservation.
- Oversized-poem summaries and synthesis.
- Checkpoint loading.
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
- Semantic and rhetorical descriptions are model-generated and can still be
  wrong despite deterministic validation.
- The validator detects structural contradictions, not full Arabic scansion.
- Chunk summaries lose some local detail in extremely long poems.
- Long reasoning increases storage and training-token cost substantially.
- Keeping all oversized poems in the master data does not make them trainable
  by every target model.
- The source dataset's title and theme metadata are too sparse to serve as the
  primary semantic supervision.

These limitations make manual review of a stratified sample important,
especially across rare meters, prose, repaired generations, metadata conflicts,
and oversized poems.

## Licensing and distribution

The upstream Ashaar dataset is published for research and development under a
fair-use, non-commercial restriction. The generated dataset retains source URLs
for provenance, but anyone distributing or training on it must review the
upstream dataset card and applicable rights. The pipeline does not change the
licensing status of the source poems.
