# Building the Ashaar instruction-following SFT dataset

## Purpose

This document describes the task-oriented pipeline implemented by the
[`ai_poet.synthetic_data`](../src/ai_poet/synthetic_data) package. The default
`poem-generation` workflow converts the poems in
`data/ashaar_classic_moroccan.parquet` into supervised fine-tuning examples
containing:

1. A long Arabic instruction describing the poem to be written.
2. A rendered Arabic editorial worklog with one structured block per couplet.
3. The exact source poem as the final answer.

The generator uses the configured chat-completions endpoint and model to infer
the subject, meaning progression, imagery, emotional atmosphere, diction, rhyme,
and other constraints from each source poem. It does not ask Gemma to reproduce
the final target. The Python code appends the original poem itself so that model
copying errors cannot corrupt the SFT answer.

Three additional workflows use the same corpus, endpoint pool, retries, tracing,
checkpointing, splits, and exporters:

- `poem-completion` augments the poem-generation instruction with a complete-
  couplet prefix chosen uniformly by a local RNG seeded from the task version
  and poem hash. It retains whole-poem editorial reasoning. Gemma never emits
  the consolidated final poem; Python appends the exact complete source poem.
  Poems with only one couplet are skipped because no complete-couplet suffix
  would remain.
- `mcq` applies templates for `poem_meter`, `poem_theme`, and `poem_title`.
  Each template contains multiple Arabic question phrasings, one of which is
  chosen uniformly with a deterministic per-poem seed. The stored metadata
  value is supplied to Gemma as ground truth and enforced by Python as the
  exact correct answer. Gemma generates three distractors and an explicit
  assessment of every option; Python shuffles and labels the choices
  reproducibly. Templates with absent or blank metadata are skipped.
- `poem-reconstruction` asks Gemma for a complete structurally preserved
  corrupted poem and detailed local repair records. Python validates every
  one-to-three-word replacement, canonicalizes unambiguous zero-based indices
  and diacritic-only fragment variants from the exact local diff, and appends
  the exact source poem; Gemma never emits the complete corrected poem.

All three commands accept `--task poem-generation`, `--task poem-completion`,
`--task mcq`, or `--task poem-reconstruction`. The first remains the default.
Each task has its own contract version and fingerprints, so capacity reports,
pilot artifacts, checkpoints, and output directories cannot be mixed.

## Source corpus findings

The current input Parquet contains 50,199 rows. Inspection of the actual file
found:

| Property | Value |
| --- | ---: |
| Source rows | 50,199 |
| Unique poem texts | 50,193 |
| Duplicate source rows beyond the unique texts | 6 |
| Distinct poem-text groups with conflicting meter metadata | 1 |
| Poems at or below the default 24-couplet SFT cap | 45,161 |
| Poems excluded by the default couplet cap | 5,032 |
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

Consequently, poem-generation must infer most semantic constraints from the
poem itself. MCQ title and theme templates instead run only where those trusted
labels exist.

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

### Why concrete templates are used

A single fixed surface instruction would make the SFT corpus formulaic. The
implementation therefore provides six independently authored, complete Arabic
templates. Each uses the same placeholders for meter, count, minimum length,
form guidance, meter explanation, composition-plan heading, and source poem.
Each also carries the full task, grounding rules, six poetic dimensions,
required instruction layout, and output contract, while changing the workflow
and structure used to express them.

`METER_DEFINITIONS` covers all 16 classical meters and the corpus's prose
category. Each entry provides a definition, base full-verse weight, and an
approximate long/short sound pattern. These data are injected into the selected
template; prose receives an explicit non-applicable weight and internal-rhythm
guidance. The base patterns follow Ahmad al-Hashimi's *Mizan al-Dhahab fi
Sina'at Shi'r al-Arab*, while valid zihaf and `illa` variants remain allowed.

The templates are:

| Template ID | Primary emphasis |
| --- | --- |
| `prosody_rhyme` | Meter, rhyme, recitation, and sound |
| `semantic_arc` | Meaning progression and poem-level unity |
| `imagery_rhetoric` | Imagery, metaphor, comparison, and rhetorical relations |
| `emotion_voice` | Emotional register, speaker, and changes in tone |
| `occasion_addressee` | Occasion, addressee, and communicative purpose |
| `diction_revision` | Lexicon, syntax, draft alternatives, and revision choices |

One template is selected uniformly without a seed at the start of each poem
generation call and is retained for all repairs in that call. A later rerun of
an unresolved poem may choose another template. Every template is eligible for
prose; its form guidance replaces meter and scansion with internal rhythm,
parallelism, sound, and observable rhyme.

### Two separate few-shot stages

Instruction generation uses the existing six-message chat sequence: a system
policy, two source/JSON demonstrations, and the selected concrete template. Its
assistant demonstrations and actual result now contain only:

```json
{"instruction": "A detailed Arabic instruction"}
```

Editorial-work generation is a separate conversation after the instruction has
passed validation. It uses a multi-verse metered demonstration or a multi-unit
prose demonstration and requests structured, source-linked work for at most
three target couplets. This separation prevents the model from narrating how it
created the `instruction` when it should be acting as the poet.

### Actual instruction requirements

The system and final user messages tell Gemma that every generated instruction
must use these sections in order:

1. `الموضوع العام:`
2. `الجو العاطفي المطلوب:`
3. `ألفاظ وصور يُستحسن استعمالها أو الدوران حولها:`
4. `القافية:`
5. `شرح البحر المطلوب:` including `وزنه في كل بيت كامل:`
6. `الصورة الصوتية التقريبية:`
7. A meter-specific practical composition plan; prose receives a prose plan.

Across those sections it must cover:

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

## Editorial work and exact final targets

The second stage returns an `overview` for the first chunk and one
`verse_reasoning` object per couplet. Every object contains the intended
meaning, its link to the preceding context, imagery and diction, a genuine
first draft, a specific reason for rejecting that draft, the exact accepted
couplet, separate sound checks for both hemistichs, and a rhyme check.

This is an explicit synthetic editorial reconstruction. Because the pipeline
starts from an existing poem, it is not represented as access to the historical
poet's private thoughts.

The renderer preserves accepted couplets quoted inside these work blocks. It
removes only a leading or explicitly labeled accidental full-poem dump and
anything after an accidental result marker, then appends exactly one canonical
marker and the exact source poem. The endpoint therefore cannot rewrite, normalize,
re-diacritize, or hallucinate the final target.

For poem completion, the validated instruction is augmented deterministically
with `بداية القصيدة:` and `مهمة الإكمال:` sections. These contain the exact
prefix plus the total, provided, and remaining couplet counts. Reasoning still
covers every couplet, including the supplied beginning. The endpoint is asked
only for structured reasoning chunks; the same Python renderer appends the
trusted complete source poem.

## Poem-size controls

The SFT generator excludes poems above `--max-couplets` before applying
`--limit`. Its default of 24 bounds one training record to at most eight
three-couplet reasoning chunks. Raising the cap is an explicit choice to target
a larger downstream training context.

For an admitted poem, the direct source limit defaults to 24,000 characters.
Only poem generation can split a longer poem at complete-couplet boundaries
into chunks of at most 12,000 characters. Completion, MCQ, and reconstruction
require the complete selected poem to fit the direct source limit.

For each source-size chunk, the endpoint receives a compact-analysis request.
The ordered summaries are used only by the global instruction stage. Editorial
work is always generated separately in groups of at most three exact couplets,
regardless of whether the source exceeded the direct-source limit.

The result remains one instruction/answer pair for the complete poem, and the
complete original poem is still appended to the response. Such rows receive:

```text
oversized_for_sft = true
```

This flag remains useful when the couplet cap is raised: admitting a poem does
not by itself prove the finished record fits a downstream model's context.

### Authentication and TLS

Production pooling loads exactly three indexed records:
`GEMMA_ENDPOINT_1..3`, `GEMMA_API_KEY_1..3`, and the corresponding served-model
aliases in `GEMMA_MODEL_1..3`. If every deployment exposes the same model alias,
one shared `GEMMA_MODEL` may replace the three indexed model variables; the two
forms cannot be mixed. Optional administrative ceilings are configured through
`GEMMA_MAX_CONCURRENCY_1..3`. Optional
`GEMMA_METRICS_URL_1..3` values enable vLLM telemetry in capacity reports.
Indexed and legacy variables may not be mixed. The legacy `GEMMA_ENDPOINT` and
`GEMMA_API_KEY` pair remains supported for a single endpoint. Existing
process-environment values take precedence over `.env`. Every configured key
is scrubbed from checkpoints, output records, manifests, traces, and errors.

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

- Run settings, source fingerprint, checkpoint reuse count, template version,
  the six concrete prompts, and the shared focus contract.
- Per-poem provenance, all eligible template IDs, the fresh random selection,
  and why the selected template is retained through repairs.
- Every request kind: oversized-poem analysis, instruction generation and
  repair, verse-chunk generation and repair, and both quality-review phases.
- The full OpenAI-compatible request body, including every system, few-shot,
  and final user message, seed, and decoding settings.
- The complete decoded API response payload. This retains endpoint-provided
  fields such as `message.reasoning_content`, `finish_reason`, and `usage` when
  the server supplies them.
- Network-attempt counts, retry errors, and elapsed time.
- Raw generation content, parsed structures, deterministic contract errors,
  raw and parsed Gemma verdicts, and whether a phase-specific repair is needed.
- Parsed instruction, structured verse-work blocks, rendered editorial text,
  and the final assistant response after the exact source poem is appended.
- Final success, failure, template, and validation-status counts.

The request body is the complete prompt representation available to this
client. Any later conversion of those chat messages into Gemma's tokenized chat
template happens inside the serving endpoint and cannot be observed here.

The verse-work fields are ordinary structured response content, not private
server reasoning. Gemma also does not generate the consolidated final poem;
the program appends the trusted source poem after rendering the validated work.

The request headers are never included. Before an event is printed or written,
the configured API key is recursively replaced with `[REDACTED]` anywhere it
might appear in request, response, or error text.

## Response validation and repair

Python performs deterministic validation before either semantic review. For
instructions it checks the sole-field schema, minimum length, exact heading
names and order, numeric count, meter name, and absence of complete source
hemistichs. Missing-heading repair feedback names every required heading that
was absent. For every
reasoning chunk it checks the exact schema and block count, contiguous indices,
non-empty fields, exact source equality of every `revised_draft`, a distinct
`first_draft`, prospective rather than retrospective wording before that draft,
and absence of dataset-generation metatext. The semantic review then checks
that the stated defect really belongs to the first draft and that the proposed
change leads coherently to the accepted verse.

Gemma then reviews the instruction or reasoning chunk at temperature zero for
semantic and rhetorical quality. The instruction judge accepts the fixed seven
heading contract and may not invent additional headings. The reasoning judge
does not receive the two scansion fields, so exact scansion is not a same-model
hard rejection gate; Python still requires those fields to be present and
substantive. Each review returns exactly:

```json
{"passed": true, "errors": []}
```

Instruction failures repair only the instruction before verse generation
begins. A failed reasoning chunk is repaired independently, so accepted chunks
are not regenerated. Malformed verdicts are retried up to three times and
remain separate from candidate repairs and network retries.

## Checkpoint and resume semantics

`generation_checkpoint.jsonl` is a versioned append-only event log. It records
accepted stages immediately:

```json
{"checkpoint_version": 2, "event": "instruction_success", "sample_id": "..."}
```

```json
{"checkpoint_version": 2, "event": "reasoning_chunk_success", "sample_id": "...", "start_offset": 0}
```

Final `sample_success` and `sample_failure` events complete the workflow. On
restart, compatible accepted instructions and chunks are reused; missing stages
alone are submitted again. Compatibility is enforced through the model,
endpoint, prompt-template, and generation-settings fingerprint. Legacy
success/failure events remain readable, and a successful record is reused only
when its `template_version` is 8.

All checkpoint appends pass through a locked writer and are flushed, so
concurrent chunk completions cannot interleave JSON lines. Each successful SFT
record is appended and flushed to `ashaar_sft.jsonl`; the finished file is
rewritten in canonical source order.

Poems above `--max-couplets` are excluded before `--limit` is applied. The
default is 24 couplets; the manifest records both the configured cap and the
number excluded. Eligible poems are scheduled shortest-first. When `--limit`
is used, checkpoint entries outside the selected prefix are ignored for that
run. The command returns exit code 1 while any selected sample
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
| `record_id` | Task-qualified record identifier; MCQ also appends its metadata field |
| `task_type` | `poem-generation`, `poem-completion`, `mcq`, or `poem-reconstruction` |
| `task_version` | Version of the selected task contract |
| `source_row_indices` | All original rows represented by the record |
| `source_urls` | All retained source URLs |
| `poet_name` | Canonical source poet name for provenance |
| `poem_title` | Canonical title when available |
| `poem_theme` | Canonical theme when available |
| `meter_id` | Original numeric base-meter class |
| `meter_name` | Decoded Arabic base-meter name |
| `couplet_count` | Number of hemistich pairs |
| `poem_beginning` | Exact seeded complete-couplet prefix for poem completion |
| `provided_couplet_count` | Number of couplets supplied in that prefix |
| `remaining_couplet_count` | Number of couplets left to complete |
| `template_id` | Selected concrete prompt template |
| `template_version` | Prompt and validation contract version; currently `8` |
| `instruction` | Generated Arabic user instruction |
| `response` | Generated reasoning plus exact source poem |
| `messages` | Two-message chat SFT representation |
| `sft_split` | Stable train, validation, or test assignment |
| `oversized_for_sft` | Whether chunk analysis was required |
| `metadata_conflict` | Whether duplicate sources disagreed on meter |
| `generation_attempts` | Total instruction and reasoning generation attempts |
| `instruction_generation_attempts` | Instruction generation attempts including repairs |
| `reasoning_generation_attempts` | Sum of generation attempts across reasoning chunks |
| `reasoning_chunk_count` | Number of bounded verse-work chunks |
| `validation_status` | Whether both phases passed directly or after any repair |
| `generation_endpoint_ids` | Endpoint IDs that served requests for this sample |
| `endpoint_failover_count` | Cross-endpoint failovers used by successful logical requests |
| `network_attempts` | Total HTTP attempts associated with the sample |
| `truncated_completions` | Responses whose endpoint finish reason was `length` |

Poem generation and completion retain `template_id`, `template_version`, and
their instruction/reasoning attempt counters. The three prefix fields above
appear only on completion records. MCQ adds `metadata_field`, `prompt_id`,
`ground_truth_answer`, `question`, four labeled `choices`, and
`correct_choice_label`. Its record ID is
`mcq:<sample_id>:<metadata_field>`, allowing each applicable template to be
checkpointed independently. Reconstruction adds `corrupted_poem` and the
reproducible `corruption_count`.

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
- `generation_metrics.jsonl` for lightweight 60-second pool telemetry
- `pilot_report.json` and `pilot_review.json` for the strict production gate
- `failures.jsonl`, updated live with `sample_id`, failure `category`, and error
- `manifest.json`

The manifest records the selected task and refuses cross-task directory reuse.
The checkpoint format is version 3 and stores generic accepted-stage events;
legacy version-1 and version-2 poem-generation events are replayed into the
same stage view.

The manifest records completion status, source SHA-256, template version, model
and endpoint, decoding and length settings, output counts, split distribution,
template distribution, couplet-cap selection counts, oversized count, and
metadata-conflict count.

The trace file is intentionally a sidecar rather than part of the training
schema. Complete prompts repeat the few-shot demonstrations, rejected outputs
are not training targets, and both substantially increase storage. Therefore
`ashaar_sft.jsonl` and `ashaar_sft.parquet` remain identical and
trainer-focused; audit events are joined by `sample_id` when an investigation
requires them. The manifest's `trace` object reports whether tracing was
enabled, its `run_id`, and the sidecar filename.

## Running the pipeline

Create a local `.env` from the committed example, then set the model and three
endpoint records:

```powershell
Copy-Item .env_example .env
```

```dotenv
GEMMA_ENDPOINT_1=https://host-1.example/v1/chat/completions
GEMMA_MODEL_1=your-endpoint-1-model-name
GEMMA_API_KEY_1=replace-with-endpoint-1-token
GEMMA_MAX_CONCURRENCY_1=32
GEMMA_ENDPOINT_2=https://host-2.example/v1/chat/completions
GEMMA_MODEL_2=your-endpoint-2-model-name
GEMMA_API_KEY_2=replace-with-endpoint-2-token
GEMMA_MAX_CONCURRENCY_2=32
GEMMA_ENDPOINT_3=https://host-3.example/v1/chat/completions
GEMMA_MODEL_3=your-endpoint-3-model-name
GEMMA_API_KEY_3=replace-with-endpoint-3-token
GEMMA_MAX_CONCURRENCY_3=32
```

### Running with one endpoint

For a single-endpoint run, replace the indexed endpoint settings above with
the legacy, unindexed variables:

```dotenv
GEMMA_ENDPOINT=https://your-host.example/v1/chat/completions
GEMMA_MODEL=your-model-name
GEMMA_API_KEY=replace-with-endpoint-token
```

Do not leave any `GEMMA_ENDPOINT_1..3`, `GEMMA_MODEL_1..3`,
`GEMMA_API_KEY_1..3`, `GEMMA_MAX_CONCURRENCY_1..3`, or
`GEMMA_METRICS_URL_1..3` variables in `.env` or the process environment;
indexed and unindexed settings cannot be mixed. Then run the generator without
capacity or pilot artifacts:

```powershell
uv run ai-poet-generate-sft `
  --input data/ashaar_classic_moroccan.parquet `
  --output-dir data/ashaar_sft_single `
  --concurrency 32 `
  --max-couplets 24 `
  --insecure
```

In single-endpoint mode, `--concurrency` sets the number of concurrent sample
workers; choose a value within that endpoint's safe capacity. Omit `--insecure`
when the endpoint presents a certificate trusted by the local system.

Certify the endpoint capacities:

```powershell
uv run ai-poet-benchmark-endpoints `
  --input data/ashaar_classic_moroccan.parquet `
  --output-dir data/gemma_capacity `
  --insecure
```

Run the strict 300-poem pilot:

```powershell
uv run ai-poet-pilot-sft `
  --input data/ashaar_classic_moroccan.parquet `
  --output-dir data/ashaar_sft `
  --capacity-report data/gemma_capacity/endpoint_capacity.json `
  --insecure
```

After the automatic gate passes, inspect all records named by
`pilot_review.json` and set their `approved` values to `true`. Then run the
corpus:

```powershell
uv run ai-poet-generate-sft `
  --input data/ashaar_classic_moroccan.parquet `
  --output-dir data/ashaar_sft `
  --capacity-report data/gemma_capacity/endpoint_capacity.json `
  --pilot-report data/ashaar_sft/pilot_report.json `
  --pilot-review data/ashaar_sft/pilot_review.json `
  --max-couplets 24 `
  --insecure
```

Re-running the same command resumes from the checkpoint. See every available
override with:

```powershell
uv run ai-poet-generate-sft --help
```

### Running completion, MCQ, or reconstruction

Use the same task value for capacity certification, the pilot, and the final
run. For example, the MCQ production sequence is:

```powershell
uv run ai-poet-benchmark-endpoints --task mcq `
  --output-dir data/gemma_capacity_mcq
uv run ai-poet-pilot-sft --task mcq `
  --capacity-report data/gemma_capacity_mcq/endpoint_capacity.json
uv run ai-poet-generate-sft --task mcq `
  --capacity-report data/gemma_capacity_mcq/endpoint_capacity.json `
  --pilot-report data/ashaar_mcq_sft/pilot_report.json `
  --pilot-review data/ashaar_mcq_sft/pilot_review.json
```

Replace `mcq` with `poem-completion` or `poem-reconstruction` for those tasks.
Their default output directories are `data/ashaar_completion_sft`,
`data/ashaar_mcq_sft`, and `data/ashaar_reconstruction_sft`, respectively. All
three require the complete poem and fail preflight if any selected poem exceeds
`--max-source-chars`; only poem generation may replace an oversized source with
chunk summaries.

## Verification

The automated suite is offline and uses fake API clients:

```powershell
uv run python -m unittest discover -s tests -v
```

It covers:

- Rendering and fresh random selection of all six concrete templates.
- Comprehensive metered and prose few-shot examples.
- Meter decoding and invalid IDs.
- Exact hemistich pairing.
- Stable hashes and splits.
- Deduplication and majority conflict resolution.
- JSON and fenced-JSON extraction.
- Strict Gemma verdict parsing, rejection feedback, and repair.
- Exact source-poem preservation.
- Seeded complete-couplet completion prefixes, one-couplet filtering, full-
  poem reasoning, generic stage resume, and Python-owned exact final poems.
- Oversized-poem summaries and synthesis.
- Template-version-aware checkpoint reuse.
- JSONL and Parquet round trips.
- Complete MCQ metadata-template expansion, missing-metadata skips, uniform
  prompt selection, ground-truth enforcement, deterministic choice ordering,
  unique-answer validation, and exact final-answer formatting.
- Stable reconstruction severity, localized diff enforcement, reasoning
  coverage, index and Arabic-diacritic canonicalization, and byte-exact
  source-poem appending.
- Task-isolated fingerprints, manifests, pilots, and version-3 checkpoints.

Before a full production run is accepted, verify that:

1. A live smoke sample has been inspected manually.
2. The manifest reports `complete: true`.
3. `generated_poems` equals the selected eligible count (45,161 with the
   current corpus and default 24-couplet cap).
4. For MCQ, `generated_records` equals `target_records`.
5. `unresolved_failures` is zero.
6. The JSONL and Parquet files both load successfully.
7. Oversized records are filtered or retained according to the downstream
   trainer's real context limit.

## Known limitations

- The base-meter label does not prove a specific metrical form.
- Semantic and rhetorical descriptions and their quality verdicts come from
  the same Gemma model, so correlated mistakes remain possible.
- The pipeline does not certify exact Arabic scansion. The generated scansion
  fields still need expert or purpose-built prosody review, and semantic Gemma
  validation adds a request for every parseable instruction or reasoning chunk
  attempt.
- Chunk summaries can lose global detail in instructions for extremely long
  poems, although verse-work generation still receives each exact couplet.
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
