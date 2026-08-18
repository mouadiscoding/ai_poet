# Arabic poetry SFT generation

This project builds Arabic supervised fine-tuning datasets from
`data/ashaar_classic_moroccan.parquet`. A run selects one of four workflows:

- `poem-generation` reverse-constructs a writing instruction and editorial
  worklog, then appends the trusted source poem.
- `poem-completion` adds a reproducibly selected complete-couplet beginning to
  the same detailed writing instruction. Its editorial worklog covers only the
  missing couplets; Gemma does not generate the consolidated final poem, and
  Python appends the trusted source.
- `mcq` applies meter, theme, and title question templates to each poem. Each
  template has multiple uniformly selected question phrasings; the stored
  metadata is supplied as the exact ground-truth answer, and a template is
  skipped when its metadata is absent.
- `poem-reconstruction` creates localized corruptions, explains their repair,
  and lets Python append the trusted original poem.

The poem-generation workflow randomly chooses one of six concrete prompt
templates for each poem. Every template and every few-shot example covers
prosody or internal rhythm, semantic progression, imagery, voice, occasion,
and diction/revision.
Metered and prose poems have separate demonstrations.

See [the complete SFT generation guide](docs/sft_dataset_generation.md) for the
corpus audit, prompt architecture, validation contract, output schema, and
operational details.

Production code uses a `src` layout. Synthetic generation lives in
`src/ai_poet/synthetic_data`, with separate modules for configuration, source
corpus handling, API transport, prompts, validation, generation, persistence,
and orchestration. Tests mirror those boundaries under `tests/unit` and
`tests/integration`. Future workflows such as fine-tuning can be added as
sibling packages under `src/ai_poet` without coupling them to data generation.

## Configure API credentials

Never put an API key in the repository or command line. Revoke any key that has
been pasted into chat, shell history, or logs. Create a local configuration
from the committed example, then keep only the variables for the endpoint mode
you intend to use:

```powershell
Copy-Item .env_example .env
```

### One API endpoint

Use only the unindexed variables:

```dotenv
GEMMA_ENDPOINT=https://host.example/v1/chat/completions
GEMMA_MODEL=your-model-name
GEMMA_API_KEY=replace-with-endpoint-token
```

Remove `GEMMA_ENDPOINT_1..3`, `GEMMA_MODEL_1..3`, and
`GEMMA_API_KEY_1..3` from `.env`; indexed and unindexed endpoint settings
cannot be mixed.

### Three API endpoints

Use exactly the indexed endpoint records 1 through 3:

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

Remove `GEMMA_ENDPOINT` and `GEMMA_API_KEY`. If all three deployments use the
same served-model name, one shared `GEMMA_MODEL` may replace
`GEMMA_MODEL_1..3`; do not use both model forms.

The underlying CLI verifies TLS certificates by default. The Just recipes in
this guide pass `--insecure`, so use them only with internal endpoints you
trust. Process-environment values override matching `.env` values.

## Run commands with Just

Install [`just`](https://just.systems/man/en/packages.html) once with the `uv`
already used by this project:

```powershell
uv tool install rust-just
```

Run `just` in the repository root to list every recipe. Recipes accept one of
these task names and default to `poem-generation` when it is omitted:

- `poem-generation`
- `poem-completion`
- `mcq`
- `poem-reconstruction`

The recipes keep each task's output, capacity, and pilot paths together because
artifacts cannot be reused across tasks. The examples below use
`poem-generation`; replace it with another task name when needed. Follow
exactly one endpoint-mode sequence below.

| Endpoint mode | Run benchmark? | Run pilot? | Generation gate arguments |
| --- | --- | --- | --- |
| One endpoint | No | No | `--concurrency N` |
| Three endpoints, standard safeguards | Yes | Yes | `--capacity-report`, `--pilot-report`, and `--pilot-review` |
| Three endpoints, safeguards skipped | No | No | `--skip-pilot-review` |

### Using one API endpoint

After configuring the unindexed `.env` variables, run generation directly.
Do not pass capacity or pilot reports, and do not pass `--skip-pilot-review`:

```powershell
just generate-single poem-generation 32
```

Choose `--concurrency` within the single endpoint's tested capacity.

### Using three API endpoints with benchmark and pilot reports

This is the safeguarded production sequence. Do not add
`--skip-pilot-review` to any command.

1. Benchmark the three endpoints. The resumable benchmark tests each endpoint
   separately and all three together, then writes
   `endpoint_capacity.json` when its certification gates pass.

   ```powershell
   just benchmark poem-generation
   ```

2. Run the deterministic pilot using that capacity report. Accepted pilot
   records are written to the final output checkpoint and reused later.

   ```powershell
   just pilot poem-generation
   ```

3. Open the task output directory's `pilot_review.json`, inspect the 30 selected
   records, set each accepted record's `approved` value to `true`, and add notes
   where useful. The pilot must have reached at least 98% success, no more than
   25% repaired records, and zero truncations.

4. Start or resume full generation with all three artifacts.

   ```powershell
   just generate poem-generation
   ```

The benchmark defaults to a 30-second warmup and a five-minute measurement at
each concurrency level. Shorter values are useful for development but are not
representative production evidence. Full-run concurrency comes from the
certified capacity report; `--concurrency` does not control the three-endpoint
pool.

### Using three API endpoints with `--skip-pilot-review`

Use this route only when you deliberately want to bypass the benchmark
capacity report, automated pilot report, and human pilot review. Despite its
name, `--skip-pilot-review` skips all three gates, so neither the benchmark nor
pilot command is needed:

```powershell
just generate-unsafe poem-generation
```

This command prints a warning and uses each endpoint's
`GEMMA_MAX_CONCURRENCY_N` value. Omit `--capacity-report`, `--pilot-report`,
and `--pilot-review`; also omit `--concurrency`, which does not control the
three-endpoint pool.

For a small inspection run, append additional CLI arguments to the recipe:

```powershell
just generate poem-generation --limit 10 --trace
```

MCQ and reconstruction always send the complete selected poem and fail before
contacting Gemma if it exceeds `--max-source-chars`. For MCQ, `--limit` counts
source poems rather than output records; each selected poem produces a meter
record and, when available, theme and title records.

## Resume and failure handling

Accepted stages are appended immediately to the version-3
`generation_checkpoint.jsonl`, followed by the final sample event. Legacy
version-1 and version-2 poem-generation checkpoints remain readable.
Re-running skips compatible accepted stages as well as successful
template-version-8 samples. Transient failures retry across healthy endpoints;
429s, timeouts, latency pressure, and repeated server failures reduce only the
affected endpoint's capacity. The run stops for a connection outage only when
all endpoints exhaust the shared retry budget. Responses that cannot be
parsed receive phase-specific repair prompts. Python first validates the
instruction layout and then requires exactly
one structured work block per source couplet. Every accepted revision is
canonicalized back to its source couplet after Unicode normalization and
harmless spacing around `=` are checked, while its first draft must differ, and
generation metatext is rejected. Gemma then performs a separate semantic review
of the instruction and the editorial content of each bounded reasoning chunk.
Exact scansion remains outside that same-model semantic gate; the scansion
fields are retained in the response but require purpose-built or expert review.
Pre-draft imagery is required to use planning language, while the later
revision decision must identify a real defect in the first draft and the
concrete change that leads to the accepted verse. That decision must cite
wording present in both drafts. Malformed semantic-verdict JSON is retried up to
three times without consuming a candidate-repair attempt.

Each successful record is also appended and flushed immediately to
`ashaar_sft.jsonl`, so the training data can be inspected while generation is
still running. The file is rewritten in source order when the run finishes.

The run returns a non-zero exit status while any selected poem remains
unresolved. `failures.jsonl` is updated during the run and records a stable
failure category with each error, without storing request headers or credentials.

By default, poems above 24 couplets are excluded before `--limit` is applied;
override this with `--max-couplets`. For the current corpus this selects 45,161
poems and excludes 5,032. Eligible work is scheduled shortest-first so useful
records and acceptance-rate evidence appear early. If the cap is raised enough
to admit a poem longer than 24,000 characters, it is analyzed in
couplet-aligned chunks for the global instruction. Three-couplet reasoning
chunks are assembled in source order and checkpointed independently.

## Outputs

The output directory contains:

- `ashaar_sft.jsonl`: trainer-friendly records.
- `ashaar_sft.parquet`: the same records in columnar form.
- `generation_checkpoint.jsonl`: append-only resume state.
- `generation_trace.jsonl`: optional append-only prompt/response audit created
  by `--trace`.
- `generation_metrics.jsonl`: lightweight 60-second endpoint-pool snapshots.
- `pilot_report.json` and `pilot_review.json`: strict full-run gate artifacts.
- `failures.jsonl`: currently unresolved samples.
- `manifest.json`: generation settings and aggregate counts.

Each training record includes `task_type`, `task_version`, a task-qualified
`record_id`, the source hash and provenance, endpoint IDs, network-attempt and
failover counts, meter ID and name, couplet count, generated `instruction`,
composed `response`, OpenAI-style `messages`, deterministic `sft_split`, and
quality flags. Poem-generation records retain their concrete template fields.
Completion records also add `poem_beginning`, `provided_couplet_count`, and
`remaining_couplet_count`; one-couplet poems are skipped because no complete
couplet remains to generate. MCQ records add the metadata field, selected prompt
ID, trusted ground-truth answer, question, choices, and correct label;
reconstruction records add the corrupted poem and corruption count. Exact
duplicate poem texts share one canonical source poem and retain all source row
indices and URLs. MCQ can emit up to three records from that canonical poem.
Splits are assigned from the poem hash using 98% train, 1% validation, and 1%
test buckets.

Full prompts and raw model responses are deliberately excluded from
`ashaar_sft.jsonl` and `ashaar_sft.parquet`. Keeping them in the separate trace
prevents accidental training on rejected attempts or few-shot demonstrations
and avoids inflating every trainer-facing row. Each trace run has a unique
`run_id`, which is also recorded in the manifest. Request headers and API keys
are never traced; configured secrets are recursively redacted from event data.

The assistant message contains synthetic editorial reasoning followed by the
source poem formatted as:

```text
مرحلة التفكير والتحرير:

<خطة القصيدة>

البيت 1:
<المعنى، المسودة، سبب المراجعة، الصياغة المنقحة، والفحص العروضي والقافية>

النتيجة النهائية:

صدر البيت = عجز البيت
```

Because the pipeline starts from an existing poem, the worklog is a synthetic
editorial reconstruction, not a claim about the historical poet's private
thoughts.

For poem completion, the beginning ends at a complete-couplet boundary selected
uniformly by a local RNG seeded from the task version and poem hash. The same
poem therefore receives the same beginning across retries and reruns. Gemma
returns only the validated instruction and structured editorial work for the
new couplets, using the final provided couplet as context. Python owns the
single exact full-poem section at the end of the assistant response.

## Tests

The test suite uses fake API clients and does not require a credential or
network connection:

```powershell
just test
```

## Data-use notice

The upstream Ashaar dataset is released for research and development under a
fair-use, non-commercial restriction. Preserve the source URLs in derived data
and review the upstream dataset card before distributing or using the generated
dataset.
