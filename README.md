# Arabic poetry SFT generation

This project builds Arabic supervised fine-tuning datasets from
`data/ashaar_classic_moroccan.parquet`. A run selects one of three workflows:

- `poem-generation` reverse-constructs a writing instruction and editorial
  worklog, then appends the trusted source poem.
- `mcq` creates one poem-grounded question with four choices, detailed answer
  analysis, and exactly one correct answer.
- `poem-reconstruction` creates localized corruptions, explains their repair,
  and lets Python append the trusted original poem.

The generator randomly chooses one of six concrete prompt templates for each
poem. Every template and every few-shot example covers prosody or internal
rhythm, semantic progression, imagery, voice, occasion, and diction/revision.
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

## Security first

Never put an API key in the repository or command line. Revoke any key that has
been pasted into chat, shell history, or logs, then create a local `.env` from
the committed example and fill in the model plus all three endpoint records:

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

The indexed configuration must contain exactly endpoints 1 through 3 and must
not be mixed with the legacy variables. Use `GEMMA_MODEL_1..3` when deployments
expose different served-model aliases. If all aliases are identical, one shared
`GEMMA_MODEL` may replace them; the two model forms cannot be mixed. A
single-endpoint run remains available through `GEMMA_ENDPOINT`,
`GEMMA_API_KEY`, and `GEMMA_MODEL`.

The client verifies TLS certificates by default. An internal endpoint may
require `--insecure`, which is equivalent to `curl -k`; use that flag only for
an endpoint you trust. Process-environment values override matching `.env`
values.

## Benchmark, pilot, and generate

First measure each endpoint in isolation and all three together. The command is
resumable and writes a certified `endpoint_capacity.json` only when the
throughput, error-rate, convergence, model, and combined-speedup gates pass:

```powershell
uv run ai-poet-benchmark-endpoints `
  --input data/ashaar_classic_moroccan.parquet `
  --output-dir data/gemma_capacity `
  --insecure
```

The default benchmark warms each level for 30 seconds and measures it for five
minutes. Use `--duration-per-level` and `--warmup-seconds` only for development;
a shortened benchmark is not representative production evidence.

Run the deterministic 300-poem pilot into the final output directory. Accepted
pilot records are written to the normal checkpoint and reused by the corpus
run:

```powershell
uv run ai-poet-pilot-sft `
  --input data/ashaar_classic_moroccan.parquet `
  --output-dir data/ashaar_sft `
  --capacity-report data/gemma_capacity/endpoint_capacity.json `
  --insecure
```

The pilot must reach 98% final success, at most 25% repaired samples, and zero
truncations. Inspect the 30 records listed in `pilot_review.json`, set every
accepted entry's `approved` value to `true`, and add notes where useful. Then
start the complete run:

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

To deliberately run without either pilot artifact, pass
`--skip-pilot-review`. This bypasses the capacity report, automated pilot
report, and human review gate and prints a yellow warning. Without a capacity
report, the run uses each endpoint's configured `GEMMA_MAX_CONCURRENCY_N`.

Request concurrency comes from the capacity report. `--concurrency` remains a
legacy single-endpoint option. Run any command with `--help` for its complete
set of controls.

### MCQ and reconstruction runs

`poem-generation` remains the default, so existing commands continue to work.
For another workflow, pass the same `--task` value to the benchmark, pilot, and
generation commands. Capacity and pilot artifacts are task-specific and cannot
be reused across workflows. When `--output-dir` is omitted, MCQ and
reconstruction default to `data/ashaar_mcq_sft` and
`data/ashaar_reconstruction_sft` respectively.

For example, a single-endpoint smoke run can skip the production gate:

```powershell
uv run ai-poet-generate-sft `
  --task mcq `
  --limit 10 `
  --skip-pilot-review `
  --trace
```

```powershell
uv run ai-poet-generate-sft `
  --task poem-reconstruction `
  --limit 10 `
  --skip-pilot-review `
  --trace
```

MCQ and reconstruction always send the complete selected poem. They fail
before contacting Gemma if a selected poem exceeds `--max-source-chars`; they
never substitute a summary for the required poem text.

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
quality flags. Poem-generation records retain their concrete template fields;
MCQ records add the question domain, choices, and correct label; reconstruction
records add the corrupted poem and corruption count. Exact
duplicate poem texts share one record and retain all source row indices and
URLs. Splits are assigned from the poem hash using 98% train, 1% validation,
and 1% test buckets.

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

## Tests

The test suite uses fake API clients and does not require a credential or
network connection:

```powershell
uv run python -m unittest discover -s tests -v
```

## Data-use notice

The upstream Ashaar dataset is released for research and development under a
fair-use, non-commercial restriction. Preserve the source URLs in derived data
and review the upstream dataset card before distributing or using the generated
dataset.
