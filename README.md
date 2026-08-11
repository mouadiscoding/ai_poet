# Arabic poetry SFT generation

This project builds a supervised fine-tuning dataset from
`data/ashaar_classic_moroccan.parquet`. For every unique poem it asks an
OpenAI-compatible Gemma endpoint to reverse-construct a detailed Arabic writing
instruction. In a separate stage, it generates a structured editorial worklog
for every couplet, including a rejected draft, revision reason, accepted line,
prosodic or rhythmic review, and rhyme review. The program then renders those
blocks and appends the original poem itself.

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
the committed example and fill in all three required values:

```powershell
Copy-Item .env_example .env
```

```dotenv
GEMMA_ENDPOINT=https://your-host.example/v1/chat/completions
GEMMA_MODEL=your-model-name
GEMMA_API_KEY=replace-with-a-new-token
```

The client verifies TLS certificates by default. An internal endpoint may
require `--insecure`, which is equivalent to `curl -k`; use that flag only for
an endpoint you trust. Process-environment values override matching `.env`
values.

## Generate a smoke sample

Start with a small run:

```powershell
uv run ai-poet-generate-sft `
  --input data/ashaar_classic_moroccan.parquet `
  --output-dir data/ashaar_sft_smoke `
  --limit 10 `
  --trace `
  --insecure
```

`--trace` prints a full audit block for each generation step and appends the
same structured events to `generation_trace.jsonl`. The trace shows the
concrete-template catalog and rationale, the fresh random selection, every
complete message array sent to Gemma, raw generation and Gemma-validation
payloads, repair results, and the final response after the source poem is
appended. Use it for smoke runs and audits; full-corpus traces are very large.

Inspect the generated instructions and reasoning before starting the complete
run:

```powershell
uv run ai-poet-generate-sft `
  --input data/ashaar_classic_moroccan.parquet `
  --output-dir data/ashaar_sft `
  --concurrency 4 `
  --insecure
```

The endpoint and model are required in `.env`; neither has a default embedded
in the code. Run
`uv run ai-poet-generate-sft --help` for generation, validation, retry, and
context-size controls.

## Resume and failure handling

Every completed request is appended immediately to
`generation_checkpoint.jsonl`. Re-running the same command skips successful
template-version-7 sample IDs and retries unresolved or legacy successes.
Transient HTTP failures are retried up to three times with exponential
backoff. Connection failures use the same backoff and stop the entire script if
Gemma is still unreachable after the third retry. Responses that cannot be
parsed receive phase-specific repair prompts. Python first validates the
instruction layout and then requires exactly
one structured work block per source couplet. Every accepted revision must
match its source couplet exactly, while its first draft must differ, and
generation metatext is rejected. Gemma then performs a separate semantic review
of the instruction and the editorial content of each bounded reasoning chunk.
Exact scansion remains outside that same-model semantic gate; the scansion
fields are retained in the response but require purpose-built or expert review.
Pre-draft imagery is required to use planning language, while the later
revision decision must identify a real defect in the first draft and the
concrete change that leads to the accepted verse.

Each successful record is also appended and flushed immediately to
`ashaar_sft.jsonl`, so the training data can be inspected while generation is
still running. The file is rewritten in source order when the run finishes.

The run returns a non-zero exit status while any selected poem remains
unresolved. `failures.jsonl` records those failures without storing request
headers or credentials.

Poems longer than 24,000 characters are analyzed in couplet-aligned chunks for
the global instruction. Editorial reasoning is always generated in bounded
three-couplet chunks, so long poems do not require one oversized completion. The
final SFT record still includes the complete source poem and marks oversized
source texts with `oversized_for_sft=true`.

## Outputs

The output directory contains:

- `ashaar_sft.jsonl`: trainer-friendly records.
- `ashaar_sft.parquet`: the same records in columnar form.
- `generation_checkpoint.jsonl`: append-only resume state.
- `generation_trace.jsonl`: optional append-only prompt/response audit created
  by `--trace`.
- `failures.jsonl`: currently unresolved samples.
- `manifest.json`: generation settings and aggregate counts.

Each training record includes the source hash and provenance, meter ID and
name, couplet count, concrete template ID and version, generated `instruction`,
composed `response`, OpenAI-style `messages`, deterministic `sft_split`, and
quality flags. Exact
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
