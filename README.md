# Arabic poetry SFT generation

This project builds a supervised fine-tuning dataset from
`data/ashaar_classic_moroccan.parquet`. For every unique poem it asks an
OpenAI-compatible Gemma endpoint to reverse-construct a detailed Arabic writing
instruction and an editorial reasoning section. The program then appends the
original poem itself, rather than allowing the language model to reproduce or
modify the training target.

The generator uses six prompt families and two few-shot user/assistant examples
inside every request. Metered and prose poems have separate demonstrations.

See [the complete SFT generation guide](docs/sft_dataset_generation.md) for the
corpus audit, prompt architecture, validation contract, output schema, and
operational details.

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
uv run python generate_sft.py `
  --input data/ashaar_classic_moroccan.parquet `
  --output-dir data/ashaar_sft_smoke `
  --limit 10 `
  --trace `
  --insecure
```

`--trace` prints a full audit block for each generation step and appends the
same structured events to `generation_trace.jsonl`. The trace shows the
meta-template catalog and rationale, the selected family and deterministic
selection calculation, every complete message array sent to Gemma, the raw API
response payload, validation and repair results, and the final response after
the source poem is appended. Use it for smoke runs and audits; full-corpus
traces are very large.

Inspect the generated instructions and reasoning before starting the complete
run:

```powershell
uv run python generate_sft.py `
  --input data/ashaar_classic_moroccan.parquet `
  --output-dir data/ashaar_sft `
  --concurrency 4 `
  --insecure
```

The endpoint and model are required in `.env`; neither has a default embedded
in the code. Run
`uv run python generate_sft.py --help` for generation, validation, retry, and
context-size controls.

## Resume and failure handling

Every completed request is appended immediately to
`generation_checkpoint.jsonl`. Re-running the same command skips successful
sample IDs and retries unresolved failures. Transient HTTP failures are retried
up to three times with exponential backoff. Connection failures use the same
backoff and stop the entire script if Gemma is still unreachable after the
third retry. Structurally or semantically invalid model responses receive up to
two repair prompts.

Each successful record is also appended and flushed immediately to
`ashaar_sft.jsonl`, so the training data can be inspected while generation is
still running. The file is rewritten in source order when the run finishes.

The run returns a non-zero exit status while any selected poem remains
unresolved. `failures.jsonl` records those failures without storing request
headers or credentials.

Poems longer than 24,000 characters are analyzed in couplet-aligned chunks.
Their final SFT record still includes the complete source poem and is marked
with `oversized_for_sft=true`, allowing trainers to exclude it according to the
target model's context length.

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
name, couplet count, template ID, generated `instruction`, composed `response`,
OpenAI-style `messages`, deterministic `sft_split`, and quality flags. Exact
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
النتيجة النهائية:

صدر البيت = عجز البيت
```

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
