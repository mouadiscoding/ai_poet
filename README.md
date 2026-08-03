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
been pasted into chat, shell history, or logs, create a replacement, and expose
the replacement only through an environment variable:

```powershell
$env:GEMMA_API_KEY="replace-with-a-new-token"
```

The client verifies TLS certificates by default. The internal endpoint shown
below may require `--insecure`, which is equivalent to `curl -k`; use that flag
only for an endpoint you trust.

## Generate a smoke sample

Start with a small run:

```powershell
uv run python generate_sft.py `
  --input data/ashaar_classic_moroccan.parquet `
  --output-dir data/ashaar_sft_smoke `
  --limit 10 `
  --insecure
```

Inspect the generated instructions and reasoning before starting the complete
run:

```powershell
uv run python generate_sft.py `
  --input data/ashaar_classic_moroccan.parquet `
  --output-dir data/ashaar_sft `
  --concurrency 4 `
  --insecure
```

The endpoint and model default to:

- `https://vllm-gemma4-31b-mtrna-ns1.apps.olympus.atlasxai.ma/v1/chat/completions`
- `gemma-4-31B`

Use `--endpoint` or `--model` to override them. Run
`uv run python generate_sft.py --help` for generation, validation, retry, and
context-size controls.

## Resume and failure handling

Every completed request is appended immediately to
`generation_checkpoint.jsonl`. Re-running the same command skips successful
sample IDs and retries unresolved failures. Transient HTTP failures are retried
up to five times; structurally or semantically invalid model responses receive
up to two repair prompts.

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
- `failures.jsonl`: currently unresolved samples.
- `manifest.json`: generation settings and aggregate counts.

Each training record includes the source hash and provenance, meter ID and
name, couplet count, template ID, generated `instruction`, composed `response`,
OpenAI-style `messages`, deterministic `sft_split`, and quality flags. Exact
duplicate poem texts share one record and retain all source row indices and
URLs. Splits are assigned from the poem hash using 98% train, 1% validation,
and 1% test buckets.

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
