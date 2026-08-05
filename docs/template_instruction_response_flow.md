# How templates become instruction-response pairs

The SFT pipeline performs a reverse-generation task: it starts with an existing
poem, asks Gemma to infer a detailed writing instruction and editorial
rationale, and then converts those generated fields into a two-message
supervised fine-tuning example.

```text
Source poem
   |
   v
Choose a concrete template randomly
   |
   v
Build a six-message few-shot prompt
   |
   v
Gemma returns {instruction, reasoning}
   |
   v
Validate and, if necessary, repair the result
   |
   v
Append the exact source poem to the reasoning
   |
   v
Store {user: instruction, assistant: response}
```

## 1. Source preparation

[`download_ashaar.py`](../download_ashaar.py) only downloads the Ashaar dataset
and writes it to Parquet. It does not use the prompt templates.

[`load_poems()`](../src/ai_poet/synthetic_data/corpus.py) reads the Parquet file
and:

- Treats `poem_verses` as alternating hemistichs.
- Formats each pair as `first hemistich = second hemistich`.
- Deduplicates identical verse sequences.
- Resolves meter metadata, including duplicate rows with conflicting labels.
- Computes a content-based SHA-256 `sample_id`.

Consequently, each unique poem normally produces exactly one training pair.
Duplicate source rows retain their provenance but do not create duplicate
targets.

## 2. Concrete templates

[`prompts/templates.py`](../src/ai_poet/synthetic_data/prompts/templates.py)
defines six complete, renderable prompt templates:

| Template ID | Main emphasis |
| --- | --- |
| `prosody_rhyme` | Meter, rhyme, recitation, and sound |
| `semantic_arc` | Meaning progression and poem-level unity |
| `imagery_rhetoric` | Imagery and rhetorical devices |
| `emotion_voice` | Emotional register, speaker, and tone |
| `occasion_addressee` | Occasion, addressee, and communicative purpose |
| `diction_revision` | Diction, syntax, alternatives, and revision |

Every template contains a literal `{poem}` placeholder plus placeholders for
the form, numeric unit count, and minimum field length. The templates vary the
order and framing of the analysis, but every one explicitly requires all six
dimensions: sound/form, semantic progression, imagery/rhetoric, emotion/voice,
occasion/addressee, and diction/revision.

[`choose_template()`](../src/ai_poet/synthetic_data/assignment.py) makes a fresh
uniform random choice once per poem-generation call. Repairs retain that
choice; rerunning an unresolved poem may choose another template. All six are
eligible for prose, where prosody is adapted to internal rhythm, parallelism,
sound, and observable rhyme rather than a classical meter.

## 3. Few-shot prompt construction

[`build_messages()`](../src/ai_poet/synthetic_data/prompts/builder.py) constructs
a six-message conversation for each poem:

```text
system:    generation policy, JSON contract, and all six focuses
user:      demonstration source poem 1
assistant: demonstration JSON 1
user:      demonstration source poem 2
assistant: demonstration JSON 2
user:      actual source poem or long-poem analysis notes
```

The system prompt requires the model to:

- Reverse-construct an instruction from the reference poem.
- Return only JSON with the keys `instruction` and `reasoning`.
- Use the correct meter and numeric couplet count.
- Avoid mentioning the source poet, title, or URL.
- Avoid copying a complete source hemistich into the instruction.
- Produce a long editorial rationale about meaning, imagery, drafting, and
  revision.
- Discuss meter and rhyme when the source is metered.
- End the reasoning with `النتيجة النهائية:`.
- Omit the complete final poem because the program will append it itself.

There are separate example banks for
[`METERED_FEW_SHOTS`](../src/ai_poet/synthetic_data/prompts/examples.py) and
[`PROSE_FEW_SHOTS`](../src/ai_poet/synthetic_data/prompts/examples.py). Every
example instruction and reasoning trace demonstrates all six focuses. The
prose demonstrations replace metrical scansion with internal rhythm and avoid
inventing a classical meter.

The final user message provides only the material needed for generation:

- The base meter, or the prose marker.
- The exact number of couplets or prose units.
- The formatted source poem.
- For an oversized poem, ordered summaries in place of the full source text.

Poet, title, and URL metadata are retained for provenance in the output record,
but are not supplied as content for the generated instruction.

## 4. Model generation and validation

[`generate_one()`](../src/ai_poet/synthetic_data/generation.py) sends the
few-shot conversation to the OpenAI-compatible Gemma endpoint. The expected
result is a JSON object:

```json
{
  "instruction": "A detailed Arabic poetry-writing request",
  "reasoning": "A synthetic Arabic editorial rationale"
}
```

Python performs only the JSON extraction and string-type checks required to
obtain `instruction` and `reasoning`; it does not judge their poetic content.

Every parseable pair is sent to Gemma in a separate zero-temperature validation
request together with the same reference material, expected form, exact unit
count, minimum lengths, and six-focus contract. Gemma returns separate
`passed`/`errors` verdicts for `instruction` and `reasoning`, and both must pass.
It checks schema, Arabic language, grounding, correct form and count, all six
focuses, instruction quality, editorial reasoning quality, and cross-field
consistency. A rejection becomes repair feedback in the original generation
conversation. A malformed validator response fails the sample safely instead
of approving it. Network retries remain separate.

## 5. Forming the final instruction-response pair

The generated `instruction` becomes the training user message directly. The
generated `reasoning` is not used unchanged.

[`compose_response()`](../src/ai_poet/synthetic_data/responses.py) removes:

- Any complete source poem echoed by the model.
- Standalone source couplets or hemistichs.
- Duplicate final-result markers.
- Standalone headers such as `القصيدة النهائية:`.

It then constructs the assistant response in one invariant order:

```text
<cleaned model-generated editorial reasoning>

النتيجة النهائية:

<exact original poem>
```

The poem is appended from the source record rather than copied from the model's
answer. This prevents the target poem from being rewritten, normalized,
re-diacritized, or hallucinated by the generation endpoint.

The successful record assembled in
[`generate_one()`](../src/ai_poet/synthetic_data/generation.py) contains both
flat fields and a chat-format representation:

```json
{
  "instruction": "<generated instruction>",
  "response": "<cleaned reasoning plus exact source poem>",
  "messages": [
    {
      "role": "user",
      "content": "<generated instruction>"
    },
    {
      "role": "assistant",
      "content": "<cleaned reasoning plus exact source poem>"
    }
  ]
}
```

The six-message template conversation is generation-time scaffolding only. The
exported SFT example is the final two-message instruction-response
conversation.

## 6. Oversized poems

When a poem exceeds the configured direct-source limit,
[`_chunk_analysis()`](../src/ai_poet/synthetic_data/generation.py) divides it at
complete-couplet boundaries and requests a compact analysis for each chunk.
The ordered summaries replace the full source poem in both the main generation
prompt and its Gemma validation prompt.

This still produces one instruction-response pair for the complete poem. The
complete original poem is appended to the assistant response, and the record is
marked with `oversized_for_sft=true` so downstream training code can apply its
own context-length policy.

## 7. Checkpointing and exports

[`run()`](../src/ai_poet/synthetic_data/runner.py) processes pending poems
concurrently and checkpoints every success or failure. Existing successful
sample IDs are skipped on resume only when their `template_version` is `2`;
older records remain in the append-only checkpoint and are regenerated.

[`write_outputs()`](../src/ai_poet/synthetic_data/outputs.py) orders successful
records by source order and writes them to:

- `ashaar_sft.jsonl`
- `ashaar_sft.parquet`

It also writes unresolved failures and a manifest containing generation
settings, template distribution, dataset split counts, and quality flags.
