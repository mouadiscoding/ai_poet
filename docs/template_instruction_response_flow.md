# How templates become instruction-response pairs

The SFT pipeline performs a reverse-generation task: it starts with an existing
poem, asks Gemma to infer a detailed writing instruction and editorial
rationale, and then converts those generated fields into a two-message
supervised fine-tuning example.

```text
Source poem
   |
   v
Choose a template family
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

## 2. Template families

[`prompts/families.py`](../src/ai_poet/synthetic_data/prompts/families.py)
defines six template families:

| Template ID | Main emphasis |
| --- | --- |
| `prosody_rhyme` | Meter, rhyme, recitation, and sound |
| `semantic_arc` | Meaning progression and poem-level unity |
| `imagery_rhetoric` | Imagery and rhetorical devices |
| `emotion_voice` | Emotional register, speaker, and tone |
| `occasion_addressee` | Occasion, addressee, and communicative purpose |
| `diction_revision` | Diction, syntax, alternatives, and revision |

These are meta-templates rather than final instructions with simple
placeholders. Each family provides an Arabic focus statement that tells the
model which aspect of the reference poem to foreground while retaining the
shared requirements for meaning, form, imagery, diction, and revision.

[`choose_family()`](../src/ai_poet/synthetic_data/assignment.py) selects a
family deterministically from the poem hash:

```python
offset = int(poem.sample_id[8:16], 16) % len(families)
```

The same poem therefore receives the same family across runs. Prose poetry
excludes `prosody_rhyme` because that family assumes a classical Arabic meter.

## 3. Few-shot prompt construction

[`build_messages()`](../src/ai_poet/synthetic_data/prompts/builder.py) constructs
a six-message conversation for each poem:

```text
system:    generation policy, JSON contract, and template-family focus
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
[`PROSE_FEW_SHOTS`](../src/ai_poet/synthetic_data/prompts/examples.py). The
first example demonstrates the general output contract. The second example is
specialized through
[the family-specific demonstration fields](../src/ai_poet/synthetic_data/prompts/families.py),
which add guidance to both its example instruction and reasoning.

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

[`validate_generation()`](../src/ai_poet/synthetic_data/validation.py) checks
that:

- The object has exactly the two expected fields.
- Both values are strings and meet the configured minimum length.
- The combined content is predominantly Arabic.
- The instruction contains the exact numeric couplet count.
- The instruction names the correct meter or explicitly requests prose.
- The instruction does not name a conflicting meter.
- The instruction does not reproduce a complete source hemistich.
- The reasoning discusses meaning, imagery or rhetoric, and revision.
- Metered reasoning discusses meter, rhythm, or prosody.
- The reasoning contains the final-result transition.

If parsing or validation fails, the generator adds the invalid response and the
specific validation errors to the original conversation, then asks the model
for a complete corrected JSON object. Network retries and semantic repair calls
are handled separately.

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
The ordered summaries replace the full source poem in the main generation
prompt.

This still produces one instruction-response pair for the complete poem. The
complete original poem is appended to the assistant response, and the record is
marked with `oversized_for_sft=true` so downstream training code can apply its
own context-length policy.

## 7. Checkpointing and exports

[`run()`](../src/ai_poet/synthetic_data/runner.py) processes pending poems
concurrently and checkpoints every success or failure. Existing successful
sample IDs are skipped when a run resumes.

[`write_outputs()`](../src/ai_poet/synthetic_data/outputs.py) orders successful
records by source order and writes them to:

- `ashaar_sft.jsonl`
- `ashaar_sft.parquet`

It also writes unresolved failures and a manifest containing generation
settings, template distribution, dataset split counts, and quality flags.
