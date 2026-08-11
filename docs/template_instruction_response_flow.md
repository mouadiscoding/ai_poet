# How templates become instruction-response pairs

The pipeline starts from an existing poem and produces a two-message SFT pair.
Instruction reconstruction and editorial-work reconstruction are deliberately
separate so that the latter cannot drift into explaining how the dataset fields
were created.

```text
Source poem
   |
   v
Choose one instruction template
   |
   v
Gemma generates {instruction}
   |
   +--> deterministic instruction checks
   +--> Gemma instruction-quality review
   |
   v
Split the poem into groups of at most three couplets
   |
   v
Gemma generates {overview?, verse_reasoning[]} for each group
   |
   +--> deterministic source-linked checks
   +--> Gemma chunk-quality review
   |
   v
Render the validated work blocks
   |
   v
Append one canonical result marker and the exact source poem
   |
   v
Store {user: instruction, assistant: response}
```

## 1. Source preparation

[`load_poems()`](../src/ai_poet/synthetic_data/corpus.py) reads alternating
hemistichs from the source dataset. Each adjacent pair becomes one canonical
line in the form `صدر = عجز`. Identical verse sequences are deduplicated while
their provenance is retained.

## 2. Instruction generation

[`prompts/templates.py`](../src/ai_poet/synthetic_data/prompts/templates.py)
contains six independently worded templates emphasizing prosody, semantic arc,
imagery, voice, occasion, or diction. Every template still covers all six
dimensions and produces one JSON field:

```json
{"instruction": "<detailed Arabic writing request>"}
```

[`build_messages()`](../src/ai_poet/synthetic_data/prompts/builder.py) combines
the chosen template with two form-appropriate instruction demonstrations. The
instruction must use the configured section order, numeric couplet count, and
trusted meter definition. It must not quote a complete source hemistich or
invent unsupported context.

Python rejects malformed instructions, missing or reordered headings, a missing
numeric count or meter, insufficient length, and complete source-hemistich
copies. Gemma then performs a separate semantic review. Only the instruction is
regenerated when this phase fails.

## 3. Verse-level editorial work

[`build_reasoning_messages()`](../src/ai_poet/synthetic_data/prompts/builder.py)
receives the validated instruction and no more than three exact target couplets.
Its metered few-shot is a multi-verse drafting demonstration; prose uses a
parallel example based on internal rhythm.

The first chunk returns a poem-level `overview`; every chunk returns one object
per target couplet:

```json
{
  "overview": "<present only for the first chunk>",
  "verse_reasoning": [
    {
      "verse_index": 1,
      "intended_meaning": "...",
      "connection_to_previous": "...",
      "imagery_and_diction": "...",
      "first_draft": "... = ...",
      "problem_with_first_draft": "...",
      "revised_draft": "<exact target couplet>",
      "first_hemistich_scansion": "...",
      "second_hemistich_scansion": "...",
      "rhyme_check": "..."
    }
  ]
}
```

The content is an explicit synthetic editorial reconstruction. It is not
presented as access to the historical poet's private thoughts.

## 4. Deterministic and semantic validation

[`validation.py`](../src/ai_poet/synthetic_data/validation.py) rejects a
reasoning chunk unless:

- It has exactly the requested keys and number of blocks.
- Verse indices are contiguous and in source order.
- Every required field is a non-empty string.
- Each `revised_draft` exactly matches its source couplet.
- Each `first_draft` differs from the accepted version.
- Pre-draft imagery and diction are expressed as a plan, not as retrospective
  claims about wording that has not appeared yet.
- The content contains no instruction-generation metatext such as references to
  fields, character counts, or building the `instruction`.

Gemma then judges whether the meaning, alternative draft, rejection reason,
concrete revision decision, and rhyme discussion are chronological and
plausible. Python checks that both
scansion fields exist and are substantive, but omits them from the semantic
review: Gemma is not used as a hard gate for exact Arabic scansion. A rejected
chunk is repaired independently; already accepted chunks are not regenerated.
A malformed validator response fails safely.

## 5. Rendering the assistant response

[`render_reasoning()`](../src/ai_poet/synthetic_data/responses.py) turns the
validated objects into Arabic sections such as:

```text
مرحلة التفكير والتحرير:

<overview>

البيت 1:

المعنى المقصود:
...

خطة الصورة والمعجم:
...

صياغة أولى:
...

قرار المراجعة:
...

الصياغة المنقحة:
...

فحص الصدر عروضيًا:
...
```

[`compose_response()`](../src/ai_poet/synthetic_data/responses.py) preserves
source couplets quoted in these validated work blocks. It discards anything
after an accidental result marker and removes only a leading or explicitly
labeled accidental full-poem dump. It then appends one canonical section:

```text
النتيجة النهائية:

<exact source poem>
```

The exported record keeps the existing trainer-facing contract:

```json
{
  "instruction": "<generated instruction>",
  "response": "<rendered worklog plus exact poem>",
  "messages": [
    {"role": "user", "content": "<generated instruction>"},
    {"role": "assistant", "content": "<rendered worklog plus exact poem>"}
  ]
}
```

The structured objects and raw model exchanges remain generation-time or trace
data; they are not added to the SFT message schema.

## 6. Long poems and checkpoint compatibility

Reasoning is always generated in groups of at most three couplets, independent
of source length. For source poems above the configured direct-source limit,
[`_chunk_analysis()`](../src/ai_poet/synthetic_data/generation.py) additionally
creates ordered summaries for the global instruction stage. Exact couplets,
not summaries, are still used by the verse-work stage.

Template version `5` invalidates older checkpoint successes so they are
regenerated under the verse-level contract. Successful records include total,
instruction, and reasoning generation-attempt counts plus the reasoning chunk
count.
