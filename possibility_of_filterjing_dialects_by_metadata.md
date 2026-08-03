No— it's not possible, not with sufficient coverage and accuracy. The metadata can produce high-confidence subsets, but it cannot completely separate MSA from Moroccan dialects.

### What is available

The internal pre-template corpus contains fields such as `poem_language`, `location`, `poet_name`, `poet_description`, `poet_era`, `meter`, and `dataset_name`. However, their availability depends heavily on the source dataset. The final corpus combines 427,337 poems from 11 source groups. [InstructPoet-Ar paper](https://aclanthology.org/2026.findings-acl.1931/), [official preprocessing code](https://github.com/mbzuai-nlp/instructpoet-ar/blob/main/scripts/data_preprocessing/combining_data.ipynb)

| Metadata                  | What it can identify                                 | Main limitation                                              |
| ------------------------- | ---------------------------------------------------- | ------------------------------------------------------------ |
| `poem_language`           | Formal versus generic colloquial poetry              | Usually missing; colloquial labels do not specify the region |
| `location`                | Poet associated with Morocco                         | Poet origin is not the poem’s language                       |
| `poet_description`        | Moroccan nationality/origin from biographies         | Sparse and still does not prove the poem is Darija           |
| `poet_era`                | Modern versus historical poetry                      | Does not reliably distinguish MSA from Classical Arabic      |
| `meter`, `poem_type`      | Occasional signals such as `زجل`, `عامي`, or `ملحون` | Labels are rare and inconsistent                             |
| `dataset_name` / `source` | Source-specific priors                               | A source can contain several varieties                       |

### The strongest metadata exists in raw Ashaar—but is limited

The original [Ashaar dataset](https://huggingface.co/datasets/arbml/ashaar) has two particularly relevant fields:

* `poem language type`
* `poet location`

Across its 254,630 raw poems:

* `poem language type` is available for 183,407 poems, or about 72%:

  * `فصيح`: 153,722
  * `فصحى`: 20,852
  * `عامي`: 8,503
  * `شعبي`: 298
  * missing: 71,223
* `poet location` is available for only 64,028 poems, or about 25%.
* `poet location = المغرب` occurs for 2,175 poems.

These counts come from the official [Ashaar dataset statistics](https://datasets-server.huggingface.co/statistics?dataset=arbml%2Fashaar&config=default&split=train).

Even before considering missing values, the labels are insufficient:

* `فصيح` and `فصحى` combine Classical Arabic and modern formal Arabic.
* `عامي` and `شعبي` do not distinguish Moroccan Darija from Egyptian, Gulf, Levantine, or other dialects.
* `المغرب` describes the poet, not necessarily the poem. Moroccan poets frequently write in formal Arabic.

### Important preprocessing issue

The official InstructPoet-Ar combination notebook drops Ashaar’s two useful fields before trying to rename them:

```python
df.drop(
    columns=["poet location", "poem language type"],
    errors="ignore",
    inplace=True,
)
```

Consequently, the final 123,581 Ashaar poems used by InstructPoet-Ar do not retain those labels in the checked preprocessing workflow. In the unified corpus, `poem_language` is shown as populated only for the `mawsooaa` source—at most 18,002 of 427,337 poems, or 4.2%. [Preprocessing notebook](https://github.com/mbzuai-nlp/instructpoet-ar/blob/main/scripts/data_preprocessing/combining_data.ipynb)

Also, the `MSA`, `North Africa`, `Gulf`, `Levant`, and `Nile Valley` columns in the released InstructPoet-Ar files describe the language of the **instruction templates**, not the source poems. `North Africa` is neither a poem label nor specifically Moroccan. [Dataset card](https://huggingface.co/datasets/MBZUAI/instructpoet-ar)

### Final assessment

Metadata alone can construct:

* A **formal-Arabic seed set** using `poem_language ∈ {فصيح, فصحى}`. This combines Classical Arabic and MSA.
* A **probable Moroccan-dialect seed set** using:

$$
\left[
(\text{language} \in \{\text{عامي، شعبي}\} \lor \text{type/meter} \in \{\text{زجل، ملحون}\}) \land \text{Moroccan poet metadata}
\right]
$$

This would have relatively high precision but very low recall.

Therefore, the appropriate design is a three-way metadata filter:

1. **High-confidence keep**
2. **High-confidence reject**
3. **Uncertain — classify from the poem text**

A text-based language-variety classifier will be necessary for most of the 427,337 poems. If rebuilding from raw sources, Ashaar’s `poem language type` and `poet location` should be preserved before unification.
