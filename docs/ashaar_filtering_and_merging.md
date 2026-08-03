# Filtering and merging the Ashaar dataset

This document describes the workflow implemented in [`notebooks/ashaar.ipynb`](../notebooks/ashaar.ipynb) to create:

- A strict classical Arabic poetry subset.
- A high-confidence subset containing poems by authors of Moroccan origin.
- A combined dataset containing the union of both subsets without duplicating poems that appear in both.

## Results

| Dataset | Rows | Unique `poet_name` values | Output |
| --- | ---: | ---: | --- |
| Classical | 49,004 | 3,466 | `data/ashaar_classic.parquet` |
| Moroccan authors | 2,157 | 62 | `data/ashaar_moroccan.parquet` |
| Overlap | 962 | — | Present in both subsets |
| Merged union | 50,199 | 3,500 | `data/ashaar_classic_moroccan.parquet` |

The original `data/ashaar_dataset.parquet` file remains unchanged. All output files preserve its 14 columns and their order. The repository ignores `data/`, so the generated Parquet files remain local unless the ignore rule is changed.

## 1. Load the source dataset

The notebook is run from the `notebooks/` directory, so the input path is relative to that directory.

```python
from pathlib import Path

import pandas as pd

dataset_path = Path("../data/ashaar_dataset.parquet")
ashaar = pd.read_parquet(dataset_path)
```

The current source contains 212,499 rows. The columns used by the filters are:

- `poet_name`
- `poet_description`
- `poet_era`
- `poet_location`
- `poem_language_type`

## 2. Create the classical subset

For this workflow, a poem is considered classical only when both conditions below hold:

1. `poet_era` is present and is not `العصر الحديث`.
2. `poem_language_type` is explicitly `فصيح` or `فصحى`.

This is intentionally strict. Poems with a missing era or missing language label are excluded, even when their text may actually be classical Arabic.

```python
MODERN_ERA = "العصر الحديث"
FORMAL_ARABIC_LABELS = {"فصيح", "فصحى"}

classic_mask = (
    ashaar["poet_era"].notna()
    & ashaar["poet_era"].ne(MODERN_ERA)
    & ashaar["poem_language_type"].isin(FORMAL_ARABIC_LABELS)
)

classic_ashaar = (
    ashaar.loc[classic_mask]
    .copy()
    .reset_index(drop=True)
)
```

The result is checked against the current source:

```python
assert len(classic_ashaar) == 49_004
assert classic_ashaar["poet_era"].notna().all()
assert classic_ashaar["poet_era"].ne(MODERN_ERA).all()
assert classic_ashaar["poem_language_type"].isin(
    FORMAL_ARABIC_LABELS
).all()
assert list(classic_ashaar.columns) == list(ashaar.columns)
```

## 3. Identify authors already labelled as Moroccan

An author is treated as an exact-location match when at least one of their rows has:

```python
poet_location == "المغرب"
```

The filter is applied at author level rather than row level. Once an author has an exact Moroccan location on any row, every poem by that author is eligible for the Moroccan subset, including rows where `poet_location` is missing.

```python
MOROCCO = "المغرب"

exact_location_poets = set(
    ashaar.loc[
        ashaar["poet_location"].eq(MOROCCO),
        "poet_name",
    ]
)

assert len(exact_location_poets) == 53
```

This author-level step is important because some authors have inconsistent location metadata across their poems.

## 4. Generate biography candidates

Location metadata is sparse, so biographies were searched for Morocco, Moroccan demonyms, and Moroccan cities or regions. These matches create a review queue only; they do not automatically include an author.

The search terms are:

```python
MOROCCAN_BIOGRAPHY_TERMS = [
    "المغرب", "مغربي", "مغربية", "المغاربة",
    "الدار البيضاء", "فاس", "مراكش", "الرباط",
    "تطوان", "مكناس", "مكناسة", "سلا", "طنجة",
    "وجدة", "أكادير", "الصويرة", "آسفي", "تارودانت",
    "شفشاون", "تازة", "أزمور", "أغمات", "دكالة",
    "تزنيت", "الصمارة", "القنيطرة", "الجديدة",
    "الحسيمة", "الناظور", "بني ملال", "ورزازات",
    "خنيفرة", "تافيلالت", "تادلة", "درعة", "سوس",
    "مولاي إدريس", "خريبكة", "العرائش", "أصيلة",
    "إفران", "إفني", "الريف",
]
```

A boundary-aware regular expression prevents short place names such as `فاس` from matching inside unrelated Arabic words:

```python
import re

moroccan_biography_pattern = re.compile(
    r"(?<![ء-ي])(?:"
    + "|".join(map(re.escape, MOROCCAN_BIOGRAPHY_TERMS))
    + r")(?![ء-ي])"
)
```

Authors already covered by `exact_location_poets` are removed before candidate generation. One biography is retained per remaining exact `poet_name` value.

```python
author_biographies = (
    ashaar.loc[
        ashaar["poet_description"].notna()
        & ~ashaar["poet_name"].isin(exact_location_poets),
        ["poet_name", "poet_location", "poet_description"],
    ]
    .drop_duplicates("poet_name")
    .copy()
)

biography_candidate_mask = author_biographies[
    "poet_description"
].map(
    lambda description: bool(
        moroccan_biography_pattern.search(description)
    )
)

moroccan_biography_candidates = author_biographies.loc[
    biography_candidate_mask
].copy()

assert len(moroccan_biography_candidates) == 37
```

An initial row-level inspection found 45 names. After treating location as an author-level property, eight of those names were found to be already represented among the 53 exact-location authors. The final biography-only review queue therefore contains 37 names.

## 5. Manually review biography candidates

The review uses present-day Morocco as its geographical boundary.

An author is approved when an external biography establishes at least one of the following:

- Moroccan nationality or identity.
- Birth in present-day Morocco.
- Ancestry explicitly connected to present-day Morocco.

An author is rejected when the match refers only to:

- Travel or temporary residence in Morocco.
- A place of death or burial in Morocco.
- Service under a Moroccan ruler.
- A manuscript or book held in Rabat or another Moroccan city.
- A literary title or a generic historical use of “Maghreb.”
- Ambiguous ancestry that cannot be tied to present-day Morocco.

The notebook stores every decision in `moroccan_author_review` with these fields:

| Field | Meaning |
| --- | --- |
| `poet_name` | Exact spelling used in the dataset |
| `is_moroccan` | Final Boolean decision |
| `rationale` | Reason for inclusion or exclusion |
| `source_url` | External biographical source |

Nine exact dataset spellings, representing eight people, were approved:

| Dataset name | Reason |
| --- | --- |
| `أبو العباس الجراوي` | Born in Tadla |
| `أبو العباسِ الجَراوي` | Diacritized variant of the same poet |
| `أَحمَد بن المَأمون البلغيثي` | Born in Fez |
| `ابن الونان` | Born, raised, and died in Fez |
| `ابن الياسمين` | From Marrakesh |
| `ابن زاكور` | Born in Fez |
| `ابن عمرو الأغماتي` | Born in Aghmat and raised in Fez |
| `الشريشي السلوي` | Born in Salé and raised in Marrakesh |
| `حسن قويدر الخليلي` | Documented Moroccan ancestry |

The remaining 28 candidates were rejected. Examples include Andalusian poets who only lived or died in Morocco, non-Moroccan poets whose manuscripts are held in Rabat, and `محمد وفا`, whose broad Maghrebi ancestry could not be tied confidently to present-day Morocco. The complete decisions, rationales, and sources are kept in the notebook's `moroccan_author_review` table.

The audit is validated before it is used:

```python
candidate_names = set(moroccan_biography_candidates["poet_name"])
reviewed_names = set(moroccan_author_review["poet_name"])

assert len(moroccan_author_review) == 37
assert moroccan_author_review["poet_name"].is_unique
assert reviewed_names == candidate_names
assert moroccan_author_review["is_moroccan"].notna().all()
assert moroccan_author_review["rationale"].str.strip().ne("").all()
assert moroccan_author_review["source_url"].str.match(
    r"https?://"
).all()
```

## 6. Create the Moroccan-author subset

The final author set is the union of exact-location authors and approved biography candidates.

```python
approved_biography_poets = set(
    moroccan_author_review.loc[
        moroccan_author_review["is_moroccan"],
        "poet_name",
    ]
)

rejected_biography_poets = (
    reviewed_names - approved_biography_poets
)
moroccan_poets = exact_location_poets | approved_biography_poets

moroccan_ashaar = (
    ashaar.loc[ashaar["poet_name"].isin(moroccan_poets)]
    .copy()
    .reset_index(drop=True)
)
```

All poems by an approved author are retained regardless of `poet_era` or `poem_language_type`.

```python
moroccan_result_poets = set(moroccan_ashaar["poet_name"])

assert exact_location_poets <= moroccan_result_poets
assert approved_biography_poets <= moroccan_result_poets
assert not (rejected_biography_poets & moroccan_result_poets)
assert set(moroccan_ashaar["poet_name"]) <= moroccan_poets
assert list(moroccan_ashaar.columns) == list(ashaar.columns)
assert len(moroccan_ashaar) == 2_157
assert moroccan_ashaar["poet_name"].nunique() == 62
```

## 7. Save and verify the filtered subsets

Both subsets are saved without DataFrame indexes.

```python
classic_output_path = dataset_path.with_name(
    "ashaar_classic.parquet"
)
moroccan_output_path = dataset_path.with_name(
    "ashaar_moroccan.parquet"
)

classic_ashaar.to_parquet(classic_output_path, index=False)
moroccan_ashaar.to_parquet(moroccan_output_path, index=False)
```

Each file is read back and compared with its in-memory DataFrame. `check_dtype=False` permits harmless Parquet normalization of object-backed null values while still checking the data, index, columns, and shape.

```python
saved_classic_ashaar = pd.read_parquet(classic_output_path)
saved_moroccan_ashaar = pd.read_parquet(moroccan_output_path)

pd.testing.assert_frame_equal(
    saved_classic_ashaar,
    classic_ashaar,
    check_dtype=False,
)
pd.testing.assert_frame_equal(
    saved_moroccan_ashaar,
    moroccan_ashaar,
    check_dtype=False,
)
```

## 8. Merge the subsets without duplicates

There are 962 source poems that satisfy both filters. Concatenating the two DataFrames would add those rows twice. Calling `drop_duplicates()` across every column is also unsuitable because `poem_verses` contains list values, which are not hashable.

Instead, the two conditions are combined into a Boolean union mask against the original DataFrame. Each original row can then be selected at most once.

```python
moroccan_mask = ashaar["poet_name"].isin(moroccan_poets)
merged_mask = classic_mask | moroccan_mask

merged_ashaar = (
    ashaar.loc[merged_mask]
    .copy()
    .reset_index(drop=True)
)
```

The expected union size is calculated with inclusion-exclusion:

```text
49,004 classical rows
+ 2,157 Moroccan-author rows
-   962 rows present in both
= 50,199 merged rows
```

```python
overlap_count = int((classic_mask & moroccan_mask).sum())
expected_merged_rows = (
    len(classic_ashaar)
    + len(moroccan_ashaar)
    - overlap_count
)

assert len(merged_ashaar) == expected_merged_rows == 50_199
assert list(merged_ashaar.columns) == list(ashaar.columns)
```

The merged result is saved and read back for verification:

```python
merged_output_path = dataset_path.with_name(
    "ashaar_classic_moroccan.parquet"
)
merged_ashaar.to_parquet(merged_output_path, index=False)

saved_merged_ashaar = pd.read_parquet(merged_output_path)
pd.testing.assert_frame_equal(
    saved_merged_ashaar,
    merged_ashaar,
    check_dtype=False,
)
```

## 9. Reproducing the workflow

Run all cells in `notebooks/ashaar.ipynb` from top to bottom. The notebook will:

1. Load the original Parquet dataset.
2. Build and validate `classic_ashaar`.
3. Generate the Moroccan biography review queue.
4. Validate the completed author audit.
5. Build and validate `moroccan_ashaar`.
6. Save and read back both filtered datasets.
7. Build the deduplicated union as `merged_ashaar`.
8. Save and read back the merged dataset.

The fixed row-count assertions are regression checks for the current source file. If the source dataset changes, the counts and biography candidate audit must be reviewed before updating those assertions.
