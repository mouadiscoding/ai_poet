"""Load, validate, deduplicate, and canonicalize source poems."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .meters import meter_name
from .poems import PoemRecord, poem_hash


def _metadata_score(row: dict[str, Any], selected_meter: int) -> tuple[int, int]:
    """Score a duplicate source row for selection as canonical metadata.

    Rows using the majority meter receive highest priority. Within that group,
    rows are ranked by how many of title, theme, poet name, and poem URL are
    populated.
    Returning a tuple lets Python compare these criteria lexicographically.

    Args:
        row: A source row containing poem metadata.
        selected_meter: The meter chosen by majority vote for the poem group.

    Returns:
        ``(meter_matches, populated_field_count)``, where the first element is
        either zero or one and the second ranges from zero to three.
    """
    return (
        int(row["poem_meter"] == selected_meter),
        sum(
            bool(row.get(key))
            for key in ("poem_title", "poem_theme", "poet_name", "poem_url")
        ),
    )


def load_poems(path: Path) -> list[PoemRecord]:
    """Load, validate, deduplicate, and canonicalize poems from Parquet.

    Only the columns required for generation are read. Rows with identical
    verse tuples are grouped as the same poem. Within each group, the meter is
    selected by an unambiguous majority vote, and canonical descriptive
    metadata comes from the best-populated row that uses that meter. All source
    row indices and distinct non-empty URLs are retained for provenance. The
    resulting sample identifier is based solely on the verses, and records are
    sorted by their first source-row position to preserve source order.

    Args:
        path: Parquet dataset containing ``poem_title``, ``poem_theme``,
            ``poem_meter``, ``poem_verses``, ``poem_url``, and ``poet_name``
            columns.

    Returns:
        One immutable :class:`PoemRecord` per distinct verse sequence.

    Raises:
        ValueError: If a row has no verses, has an odd number of hemistichs, a
            duplicate group has a tied meter vote, or the selected meter ID is
            unsupported.
        OSError: If the source file cannot be read.
    """
    columns = [
        "poem_title",
        "poem_theme",
        "poem_meter",
        "poem_verses",
        "poem_url",
        "poet_name",
    ]
    rows = pq.read_table(path, columns=columns).to_pylist()
    grouped: dict[tuple[str, ...], list[tuple[int, dict[str, Any]]]] = {}
    for index, row in enumerate(rows):
        verses = tuple(row["poem_verses"] or ())
        if not verses or len(verses) % 2:
            raise ValueError(f"Source row {index} has an invalid poem_verses list")
        grouped.setdefault(verses, []).append((index, row))

    poems: list[PoemRecord] = []
    for verses, group in grouped.items():
        meter_counts = Counter(int(row["poem_meter"]) for _, row in group)
        top_count = max(meter_counts.values())
        top_meters = sorted(
            meter for meter, count in meter_counts.items() if count == top_count
        )
        if len(top_meters) != 1:
            indices = [index for index, _ in group]
            raise ValueError(f"Tied meter metadata for source rows {indices}")
        selected_meter = top_meters[0]
        canonical_index, canonical = max(
            group,
            key=lambda item: _metadata_score(item[1], selected_meter),
        )
        del canonical_index
        urls = tuple(
            dict.fromkeys(
                str(row["poem_url"])
                for _, row in group
                if row.get("poem_url")
            )
        )
        poems.append(
            PoemRecord(
                sample_id=poem_hash(verses),
                source_row_indices=tuple(index for index, _ in group),
                source_urls=urls,
                poet_name=str(canonical.get("poet_name") or ""),
                poem_title=canonical.get("poem_title"),
                poem_theme=canonical.get("poem_theme"),
                meter_id=selected_meter,
                meter_name=meter_name(selected_meter),
                verses=verses,
                metadata_conflict=len(meter_counts) > 1,
            )
        )
    poems.sort(key=lambda poem: poem.source_row_indices[0])
    return poems
