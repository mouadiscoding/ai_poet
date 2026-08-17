"""Canonical poem representation and pure poem transformations."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Sequence


@dataclass(frozen=True)
class PoemRecord:
    sample_id: str
    source_row_indices: tuple[int, ...]
    source_urls: tuple[str, ...]
    poet_name: str
    poem_title: str | None
    poem_theme: str | None
    meter_id: int
    meter_name: str
    verses: tuple[str, ...]
    metadata_conflict: bool

    @property
    def couplet_count(self) -> int:
        """Return the number of complete verse pairs stored in the record.

        ``verses`` stores each hemistich as a separate item, with consecutive
        items forming one couplet. Source validation guarantees an even number
        of items, so the count is exactly half the sequence length.
        """
        return len(self.verses) // 2

    @property
    def poem_text(self) -> str:
        """Return the poem in the line-oriented format used by SFT prompts.

        Each pair of hemistichs is joined with ``" = "``, and the resulting
        couplets are separated by newlines. Formatting is delegated to
        :func:`format_poem`, which also enforces the non-empty/even invariant.
        """
        return format_poem(self.verses)



def poem_hash(verses: Sequence[str]) -> str:
    """Compute a stable content identifier for an ordered verse sequence.

    Hemistichs are joined with the Unicode record-separator symbol ``U+241E``
    before hashing, which preserves item boundaries that ordinary string
    concatenation would lose. The digest depends only on verse content and
    order, not on metadata such as poet, title, URL, or meter.

    Args:
        verses: Hemistich strings in their original order.

    Returns:
        The lowercase hexadecimal SHA-256 digest of the joined UTF-8 text.
    """
    joined = "\u241e".join(verses)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def format_poem(verses: Sequence[str]) -> str:
    """Format alternating hemistichs as newline-separated couplets.

    Items at positions ``0`` and ``1`` form the first couplet, positions ``2``
    and ``3`` form the second, and so on. The two sides are separated by
    ``" = "`` to match the representation used in prompts and output records.

    Args:
        verses: A non-empty, even-length sequence of hemistich strings.

    Returns:
        A single string containing one formatted couplet per line.

    Raises:
        ValueError: If ``verses`` is empty or contains an odd number of items.
    """
    if not verses or len(verses) % 2:
        raise ValueError("poem_verses must contain a non-empty even number of items")
    return "\n".join(
        f"{verses[index]} = {verses[index + 1]}"
        for index in range(0, len(verses), 2)
    )



def split_poem_chunks(verses: Sequence[str], max_chars: int) -> list[str]:
    """Partition a poem into size-limited chunks without splitting couplets.

    Each adjacent hemistich pair is first formatted as ``left = right``. Lines
    are accumulated until adding the next complete couplet, including its
    separating newline, would exceed ``max_chars``. A single couplet longer
    than the limit remains intact in its own oversized chunk because preserving
    semantic and metrical pairs takes precedence over the character target.

    Args:
        verses: Alternating left and right hemistichs. Callers are expected to
            supply a complete, even-length sequence.
        max_chars: Maximum preferred character count per chunk.

    Returns:
        Formatted poem chunks in source order. An empty sequence produces an
        empty list.

    Raises:
        ValueError: If ``max_chars`` is zero or negative.
        IndexError: If ``verses`` contains an unmatched final hemistich.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    chunks: list[str] = []
    current: list[str] = []
    current_chars = 0
    for index in range(0, len(verses), 2):
        line = f"{verses[index]} = {verses[index + 1]}"
        added = len(line) + int(bool(current))
        if current and current_chars + added > max_chars:
            chunks.append("\n".join(current))
            current = []
            current_chars = 0
        current.append(line)
        current_chars += len(line) + int(len(current) > 1)
    if current:
        chunks.append("\n".join(current))
    return chunks
