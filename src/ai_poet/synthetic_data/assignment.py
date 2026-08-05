"""Deterministic dataset-split and prompt-family assignment."""

from __future__ import annotations

from .poems import PoemRecord
from .prompts.families import TemplateFamily, eligible_families


def sft_split(sample_id: str) -> str:
    """Assign a sample deterministically to the train, validation, or test set.

    The first eight hexadecimal digits of the content-derived sample ID are
    mapped into one hundred buckets. Buckets 0--97 are training data, bucket 98
    is validation data, and bucket 99 is test data, yielding a stable 98/1/1
    allocation that does not depend on input ordering.

    Args:
        sample_id: Hexadecimal sample identifier, normally from
            :func:`poem_hash`.

    Returns:
        ``"train"``, ``"validation"``, or ``"test"``.

    Raises:
        ValueError: If the first eight characters are not valid hexadecimal.
    """
    bucket = int(sample_id[:8], 16) % 100
    if bucket < 98:
        return "train"
    if bucket == 98:
        return "validation"
    return "test"


def choose_family(poem: PoemRecord) -> TemplateFamily:
    """Choose a deterministic prompt-template family for a poem.

    Eligibility is determined by the poem's meter; notably, prose poems cannot
    use the prosody-and-rhyme family. The next eight hexadecimal digits of the
    sample ID select uniformly by modulo from the eligible tuple, so duplicate
    content always receives the same template across runs.

    Args:
        poem: Canonical poem whose meter and sample ID drive selection.

    Returns:
        The selected :class:`TemplateFamily`.

    Raises:
        ValueError: If the relevant sample-ID characters are not hexadecimal.
    """
    families = eligible_families(poem.meter_name)
    offset = int(poem.sample_id[8:16], 16) % len(families)
    return families[offset]
