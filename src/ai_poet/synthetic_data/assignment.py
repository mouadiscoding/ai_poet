"""Deterministic dataset-split assignment."""

from __future__ import annotations


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
