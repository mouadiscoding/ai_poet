"""Canonical Arabic poetry meter vocabulary."""

from __future__ import annotations


METER_NAMES = (
    "البسيط",
    "الخفيف",
    "الرجز",
    "الرمل",
    "السريع",
    "الطويل",
    "الكامل",
    "المتدارك",
    "المتقارب",
    "المجتث",
    "المديد",
    "المضارع",
    "المقتضب",
    "المنسرح",
    "النثر",
    "الهزج",
    "الوافر",
)


def meter_name(meter_id: int) -> str:
    """Resolve a numeric meter identifier to its canonical Arabic name.

    Meter identifiers are zero-based indices into :data:`METER_NAMES`. Boolean
    values technically satisfy Python's ``int`` check and therefore resolve as
    indices in the same way as integers.

    Args:
        meter_id: Zero-based index of a supported poetic meter.

    Returns:
        The Arabic meter name at the requested index.

    Raises:
        ValueError: If the identifier is not an integer or lies outside the
            configured meter-name table.
    """
    if not isinstance(meter_id, int) or not 0 <= meter_id < len(METER_NAMES):
        raise ValueError(f"Unknown meter ID: {meter_id}")
    return METER_NAMES[meter_id]
