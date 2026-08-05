"""Compose trusted SFT responses from generated reasoning and source poems."""

from __future__ import annotations

import re

from .poems import PoemRecord


def compose_response(reasoning: str, poem: PoemRecord) -> str:
    """Combine cleaned editorial reasoning with the exact source poem.

    Any occurrence of the fully formatted poem is removed from the generated
    reasoning. The cleanup also drops lines that exactly duplicate a source
    couplet or hemistich, repeated final-result markers, and standalone headers
    that would introduce a model-generated poem. Excess blank lines are
    collapsed before one canonical Arabic final-result marker and the source
    poem are appended verbatim. This guarantees that training targets end with
    trusted source text rather than a potentially altered model reproduction.

    Args:
        reasoning: Model-generated editorial analysis, possibly containing
            redundant poem text or result headings.
        poem: Canonical source poem to append.

    Returns:
        The cleaned reasoning followed by ``النتيجة النهائية:`` and the
        formatted source poem.
    """
    marker = "النتيجة النهائية:"
    reasoning = reasoning.replace(poem.poem_text, "")

    source_lines = {
        line.strip() for line in poem.poem_text.splitlines() if line.strip()
    }
    source_lines.update(verse.strip() for verse in poem.verses if verse.strip())
    standalone_poem_headers = {
        "القصيدة:",
        "القصيدة النهائية:",
        "النص النهائي:",
    }
    retained_lines = []
    for line in reasoning.splitlines():
        stripped = line.strip()
        if stripped == marker or stripped in source_lines:
            continue
        if stripped in standalone_poem_headers:
            continue
        retained_lines.append(line.rstrip())

    editorial_reasoning = "\n".join(retained_lines).strip()
    editorial_reasoning = re.sub(r"\n{3,}", "\n\n", editorial_reasoning)
    return (
        f"{editorial_reasoning}\n\n{marker}\n\n{poem.poem_text}"
    )
