"""Exceptions raised by the synthetic data generation pipeline."""


class GenerationError(RuntimeError):
    pass


class GemmaConnectionError(GenerationError):
    """Raised when Gemma remains unreachable after all connection retries."""


def classify_generation_failure(message: str) -> str:
    """Return a stable, low-cardinality category for generation telemetry."""
    if "validation response" in message and "invalid" in message:
        return "validator_format"
    if "revised_draft must exactly match" in message:
        return "source_copy"
    if "Gemma rejected" in message:
        return "semantic_rejection"
    if "remained invalid after repairs" in message:
        return "deterministic_contract"
    return "generation_error"
