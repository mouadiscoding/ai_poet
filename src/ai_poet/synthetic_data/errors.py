"""Exceptions raised by the synthetic data generation pipeline."""


class GenerationError(RuntimeError):
    pass


class GemmaConnectionError(GenerationError):
    """Raised when Gemma remains unreachable after all connection retries."""
