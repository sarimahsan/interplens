"""Custom Exception Hierarchy for InterpLens."""


class InterpLensError(Exception):
    """Base exception class for all InterpLens errors."""
    pass


class ModelLoadError(InterpLensError):
    """Raised when loading a model or tokenizer fails."""
    pass


class AdapterNotFoundError(InterpLensError):
    """Raised when no suitable model adapter can be resolved for a model."""
    pass


class CapabilityError(InterpLensError):
    """Raised when a model or adapter lacks required interpretability capabilities."""
    pass


class UnembeddingNotFoundError(CapabilityError):
    """Raised when unembedding matrix W_U cannot be extracted from a model."""
    pass


class ServerExecutionError(InterpLensError):
    """Raised when the debugger web server fails to start or encounter runtime errors."""
    pass
