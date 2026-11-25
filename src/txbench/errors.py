"""Typed errors. Every error message is safe to log: no credentials, no URLs with keys."""


class TxBenchError(Exception):
    """Base class for all project errors."""


class ConfigError(TxBenchError):
    pass


class ReleaseError(TxBenchError):
    """The upstream release is missing, incomplete or fails its checksums."""


class SchemaMappingError(TxBenchError):
    """An upstream column could not be mapped to a canonical field."""


class ValidationError(TxBenchError):
    """A record failed id / hash / cross-source validation."""


class CacheError(TxBenchError):
    pass


class ProviderError(TxBenchError):
    """A provider call failed. Carries a classification, never a credential."""

    def __init__(self, message: str, *, retryable: bool = False, unsupported: bool = False,
                 status: int | None = None, retry_after: float | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.unsupported = unsupported
        self.status = status
        self.retry_after = retry_after


class BudgetExceeded(TxBenchError):
    """A configured call / CU / byte cap was reached. Not a failure of the data."""


class EvaluationError(TxBenchError):
    pass
