"""Domain-specific exceptions for external data retrieval and validation."""


class NoaaError(RuntimeError):
    """Base class for NOAA-related failures."""


class NoaaApiError(NoaaError):
    """Raised when NOAA cannot be reached or returns an API error."""


class NoaaValidationError(NoaaError):
    """Raised when NOAA data does not satisfy the expected scientific contract."""
