"""Domain-specific exceptions for external data retrieval and validation."""


class NoaaError(RuntimeError):
    """Base class for NOAA-related failures."""


class NoaaApiError(NoaaError):
    """Raised when NOAA cannot be reached or returns an API error."""


class NoaaValidationError(NoaaError):
    """Raised when NOAA data does not satisfy the expected scientific contract."""


class UsgsError(RuntimeError):
    """Base class for USGS high-water-mark failures."""


class UsgsApiError(UsgsError):
    """Raised when the USGS STN service cannot be reached or parsed."""


class UsgsValidationError(UsgsError):
    """Raised when USGS records violate the requested scientific contract."""
