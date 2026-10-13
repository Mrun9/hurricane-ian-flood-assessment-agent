"""Configuration loading for the flood-assessment application."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


DEFAULT_BASE_URL = "https://api.ai.it.ufl.edu/v1"
DEFAULT_MODEL = "mistral-small-3.1"


class ConfigurationError(ValueError):
    """Raised when required application configuration is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    """Runtime settings shared by the model and future application layers."""

    navigator_api_key: str
    navigator_base_url: str = DEFAULT_BASE_URL
    navigator_model: str = DEFAULT_MODEL


def load_settings(
    *,
    env_file: str | Path | None = ".env",
    require_api_key: bool = True,
) -> Settings:
    """Load settings from a local ``.env`` file and process environment.

    Existing process environment variables take precedence over values in the
    file. Passing ``env_file=None`` skips file loading, which is useful in tests.
    """

    if env_file is not None:
        env_path = Path(env_file)
        if env_path.is_file():
            load_dotenv(dotenv_path=env_path, override=False)

    api_key = os.getenv("NAVIGATOR_TOOLKIT_API_KEY", "").strip()
    base_url = os.getenv("NAVIGATOR_BASE_URL", DEFAULT_BASE_URL).strip().rstrip("/")
    model = os.getenv("NAVIGATOR_MODEL", DEFAULT_MODEL).strip()

    if require_api_key and not api_key:
        raise ConfigurationError(
            "NAVIGATOR_TOOLKIT_API_KEY is not set. Add it to your shell or a "
            "local .env file; never commit the credential."
        )
    if not base_url.startswith("https://"):
        raise ConfigurationError("NAVIGATOR_BASE_URL must use HTTPS.")
    if not model:
        raise ConfigurationError("NAVIGATOR_MODEL cannot be empty.")

    return Settings(
        navigator_api_key=api_key,
        navigator_base_url=base_url,
        navigator_model=model,
    )
