"""Tests for environment-based configuration."""

import pytest

from hurricane_flood_agent.config import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    ConfigurationError,
    load_settings,
)


def test_load_settings_uses_safe_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NAVIGATOR_TOOLKIT_API_KEY", "placeholder-test-key")
    monkeypatch.delenv("NAVIGATOR_BASE_URL", raising=False)
    monkeypatch.delenv("NAVIGATOR_MODEL", raising=False)

    settings = load_settings(env_file=None)

    assert settings.navigator_api_key == "placeholder-test-key"
    assert settings.navigator_base_url == DEFAULT_BASE_URL
    assert settings.navigator_model == DEFAULT_MODEL


def test_load_settings_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NAVIGATOR_TOOLKIT_API_KEY", raising=False)

    with pytest.raises(ConfigurationError, match="NAVIGATOR_TOOLKIT_API_KEY"):
        load_settings(env_file=None)


def test_load_settings_rejects_non_https_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NAVIGATOR_TOOLKIT_API_KEY", "placeholder-test-key")
    monkeypatch.setenv("NAVIGATOR_BASE_URL", "http://example.invalid/v1")

    with pytest.raises(ConfigurationError, match="HTTPS"):
        load_settings(env_file=None)
