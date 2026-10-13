"""Tests for the LangChain model factory without making network calls."""

from unittest.mock import Mock

import hurricane_flood_agent.llm as llm_module
from hurricane_flood_agent.config import Settings


def test_create_chat_model_passes_navigator_configuration(monkeypatch) -> None:
    chat_openai = Mock(return_value=Mock(name="navigator_model"))
    monkeypatch.setattr(llm_module, "ChatOpenAI", chat_openai)
    settings = Settings(
        navigator_api_key="placeholder-test-key",
        navigator_base_url="https://navigator.example/v1",
        navigator_model="test-model",
    )

    result = llm_module.create_chat_model(settings)

    assert result is chat_openai.return_value
    chat_openai.assert_called_once_with(
        model="test-model",
        api_key="placeholder-test-key",
        base_url="https://navigator.example/v1",
        temperature=0,
        max_retries=2,
        timeout=60,
    )
