"""LangChain model construction for UF NaviGator AI Toolkit."""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from hurricane_flood_agent.config import Settings, load_settings


def create_chat_model(settings: Settings | None = None) -> ChatOpenAI:
    """Create the shared NaviGator chat model without embedding credentials.

    Keeping model construction here prevents scientific tools from depending on
    a particular model provider. Those tools will receive and return ordinary
    Python data independently of this class.
    """

    active_settings = settings or load_settings()
    return ChatOpenAI(
        model=active_settings.navigator_model,
        api_key=active_settings.navigator_api_key,
        base_url=active_settings.navigator_base_url,
        temperature=0,
        max_retries=2,
        timeout=60,
    )
