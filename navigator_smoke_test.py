"""Live smoke test for LangChain, UF NaviGator, and automatic tool choice."""

import sys

from langchain_core.tools import tool

from hurricane_flood_agent.config import ConfigurationError
from hurricane_flood_agent.llm import create_chat_model

TEST_CASES = [
    {
        "question": "Retrieve NOAA data for Fort Myers station 8725520.",
        "expect_tool": True,
    },
    {
        "question": "Explain why a tide gauge cannot directly measure street flood depth.",
        "expect_tool": False,
    },
]


@tool
def retrieve_noaa_data(station_id: str) -> str:
    """Retrieve historical water levels from NOAA for a station ID."""

    return f"Smoke-test placeholder for NOAA station {station_id}."


def main() -> int:
    try:
        model = create_chat_model()
    except ConfigurationError as error:
        print(f"CONFIGURATION ERROR: {error}")
        return 2

    model_with_tools = model.bind_tools([retrieve_noaa_data], tool_choice="auto")
    failures = 0

    for test in TEST_CASES:
        question = test["question"]
        expect_tool = test["expect_tool"]
        print(f"\nQUESTION: {question}")

        try:
            message = model_with_tools.invoke(
                [
                    {
                        "role": "system",
                        "content": (
                            "Use tools when external data is necessary. "
                            "Otherwise answer without using tools."
                        ),
                    },
                    {"role": "user", "content": question},
                ]
            )
        except Exception as error:
            print(f"API ERROR: {type(error).__name__}: {error}")
            failures += 1
            continue

        selected_tool = bool(message.tool_calls)

        if selected_tool:
            for call in message.tool_calls:
                print(f"TOOL SELECTED: {call['name']}")
                print("ARGUMENTS:", call["args"])
        else:
            print("NO TOOL SELECTED")
            print("ANSWER:", message.content)

        if selected_tool == expect_tool:
            print("RESULT: PASS")
        else:
            expected = "a tool call" if expect_tool else "a direct answer"
            print(f"RESULT: FAIL (expected {expected})")
            failures += 1

    passed = len(TEST_CASES) - failures
    print(f"\nSUMMARY: {passed}/{len(TEST_CASES)} tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
