"""Live UF NaviGator integration checks for the two required demo questions."""

from __future__ import annotations

import json
import sys

from hurricane_flood_agent.agent import FloodAssessmentAgent
from hurricane_flood_agent.config import ConfigurationError


QUESTIONS = (
    "How severe was coastal flooding in Lee County, Florida, during Hurricane Ian?",
    (
        "Can the Fort Myers tide-gauge observations tell us how deep flooding was "
        "in Fort Myers Beach or Sanibel Island? Why or why not?"
    ),
)


def main() -> int:
    failures = 0
    results = []
    for index, question in enumerate(QUESTIONS, start=1):
        try:
            # Each professor-required question is an independent live test case.
            # A fresh conversation prevents one case's revision history from
            # influencing the other case's tool-selection decision.
            agent = FloodAssessmentAgent()
        except ConfigurationError as error:
            print(f"CONFIGURATION ERROR: {error}")
            return 2
        print(f"\nQUESTION {index}: {question}")
        result = agent.run(question)
        results.append(result)
        print("TOOLS:", ", ".join(result.tool_names) or "none")
        print("ANSWER:", result.response)
        print("LOG:", result.log_path)

        if result.status != "ok":
            print("RESULT: FAIL (agent run failed)")
            failures += 1
            continue

        response_lower = result.response.lower()
        if index == 1:
            required_tools = {
                "retrieve_noaa_data",
                "get_flood_thresholds",
                "calculate_flood_indicators",
            }
            passed = (
                required_tools.issubset(set(result.tool_names))
                and ("7.257" in result.response or "7.26" in result.response)
                and ("7.249" in result.response or "7.25" in result.response)
                and "8725520" in result.response
                and "8725510" not in result.response
                and "6.43" not in result.response
                and "major" in response_lower
                and "mhhw" in response_lower
                and (
                    "national hurricane center" in response_lower
                    or "nhc" in response_lower
                )
            )
        else:
            passed = (
                not result.tool_names
                and ("cannot" in response_lower or "not" in response_lower)
                and "elevation" in response_lower
                and ("ground" in response_lower or "topograph" in response_lower)
            )

        event_sequences = [entry["sequence"] for entry in result.execution_log]
        event_types = [entry["event_type"] for entry in result.execution_log]
        if event_sequences != list(range(1, len(event_sequences) + 1)):
            passed = False
            print("LOG FAILURE: execution events are not chronological")
        if index == 1:
            successful_results = {
                entry["details"].get("tool_name")
                for entry in result.execution_log
                if entry["event_type"] == "tool_result" and entry["status"] == "ok"
            }
            if not required_tools.issubset(successful_results):
                passed = False
                print("LOG FAILURE: required tools lack successful result events")
        elif "tool_call" in event_types:
            passed = False
            print("LOG FAILURE: conceptual answer unexpectedly called a tool")

        serialized_log = json.dumps(result.execution_log)
        if "sk-" in serialized_log:
            passed = False
            print("SECURITY FAILURE: possible credential pattern in log")
        if passed:
            print("RESULT: PASS")
        else:
            print("RESULT: FAIL (selection or answer checks failed)")
            failures += 1

    print(f"\nSUMMARY: {len(results) - failures}/{len(QUESTIONS)} live tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
