"""Run the Milestone 4 LangChain flood-assessment agent from the command line."""

from __future__ import annotations

import argparse
import json

from hurricane_flood_agent.agent import FloodAssessmentAgent
from hurricane_flood_agent.config import ConfigurationError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Question for the flood-assessment agent.")
    parser.add_argument(
        "--show-log",
        action="store_true",
        help="Print the structured chronological execution log.",
    )
    arguments = parser.parse_args()

    try:
        agent = FloodAssessmentAgent()
        result = agent.run(arguments.question)
    except ConfigurationError as error:
        print(f"CONFIGURATION ERROR: {error}")
        return 2

    print("\nANSWER\n")
    print(result.response)
    print("\nTOOLS:", ", ".join(result.tool_names) or "none")
    print("LOG:", result.log_path)
    if arguments.show_log:
        print("\nEXECUTION LOG\n")
        print(json.dumps(result.execution_log, indent=2))
    return 0 if result.status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
