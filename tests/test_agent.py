"""Deterministic tests for LangChain tools, state, feedback, and logs."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from hurricane_flood_agent.agent import (
    FloodAssessmentAgent,
    NHC_REPORT_URL,
    _redact,
)
from hurricane_flood_agent.agent_tools import AgentSessionState, create_flood_tools
from hurricane_flood_agent.artifacts import ArtifactStore
from hurricane_flood_agent.schemas import FloodThresholdSet, NoaaDataRequest


class ScriptedModel:
    """Small model double that still exercises LangChain messages and tools."""

    def __init__(self, responses: list[AIMessage | Callable[[list[Any]], AIMessage]]) -> None:
        self.responses = responses
        self.bound_tools = []
        self.tool_choice = None

    def bind_tools(self, tools, *, tool_choice=None):
        self.bound_tools = list(tools)
        self.tool_choice = tool_choice
        return self

    def invoke(self, messages):
        response = self.responses.pop(0)
        return response(messages) if callable(response) else response


def _build_synthetic_artifacts(root: Path) -> tuple[str, str]:
    request = NoaaDataRequest(
        station_id="8725520",
        begin_date=date(2022, 9, 27),
        end_date=date(2022, 9, 27),
    )
    timestamps = pd.date_range("2022-09-27", periods=4, freq="6min", tz="UTC")
    observations = pd.DataFrame(
        {
            "timestamp": timestamps,
            "value": [1.0, 2.0, 6.0, 3.0],
            "sigma": [0.01, 0.02, 0.03, 0.02],
            "flags": ["0,0,0,0"] * 4,
            "quality": ["v"] * 4,
            "status": ["valid"] * 4,
        }
    )
    predictions = pd.DataFrame(
        {
            "timestamp": timestamps,
            "value": [0.5, 1.0, 1.0, 2.0],
            "sigma": [float("nan")] * 4,
            "flags": [None] * 4,
            "quality": [None] * 4,
            "status": ["valid"] * 4,
        }
    )
    store = ArtifactStore(root)
    data_result = store.save_water_level_data(
        request=request,
        station={"id": "8725520", "name": "Fort Myers"},
        observed_raw={"data": []},
        predicted_raw={"predictions": []},
        observed=observations,
        predicted=predictions,
        validation={
            "observations": {
                "valid_value_count": 4,
                "expected_record_count": 4,
                "missing_or_invalid_count": 0,
                "quality_code_counts": {"v": 4},
                "warnings": [],
            },
            "predictions": {
                "valid_value_count": 4,
                "expected_record_count": 4,
                "missing_or_invalid_count": 0,
                "quality_code_counts": {"not_applicable": 4},
                "warnings": [],
            },
        },
    )
    thresholds = FloodThresholdSet(
        station_id="8725520",
        station_name="Fort Myers",
        datum="MHHW",
        units="feet",
        minor=1.64,
        moderate=3.28,
        major=4.92,
        retrieved_at_utc="2022-10-01T00:00:00+00:00",
        source_urls=("https://example.invalid/noaa-threshold-source",),
    )
    threshold_result = store.save_thresholds(
        thresholds,
        {
            "mhhw_above_station_datum_meters": 1.0,
            "absolute_nos_thresholds_above_station_datum_meters": {
                "minor": 1.5,
                "moderate": 2.0,
                "major": 2.5,
            },
            "raw_datums_response": {
                "max": 2.8288,
                "maxdate": "20220927",
                "maxtime": "00:12",
            },
        },
    )
    return data_result["artifact_id"], threshold_result["artifact_id"]


def _tool_call(name: str, args: dict[str, Any], call_id: str) -> dict[str, Any]:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def test_four_tool_schemas_and_dependency_errors(tmp_path) -> None:
    state = AgentSessionState(tmp_path)
    tools = {tool.name: tool for tool in create_flood_tools(state)}

    assert list(tools) == [
        "retrieve_noaa_data",
        "calculate_flood_indicators",
        "get_flood_thresholds",
        "generate_visualization",
    ]
    assert set(tools["retrieve_noaa_data"].args) == {
        "station_id",
        "begin_date",
        "end_date",
        "datum",
        "units",
        "use_cached",
    }
    navigator_schemas = [convert_to_openai_tool(tool) for tool in tools.values()]
    assert all(schema["type"] == "function" for schema in navigator_schemas)
    assert {schema["function"]["name"] for schema in navigator_schemas} == set(tools)
    retrieve_schema = next(
        schema
        for schema in navigator_schemas
        if schema["function"]["name"] == "retrieve_noaa_data"
    )
    assert retrieve_schema["function"]["parameters"]["required"] == [
        "station_id",
        "begin_date",
        "end_date",
    ]

    calculation_error = json.loads(tools["calculate_flood_indicators"].invoke({}))
    plot_error = json.loads(tools["generate_visualization"].invoke({}))
    assert calculation_error["error_type"] == "missing_prerequisite"
    assert calculation_error["suggested_tools"] == [
        "retrieve_noaa_data",
        "get_flood_thresholds",
    ]
    assert plot_error["suggested_tools"] == ["calculate_flood_indicators"]


def test_multi_step_tool_feedback_state_and_chronological_logs(tmp_path) -> None:
    data_id, threshold_id = _build_synthetic_artifacts(tmp_path)
    state = AgentSessionState(tmp_path)

    def final_response(messages) -> AIMessage:
        tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
        outputs = [json.loads(message.content) for message in tool_messages]
        analysis = next(
            output for output in outputs if output.get("tool") == "calculate_flood_indicators"
        )
        plot = next(
            output for output in outputs if output.get("tool") == "generate_visualization"
        )
        assert analysis["peak_observed"]["water_level_ft_mhhw"] == 6.0
        assert analysis["maximum_residual"]["residual_ft"] == 5.0
        assert Path(plot["visualization_path"]).is_file()
        return AIMessage(
            content=(
                "The validated peak was 6.0 ft above MHHW and the exact-time residual "
                "was 5.0 ft, classified as major flooding. Low-lying roads and "
                "causeways may have been exposed. A single gauge cannot represent "
                "every location. Gauge water-surface elevation and residual are not "
                "local inundation depth, which requires local ground or infrastructure "
                "elevation. Sigma is a variability indicator, not total measurement "
                f"uncertainty. Source: {NHC_REPORT_URL}"
            )
        )

    model = ScriptedModel(
        [
            AIMessage(
                content=(
                    "Fort Myers station 8725510 recorded 6.43 ft above MHHW."
                ),
                response_metadata={
                    "model_name": "mistral-small-3.1",
                    "finish_reason": "stop",
                },
            ),
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call(
                        "retrieve_noaa_data",
                        {
                            "station_id": "8725520",
                            "begin_date": "2022-09-27",
                            "end_date": "2022-09-27",
                        },
                        "call_data",
                    ),
                    _tool_call(
                        "get_flood_thresholds",
                        {"station_id": "8725520"},
                        "call_thresholds",
                    ),
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call("calculate_flood_indicators", {}, "call_analysis")
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call("generate_visualization", {}, "call_plot")
                ],
            ),
            final_response,
        ]
    )
    agent = FloodAssessmentAgent(model=model, state=state)

    result = agent.run(
        "How severe was coastal flooding in Lee County, Florida, during Hurricane Ian?"
    )

    assert model.tool_choice == "auto"
    assert [tool.name for tool in model.bound_tools] == [
        "retrieve_noaa_data",
        "calculate_flood_indicators",
        "get_flood_thresholds",
        "generate_visualization",
    ]
    assert result.status == "ok"
    assert result.tool_names == (
        "retrieve_noaa_data",
        "get_flood_thresholds",
        "calculate_flood_indicators",
        "generate_visualization",
    )
    assert result.state["active_data_artifact_id"] == data_id
    assert result.state["active_threshold_artifact_id"] == threshold_id
    assert result.state["active_visualization_path"] is not None
    assert Path(result.log_path).is_file()
    assert [entry["sequence"] for entry in result.execution_log] == list(
        range(1, len(result.execution_log) + 1)
    )
    event_types = [entry["event_type"] for entry in result.execution_log]
    assert event_types[0] == "user_question"
    assert event_types[1] == "tool_configuration"
    assert event_types[-1] == "final_response"
    assert event_types.count("grounding_rejection") == 1
    assert event_types.count("tool_call") == 4
    assert event_types.count("tool_result") == 4
    assert isinstance(agent.messages[0], SystemMessage)
    assert not any(
        isinstance(message, SystemMessage) for message in agent.messages[1:]
    )
    validator_feedback = [
        message
        for message in agent.messages
        if isinstance(message, HumanMessage)
        and isinstance(message.content, str)
        and message.content.startswith("Automatic validator feedback:")
    ]
    assert len(validator_feedback) == 1
    rejection_events = [
        entry
        for entry in result.execution_log
        if entry["event_type"] == "grounding_rejection"
    ]
    assert all(entry["details"]["proposed_response"] for entry in rejection_events)
    assert "local inundation depth" in result.response
    first_model_event = next(
        entry for entry in result.execution_log if entry["event_type"] == "model_response"
    )
    assert first_model_event["details"]["requested_tools"] == []
    assert first_model_event["details"]["provider_metadata"] == {
        "model_name": "mistral-small-3.1",
        "finish_reason": "stop",
    }
def test_peak_question_requires_retrieval_thresholds_and_calculation(tmp_path) -> None:
    _build_synthetic_artifacts(tmp_path)
    model = ScriptedModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call(
                        "retrieve_noaa_data",
                        {
                            "station_id": "8725520",
                            "begin_date": "2022-09-27",
                            "end_date": "2022-09-27",
                        },
                        "peak_data",
                    ),
                    _tool_call(
                        "get_flood_thresholds",
                        {"station_id": "8725520"},
                        "peak_thresholds",
                    ),
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call("calculate_flood_indicators", {}, "peak_analysis")
                ],
            ),
            AIMessage(
                content=(
                    "The deterministic result is a 6.0 ft MHHW peak and a 5.0 ft "
                    "matching-timestamp residual."
                )
            ),
        ]
    )
    agent = FloodAssessmentAgent(model=model, state=AgentSessionState(tmp_path))

    result = agent.run("What was the peak water level during Hurricane Ian?")

    assert result.status == "ok"
    assert result.tool_names == (
        "retrieve_noaa_data",
        "get_flood_thresholds",
        "calculate_flood_indicators",
    )
    assert "6.0 ft" in result.response
    assert "5.0 ft" in result.response


def test_visualization_request_recovers_prerequisites_and_creates_plot(tmp_path) -> None:
    _build_synthetic_artifacts(tmp_path)
    state = AgentSessionState(tmp_path)

    def final_plot_answer(messages) -> AIMessage:
        outputs = [
            json.loads(message.content)
            for message in messages
            if isinstance(message, ToolMessage)
        ]
        plot = next(
            output
            for output in reversed(outputs)
            if output.get("tool") == "generate_visualization"
            and output.get("status") == "ok"
        )
        assert Path(plot["visualization_path"]).is_file()
        return AIMessage(content=f"The plot was generated at {plot['visualization_path']}.")

    model = ScriptedModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call("generate_visualization", {}, "plot_too_early")
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call(
                        "retrieve_noaa_data",
                        {
                            "station_id": "8725520",
                            "begin_date": "2022-09-27",
                            "end_date": "2022-09-27",
                        },
                        "plot_data",
                    ),
                    _tool_call(
                        "get_flood_thresholds",
                        {"station_id": "8725520"},
                        "plot_thresholds",
                    ),
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call("calculate_flood_indicators", {}, "plot_analysis")
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call("generate_visualization", {}, "plot_retry")
                ],
            ),
            final_plot_answer,
        ]
    )
    agent = FloodAssessmentAgent(model=model, state=state)

    result = agent.run("Generate a visualization of the Hurricane Ian water levels.")

    assert result.status == "ok"
    assert result.tool_names == (
        "generate_visualization",
        "retrieve_noaa_data",
        "get_flood_thresholds",
        "calculate_flood_indicators",
        "generate_visualization",
    )
    assert state.active_visualization_path is not None
    assert state.active_visualization_path.is_file()
    plot_results = [
        entry
        for entry in result.execution_log
        if entry["event_type"] == "tool_result"
        and entry["details"]["tool_name"] == "generate_visualization"
    ]
    assert [entry["status"] for entry in plot_results] == ["error", "ok"]


def test_conceptual_answer_can_skip_all_tools(tmp_path) -> None:
    model = ScriptedModel(
        [
            AIMessage(
                content=(
                    "No. NOAA station 8725520 measures water-surface elevation at one "
                    "point; street depth also requires the local ground elevation."
                )
            )
        ]
    )
    agent = FloodAssessmentAgent(
        model=model,
        state=AgentSessionState(tmp_path),
    )

    result = agent.run(
        "Can the Fort Myers tide-gauge observations tell us how deep flooding was "
        "in Fort Myers Beach or Sanibel Island? Why or why not?"
    )

    assert result.status == "ok"
    assert result.tool_names == ()
    assert not any(
        entry["event_type"] == "tool_call" for entry in result.execution_log
    )


def test_tool_input_error_is_returned_to_model_and_logged(tmp_path) -> None:
    def answer_after_error(messages) -> AIMessage:
        tool_message = next(
            message for message in messages if isinstance(message, ToolMessage)
        )
        output = json.loads(tool_message.content)
        assert output["status"] == "error"
        return AIMessage(content="The station identifier was invalid, so no claim was made.")

    model = ScriptedModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call(
                        "retrieve_noaa_data",
                        {
                            "station_id": "bad",
                            "begin_date": "2022-09-27",
                            "end_date": "2022-09-30",
                        },
                        "call_bad",
                    )
                ],
            ),
            answer_after_error,
        ]
    )
    agent = FloodAssessmentAgent(model=model, state=AgentSessionState(tmp_path))

    result = agent.run("Retrieve data for an invalid station.")

    tool_result = next(
        entry for entry in result.execution_log if entry["event_type"] == "tool_result"
    )
    assert tool_result["status"] == "error"
    assert "no claim was made" in result.response


def test_log_redaction_masks_credential_patterns() -> None:
    fake_openai_style_key = "sk-" + "example123456"
    redacted = _redact(
        {
            "api_key": "not-safe",
            "message": (
                "Authorization: Bearer abcdefghijklmnop and "
                + fake_openai_style_key
            ),
        }
    )

    assert redacted["api_key"] == "[REDACTED]"
    assert "abcdefghijklmnop" not in redacted["message"]
    assert fake_openai_style_key not in redacted["message"]
