"""Deterministic tests for the optional USGS spatial extension."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from langchain_core.messages import AIMessage, ToolMessage

from hurricane_flood_agent.agent import FloodAssessmentAgent
from hurricane_flood_agent.agent_tools import AgentSessionState, create_flood_tools
from hurricane_flood_agent.spatial import (
    calculate_spatial_summary,
    create_spatial_hwm_map,
    load_usgs_artifact,
    normalize_high_water_marks,
    retrieve_and_save_usgs_high_water_marks,
)


def _records() -> list[dict]:
    base = {
        "event_id": 325,
        "eventName": "2022 Ian",
        "stateName": "FL",
        "countyName": "Lee County",
        "verticalDatumName": "NAVD88",
        "horizontalDatumName": "WGS84 (from Digital Map)",
        "verticalMethodName": "Level Gun",
        "hwmTypeName": "Seed line",
        "hwmQualityName": "Good: +/- 0.10 ft",
        "hwm_environment": "Coastal",
        "stillwater": 1.0,
        "survey_date": "2022-10-05T04:00:00",
        "height_above_gnd": 4.0,
        "waterbody": "Test waterbody",
    }
    return [
        {
            **base,
            "hwm_id": 1,
            "site_id": 10,
            "site_no": "FLLEE00010",
            "siteDescription": "Sanibel Island test point",
            "latitude_dd": 26.44,
            "longitude_dd": -82.08,
            "elev_ft": 8.0,
        },
        {
            **base,
            "hwm_id": 2,
            "site_id": 20,
            "site_no": "FLLEE00020",
            "siteDescription": "Fort Myers test point",
            "latitude_dd": 26.63,
            "longitude_dd": -81.87,
            "elev_ft": 10.0,
            "hwmQualityName": "Excellent: +/- 0.05 ft",
        },
        {
            **base,
            "hwm_id": 3,
            "site_id": 30,
            "site_no": "FLLEE00030",
            "siteDescription": "Cape Coral test point",
            "latitude_dd": 26.56,
            "longitude_dd": -81.95,
            "elev_ft": 6.0,
            "hwm_environment": "Riverine",
            "stillwater": 0.0,
        },
    ]


class _FakeUsgsClient:
    def fetch_high_water_marks(self, **_kwargs):
        return _records()


class _ScriptedModel:
    def __init__(self, responses):
        self.responses = list(responses)

    def bind_tools(self, tools, *, tool_choice=None):
        self.tools = list(tools)
        self.tool_choice = tool_choice
        return self

    def invoke(self, messages):
        response = self.responses.pop(0)
        return response(messages) if callable(response) else response


def _tool_call(name: str, call_id: str) -> dict:
    return {"name": name, "args": {}, "id": call_id, "type": "tool_call"}


def test_normalizes_and_summarizes_compatible_usgs_marks() -> None:
    frame, validation = normalize_high_water_marks(_records())
    summary = calculate_spatial_summary(frame)

    assert validation["record_count"] == 3
    assert validation["vertical_datum_counts"] == {"NAVD88": 3}
    assert summary["unique_site_count"] == 3
    assert summary["comparable_vertical_datum"] == "NAVD88"
    assert summary["elevation_range_ft"] == {
        "minimum": 6.0,
        "median": 8.0,
        "maximum": 10.0,
    }
    assert not summary["scientific_constraints"][
        "direct_comparison_to_noaa_mhhw_allowed"
    ]


def test_saves_loads_and_maps_usgs_artifact(tmp_path) -> None:
    saved = retrieve_and_save_usgs_high_water_marks(
        tmp_path, client=_FakeUsgsClient()
    )
    frame, metadata = load_usgs_artifact(Path(saved["artifact_path"]))
    summary = calculate_spatial_summary(frame)
    map_path = create_spatial_hwm_map(frame, summary, tmp_path / "map.png")

    assert metadata["request"]["event_id"] == 325
    assert len(frame) == 3
    assert map_path.is_file()
    assert map_path.stat().st_size > 10_000


def test_spatial_tools_use_artifacts_and_enforce_dependencies(tmp_path) -> None:
    retrieve_and_save_usgs_high_water_marks(tmp_path, client=_FakeUsgsClient())
    state = AgentSessionState(tmp_path)
    tools = {tool.name: tool for tool in create_flood_tools(state)}

    early = json.loads(tools["generate_spatial_visualization"].invoke({}))
    retrieved = json.loads(tools["retrieve_usgs_high_water_marks"].invoke({}))
    analyzed = json.loads(tools["analyze_spatial_flooding"].invoke({}))
    mapped = json.loads(tools["generate_spatial_visualization"].invoke({}))

    assert early["error_type"] == "missing_prerequisite"
    assert retrieved["cache_used"] is True
    assert analyzed["record_count"] == 3
    assert analyzed["comparable_vertical_datum"] == "NAVD88"
    assert Path(mapped["visualization_path"]).is_file()
    assert state.active_spatial_visualization_path is not None


def test_agent_can_generate_spatial_map_and_ground_answer(tmp_path) -> None:
    retrieve_and_save_usgs_high_water_marks(tmp_path, client=_FakeUsgsClient())

    def final_answer(messages):
        outputs = [
            json.loads(message.content)
            for message in messages
            if isinstance(message, ToolMessage)
        ]
        mapped = next(
            output
            for output in outputs
            if output.get("tool") == "generate_spatial_visualization"
        )
        assert Path(mapped["visualization_path"]).is_file()
        return AIMessage(
            content=(
                "The map contains three surveyed USGS point observations in NAVD88. "
                "They are not a continuous flood surface and cannot be compared "
                "directly with NOAA MHHW values or treated as road inundation depth."
            )
        )

    model = _ScriptedModel(
        [
            AIMessage(
                content="",
                tool_calls=[_tool_call("retrieve_usgs_high_water_marks", "usgs")],
            ),
            AIMessage(
                content="",
                tool_calls=[_tool_call("analyze_spatial_flooding", "spatial")],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call("generate_spatial_visualization", "spatial_map")
                ],
            ),
            final_answer,
        ]
    )
    agent = FloodAssessmentAgent(
        model=model,
        state=AgentSessionState(tmp_path),
    )

    result = agent.run("Generate a spatial map of Hurricane Ian USGS high-water marks.")

    assert result.status == "ok"
    assert result.tool_names == (
        "retrieve_usgs_high_water_marks",
        "analyze_spatial_flooding",
        "generate_spatial_visualization",
    )
    assert result.state["active_spatial_visualization_path"] is not None


def test_mixed_datums_are_not_reported_as_comparable() -> None:
    frame, _ = normalize_high_water_marks(_records())
    frame.loc[0, "vertical_datum"] = "NGVD29"

    summary = calculate_spatial_summary(frame)

    assert summary["comparable_vertical_datum"] is None
    assert summary["vertical_datum_counts"] == {"NAVD88": 2, "NGVD29": 1}
    assert summary["elevation_range_ft"] is None


def test_missing_spatial_columns_are_rejected() -> None:
    frame = pd.DataFrame({"latitude": [26.5], "longitude": [-81.9]})

    try:
        calculate_spatial_summary(frame)
    except Exception as error:
        assert "missing columns" in str(error)
    else:
        raise AssertionError("Incomplete spatial table was accepted")
