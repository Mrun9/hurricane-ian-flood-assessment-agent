"""Offline tests for Streamlit presentation helpers and application startup."""

from __future__ import annotations

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

from hurricane_flood_agent.agent import NHC_REPORT_URL
from hurricane_flood_agent.ui import (
    VERIFIED_SOURCES,
    analysis_from_execution_log,
    find_visualization,
    load_latest_saved_analysis,
    tool_events,
)


def _calculation_payload() -> dict:
    return {
        "peak_observed": {
            "water_level_ft_mhhw": 7.257,
            "timestamp_utc": "2022-09-28T22:30:00Z",
            "flood_category": "major",
        },
        "maximum_residual": {
            "residual_ft": 7.249,
            "timestamp_utc": "2022-09-28T22:18:00Z",
        },
        "observation_quality": {
            "status_counts": {"valid": 960},
            "missing_observed_value_count": 0,
        },
        "sigma": {
            "at_recorded_peak_ft": 0.217,
            "interpretation": "Variability indicator, not total uncertainty.",
        },
        "thresholds": {
            "minor": 1.69,
            "moderate": 2.66,
            "major": 3.89,
            "retrospective_caveat": "Applied retrospectively.",
        },
    }


def test_extracts_indicators_only_from_successful_real_tool_result() -> None:
    log = [
        {
            "event_type": "tool_result",
            "status": "error",
            "details": {
                "tool_name": "calculate_flood_indicators",
                "output": {"status": "error"},
            },
        },
        {
            "event_type": "tool_result",
            "status": "ok",
            "details": {
                "tool_name": "calculate_flood_indicators",
                "output": _calculation_payload(),
            },
        },
    ]

    summary = analysis_from_execution_log(log)

    assert summary is not None
    assert summary["peak_value_ft_mhhw"] == 7.257
    assert summary["maximum_residual_ft"] == 7.249
    assert summary["flood_category"] == "major"
    assert summary["valid_observation_count"] == 960
    assert summary["sigma_at_peak_ft"] == 0.217


def test_loads_independent_saved_analysis(tmp_path) -> None:
    path = tmp_path / "analysis" / "case" / "analysis_results.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"calculations": _calculation_payload()}), encoding="utf-8"
    )

    summary = load_latest_saved_analysis(tmp_path)

    assert summary is not None
    assert summary["peak_timestamp_utc"] == "2022-09-28T22:30:00Z"
    assert summary["analysis_path"] == str(path.resolve())


def test_visualization_lookup_prefers_safe_active_artifact(tmp_path) -> None:
    active = tmp_path / "analysis" / "active" / "water_levels.png"
    active.parent.mkdir(parents=True)
    active.write_bytes(b"active")
    packaged = tmp_path / "submission" / "hurricane_ian_water_levels.png"
    packaged.parent.mkdir(parents=True)
    packaged.write_bytes(b"packaged")

    selected = find_visualization(
        tmp_path,
        state_path=str(active),
        packaged_plot=packaged,
    )

    assert selected == active.resolve()


def test_tool_event_filter_excludes_user_text_and_final_answer() -> None:
    log = [
        {"event_type": "user_question"},
        {"event_type": "model_response"},
        {"event_type": "tool_call"},
        {"event_type": "tool_result"},
        {"event_type": "final_response"},
    ]

    assert [event["event_type"] for event in tool_events(log)] == [
        "model_response",
        "tool_call",
        "tool_result",
    ]


def test_verified_sources_include_literal_nhc_url() -> None:
    assert NHC_REPORT_URL in {source["url"] for source in VERIFIED_SOURCES}
    assert all(source["url"].startswith("https://") for source in VERIFIED_SOURCES)


def test_streamlit_application_starts_without_network_or_model_call() -> None:
    app_path = Path(__file__).resolve().parents[1] / "streamlit_app.py"

    app = AppTest.from_file(str(app_path)).run(timeout=20)

    assert not app.exception
    assert app.title[0].value == "Hurricane Ian Flood Assessment Agent"
    assert any("Assessment chat" in heading.value for heading in app.subheader)
    assert any("Validated flood indicators" in heading.value for heading in app.subheader)
    assert len(app.button) >= 4
