"""Offline tests for Streamlit presentation helpers and application startup."""

from __future__ import annotations

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

from hurricane_flood_agent.agent import NHC_REPORT_URL
from hurricane_flood_agent.ui import (
    VERIFIED_SOURCES,
    analysis_from_execution_log,
    conversation_document,
    conversation_json_bytes,
    find_spatial_visualization,
    find_visualization,
    generated_visualizations,
    load_latest_saved_analysis,
    load_latest_saved_spatial_analysis,
    save_conversation,
    spatial_summary_from_execution_log,
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


def test_spatial_summary_and_visualization_are_loaded_safely(tmp_path) -> None:
    analysis_path = (
        tmp_path
        / "spatial_analysis"
        / "spatial_analysis_usgs"
        / "spatial_analysis_results.json"
    )
    analysis_path.parent.mkdir(parents=True)
    summary = {
        "record_count": 257,
        "unique_site_count": 200,
        "vertical_datum_counts": {"NAVD88": 257},
        "elevation_range_ft": {"minimum": 2.16, "median": 7.0, "maximum": 13.8},
    }
    analysis_path.write_text(json.dumps({"summary": summary}), encoding="utf-8")
    map_path = analysis_path.parent / "high_water_marks_map.png"
    map_path.write_bytes(b"map")
    log = [
        {
            "event_type": "tool_result",
            "status": "ok",
            "details": {
                "tool_name": "analyze_spatial_flooding",
                "output": summary,
            },
        },
        {
            "event_type": "tool_result",
            "status": "ok",
            "details": {
                "tool_name": "generate_spatial_visualization",
                "output": {"visualization_path": str(map_path)},
            },
        },
    ]

    assert spatial_summary_from_execution_log(log)["record_count"] == 257
    assert load_latest_saved_spatial_analysis(tmp_path)["unique_site_count"] == 200
    assert find_spatial_visualization(tmp_path, state_path=str(map_path)) == map_path
    assert generated_visualizations(log) == [
        {"kind": "spatial_map", "path": str(map_path)}
    ]


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


def test_conversation_can_be_saved_and_downloaded(tmp_path) -> None:
    conversation_id = "a" * 32
    messages = [
        {"role": "user", "content": "What was the peak?"},
        {"role": "assistant", "content": "7.257 ft MHHW."},
    ]
    runs = [{"run_id": "run-1", "status": "ok"}]

    document = conversation_document(
        conversation_id,
        messages,
        runs,
        saved_at_utc="2026-10-14T12:00:00Z",
    )
    saved_path = save_conversation(tmp_path, conversation_id, messages, runs)
    downloaded = json.loads(conversation_json_bytes(conversation_id, messages, runs))

    assert document["message_count"] == 2
    assert document["saved_at_utc"] == "2026-10-14T12:00:00Z"
    assert json.loads(saved_path.read_text(encoding="utf-8"))["messages"] == messages
    assert downloaded["agent_runs"] == runs
    assert saved_path.parent == tmp_path.resolve() / "conversations"


def test_conversation_identifier_rejects_path_traversal() -> None:
    try:
        conversation_document("../../outside", [], [])
    except ValueError as error:
        assert "32-character hexadecimal" in str(error)
    else:
        raise AssertionError("Unsafe conversation identifier was accepted")


def test_streamlit_application_starts_without_network_or_model_call() -> None:
    app_path = Path(__file__).resolve().parents[1] / "streamlit_app.py"

    app = AppTest.from_file(str(app_path)).run(timeout=20)

    assert not app.exception
    assert app.title[0].value == "Hurricane Ian Flood Assessment Agent"
    assert any("Ask the agent" in heading.value for heading in app.subheader)
    assert any("Flood indicators" in heading.value for heading in app.subheader)
    assert any("How severe was the flooding?" == button.label for button in app.button)
    assert any("New conversation" == button.label for button in app.button)
    assert [tab.label for tab in app.tabs] == [
        "Chat",
        "Analysis",
        "Visualization",
        "Agent activity",
        "Sources & limitations",
    ]

    app.session_state["chat_messages"] = [
        {"role": "user", "content": "Saved question"},
        {"role": "assistant", "content": "Saved answer", "status": "ok"},
    ]
    app.run(timeout=20)

    assert any(
        button.label == "Download conversation" for button in app.download_button
    )
