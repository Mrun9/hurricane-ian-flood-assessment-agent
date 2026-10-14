"""Streamlit interface for the Hurricane Ian Flood Assessment Agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import streamlit as st

from hurricane_flood_agent.agent import FloodAssessmentAgent
from hurricane_flood_agent.agent_tools import AgentSessionState
from hurricane_flood_agent.config import ConfigurationError
from hurricane_flood_agent.ui import (
    QUESTION_1,
    QUESTION_2,
    VERIFIED_SOURCES,
    VISUALIZATION_QUESTION,
    analysis_from_execution_log,
    find_visualization,
    load_latest_saved_analysis,
    tool_events,
)


PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"
PACKAGED_PLOT = PROJECT_ROOT / "submission" / "hurricane_ian_water_levels.png"


st.set_page_config(
    page_title="Hurricane Ian Flood Assessment Agent",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _initialize_session() -> None:
    st.session_state.setdefault("chat_messages", [])
    st.session_state.setdefault("run_history", [])
    st.session_state.setdefault("latest_result", None)
    st.session_state.setdefault("agent", None)


def _agent() -> FloodAssessmentAgent:
    if st.session_state.agent is None:
        st.session_state.agent = FloodAssessmentAgent(
            state=AgentSessionState(ARTIFACT_ROOT)
        )
    return st.session_state.agent


def _clear_session() -> None:
    for key in ("chat_messages", "run_history", "latest_result", "agent"):
        st.session_state.pop(key, None)
    st.rerun()


def _run_question(question: str) -> None:
    st.session_state.chat_messages.append({"role": "user", "content": question})
    try:
        with st.spinner("NaviGator is selecting and executing tools…"):
            result = _agent().run(question)
    except ConfigurationError as error:
        st.session_state.chat_messages.append(
            {
                "role": "assistant",
                "content": f"Configuration error: {error}",
                "status": "error",
            }
        )
        return

    result_payload = result.to_dict()
    st.session_state.latest_result = result_payload
    st.session_state.run_history.append(result_payload)
    st.session_state.chat_messages.append(
        {
            "role": "assistant",
            "content": result.response,
            "status": result.status,
            "run_id": result.run_id,
        }
    )


def _render_chat() -> None:
    st.subheader("Assessment chat")
    messages = st.session_state.chat_messages
    if not messages:
        st.info(
            "Use a demonstration question or ask about the Fort Myers Hurricane Ian "
            "water-level record."
        )
    for message in messages:
        with st.chat_message(message["role"]):
            if message.get("status") == "error":
                st.error(message["content"])
            else:
                st.markdown(message["content"])


def _render_indicators(latest_result: Mapping[str, Any] | None) -> None:
    st.subheader("Validated flood indicators")
    summary = None
    if latest_result:
        summary = analysis_from_execution_log(latest_result.get("execution_log", []))
    if summary is None:
        summary = load_latest_saved_analysis(ARTIFACT_ROOT)
    if summary is None:
        st.info("Run the quantitative assessment to populate validated indicators.")
        return

    peak = summary.get("peak_value_ft_mhhw")
    residual = summary.get("maximum_residual_ft")
    category = str(summary.get("flood_category") or "Unavailable").title()
    valid_count = summary.get("valid_observation_count")
    missing_count = summary.get("missing_observation_count")

    first, second = st.columns(2)
    first.metric("Peak recorded level", _format_number(peak, " ft MHHW"))
    second.metric("Maximum residual", _format_number(residual, " ft"))
    st.caption(f"Peak: {_format_timestamp(summary.get('peak_timestamp_utc'))}")
    st.caption(
        "Residual: "
        f"{_format_timestamp(summary.get('residual_timestamp_utc'))} · "
        "observed minus predicted at the same timestamp"
    )
    third, fourth = st.columns(2)
    third.metric("Flood classification", category)
    fourth.metric(
        "Valid observations",
        str(valid_count) if valid_count is not None else "Unavailable",
        delta=(
            f"{missing_count} missing/invalid"
            if missing_count is not None
            else None
        ),
        delta_color="off",
    )
    sigma = summary.get("sigma_at_peak_ft")
    if sigma is not None:
        st.caption(
            f"NOAA sigma at the recorded peak: {float(sigma):.3f} ft. Sigma is a "
            "reported variability indicator, not total measurement uncertainty."
        )
    caveat = summary.get("threshold_caveat")
    if caveat:
        st.warning(str(caveat), icon="⚠️")


def _render_execution_log(latest_result: Mapping[str, Any] | None) -> None:
    if not latest_result:
        st.info("Submit a question to see actual LangChain execution events.")
        return

    st.caption(
        f"Run `{latest_result['run_id']}` · status: `{latest_result['status']}` · "
        f"saved log: `{latest_result['log_path']}`"
    )
    events = tool_events(latest_result.get("execution_log", []))
    if not events:
        st.info("The model answered directly; no data tool was required.")
    for entry in events:
        event_type = str(entry.get("event_type"))
        status = str(entry.get("status"))
        details = entry.get("details", {})
        if event_type == "model_response":
            selected = details.get("requested_tools", [])
            label = ", ".join(selected) if selected else "No tool selected"
            st.caption(f"Step {entry['sequence']} · Model decision: {label}")
            continue

        tool_name = details.get("tool_name")
        title = f"Step {entry['sequence']} · {event_type.replace('_', ' ').title()}"
        if tool_name:
            title += f" · `{tool_name}`"
        if status != "ok":
            title += " · ERROR"
        with st.expander(title, expanded=event_type in {"tool_call", "model_error"}):
            if event_type == "tool_call":
                st.markdown("**Validated input**")
                st.json(details.get("validated_input", {}))
            elif event_type == "tool_result":
                st.markdown("**Concise structured output**")
                st.json(details.get("output", {}))
                warnings = details.get("warnings", [])
                for warning in warnings:
                    st.warning(str(warning))
            else:
                st.json(details)

    log_bytes = json.dumps(
        latest_result.get("execution_log", []), indent=2, sort_keys=True
    ).encode("utf-8")
    st.download_button(
        "Download execution log",
        data=log_bytes,
        file_name=f"agent_run_{latest_result['run_id']}.json",
        mime="application/json",
    )


def _render_visualization(latest_result: Mapping[str, Any] | None) -> None:
    state_path = None
    if latest_result:
        state_path = latest_result.get("state", {}).get("active_visualization_path")
    plot_path = find_visualization(
        ARTIFACT_ROOT,
        state_path=state_path,
        packaged_plot=PACKAGED_PLOT,
    )
    if plot_path is None:
        st.info(
            "No plot is available yet. Click **Request visualization** to let the "
            "agent select the visualization tool, or run `python analyze_hurricane_ian.py`."
        )
        return
    st.image(
        str(plot_path),
        caption=(
            "Fort Myers station 8725520 · observed and predicted water levels · "
            "feet relative to MHHW · timestamps in UTC"
        ),
        width="stretch",
    )
    st.download_button(
        "Download PNG",
        data=plot_path.read_bytes(),
        file_name="hurricane_ian_fort_myers_water_levels.png",
        mime="image/png",
    )
    st.caption(f"Artifact: `{plot_path}`")


def _render_sources_and_limitations() -> None:
    st.markdown("#### Verified sources")
    st.caption(
        "These links are rendered deterministically. They remain available even when "
        "NaviGator cites a source by name but omits its literal URL."
    )
    for source in VERIFIED_SOURCES:
        st.markdown(
            f"- [{source['name']}]({source['url']}) — {source['purpose']}"
        )

    st.markdown("#### Scientific limitations")
    st.markdown(
        "- This is a retrospective assessment, not a real-time warning system.\n"
        "- The gauge reports water-surface elevation relative to MHHW at one station.\n"
        "- Residual is observed minus predicted tide at the same timestamp.\n"
        "- Neither gauge elevation nor residual is local inundation depth; local ground "
        "or infrastructure elevation is also required.\n"
        "- One Fort Myers gauge cannot represent all of Fort Myers Beach, Sanibel "
        "Island, or Lee County.\n"
        "- Current NOAA/NOS flood thresholds are applied retrospectively to 2022.\n"
        "- Transportation exposure statements are potential implications unless an "
        "authoritative source documents a specific impact."
    )


def _format_number(value: Any, suffix: str) -> str:
    return f"{float(value):.3f}{suffix}" if value is not None else "Unavailable"


def _format_timestamp(value: Any) -> str:
    if not value:
        return "timestamp unavailable"
    return str(value).replace("T", " ").replace("Z", " UTC")


_initialize_session()

st.title("Hurricane Ian Flood Assessment Agent")
st.caption(
    "Retrospective coastal flood evidence for NOAA Fort Myers station 8725520 · "
    "September 27–30, 2022 · UTC · feet relative to MHHW"
)

with st.sidebar:
    st.header("Two-minute demonstration")
    st.write("Each button sends a real user message to the LangChain agent.")
    question_1_clicked = st.button("Run Question 1", width="stretch")
    question_2_clicked = st.button("Run Question 2", width="stretch")
    visualization_clicked = st.button(
        "Request visualization", width="stretch"
    )
    st.divider()
    st.markdown(
        "**Case study**  \n"
        "Fort Myers · Station 8725520  \n"
        "Sep 27–30, 2022 · UTC  \n"
        "Observed + predicted · MHHW"
    )
    if st.button("Clear chat and agent state", width="stretch"):
        _clear_session()

typed_question = st.chat_input("Ask about Hurricane Ian coastal flooding…")
preset_question = None
if question_1_clicked:
    preset_question = QUESTION_1
elif question_2_clicked:
    preset_question = QUESTION_2
elif visualization_clicked:
    preset_question = VISUALIZATION_QUESTION

submitted_question = typed_question or preset_question
if submitted_question:
    _run_question(submitted_question)

latest = st.session_state.latest_result
left, right = st.columns([1.75, 1], gap="large")
with left:
    _render_chat()
with right:
    _render_indicators(latest)

visualization_tab, execution_tab, evidence_tab = st.tabs(
    ["Visualization", "Agent execution", "Evidence & limitations"]
)
with visualization_tab:
    _render_visualization(latest)
with execution_tab:
    _render_execution_log(latest)
with evidence_tab:
    _render_sources_and_limitations()

st.caption(
    "Deterministic calculations are performed in Python. The LLM selects tools and "
    "interprets their outputs; it does not calculate the flood indicators."
)
