"""Streamlit interface for the Hurricane Ian Flood Assessment Agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import streamlit as st

from hurricane_flood_agent.agent import FloodAssessmentAgent
from hurricane_flood_agent.agent_tools import AgentSessionState
from hurricane_flood_agent.config import ConfigurationError
from hurricane_flood_agent.ui import (
    QUESTION_1,
    QUESTION_2,
    SPATIAL_QUESTION,
    SPATIAL_VISUALIZATION_QUESTION,
    VERIFIED_SOURCES,
    VISUALIZATION_QUESTION,
    analysis_from_execution_log,
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


PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"
PACKAGED_PLOT = PROJECT_ROOT / "submission" / "hurricane_ian_water_levels.png"
PACKAGED_SPATIAL_PLOT = (
    PROJECT_ROOT / "submission" / "hurricane_ian_lee_county_usgs_hwm_map.png"
)


st.set_page_config(
    page_title="Hurricane Ian Flood Assessment Agent",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def _apply_theme() -> None:
    """Apply a minimal light treatment without replacing native Streamlit patterns."""

    st.markdown(
        """
        <style>
        .stApp {
            background: #f7f9fa;
        }

        .block-container {
            max-width: 1220px;
            padding-top: 1.7rem;
            padding-bottom: 2.5rem;
        }

        [data-testid="stHeader"] {
            background: #f7f9fa;
        }

        h1 {
            font-size: 2.15rem !important;
            letter-spacing: -0.025em;
            margin-bottom: 0.25rem !important;
        }

        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #dfe5e8;
            border-radius: 6px;
            padding: 0.85rem 1rem;
            min-height: 104px;
        }

        .stButton > button,
        .stDownloadButton > button {
            background: #ffffff;
            border: 1px solid #cfd9dd;
            border-radius: 5px;
            color: #183247;
            font-weight: 600;
        }

        [data-testid="stBaseButton-primary"] {
            background: #087f79 !important;
            border-color: #087f79 !important;
            color: #ffffff !important;
        }

        [data-testid="stTabs"] [data-baseweb="tab-list"] {
            gap: 1.6rem;
        }

        [data-testid="stTabs"] [data-baseweb="tab"] {
            padding-left: 0;
            padding-right: 0;
        }

        [data-testid="stChatMessage"] {
            background: #ffffff;
            border: 1px solid #e1e6e8;
            border-radius: 6px;
            margin-bottom: 0.65rem;
        }

        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid #e2e7e9;
        }

        [data-testid="stExpander"] {
            background: #ffffff;
            border-color: #dfe5e8;
            border-radius: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _initialize_session() -> None:
    st.session_state.setdefault("chat_messages", [])
    st.session_state.setdefault("run_history", [])
    st.session_state.setdefault("latest_result", None)
    st.session_state.setdefault("agent", None)
    st.session_state.setdefault("conversation_id", uuid4().hex)
    st.session_state.setdefault("conversation_path", None)
    st.session_state.setdefault("conversation_save_error", None)


def _agent() -> FloodAssessmentAgent:
    if st.session_state.agent is None:
        st.session_state.agent = FloodAssessmentAgent(
            state=AgentSessionState(ARTIFACT_ROOT)
        )
    return st.session_state.agent


def _new_conversation() -> None:
    for key in (
        "chat_messages",
        "run_history",
        "latest_result",
        "agent",
        "conversation_id",
        "conversation_path",
        "conversation_save_error",
    ):
        st.session_state.pop(key, None)
    st.rerun()


def _persist_conversation() -> None:
    """Save the visible transcript and structured runs without exposing secrets."""

    try:
        path = save_conversation(
            ARTIFACT_ROOT,
            st.session_state.conversation_id,
            st.session_state.chat_messages,
            st.session_state.run_history,
        )
    except OSError as error:
        st.session_state.conversation_save_error = str(error)
        return
    st.session_state.conversation_path = str(path)
    st.session_state.conversation_save_error = None


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
        _persist_conversation()
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
            "visualizations": generated_visualizations(result.execution_log),
        }
    )
    _persist_conversation()


def _render_chat() -> None:
    st.subheader("Ask the agent")
    messages = st.session_state.chat_messages
    if not messages:
        st.info(
            "No conversation yet. Choose a suggested question or enter your own."
        )
    for message in messages:
        with st.chat_message(message["role"]):
            if message.get("status") == "error":
                st.error(message["content"])
            else:
                st.markdown(message["content"])
            for item in message.get("visualizations", []):
                if item.get("kind") == "time_series":
                    image_path = find_visualization(
                        ARTIFACT_ROOT,
                        state_path=item.get("path"),
                        packaged_plot=PACKAGED_PLOT,
                    )
                    caption = "NOAA observed water level and predicted tide"
                elif item.get("kind") == "spatial_map":
                    image_path = find_spatial_visualization(
                        ARTIFACT_ROOT,
                        state_path=item.get("path"),
                        packaged_plot=PACKAGED_SPATIAL_PLOT,
                    )
                    caption = "USGS Hurricane Ian high-water marks in Lee County"
                else:
                    image_path = None
                    caption = "Generated visualization"
                if image_path is not None:
                    st.image(str(image_path), caption=caption, width="stretch")


def _render_indicators(latest_result: Mapping[str, Any] | None) -> None:
    st.subheader("Flood indicators")
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

    first, second, third, fourth = st.columns(4, gap="medium")
    with first:
        st.metric("Peak recorded level", _format_number(peak, " ft"))
        st.caption(f"MHHW · {_format_timestamp(summary.get('peak_timestamp_utc'))}")
    with second:
        st.metric("Maximum residual", _format_number(residual, " ft"))
        st.caption(
            "Observed − predicted · "
            f"{_format_timestamp(summary.get('residual_timestamp_utc'))}"
        )
    with third:
        st.metric("Flood classification", category)
        st.caption("Based on verified NOAA/NOS thresholds")
    with fourth:
        quality_value = (
            f"{valid_count} / 960" if valid_count is not None else "Unavailable"
        )
        st.metric("Valid observations", quality_value)
        st.caption(
            f"{missing_count} missing or invalid records"
            if missing_count is not None
            else "Quality count unavailable"
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


def _render_spatial_summary(latest_result: Mapping[str, Any] | None) -> None:
    summary = None
    if latest_result:
        summary = spatial_summary_from_execution_log(
            latest_result.get("execution_log", [])
        )
    if summary is None:
        summary = load_latest_saved_spatial_analysis(ARTIFACT_ROOT)
    if summary is None:
        st.info(
            "Run the USGS spatial assessment from Chat or generate the spatial map "
            "from the Visualization tab."
        )
        return

    elevation_range = summary.get("elevation_range_ft", {})
    datum = summary.get("comparable_vertical_datum") or "Mixed"
    first, second, third, fourth = st.columns(4)
    first.metric("Surveyed marks", str(summary.get("record_count", "Unavailable")))
    second.metric("Unique sites", str(summary.get("unique_site_count", "Unavailable")))
    third.metric("Vertical datum", str(datum))
    if elevation_range:
        range_value = (
            f"{float(elevation_range['minimum']):.2f}–"
            f"{float(elevation_range['maximum']):.2f} ft"
        )
    else:
        range_value = "Unavailable"
    fourth.metric("Elevation range", range_value)
    constraints = summary.get("scientific_constraints", {})
    reason = constraints.get("reason") if isinstance(constraints, Mapping) else None
    if reason:
        st.warning(str(reason), icon="⚠️")


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
            "No plot is available yet. Click **Create / update plot** to let the "
            "agent select the visualization tool, or run `python analyze_hurricane_ian.py`."
        )
        return
    st.subheader("Observed water level and predicted tide")
    st.caption(
        "Recorded water surface compared with the astronomical tide at matching "
        "six-minute timestamps. Thresholds are feet above MHHW."
    )
    st.image(
        str(plot_path),
        caption=(
            "Fort Myers station 8725520 · observed and predicted water levels · "
            "feet relative to MHHW · timestamps in UTC"
        ),
        width="stretch",
    )
    download_col, artifact_col = st.columns([1, 3], vertical_alignment="center")
    with download_col:
        st.download_button(
            "Download plot (PNG)",
            data=plot_path.read_bytes(),
            file_name="hurricane_ian_fort_myers_water_levels.png",
            mime="image/png",
            width="stretch",
        )
    with artifact_col:
        st.caption(f"Reproducible artifact: `{plot_path}`")


def _render_spatial_visualization(
    latest_result: Mapping[str, Any] | None,
) -> None:
    state_path = None
    if latest_result:
        state_path = latest_result.get("state", {}).get(
            "active_spatial_visualization_path"
        )
    plot_path = find_spatial_visualization(
        ARTIFACT_ROOT,
        state_path=state_path,
        packaged_plot=PACKAGED_SPATIAL_PLOT,
    )
    if plot_path is None:
        st.info(
            "No USGS spatial map is available yet. Generate it to retrieve and "
            "analyze the official Hurricane Ian high-water marks."
        )
        return
    st.subheader("USGS high-water-mark map")
    st.caption(
        "Surveyed point elevations for event 2022 Ian in Lee County. Color values "
        "use the datum stated on the map and are not local road flood depths."
    )
    st.image(str(plot_path), width="stretch")
    st.download_button(
        "Download spatial map (PNG)",
        data=plot_path.read_bytes(),
        file_name="hurricane_ian_lee_county_usgs_hwm_map.png",
        mime="image/png",
    )
    st.caption(f"Reproducible artifact: `{plot_path}`")


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

    st.markdown("#### Spatial extension")
    st.info(
        "The optional USGS high-water-mark extension is implemented as separate "
        "retrieval, analysis, and mapping tools. The validated marks use NAVD88; they "
        "are not numerically combined with the NOAA MHHW gauge series. Estimating "
        "inundation at a road or building still requires compatible local ground or "
        "infrastructure elevations."
    )


def _render_conversation_controls() -> None:
    st.sidebar.subheader("Conversation")
    message_count = len(st.session_state.chat_messages)
    st.sidebar.caption(f"{message_count} saved message{'s' if message_count != 1 else ''}")

    if message_count:
        payload = conversation_json_bytes(
            st.session_state.conversation_id,
            st.session_state.chat_messages,
            st.session_state.run_history,
        )
        st.sidebar.download_button(
            "Download conversation",
            data=payload,
            file_name=f"conversation_{st.session_state.conversation_id}.json",
            mime="application/json",
            width="stretch",
        )
    if st.sidebar.button("New conversation", width="stretch"):
        _new_conversation()

    if st.session_state.conversation_save_error:
        st.sidebar.warning("The local conversation copy could not be saved.")
    elif st.session_state.conversation_path:
        st.sidebar.caption("Auto-saved to the local artifacts folder.")
    st.sidebar.caption(
        "Community Cloud storage may reset. Download the JSON for a durable copy."
    )


def _format_number(value: Any, suffix: str) -> str:
    return f"{float(value):.3f}{suffix}" if value is not None else "Unavailable"


def _format_timestamp(value: Any) -> str:
    if not value:
        return "timestamp unavailable"
    return str(value).replace("T", " ").replace("Z", " UTC")


_initialize_session()
_apply_theme()

st.title("Hurricane Ian Flood Assessment Agent")
st.caption(
    "Fort Myers NOAA station 8725520 · September 27–30, 2022 · UTC · feet above MHHW"
)

with st.sidebar:
    st.subheader("Case study")
    st.markdown(
        "**Hurricane Ian**  \n"
        "Fort Myers, Florida  \n"
        "Station 8725520  \n"
        "Observed and predicted water levels"
    )
    st.divider()
_render_conversation_controls()

chat_tab, analysis_tab, visualization_tab, activity_tab, sources_tab = st.tabs(
    [
        "Chat",
        "Analysis",
        "Visualization",
        "Agent activity",
        "Sources & limitations",
    ]
)

with chat_tab:
    st.caption("Suggested questions")
    question_one_col, question_two_col, spatial_question_col = st.columns(3)
    with question_one_col:
        question_1_clicked = st.button(
            "How severe was the flooding?", type="primary", width="stretch"
        )
    with question_two_col:
        question_2_clicked = st.button(
            "Can the gauge measure local flood depth?", width="stretch"
        )
    with spatial_question_col:
        spatial_question_clicked = st.button(
            "How did flooding vary across Lee County?", width="stretch"
        )

    _render_chat()
    typed_question = st.chat_input(
        "Ask about Hurricane Ian coastal flooding…", key="agent_question"
    )
    preset_question = None
    if question_1_clicked:
        preset_question = QUESTION_1
    elif question_2_clicked:
        preset_question = QUESTION_2
    elif spatial_question_clicked:
        preset_question = SPATIAL_QUESTION
    submitted_question = typed_question or preset_question
    if submitted_question:
        _run_question(submitted_question)
        st.rerun()

latest = st.session_state.latest_result

with analysis_tab:
    st.header("Flood analysis")
    st.caption("Deterministic results from validated NOAA and USGS artifacts.")
    st.markdown("#### Fort Myers tide gauge")
    _render_indicators(latest)
    st.divider()
    st.markdown("#### Lee County high-water marks")
    _render_spatial_summary(latest)

with visualization_tab:
    st.header("Visualization")
    st.caption(
        "Generate figures through the agent or inspect previously validated artifacts."
    )
    time_series_col, spatial_map_col = st.columns(2)
    with time_series_col:
        time_series_clicked = st.button(
            "Generate NOAA time-series plot", width="stretch"
        )
    with spatial_map_col:
        spatial_map_clicked = st.button(
            "Generate USGS spatial map", width="stretch"
        )
    if time_series_clicked:
        _run_question(VISUALIZATION_QUESTION)
        st.rerun()
    if spatial_map_clicked:
        _run_question(SPATIAL_VISUALIZATION_QUESTION)
        st.rerun()

    selected_visualization = st.radio(
        "Visualization type",
        ["NOAA time series", "USGS spatial map"],
        horizontal=True,
        label_visibility="collapsed",
    )
    if selected_visualization == "NOAA time series":
        _render_visualization(latest)
    else:
        _render_spatial_visualization(latest)

with activity_tab:
    st.header("Agent activity")
    st.caption(
        "Chronological model decisions, validated tool inputs, structured outputs, "
        "warnings, and errors. Private reasoning and API keys are never shown."
    )
    _render_execution_log(latest)

with sources_tab:
    st.header("Sources and limitations")
    _render_sources_and_limitations()

st.divider()
st.caption(
    "Python performs the scientific calculations. NaviGator selects tools and "
    "interprets structured results. This is a retrospective assessment, not an "
    "emergency warning system."
)
