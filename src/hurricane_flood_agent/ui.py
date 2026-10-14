"""Pure presentation helpers shared by the Streamlit app and UI tests."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from hurricane_flood_agent.agent import (
    NHC_REPORT_URL,
    NHC_UPDATE_URL,
    NOAA_RESPONSE_HELP_URL,
)
from hurricane_flood_agent.spatial import USGS_EVENT_URL, USGS_HWM_URL


QUESTION_1 = (
    "How severe was coastal flooding in Lee County, Florida, during Hurricane Ian?"
)
QUESTION_2 = (
    "Can the Fort Myers tide-gauge observations tell us how deep flooding was "
    "in Fort Myers Beach or Sanibel Island? Why or why not?"
)
VISUALIZATION_QUESTION = (
    "Generate a visualization of observed and predicted water levels for Hurricane "
    "Ian using the validated Fort Myers analysis."
)
SPATIAL_QUESTION = (
    "Using official USGS Hurricane Ian high-water marks, assess spatial variation "
    "across Lee County and explain the datum and inundation-depth limitations."
)
SPATIAL_VISUALIZATION_QUESTION = (
    "Generate a spatial map of official USGS Hurricane Ian high-water marks across "
    "Lee County and summarize what the point observations can and cannot show."
)

NOAA_DATA_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
NOAA_THRESHOLD_URL = (
    "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/"
    "8725520.json?expand=details,floodlevels&units=metric"
)

VERIFIED_SOURCES: tuple[dict[str, str], ...] = (
    {
        "name": "NOAA Tides and Currents Data API",
        "url": NOAA_DATA_URL,
        "purpose": "Observed six-minute water levels and predicted tides.",
    },
    {
        "name": "NOAA/NOS Fort Myers flood-level metadata",
        "url": NOAA_THRESHOLD_URL,
        "purpose": "Station datum and coastal flood thresholds.",
    },
    {
        "name": "NHC Hurricane Ian Tropical Cyclone Report",
        "url": NHC_REPORT_URL,
        "purpose": "Independent official Hurricane Ian impact context.",
    },
    {
        "name": "NHC contemporaneous Hurricane Ian update",
        "url": NHC_UPDATE_URL,
        "purpose": "Event-time Fort Myers-area NOS context.",
    },
    {
        "name": "NOAA CO-OPS response field definitions",
        "url": NOAA_RESPONSE_HELP_URL,
        "purpose": "Definitions for water-level quality and sigma fields.",
    },
    {
        "name": "USGS Hurricane Ian Flood Event Viewer",
        "url": USGS_EVENT_URL,
        "purpose": "Official event context and spatial observation viewer.",
    },
    {
        "name": "USGS STN high-water-mark service",
        "url": USGS_HWM_URL,
        "purpose": "Surveyed Lee County high-water-mark records and metadata.",
    },
)


def conversation_document(
    conversation_id: str,
    chat_messages: Sequence[Mapping[str, Any]],
    run_history: Sequence[Mapping[str, Any]],
    *,
    saved_at_utc: str | None = None,
) -> dict[str, Any]:
    """Create a portable conversation record without credentials or hidden reasoning."""

    if not re.fullmatch(r"[a-f0-9]{32}", conversation_id):
        raise ValueError("conversation_id must be a 32-character hexadecimal identifier")
    timestamp = saved_at_utc or datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    return {
        "schema_version": 1,
        "conversation_id": conversation_id,
        "saved_at_utc": timestamp,
        "message_count": len(chat_messages),
        "messages": [dict(message) for message in chat_messages],
        "agent_runs": [dict(run) for run in run_history],
    }


def conversation_json_bytes(
    conversation_id: str,
    chat_messages: Sequence[Mapping[str, Any]],
    run_history: Sequence[Mapping[str, Any]],
) -> bytes:
    """Serialize a conversation for a user-controlled durable download."""

    return json.dumps(
        conversation_document(conversation_id, chat_messages, run_history),
        indent=2,
        sort_keys=True,
    ).encode("utf-8")


def save_conversation(
    artifact_root: Path,
    conversation_id: str,
    chat_messages: Sequence[Mapping[str, Any]],
    run_history: Sequence[Mapping[str, Any]],
) -> Path:
    """Atomically save the current conversation in the ignored artifact directory."""

    document = conversation_document(conversation_id, chat_messages, run_history)
    destination_dir = Path(artifact_root).resolve() / "conversations"
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"conversation_{conversation_id}.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(destination)
    return destination


def latest_successful_tool_output(
    execution_log: Sequence[Mapping[str, Any]],
    tool_name: str,
) -> dict[str, Any] | None:
    """Return the most recent successful structured result for one tool."""

    for entry in reversed(execution_log):
        if entry.get("event_type") != "tool_result" or entry.get("status") != "ok":
            continue
        details = entry.get("details", {})
        if details.get("tool_name") != tool_name:
            continue
        output = details.get("output")
        if isinstance(output, dict):
            return output
    return None


def indicator_summary(calculations: Mapping[str, Any]) -> dict[str, Any] | None:
    """Flatten validated calculations into values suitable for compact metric cards."""

    peak = calculations.get("peak_observed")
    residual = calculations.get("maximum_residual")
    quality = calculations.get("observation_quality", {})
    thresholds = calculations.get("thresholds", {})
    sigma = calculations.get("sigma", {})
    if not isinstance(peak, Mapping) or not isinstance(residual, Mapping):
        return None

    valid_count = quality.get("valid_observation_count")
    if valid_count is None:
        status_counts = quality.get("status_counts", {})
        if isinstance(status_counts, Mapping):
            valid_count = status_counts.get("valid")

    return {
        "peak_value_ft_mhhw": peak.get("water_level_ft_mhhw"),
        "peak_timestamp_utc": peak.get("timestamp_utc"),
        "flood_category": peak.get("flood_category"),
        "maximum_residual_ft": residual.get("residual_ft"),
        "residual_timestamp_utc": residual.get("timestamp_utc"),
        "valid_observation_count": valid_count,
        "missing_observation_count": quality.get("missing_observed_value_count"),
        "sigma_at_peak_ft": sigma.get("at_recorded_peak_ft"),
        "thresholds": {
            name: thresholds.get(name) for name in ("minor", "moderate", "major")
        },
        "threshold_caveat": thresholds.get("retrospective_caveat"),
        "sigma_interpretation": sigma.get("interpretation"),
    }


def analysis_from_execution_log(
    execution_log: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Extract the calculation payload returned by the real analysis tool."""

    output = latest_successful_tool_output(
        execution_log, "calculate_flood_indicators"
    )
    return indicator_summary(output) if output is not None else None


def spatial_summary_from_execution_log(
    execution_log: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Extract the deterministic summary returned by the spatial-analysis tool."""

    output = latest_successful_tool_output(execution_log, "analyze_spatial_flooding")
    if output is None:
        return None
    required = {"record_count", "vertical_datum_counts", "elevation_range_ft"}
    return dict(output) if required.issubset(output) else None


def generated_visualizations(
    execution_log: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Return only visualization artifacts generated successfully in this run."""

    supported = {
        "generate_visualization": "time_series",
        "generate_spatial_visualization": "spatial_map",
    }
    results: list[dict[str, str]] = []
    for entry in execution_log:
        if entry.get("event_type") != "tool_result" or entry.get("status") != "ok":
            continue
        details = entry.get("details", {})
        kind = supported.get(str(details.get("tool_name")))
        output = details.get("output", {})
        path = output.get("visualization_path") if isinstance(output, Mapping) else None
        if kind and isinstance(path, str):
            results.append({"kind": kind, "path": path})
    return results


def load_latest_saved_analysis(artifact_root: Path) -> dict[str, Any] | None:
    """Load the newest independently saved deterministic analysis artifact."""

    analysis_root = Path(artifact_root).resolve() / "analysis"
    candidates = [
        path
        for path in analysis_root.glob("*/analysis_results.json")
        if path.is_file() and not path.name.startswith("._")
    ]
    for path in sorted(candidates, key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        calculations = document.get("calculations") if isinstance(document, dict) else None
        if isinstance(calculations, dict):
            summary = indicator_summary(calculations)
            if summary is not None:
                summary["analysis_path"] = str(path.resolve())
                return summary
    return None


def load_latest_saved_spatial_analysis(artifact_root: Path) -> dict[str, Any] | None:
    """Load the newest independently saved USGS spatial-analysis artifact."""

    analysis_root = Path(artifact_root).resolve() / "spatial_analysis"
    candidates = [
        path
        for path in analysis_root.glob("*/spatial_analysis_results.json")
        if path.is_file() and not path.name.startswith("._")
    ]
    for path in sorted(candidates, key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        summary = document.get("summary") if isinstance(document, dict) else None
        if isinstance(summary, dict):
            result = dict(summary)
            result["analysis_path"] = str(path.resolve())
            return result
    return None


def find_visualization(
    artifact_root: Path,
    *,
    state_path: str | None = None,
    packaged_plot: Path | None = None,
) -> Path | None:
    """Find a safe generated PNG, preferring the active agent artifact."""

    root = Path(artifact_root).resolve()
    candidates: list[Path] = []
    if state_path:
        candidates.append(Path(state_path))
    analysis_root = root / "analysis"
    generated = [
        path
        for path in analysis_root.glob("*/water_levels.png")
        if path.is_file() and not path.name.startswith("._")
    ]
    candidates.extend(
        sorted(generated, key=lambda item: item.stat().st_mtime, reverse=True)
    )
    if packaged_plot is not None:
        candidates.append(Path(packaged_plot))

    allowed_roots = [root]
    if packaged_plot is not None:
        allowed_roots.append(Path(packaged_plot).resolve().parent)
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.suffix.lower() != ".png" or not resolved.is_file():
            continue
        if any(_is_relative_to(resolved, allowed) for allowed in allowed_roots):
            return resolved
    return None


def find_spatial_visualization(
    artifact_root: Path,
    *,
    state_path: str | None = None,
    packaged_plot: Path | None = None,
) -> Path | None:
    """Find a safe generated USGS map within the spatial artifact root."""

    root = Path(artifact_root).resolve()
    candidates: list[Path] = []
    if state_path:
        candidates.append(Path(state_path))
    analysis_root = root / "spatial_analysis"
    generated = [
        path
        for path in analysis_root.glob("*/high_water_marks_map.png")
        if path.is_file() and not path.name.startswith("._")
    ]
    candidates.extend(
        sorted(generated, key=lambda item: item.stat().st_mtime, reverse=True)
    )
    if packaged_plot is not None:
        candidates.append(Path(packaged_plot))
    allowed_roots = [root]
    if packaged_plot is not None:
        allowed_roots.append(Path(packaged_plot).resolve().parent)
    for candidate in candidates:
        resolved = candidate.resolve()
        if (
            resolved.suffix.lower() == ".png"
            and resolved.is_file()
            and any(_is_relative_to(resolved, allowed) for allowed in allowed_roots)
        ):
            return resolved
    return None


def tool_events(
    execution_log: Iterable[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Select user-visible execution events without exposing model reasoning."""

    visible_types = {
        "model_response",
        "tool_call",
        "tool_result",
        "invalid_tool_call",
        "tool_call_recovery_requested",
        "grounding_rejection",
        "model_error",
        "step_limit",
    }
    return [entry for entry in execution_log if entry.get("event_type") in visible_types]


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
