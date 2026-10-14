"""Pure presentation helpers shared by the Streamlit app and UI tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from hurricane_flood_agent.agent import (
    NHC_REPORT_URL,
    NHC_UPDATE_URL,
    NOAA_RESPONSE_HELP_URL,
)


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
)


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
