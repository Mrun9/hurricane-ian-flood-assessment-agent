"""Run deterministic Milestone 3 calculations on saved Milestone 2 artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from hurricane_flood_agent.analysis import (
    build_analysis_document,
    calculate_flood_indicators,
    load_milestone2_artifacts,
    save_analysis_document,
)
from hurricane_flood_agent.visualization import create_water_level_plot


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ARTIFACT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "noaa"
    / "noaa_8725520_20220927_20220930_mhhw_english_gmt"
)
THRESHOLD_ARTIFACT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "thresholds"
    / "noaa_nos_thresholds_8725520_20261013"
)
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "analysis" / "hurricane_ian_fort_myers"


def main() -> None:
    inputs = load_milestone2_artifacts(DATA_ARTIFACT_DIR, THRESHOLD_ARTIFACT_DIR)
    calculations = calculate_flood_indicators(
        inputs.observations,
        inputs.predictions,
        inputs.thresholds,
        interval_minutes=inputs.metadata["request"]["interval_minutes"],
    )
    document = build_analysis_document(inputs, calculations)
    result_path = save_analysis_document(
        document, OUTPUT_DIR / "analysis_results.json"
    )
    station = inputs.metadata["station"]
    plot_path = create_water_level_plot(
        inputs.observations,
        inputs.predictions,
        calculations,
        station_name=station["name"],
        station_id=station["id"],
        output_path=OUTPUT_DIR / "water_levels.png",
    )

    summary = {
        "peak_observed": calculations["peak_observed"],
        "maximum_residual": calculations["maximum_residual"],
        "threshold_exceedances": calculations["threshold_exceedances"],
        "sigma": calculations["sigma"],
        "analysis_path": str(result_path),
        "visualization_path": str(plot_path),
    }
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
