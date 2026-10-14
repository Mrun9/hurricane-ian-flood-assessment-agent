"""Deterministic flood calculations using validated Milestone 2 artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


FEET_PER_METER = 3.28084
THRESHOLD_NAMES = ("minor", "moderate", "major")


class FloodAnalysisError(ValueError):
    """Raised when analysis inputs cannot support a defensible calculation."""


@dataclass(frozen=True)
class AnalysisInputs:
    """Loaded and cross-validated inputs for deterministic flood analysis."""

    observations: pd.DataFrame
    predictions: pd.DataFrame
    metadata: dict[str, Any]
    thresholds: dict[str, Any]
    threshold_provenance: dict[str, Any]
    integrity: dict[str, Any]


def load_milestone2_artifacts(
    data_artifact_dir: str | Path,
    threshold_artifact_dir: str | Path,
) -> AnalysisInputs:
    """Load and cross-check saved NOAA data and threshold artifacts.

    No network access is performed. The function verifies the data-file hashes,
    station, vertical datum, units, sampling interval, and the independently
    derivable NOS threshold conversion before returning any tables.
    """

    data_dir = Path(data_artifact_dir)
    threshold_dir = Path(threshold_artifact_dir)
    metadata = _read_json(data_dir / "metadata.json")
    thresholds = _read_json(threshold_dir / "thresholds.json")
    provenance = _read_json(threshold_dir / "source_responses.json")

    files = metadata.get("files", {})
    observation_path = data_dir / str(files.get("observations", "observations.csv"))
    prediction_path = data_dir / str(files.get("predictions", "predictions.csv"))

    hash_checks: dict[str, bool] = {}
    for filename, expected_hash in metadata.get("sha256", {}).items():
        path = data_dir / filename
        if not path.is_file():
            raise FloodAnalysisError(f"Required artifact file is missing: {path}")
        hash_checks[filename] = _sha256(path) == expected_hash
    failed_hashes = [name for name, valid in hash_checks.items() if not valid]
    if failed_hashes:
        raise FloodAnalysisError(
            "Artifact integrity check failed for: " + ", ".join(failed_hashes)
        )

    observations = _read_series_csv(observation_path, "observations")
    predictions = _read_series_csv(prediction_path, "predictions")
    request = metadata.get("request", {})

    station_id = str(metadata.get("station", {}).get("id", ""))
    if station_id != str(request.get("station_id", "")):
        raise FloodAnalysisError("Station identifiers disagree within data metadata.")
    if station_id != str(thresholds.get("station_id", "")):
        raise FloodAnalysisError("Data and threshold artifacts use different stations.")
    if request.get("datum") != thresholds.get("datum"):
        raise FloodAnalysisError("Data and threshold artifacts use different datums.")
    if request.get("unit_label") != thresholds.get("units"):
        raise FloodAnalysisError("Data and threshold artifacts use different units.")
    if request.get("time_zone") != "gmt":
        raise FloodAnalysisError("Analysis requires NOAA timestamps requested in GMT/UTC.")

    interval_minutes = request.get("interval_minutes")
    if not isinstance(interval_minutes, int) or interval_minutes <= 0:
        raise FloodAnalysisError("Metadata contains an invalid sampling interval.")

    derived_thresholds = derive_thresholds_from_provenance(provenance)
    for name in THRESHOLD_NAMES:
        artifact_value = _finite_float(thresholds.get(name), f"threshold {name}")
        if not np.isclose(artifact_value, derived_thresholds[name], atol=0.005):
            raise FloodAnalysisError(
                f"Saved {name} threshold does not match its source conversion."
            )

    _validate_threshold_order(thresholds)
    integrity = {
        "all_data_file_hashes_match": all(hash_checks.values()),
        "data_file_hash_checks": hash_checks,
        "station_datum_and_units_match": True,
        "threshold_source_conversion_matches": True,
        "network_access_used": False,
    }
    return AnalysisInputs(
        observations=observations,
        predictions=predictions,
        metadata=metadata,
        thresholds=thresholds,
        threshold_provenance=provenance,
        integrity=integrity,
    )


def derive_thresholds_from_provenance(
    provenance: Mapping[str, Any],
) -> dict[str, float]:
    """Reproduce the saved NOS station-datum to MHHW conversion."""

    absolute = provenance.get("absolute_nos_thresholds_above_station_datum_meters")
    if not isinstance(absolute, Mapping):
        raise FloodAnalysisError("Threshold provenance lacks absolute NOS elevations.")
    mhhw = _finite_float(
        provenance.get("mhhw_above_station_datum_meters"),
        "MHHW station-datum elevation",
    )
    return {
        name: round(
            (_finite_float(absolute.get(name), f"absolute {name} threshold") - mhhw)
            * FEET_PER_METER,
            2,
        )
        for name in THRESHOLD_NAMES
    }


def calculate_flood_indicators(
    observations: pd.DataFrame,
    predictions: pd.DataFrame,
    thresholds: Mapping[str, Any],
    *,
    interval_minutes: int,
) -> dict[str, Any]:
    """Calculate peaks, exact-time residuals, exceedances, and sigma summaries."""

    if interval_minutes <= 0:
        raise FloodAnalysisError("interval_minutes must be positive.")
    _validate_threshold_order(thresholds)
    observed = _prepare_series(observations, "observations")
    predicted = _prepare_series(predictions, "predictions")

    valid_observed = observed.loc[observed["value"].notna()].copy()
    if valid_observed.empty:
        raise FloodAnalysisError("No valid observed water levels are available.")

    peak_index = valid_observed["value"].idxmax()
    peak_row = valid_observed.loc[peak_index]
    peak_timestamp = peak_row["timestamp"]
    peak_value = float(peak_row["value"])

    matched = observed[["timestamp", "value"]].merge(
        predicted[["timestamp", "value"]],
        on="timestamp",
        how="outer",
        suffixes=("_observed", "_predicted"),
        indicator=True,
        validate="one_to_one",
    )
    exact_time_rows = matched.loc[matched["_merge"] == "both"].copy()
    valid_residual_rows = exact_time_rows.loc[
        exact_time_rows["value_observed"].notna()
        & exact_time_rows["value_predicted"].notna()
    ].copy()
    if valid_residual_rows.empty:
        raise FloodAnalysisError(
            "No exact timestamp has both a valid observation and prediction."
        )
    valid_residual_rows["residual"] = (
        valid_residual_rows["value_observed"]
        - valid_residual_rows["value_predicted"]
    )
    residual_index = valid_residual_rows["residual"].idxmax()
    residual_row = valid_residual_rows.loc[residual_index]

    exceedances = {
        name: _summarize_exceedance(
            observed,
            threshold=float(thresholds[name]),
            interval_minutes=interval_minutes,
        )
        for name in THRESHOLD_NAMES
    }
    classification = _classify_peak(peak_value, thresholds)
    sigma_summary = _summarize_sigma(observed, peak_timestamp)

    observation_only = matched.loc[matched["_merge"] == "left_only", "timestamp"]
    prediction_only = matched.loc[matched["_merge"] == "right_only", "timestamp"]
    quality_counts = (
        observed["quality"].fillna("missing").value_counts().sort_index().to_dict()
        if "quality" in observed
        else {}
    )
    flag_counts = (
        observed["flags"].fillna("missing").value_counts().sort_index().to_dict()
        if "flags" in observed
        else {}
    )
    status_counts = (
        observed["status"].fillna("missing").value_counts().sort_index().to_dict()
        if "status" in observed
        else {}
    )
    has_saved_quality_fields = {"quality", "flags", "status"}.issubset(observed.columns)
    all_records_meet_saved_validity_criteria = (
        bool(
            observed["quality"].fillna("missing").eq("v").all()
            and observed["flags"].fillna("missing").eq("0,0,0,0").all()
            and observed["status"].fillna("missing").eq("valid").all()
            and observed["value"].notna().all()
        )
        if has_saved_quality_fields
        else None
    )

    return {
        "sampling_interval_minutes": interval_minutes,
        "peak_observed": {
            "water_level_ft_mhhw": peak_value,
            "timestamp_utc": _iso_utc(peak_timestamp),
            "flood_category": classification,
        },
        "maximum_residual": {
            "residual_ft": round(float(residual_row["residual"]), 6),
            "timestamp_utc": _iso_utc(residual_row["timestamp"]),
            "observed_ft_mhhw": float(residual_row["value_observed"]),
            "predicted_ft_mhhw": float(residual_row["value_predicted"]),
            "method": (
                "Observed minus predicted at the same exact UTC timestamp; "
                "independent maxima are never subtracted."
            ),
        },
        "thresholds": {
            "datum": thresholds.get("datum"),
            "units": thresholds.get("units"),
            **{name: float(thresholds[name]) for name in THRESHOLD_NAMES},
            "source_name": thresholds.get("source_name"),
            "retrospective_caveat": thresholds.get("retrospective_caveat"),
        },
        "threshold_exceedances": exceedances,
        "duration_interpretation": (
            "Sample-equivalent duration is qualifying sample count multiplied by "
            f"the nominal {interval_minutes}-minute interval. Crossing times between "
            "samples are unknown; first-to-last span is also reported and excludes "
            "one interval per continuous episode."
        ),
        "sigma": sigma_summary,
        "data_alignment": {
            "observation_row_count": int(len(observed)),
            "prediction_row_count": int(len(predicted)),
            "valid_observation_count": int(observed["value"].notna().sum()),
            "valid_prediction_count": int(predicted["value"].notna().sum()),
            "exact_timestamp_match_count": int(len(exact_time_rows)),
            "valid_residual_count": int(len(valid_residual_rows)),
            "observation_only_timestamp_count": int(len(observation_only)),
            "prediction_only_timestamp_count": int(len(prediction_only)),
            "observation_only_timestamps_utc": [
                _iso_utc(value) for value in observation_only
            ],
            "prediction_only_timestamps_utc": [
                _iso_utc(value) for value in prediction_only
            ],
            "matching_rule": "Exact UTC timestamp inner match; no interpolation.",
        },
        "observation_quality": {
            "quality_code_counts": {str(k): int(v) for k, v in quality_counts.items()},
            "flag_code_counts": {str(k): int(v) for k, v in flag_counts.items()},
            "status_counts": {str(k): int(v) for k, v in status_counts.items()},
            "missing_observed_value_count": int(observed["value"].isna().sum()),
            "all_records_meet_saved_validity_criteria": (
                all_records_meet_saved_validity_criteria
            ),
        },
    }


def compare_with_station_maximum(
    analysis: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare the computed peak with NOAA's rounded station maximum metadata."""

    raw_datums = provenance.get("raw_datums_response")
    if not isinstance(raw_datums, Mapping):
        raise FloodAnalysisError("Threshold provenance lacks NOAA datum metadata.")
    maximum_m = _finite_float(raw_datums.get("max"), "NOAA station maximum")
    mhhw_m = _finite_float(
        provenance.get("mhhw_above_station_datum_meters"),
        "MHHW station-datum elevation",
    )
    maximum_ft_mhhw = (maximum_m - mhhw_m) * FEET_PER_METER
    computed_peak = float(analysis["peak_observed"]["water_level_ft_mhhw"])
    return {
        "source": "NOAA/NOS/CO-OPS station datum metadata saved in Milestone 2",
        "raw_maximum_m_above_station_datum": maximum_m,
        "raw_mhhw_m_above_station_datum": mhhw_m,
        "converted_maximum_ft_mhhw": maximum_ft_mhhw,
        "computed_six_minute_peak_ft_mhhw": computed_peak,
        "computed_minus_metadata_ft": round(computed_peak - maximum_ft_mhhw, 6),
        "raw_maximum_date": raw_datums.get("maxdate"),
        "raw_maximum_time": raw_datums.get("maxtime"),
        "interpretation": (
            "The small value difference is consistent with converting NOAA metadata "
            "rounded to 0.001 meter. The metadata date/time fields are preserved but "
            "are not used to validate the exact UTC timestamp because that response "
            "does not state their time-zone interpretation."
        ),
    }


def build_analysis_document(
    inputs: AnalysisInputs,
    calculations: Mapping[str, Any],
) -> dict[str, Any]:
    """Combine provenance, integrity results, and calculations for persistence."""

    request = inputs.metadata["request"]
    return {
        "analysis_version": 1,
        "method": "Deterministic pandas calculations; no LLM involved.",
        "source_artifacts": {
            "water_level_artifact_id": inputs.metadata.get("artifact_id"),
            "threshold_station_id": inputs.thresholds.get("station_id"),
            "station": inputs.metadata.get("station"),
            "request": request,
            "input_integrity": inputs.integrity,
        },
        "calculations": dict(calculations),
        "authoritative_noaa_comparison": compare_with_station_maximum(
            calculations, inputs.threshold_provenance
        ),
    }


def save_analysis_document(document: Mapping[str, Any], path: str | Path) -> Path:
    """Save a portable JSON result with strict JSON numeric semantics."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path.resolve()


def _summarize_exceedance(
    observations: pd.DataFrame,
    *,
    threshold: float,
    interval_minutes: int,
) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    current: list[pd.Timestamp] = []
    previous_timestamp: pd.Timestamp | None = None
    expected_gap = pd.Timedelta(minutes=interval_minutes)

    for row in observations.itertuples(index=False):
        timestamp = row.timestamp
        value = row.value
        exceeds = pd.notna(value) and float(value) >= threshold
        contiguous = (
            previous_timestamp is not None
            and timestamp - previous_timestamp == expected_gap
        )
        if exceeds:
            if current and not contiguous:
                episodes.append(_episode_summary(current, interval_minutes))
                current = []
            current.append(timestamp)
        elif current:
            episodes.append(_episode_summary(current, interval_minutes))
            current = []
        previous_timestamp = timestamp
    if current:
        episodes.append(_episode_summary(current, interval_minutes))

    count = sum(episode["sample_count"] for episode in episodes)
    total_span = sum(episode["first_to_last_span_minutes"] for episode in episodes)
    return {
        "threshold_ft_mhhw": threshold,
        "sample_count_at_or_above": count,
        "approximate_sample_equivalent_duration_hours": count
        * interval_minutes
        / 60.0,
        "summed_first_to_last_span_hours": total_span / 60.0,
        "episode_count": len(episodes),
        "episodes": episodes,
    }


def _episode_summary(
    timestamps: list[pd.Timestamp], interval_minutes: int
) -> dict[str, Any]:
    count = len(timestamps)
    return {
        "start_timestamp_utc": _iso_utc(timestamps[0]),
        "end_timestamp_utc": _iso_utc(timestamps[-1]),
        "sample_count": count,
        "sample_equivalent_duration_minutes": count * interval_minutes,
        "first_to_last_span_minutes": (
            timestamps[-1] - timestamps[0]
        ).total_seconds()
        / 60.0,
    }


def _summarize_sigma(
    observations: pd.DataFrame,
    peak_timestamp: pd.Timestamp,
) -> dict[str, Any]:
    interpretation = (
        "Sigma is NOAA's reported variability/standard-deviation indicator for the "
        "six-minute water-level value. It is not treated here as total measurement "
        "uncertainty or as an error bound on the storm peak."
    )
    if "sigma" not in observations:
        return {
            "available": False,
            "interpretation": interpretation,
        }

    sigma = pd.to_numeric(observations["sigma"], errors="coerce")
    valid = observations.loc[sigma.notna(), ["timestamp"]].copy()
    valid["sigma"] = sigma.loc[sigma.notna()].astype(float)
    if valid.empty:
        return {
            "available": False,
            "interpretation": interpretation,
        }

    max_index = valid["sigma"].idxmax()
    peak_sigma_values = valid.loc[valid["timestamp"] == peak_timestamp, "sigma"]
    window_start = peak_timestamp - pd.Timedelta(minutes=30)
    window_end = peak_timestamp + pd.Timedelta(minutes=30)
    near_peak = valid.loc[
        valid["timestamp"].between(window_start, window_end, inclusive="both")
    ]
    return {
        "available": True,
        "units": "feet",
        "overall": {
            "count": int(len(valid)),
            "minimum_ft": float(valid["sigma"].min()),
            "median_ft": float(valid["sigma"].median()),
            "maximum_ft": float(valid["sigma"].max()),
            "maximum_timestamp_utc": _iso_utc(valid.loc[max_index, "timestamp"]),
        },
        "at_recorded_peak_ft": (
            float(peak_sigma_values.iloc[0]) if not peak_sigma_values.empty else None
        ),
        "within_30_minutes_of_peak": {
            "window_start_utc": _iso_utc(window_start),
            "window_end_utc": _iso_utc(window_end),
            "count": int(len(near_peak)),
            "minimum_ft": float(near_peak["sigma"].min()),
            "median_ft": float(near_peak["sigma"].median()),
            "maximum_ft": float(near_peak["sigma"].max()),
        },
        "interpretation": interpretation,
    }


def _classify_peak(value: float, thresholds: Mapping[str, Any]) -> str:
    if value >= float(thresholds["major"]):
        return "major"
    if value >= float(thresholds["moderate"]):
        return "moderate"
    if value >= float(thresholds["minor"]):
        return "minor"
    return "below_minor"


def _validate_threshold_order(thresholds: Mapping[str, Any]) -> None:
    values = [_finite_float(thresholds.get(name), f"threshold {name}") for name in THRESHOLD_NAMES]
    if not values[0] < values[1] < values[2]:
        raise FloodAnalysisError("Thresholds must be strictly ordered minor < moderate < major.")


def _prepare_series(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    missing_columns = {"timestamp", "value"} - set(frame.columns)
    if missing_columns:
        raise FloodAnalysisError(
            f"{name} is missing columns: {', '.join(sorted(missing_columns))}"
        )
    output = frame.copy()
    output["timestamp"] = pd.to_datetime(output["timestamp"], utc=True, errors="coerce")
    if output["timestamp"].isna().any():
        raise FloodAnalysisError(f"{name} contains invalid timestamps.")
    output["value"] = pd.to_numeric(output["value"], errors="coerce")
    if output["timestamp"].duplicated().any():
        raise FloodAnalysisError(f"{name} contains duplicate timestamps.")
    return output.sort_values("timestamp").reset_index(drop=True)


def _read_series_csv(path: Path, name: str) -> pd.DataFrame:
    if not path.is_file():
        raise FloodAnalysisError(f"Required {name} artifact is missing: {path}")
    frame = pd.read_csv(path, dtype={"flags": "string", "quality": "string", "status": "string"})
    return _prepare_series(frame, name)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FloodAnalysisError(f"Required artifact is missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise FloodAnalysisError(f"Expected a JSON object in {path}")
    return data


def _finite_float(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise FloodAnalysisError(f"{label} must be numeric.") from error
    if not np.isfinite(number):
        raise FloodAnalysisError(f"{label} must be finite.")
    return number


def _iso_utc(value: Any) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()
