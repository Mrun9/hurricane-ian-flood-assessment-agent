from __future__ import annotations

import pandas as pd
import pytest

from hurricane_flood_agent.analysis import (
    calculate_flood_indicators,
    derive_thresholds_from_provenance,
)


THRESHOLDS = {
    "minor": 1.5,
    "moderate": 2.5,
    "major": 3.5,
    "datum": "MHHW",
    "units": "feet",
}


def _frame(
    times: list[str], values: list[float | None], **columns: list[object]
) -> pd.DataFrame:
    return pd.DataFrame({"timestamp": times, "value": values, **columns})


def test_known_peak_residual_threshold_counts_and_sigma() -> None:
    times = [
        "2022-09-28T00:00:00Z",
        "2022-09-28T00:06:00Z",
        "2022-09-28T00:12:00Z",
        "2022-09-28T00:18:00Z",
    ]
    observations = _frame(times, [1.0, 2.0, 4.0, 3.0], sigma=[0.1, 0.2, 0.3, 0.2])
    predictions = _frame(times, [0.5, 1.0, 1.5, 2.0])

    result = calculate_flood_indicators(
        observations, predictions, THRESHOLDS, interval_minutes=6
    )

    assert result["peak_observed"] == {
        "water_level_ft_mhhw": 4.0,
        "timestamp_utc": "2022-09-28T00:12:00Z",
        "flood_category": "major",
    }
    assert result["maximum_residual"]["residual_ft"] == 2.5
    assert result["maximum_residual"]["timestamp_utc"] == "2022-09-28T00:12:00Z"
    assert result["threshold_exceedances"]["minor"]["sample_count_at_or_above"] == 3
    assert result["threshold_exceedances"]["moderate"]["sample_count_at_or_above"] == 2
    assert result["threshold_exceedances"]["major"]["sample_count_at_or_above"] == 1
    assert result["threshold_exceedances"]["minor"][
        "approximate_sample_equivalent_duration_hours"
    ] == pytest.approx(0.3)
    assert result["sigma"]["at_recorded_peak_ft"] == 0.3


def test_residual_uses_matching_timestamp_not_independent_maxima() -> None:
    times = ["2022-09-28T00:00:00Z", "2022-09-28T00:06:00Z"]
    observations = _frame(times, [10.0, 9.0])
    predictions = _frame(times, [8.0, 9.0])

    result = calculate_flood_indicators(
        observations, predictions, THRESHOLDS, interval_minutes=6
    )

    # An incorrect independent-maxima calculation is 10 - 9 = 1.
    # Exact timestamp matching correctly finds 10 - 8 = 2 at 00:00 UTC.
    assert result["maximum_residual"]["residual_ft"] == 2.0
    assert result["maximum_residual"]["timestamp_utc"] == "2022-09-28T00:00:00Z"


def test_missing_values_are_excluded_without_imputation() -> None:
    times = [
        "2022-09-28T00:00:00Z",
        "2022-09-28T00:06:00Z",
        "2022-09-28T00:12:00Z",
    ]
    observations = _frame(times, [1.0, None, 4.0])
    predictions = _frame(times, [0.5, 0.7, None])

    result = calculate_flood_indicators(
        observations, predictions, THRESHOLDS, interval_minutes=6
    )

    assert result["peak_observed"]["water_level_ft_mhhw"] == 4.0
    assert result["maximum_residual"]["residual_ft"] == 0.5
    assert result["data_alignment"]["valid_residual_count"] == 1
    assert result["observation_quality"]["missing_observed_value_count"] == 1
    assert result["threshold_exceedances"]["major"]["sample_count_at_or_above"] == 1


def test_mismatched_timestamps_only_use_exact_matches() -> None:
    observations = _frame(
        ["2022-09-28T00:00:00Z", "2022-09-28T00:06:00Z"],
        [2.0, 8.0],
    )
    predictions = _frame(
        ["2022-09-28T00:00:00Z", "2022-09-28T00:12:00Z"],
        [0.5, -1.0],
    )

    result = calculate_flood_indicators(
        observations, predictions, THRESHOLDS, interval_minutes=6
    )

    assert result["maximum_residual"]["residual_ft"] == 1.5
    assert result["data_alignment"]["exact_timestamp_match_count"] == 1
    assert result["data_alignment"]["observation_only_timestamp_count"] == 1
    assert result["data_alignment"]["prediction_only_timestamp_count"] == 1


def test_threshold_conversion_uses_source_values() -> None:
    provenance = {
        "mhhw_above_station_datum_meters": 1.0,
        "absolute_nos_thresholds_above_station_datum_meters": {
            "minor": 1.5,
            "moderate": 2.0,
            "major": 2.5,
        },
    }

    assert derive_thresholds_from_provenance(provenance) == {
        "minor": 1.64,
        "moderate": 3.28,
        "major": 4.92,
    }
