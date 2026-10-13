"""Tests for the harmless schema used by the live tool-selection smoke test."""

from navigator_smoke_test import retrieve_noaa_data


def test_smoke_tool_has_expected_name_and_argument() -> None:
    assert retrieve_noaa_data.name == "retrieve_noaa_data"
    assert "station_id" in retrieve_noaa_data.args
