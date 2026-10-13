"""Tests for NOAA/NOS flood-threshold datum conversion."""

from unittest.mock import Mock

import pytest

from hurricane_flood_agent.exceptions import NoaaValidationError
from hurricane_flood_agent.thresholds import retrieve_flood_thresholds


def _client() -> Mock:
    client = Mock()
    client.fetch_station_floodlevels.return_value = {
        "stations": [
            {
                "id": "8725520",
                "name": "Fort Myers",
                "floodlevels": {
                    "nos_minor": 2.246,
                    "nos_moderate": 2.542,
                    "nos_major": 2.916,
                },
            }
        ]
    }
    client.fetch_station_datums.return_value = {
        "datums": [{"name": "MHHW", "value": 1.730}]
    }
    return client


def test_thresholds_are_converted_to_feet_above_mhhw() -> None:
    thresholds, provenance = retrieve_flood_thresholds("8725520", _client())

    assert thresholds.station_name == "Fort Myers"
    assert thresholds.datum == "MHHW"
    assert thresholds.units == "feet"
    assert thresholds.minor == 1.69
    assert thresholds.moderate == 2.66
    assert thresholds.major == 3.89
    assert provenance["mhhw_above_station_datum_meters"] == 1.73


def test_thresholds_require_mhhw_datum() -> None:
    client = _client()
    client.fetch_station_datums.return_value = {
        "datums": [{"name": "MSL", "value": 1.5}]
    }

    with pytest.raises(NoaaValidationError, match="MHHW"):
        retrieve_flood_thresholds("8725520", client)
