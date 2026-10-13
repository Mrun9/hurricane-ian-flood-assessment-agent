"""Tests for NOAA request validation, parsing, and failure behavior."""

from datetime import date
from unittest.mock import Mock

import pytest
import requests

from hurricane_flood_agent.exceptions import NoaaApiError, NoaaValidationError
from hurricane_flood_agent.noaa import NoaaClient, parse_series_payload
from hurricane_flood_agent.schemas import NoaaDataRequest


def _request() -> NoaaDataRequest:
    return NoaaDataRequest("8725520", date(2022, 9, 27), date(2022, 9, 27))


def _full_day_records() -> list[dict[str, str]]:
    records = []
    for index in range(24 * 10):
        hour, interval = divmod(index, 10)
        records.append(
            {
                "t": f"2022-09-27 {hour:02d}:{interval * 6:02d}",
                "v": f"{index / 100:.2f}",
                "s": "0.01",
                "f": "0,0,0,0",
                "q": "v",
            }
        )
    return records


def test_request_validation_rejects_bad_station_and_dates() -> None:
    with pytest.raises(ValueError, match="seven digits"):
        NoaaDataRequest("bad", date(2022, 9, 27), date(2022, 9, 30))
    with pytest.raises(ValueError, match="on or before"):
        NoaaDataRequest("8725520", date(2022, 9, 30), date(2022, 9, 27))


def test_parse_complete_observations() -> None:
    frame, validation = parse_series_payload(
        {"data": _full_day_records()},
        _request(),
        product="water_level",
    )

    assert len(frame) == 240
    assert validation["valid_value_count"] == 240
    assert validation["missing_or_invalid_count"] == 0
    assert validation["completeness_percent"] == 100.0
    assert str(frame["timestamp"].dt.tz) == "UTC"


def test_parse_preserves_invalid_values_and_detects_gap() -> None:
    payload = {
        "data": [
            {"t": "2022-09-27 00:00", "v": "1.0", "s": "0.01", "f": "0", "q": "v"},
            {"t": "2022-09-27 00:06", "v": "", "s": "", "f": "1", "q": "p"},
            {"t": "2022-09-27 00:24", "v": "1.2", "s": "0.01", "f": "0", "q": "v"},
        ]
    }

    frame, validation = parse_series_payload(payload, _request(), product="water_level")

    first_gap = validation["gaps"][0]
    assert frame.loc[frame["timestamp"].dt.minute == 6, "value"].isna().all()
    assert validation["invalid_value_count"] == 1
    assert validation["missing_timestamp_count"] == 237
    assert validation["flagged_record_count"] == 1
    assert validation["quality_code_counts"] == {"v": 2, "p": 1}
    assert first_gap["start_utc"] == "2022-09-27T00:06:00+00:00"
    assert first_gap["end_utc"] == "2022-09-27T00:18:00+00:00"
    assert first_gap["is_substantial"] is True


def test_duplicate_timestamp_is_rejected() -> None:
    duplicate = {"t": "2022-09-27 00:00", "v": "1.0"}
    with pytest.raises(NoaaValidationError, match="duplicate"):
        parse_series_payload(
            {"predictions": [duplicate, duplicate]},
            _request(),
            product="predictions",
        )


def test_client_wraps_timeout() -> None:
    session = Mock()
    session.get.side_effect = requests.Timeout("timed out")
    client = NoaaClient(session=session, timeout_seconds=1)

    with pytest.raises(NoaaApiError, match="request failed"):
        client.fetch_water_levels(_request())


def test_client_recognizes_noaa_error_payload() -> None:
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"error": {"message": "No data was found"}}
    session = Mock()
    session.get.return_value = response
    client = NoaaClient(session=session)

    with pytest.raises(NoaaApiError, match="No data was found"):
        client.fetch_predictions(_request())
