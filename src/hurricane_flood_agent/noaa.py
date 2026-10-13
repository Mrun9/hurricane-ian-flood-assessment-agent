"""NOAA CO-OPS retrieval and deterministic response validation."""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from hurricane_flood_agent.exceptions import NoaaApiError, NoaaValidationError
from hurricane_flood_agent.schemas import NoaaDataRequest


DATA_API_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
METADATA_API_URL = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi"
APPLICATION_NAME = "CGN6905_Hurricane_Flood_Assessment"
TRANSIENT_STATUS_CODES = (429, 500, 502, 503, 504)


def build_retrying_session() -> requests.Session:
    """Create an HTTP session that retries safe GETs after transient failures."""

    retry = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=0.5,
        status_forcelist=TRANSIENT_STATUS_CODES,
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.headers.update({"User-Agent": f"{APPLICATION_NAME}/0.1"})
    session.mount("https://", adapter)
    return session


class NoaaClient:
    """Narrow client for the NOAA endpoints required by this project."""

    def __init__(
        self,
        session: requests.Session | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.session = session or build_retrying_session()
        self.timeout_seconds = timeout_seconds

    def _get_json(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self.session.get(
                url,
                params=params,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as error:
            raise NoaaApiError(f"NOAA request failed: {error}") from error
        except ValueError as error:
            raise NoaaApiError("NOAA returned a response that was not valid JSON.") from error

        if not isinstance(payload, dict):
            raise NoaaValidationError("NOAA returned an unexpected non-object response.")
        if "error" in payload:
            error = payload["error"]
            message = error.get("message", str(error)) if isinstance(error, dict) else str(error)
            raise NoaaApiError(f"NOAA API error: {message}")
        return payload

    def fetch_water_levels(self, request: NoaaDataRequest) -> dict[str, Any]:
        return self._get_json(DATA_API_URL, _series_params(request, "water_level"))

    def fetch_predictions(self, request: NoaaDataRequest) -> dict[str, Any]:
        params = _series_params(request, "predictions")
        params["interval"] = str(request.interval_minutes)
        return self._get_json(DATA_API_URL, params)

    def fetch_station_floodlevels(self, station_id: str) -> dict[str, Any]:
        _validate_station_id(station_id)
        return self._get_json(
            f"{METADATA_API_URL}/stations/{station_id}.json",
            {"expand": "details,floodlevels", "units": "metric"},
        )

    def fetch_station_datums(self, station_id: str) -> dict[str, Any]:
        _validate_station_id(station_id)
        return self._get_json(
            f"{METADATA_API_URL}/stations/{station_id}/datums.json",
            {"units": "metric"},
        )


def _validate_station_id(station_id: str) -> None:
    NoaaDataRequest(
        station_id=station_id,
        begin_date=datetime(2000, 1, 1).date(),
        end_date=datetime(2000, 1, 1).date(),
    )


def _series_params(request: NoaaDataRequest, product: str) -> dict[str, str]:
    return {
        "product": product,
        "application": APPLICATION_NAME,
        "begin_date": request.begin_date.strftime("%Y%m%d"),
        "end_date": request.end_date.strftime("%Y%m%d"),
        "datum": request.datum,
        "station": request.station_id,
        "time_zone": request.time_zone,
        "units": request.units,
        "format": "json",
    }


def validate_station_metadata(payload: dict[str, Any], station_id: str) -> dict[str, Any]:
    """Validate and normalize the station metadata returned with observations."""

    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise NoaaValidationError("NOAA water-level response is missing station metadata.")
    if str(metadata.get("id")) != station_id:
        raise NoaaValidationError(
            f"NOAA returned station {metadata.get('id')!r}, expected {station_id!r}."
        )
    return {
        "id": station_id,
        "name": str(metadata.get("name", "")),
        "latitude": _optional_float(metadata.get("lat")),
        "longitude": _optional_float(metadata.get("lon")),
    }


def parse_series_payload(
    payload: dict[str, Any],
    request: NoaaDataRequest,
    *,
    product: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Parse one NOAA series and explicitly represent absent/invalid values."""

    if product not in {"water_level", "predictions"}:
        raise ValueError(f"Unsupported product: {product}")
    record_key = "data" if product == "water_level" else "predictions"
    records = payload.get(record_key)
    if not isinstance(records, list) or not records:
        raise NoaaValidationError(f"NOAA {product} response contains no records.")

    parsed_rows: list[dict[str, Any]] = []
    invalid_timestamp_count = 0
    for record in records:
        if not isinstance(record, dict):
            raise NoaaValidationError(f"NOAA {product} contains a non-object record.")
        timestamp = pd.to_datetime(
            record.get("t"),
            format="%Y-%m-%d %H:%M",
            errors="coerce",
            utc=True,
        )
        if pd.isna(timestamp):
            invalid_timestamp_count += 1
            continue
        parsed_rows.append(
            {
                "timestamp": timestamp,
                "value": pd.to_numeric(record.get("v"), errors="coerce"),
                "sigma": (
                    pd.to_numeric(record.get("s"), errors="coerce")
                    if product == "water_level"
                    else float("nan")
                ),
                "flags": record.get("f") if product == "water_level" else None,
                "quality": record.get("q") if product == "water_level" else None,
            }
        )

    if not parsed_rows:
        raise NoaaValidationError(f"NOAA {product} has no valid timestamps.")

    frame = pd.DataFrame(parsed_rows).sort_values("timestamp")
    duplicate_count = int(frame["timestamp"].duplicated().sum())
    if duplicate_count:
        raise NoaaValidationError(
            f"NOAA {product} contains {duplicate_count} duplicate timestamps."
        )

    expected_index = expected_timestamps(request)
    unexpected = frame.loc[~frame["timestamp"].isin(expected_index), "timestamp"]
    if not unexpected.empty:
        raise NoaaValidationError(
            f"NOAA {product} contains timestamps outside the requested six-minute grid."
        )

    received_timestamps = pd.DatetimeIndex(frame["timestamp"])
    missing_timestamp_count = int(len(expected_index.difference(received_timestamps)))
    invalid_value_count = int(frame["value"].isna().sum())
    quality_counts = {
        str(quality): int(count)
        for quality, count in frame["quality"]
        .fillna("not_applicable")
        .value_counts()
        .items()
    }
    flagged_record_count = int(
        frame["flags"]
        .fillna("")
        .map(
            lambda flags: any(
                part.strip() not in {"", "0"} for part in str(flags).split(",")
            )
        )
        .sum()
    )

    completed = frame.set_index("timestamp").reindex(expected_index)
    completed.index.name = "timestamp"
    completed = completed.reset_index()
    completed["status"] = "valid"
    completed.loc[completed["value"].isna(), "status"] = "missing_or_invalid"

    valid_value_count = int(completed["value"].notna().sum())
    expected_count = len(expected_index)
    gaps = _find_missing_runs(completed, request.interval_minutes)
    warnings: list[str] = []
    if invalid_timestamp_count:
        warnings.append(f"Discarded {invalid_timestamp_count} records with invalid timestamps.")
    if invalid_value_count:
        warnings.append(f"Preserved {invalid_value_count} invalid values as missing.")
    if flagged_record_count:
        warnings.append(
            f"Preserved {flagged_record_count} numeric record(s) carrying NOAA quality flags."
        )
    if missing_timestamp_count:
        warnings.append(
            f"Inserted {missing_timestamp_count} absent timestamps as explicit missing values."
        )
    substantial_gap_count = sum(gap["is_substantial"] for gap in gaps)
    if substantial_gap_count:
        warnings.append(
            f"Detected {substantial_gap_count} substantial gap(s) of at least 18 minutes."
        )

    validation = {
        "product": product,
        "expected_interval_minutes": request.interval_minutes,
        "expected_record_count": expected_count,
        "received_record_count": len(records),
        "valid_value_count": valid_value_count,
        "missing_or_invalid_count": expected_count - valid_value_count,
        "missing_timestamp_count": missing_timestamp_count,
        "invalid_value_count": invalid_value_count,
        "invalid_timestamp_count": invalid_timestamp_count,
        "flagged_record_count": flagged_record_count,
        "quality_code_counts": quality_counts,
        "completeness_percent": round(100.0 * valid_value_count / expected_count, 2),
        "gaps": gaps,
        "warnings": warnings,
    }
    return completed, validation


def expected_timestamps(request: NoaaDataRequest) -> pd.DatetimeIndex:
    start = datetime.combine(request.begin_date, time.min, tzinfo=timezone.utc)
    end_exclusive = datetime.combine(
        request.end_date + timedelta(days=1),
        time.min,
        tzinfo=timezone.utc,
    )
    return pd.date_range(
        start=start,
        end=end_exclusive,
        freq=f"{request.interval_minutes}min",
        inclusive="left",
    )


def _find_missing_runs(
    frame: pd.DataFrame,
    interval_minutes: int,
) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    run_start: int | None = None
    missing = frame["value"].isna().tolist()
    for index, is_missing in enumerate(missing + [False]):
        if is_missing and run_start is None:
            run_start = index
        elif not is_missing and run_start is not None:
            run_length = index - run_start
            start_timestamp = frame.iloc[run_start]["timestamp"]
            end_timestamp = frame.iloc[index - 1]["timestamp"]
            gaps.append(
                {
                    "start_utc": start_timestamp.isoformat(),
                    "end_utc": end_timestamp.isoformat(),
                    "missing_intervals": run_length,
                    "duration_minutes": run_length * interval_minutes,
                    "is_substantial": run_length >= 3,
                }
            )
            run_start = None
    return gaps


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
