"""Retrieval and datum conversion for authoritative NOAA/NOS thresholds."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from hurricane_flood_agent.exceptions import NoaaValidationError
from hurricane_flood_agent.noaa import METADATA_API_URL, NoaaClient
from hurricane_flood_agent.schemas import FloodThresholdSet


METERS_TO_FEET = 3.28084


def retrieve_flood_thresholds(
    station_id: str,
    client: NoaaClient | None = None,
) -> tuple[FloodThresholdSet, dict[str, Any]]:
    """Retrieve current NOS thresholds and express them in feet above MHHW.

    NOAA returns flood levels and tidal datums as elevations above station
    datum in meters. Subtracting the MHHW elevation produces thresholds relative
    to MHHW; converting that difference to feet matches the water-level request.
    """

    active_client = client or NoaaClient()
    station_payload = active_client.fetch_station_floodlevels(station_id)
    datums_payload = active_client.fetch_station_datums(station_id)

    stations = station_payload.get("stations")
    if not isinstance(stations, list) or len(stations) != 1:
        raise NoaaValidationError("NOAA station metadata did not contain one station.")
    station = stations[0]
    if str(station.get("id")) != station_id:
        raise NoaaValidationError("NOAA flood-level station ID did not match the request.")

    floodlevels = station.get("floodlevels")
    datums = datums_payload.get("datums")
    if not isinstance(floodlevels, dict) or not isinstance(datums, list):
        raise NoaaValidationError("NOAA threshold response is missing flood levels or datums.")

    mhhw_record = next(
        (datum for datum in datums if isinstance(datum, dict) and datum.get("name") == "MHHW"),
        None,
    )
    if not mhhw_record:
        raise NoaaValidationError("NOAA datum response does not include MHHW.")

    try:
        mhhw_station_datum_m = float(mhhw_record["value"])
        absolute_thresholds_m = {
            severity: float(floodlevels[f"nos_{severity}"])
            for severity in ("minor", "moderate", "major")
        }
    except (KeyError, TypeError, ValueError) as error:
        raise NoaaValidationError("NOAA NOS flood thresholds are missing or invalid.") from error

    relative_thresholds_ft = {
        severity: round((value - mhhw_station_datum_m) * METERS_TO_FEET, 2)
        for severity, value in absolute_thresholds_m.items()
    }
    ordered = [relative_thresholds_ft[name] for name in ("minor", "moderate", "major")]
    if not (0 < ordered[0] < ordered[1] < ordered[2]):
        raise NoaaValidationError("NOAA NOS thresholds are not positive and increasing.")

    retrieved_at = datetime.now(timezone.utc).isoformat()
    source_urls = (
        f"{METADATA_API_URL}/stations/{station_id}.json?expand=details,floodlevels&units=metric",
        f"{METADATA_API_URL}/stations/{station_id}/datums.json?units=metric",
        f"https://tidesandcurrents.noaa.gov/inundationdb/inundation.html?id={station_id}",
    )
    threshold_set = FloodThresholdSet(
        station_id=station_id,
        station_name=str(station.get("name", "")),
        datum="MHHW",
        units="feet",
        minor=relative_thresholds_ft["minor"],
        moderate=relative_thresholds_ft["moderate"],
        major=relative_thresholds_ft["major"],
        retrieved_at_utc=retrieved_at,
        source_urls=source_urls,
    )
    provenance = {
        "method": (
            "For each NOS threshold: (flood-level elevation above station datum "
            "in meters - MHHW elevation above station datum in meters) * 3.28084."
        ),
        "mhhw_above_station_datum_meters": mhhw_station_datum_m,
        "absolute_nos_thresholds_above_station_datum_meters": absolute_thresholds_m,
        "raw_station_response": station_payload,
        "raw_datums_response": datums_payload,
    }
    return threshold_set, provenance
