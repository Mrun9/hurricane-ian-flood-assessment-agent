"""Use-case functions that connect NOAA retrieval, validation, and artifacts."""

from __future__ import annotations

from hurricane_flood_agent.artifacts import ArtifactStore
from hurricane_flood_agent.exceptions import NoaaValidationError
from hurricane_flood_agent.noaa import (
    NoaaClient,
    parse_series_payload,
    validate_station_metadata,
)
from hurricane_flood_agent.schemas import NoaaDataRequest
from hurricane_flood_agent.thresholds import retrieve_flood_thresholds


def retrieve_noaa_data(
    request: NoaaDataRequest,
    *,
    client: NoaaClient | None = None,
    artifact_store: ArtifactStore | None = None,
) -> dict:
    """Retrieve and persist validated observations and tide predictions."""

    active_client = client or NoaaClient()
    active_store = artifact_store or ArtifactStore()

    observed_raw = active_client.fetch_water_levels(request)
    predicted_raw = active_client.fetch_predictions(request)
    station = validate_station_metadata(observed_raw, request.station_id)
    observed, observed_validation = parse_series_payload(
        observed_raw,
        request,
        product="water_level",
    )
    predicted, predicted_validation = parse_series_payload(
        predicted_raw,
        request,
        product="predictions",
    )
    if not observed["timestamp"].equals(predicted["timestamp"]):
        raise NoaaValidationError(
            "Validated observations and predictions do not share the same UTC grid."
        )
    validation = {
        "observations": observed_validation,
        "predictions": predicted_validation,
    }
    return active_store.save_water_level_data(
        request=request,
        station=station,
        observed_raw=observed_raw,
        predicted_raw=predicted_raw,
        observed=observed,
        predicted=predicted,
        validation=validation,
    )


def retrieve_and_save_thresholds(
    station_id: str,
    *,
    client: NoaaClient | None = None,
    artifact_store: ArtifactStore | None = None,
) -> dict:
    """Retrieve current NOAA/NOS thresholds and persist full provenance."""

    threshold_set, provenance = retrieve_flood_thresholds(station_id, client)
    return (artifact_store or ArtifactStore()).save_thresholds(threshold_set, provenance)
