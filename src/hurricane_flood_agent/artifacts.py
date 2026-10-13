"""Local, transparent artifact persistence for NOAA source data."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from hurricane_flood_agent.schemas import FloodThresholdSet, NoaaDataRequest


class ArtifactStore:
    """Save raw responses, validated tables, and provenance under one root."""

    def __init__(self, root: str | Path = "artifacts") -> None:
        self.root = Path(root)

    def save_water_level_data(
        self,
        *,
        request: NoaaDataRequest,
        station: dict[str, Any],
        observed_raw: dict[str, Any],
        predicted_raw: dict[str, Any],
        observed: pd.DataFrame,
        predicted: pd.DataFrame,
        validation: dict[str, Any],
    ) -> dict[str, Any]:
        artifact_id = (
            f"noaa_{request.station_id}_{request.begin_date:%Y%m%d}_"
            f"{request.end_date:%Y%m%d}_{request.datum.lower()}_"
            f"{request.units}_{request.time_zone}"
        )
        directory = self.root / "noaa" / artifact_id
        directory.mkdir(parents=True, exist_ok=True)

        raw_observed_path = directory / "observations_raw.json"
        raw_predicted_path = directory / "predictions_raw.json"
        observations_path = directory / "observations.csv"
        predictions_path = directory / "predictions.csv"
        metadata_path = directory / "metadata.json"

        _write_json(raw_observed_path, observed_raw)
        _write_json(raw_predicted_path, predicted_raw)
        _write_frame(observations_path, observed)
        _write_frame(predictions_path, predicted)

        metadata = {
            "artifact_id": artifact_id,
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": "NOAA/NOS/CO-OPS Data API",
            "source_url": "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter",
            "station": station,
            "request": request.to_dict(),
            "timestamp_interpretation": "All timestamps are UTC (NOAA time_zone=gmt).",
            "vertical_reference": (
                f"Values are {request.unit_label} relative to {request.datum}."
            ),
            "validation": validation,
            "files": {
                "observations_raw": raw_observed_path.name,
                "predictions_raw": raw_predicted_path.name,
                "observations": observations_path.name,
                "predictions": predictions_path.name,
            },
        }
        metadata["sha256"] = {
            path.name: _sha256(path)
            for path in (
                raw_observed_path,
                raw_predicted_path,
                observations_path,
                predictions_path,
            )
        }
        _write_json(metadata_path, metadata)
        return {
            "artifact_id": artifact_id,
            "artifact_path": str(directory.resolve()),
            "metadata_path": str(metadata_path.resolve()),
            "station": station,
            "request": request.to_dict(),
            "validation": validation,
        }

    def save_thresholds(
        self,
        threshold_set: FloodThresholdSet,
        provenance: dict[str, Any],
    ) -> dict[str, Any]:
        retrieval_date = threshold_set.retrieved_at_utc[:10].replace("-", "")
        artifact_id = f"noaa_nos_thresholds_{threshold_set.station_id}_{retrieval_date}"
        directory = self.root / "thresholds" / artifact_id
        directory.mkdir(parents=True, exist_ok=True)
        threshold_path = directory / "thresholds.json"
        raw_path = directory / "source_responses.json"
        _write_json(threshold_path, threshold_set.to_dict())
        _write_json(raw_path, provenance)
        return {
            "artifact_id": artifact_id,
            "artifact_path": str(directory.resolve()),
            "thresholds": threshold_set.to_dict(),
            "sha256": {
                threshold_path.name: _sha256(threshold_path),
                raw_path.name: _sha256(raw_path),
            },
        }


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    output = frame.copy()
    output["timestamp"] = output["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    output.to_csv(path, index=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()
