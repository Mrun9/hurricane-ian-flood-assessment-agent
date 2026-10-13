"""Tests for transparent, reproducible local NOAA artifacts."""

import json
from datetime import date

import pandas as pd

from hurricane_flood_agent.artifacts import ArtifactStore
from hurricane_flood_agent.schemas import NoaaDataRequest


def test_water_level_artifact_contains_data_and_provenance(tmp_path) -> None:
    request = NoaaDataRequest("8725520", date(2022, 9, 27), date(2022, 9, 27))
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2022-09-27 00:00"], utc=True),
            "value": [1.0],
            "sigma": [0.01],
            "flags": ["0,0,0,0"],
            "quality": ["v"],
            "status": ["valid"],
        }
    )
    store = ArtifactStore(tmp_path)

    result = store.save_water_level_data(
        request=request,
        station={"id": "8725520", "name": "Fort Myers"},
        observed_raw={"data": [{"t": "2022-09-27 00:00", "v": "1.0"}]},
        predicted_raw={"predictions": [{"t": "2022-09-27 00:00", "v": "0.5"}]},
        observed=frame,
        predicted=frame,
        validation={"observations": {}, "predictions": {}},
    )

    metadata = json.loads((tmp_path / "noaa" / result["artifact_id"] / "metadata.json").read_text())
    assert metadata["request"]["datum"] == "MHHW"
    assert metadata["request"]["unit_label"] == "feet"
    assert metadata["timestamp_interpretation"].startswith("All timestamps are UTC")
    assert set(metadata["sha256"]) == {
        "observations_raw.json",
        "predictions_raw.json",
        "observations.csv",
        "predictions.csv",
    }
