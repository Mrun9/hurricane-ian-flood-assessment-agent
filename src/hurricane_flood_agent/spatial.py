"""USGS Hurricane Ian high-water-mark retrieval, analysis, and mapping."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import tempfile
from typing import Any, Mapping

import pandas as pd
import requests

from hurricane_flood_agent.exceptions import UsgsApiError, UsgsValidationError
from hurricane_flood_agent.noaa import build_retrying_session


USGS_HWM_URL = "https://stn.wim.usgs.gov/STNServices/HWMs/FilteredHWMs.json"
USGS_EVENT_URL = "https://apps.usgs.gov/fev/event/2022-ian"
IAN_EVENT_ID = 325
DEFAULT_STATE = "FL"
DEFAULT_COUNTY = "Lee County"

_matplotlib_cache = Path(tempfile.gettempdir()) / "hurricane-flood-agent-matplotlib"
_matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_matplotlib_cache))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class UsgsClient:
    """Narrow client for official USGS Short-Term Network high-water marks."""

    def __init__(
        self,
        session: requests.Session | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.session = session or build_retrying_session()
        self.timeout_seconds = timeout_seconds

    def fetch_high_water_marks(
        self,
        *,
        event_id: int = IAN_EVENT_ID,
        state: str = DEFAULT_STATE,
        county: str = DEFAULT_COUNTY,
    ) -> list[dict[str, Any]]:
        params = {
            "Event": str(event_id),
            "EventType": "",
            "EventStatus": "",
            "States": state,
            "County": county,
            "HWMType": "",
            "HWMQuality": "",
            "HWMEnvironment": "",
            "SurveyComplete": "true",
            "StillWater": "",
        }
        try:
            response = self.session.get(
                USGS_HWM_URL,
                params=params,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as error:
            raise UsgsApiError(f"USGS STN request failed: {error}") from error
        except ValueError as error:
            raise UsgsApiError("USGS STN returned invalid JSON.") from error
        if not isinstance(payload, list):
            raise UsgsValidationError("USGS STN returned an unexpected response type.")
        return payload


def normalize_high_water_marks(
    records: list[dict[str, Any]],
    *,
    event_id: int = IAN_EVENT_ID,
    state: str = DEFAULT_STATE,
    county: str = DEFAULT_COUNTY,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate and normalize surveyed high-water marks for one event and county."""

    if not records:
        raise UsgsValidationError("USGS returned no surveyed high-water marks.")
    rows: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            raise UsgsValidationError("USGS returned a non-object high-water mark.")
        if int(record.get("event_id", -1)) != event_id:
            raise UsgsValidationError("USGS response contains a different event.")
        if str(record.get("stateName", "")) != state:
            raise UsgsValidationError("USGS response contains a different state.")
        if str(record.get("countyName", "")) != county:
            raise UsgsValidationError("USGS response contains a different county.")
        rows.append(
            {
                "hwm_id": _required_int(record, "hwm_id"),
                "site_id": _required_int(record, "site_id"),
                "site_no": str(record.get("site_no", "")),
                "site_description": str(record.get("siteDescription") or ""),
                "waterbody": str(record.get("waterbody") or ""),
                "latitude": _required_float(record, "latitude_dd"),
                "longitude": _required_float(record, "longitude_dd"),
                "elevation_ft": _required_float(record, "elev_ft"),
                "vertical_datum": str(record.get("verticalDatumName") or "Unknown"),
                "height_above_ground_ft": _optional_float(
                    record.get("height_above_gnd")
                ),
                "environment": str(record.get("hwm_environment") or "Unknown"),
                "mark_type": str(record.get("hwmTypeName") or "Unknown"),
                "quality": str(record.get("hwmQualityName") or "Unknown"),
                "stillwater": bool(record.get("stillwater") == 1.0),
                "survey_date": str(record.get("survey_date") or ""),
                "vertical_method": str(record.get("verticalMethodName") or "Unknown"),
                "horizontal_datum": str(
                    record.get("horizontalDatumName") or "Unknown"
                ),
            }
        )

    frame = pd.DataFrame(rows).sort_values(["site_no", "hwm_id"]).reset_index(drop=True)
    if frame["hwm_id"].duplicated().any():
        raise UsgsValidationError("USGS response contains duplicate HWM identifiers.")
    if not frame["latitude"].between(-90, 90).all() or not frame[
        "longitude"
    ].between(-180, 180).all():
        raise UsgsValidationError("USGS response contains invalid coordinates.")

    validation = {
        "record_count": int(len(frame)),
        "unique_site_count": int(frame["site_id"].nunique()),
        "elevation_value_count": int(frame["elevation_ft"].notna().sum()),
        "vertical_datum_counts": _counts(frame["vertical_datum"]),
        "environment_counts": _counts(frame["environment"]),
        "quality_counts": _counts(frame["quality"]),
        "mark_type_counts": _counts(frame["mark_type"]),
        "stillwater_count": int(frame["stillwater"].sum()),
        "warnings": [],
    }
    if len(validation["vertical_datum_counts"]) > 1:
        validation["warnings"].append(
            "Multiple vertical datums are present; elevations must be separated or converted."
        )
    if any(value == "Unknown" for value in frame["vertical_datum"]):
        validation["warnings"].append("Some records have an unknown vertical datum.")
    return frame, validation


def retrieve_and_save_usgs_high_water_marks(
    artifact_root: Path,
    *,
    event_id: int = IAN_EVENT_ID,
    state: str = DEFAULT_STATE,
    county: str = DEFAULT_COUNTY,
    client: UsgsClient | None = None,
) -> dict[str, Any]:
    """Retrieve, validate, and persist USGS marks with full source provenance."""

    records = (client or UsgsClient()).fetch_high_water_marks(
        event_id=event_id, state=state, county=county
    )
    frame, validation = normalize_high_water_marks(
        records, event_id=event_id, state=state, county=county
    )
    artifact_id = f"usgs_hwm_event_{event_id}_{state.lower()}_lee_county"
    directory = Path(artifact_root).resolve() / "usgs" / artifact_id
    directory.mkdir(parents=True, exist_ok=True)
    raw_path = directory / "high_water_marks_raw.json"
    data_path = directory / "high_water_marks.csv"
    metadata_path = directory / "metadata.json"
    raw_path.write_text(
        json.dumps(records, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    frame.to_csv(data_path, index=False)
    metadata = {
        "artifact_id": artifact_id,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "USGS Short-Term Network / Flood Event Viewer",
        "source_url": USGS_HWM_URL,
        "event_url": USGS_EVENT_URL,
        "request": {
            "event_id": event_id,
            "event_name": "2022 Ian",
            "state": state,
            "county": county,
            "survey_complete": True,
        },
        "validation": validation,
        "files": {"raw": raw_path.name, "normalized": data_path.name},
        "sha256": {
            raw_path.name: _sha256(raw_path),
            data_path.name: _sha256(data_path),
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "artifact_id": artifact_id,
        "artifact_path": str(directory),
        "metadata": metadata,
    }


def load_usgs_artifact(directory: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load and integrity-check an independently saved USGS artifact."""

    root = Path(directory).resolve()
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    for filename, expected in metadata.get("sha256", {}).items():
        path = root / filename
        if not path.is_file() or _sha256(path) != expected:
            raise UsgsValidationError(f"USGS artifact failed integrity check: {filename}")
    frame = pd.read_csv(root / metadata["files"]["normalized"])
    return frame, metadata


def calculate_spatial_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """Calculate deterministic spatial coverage and comparable-datum summaries."""

    required = {
        "hwm_id",
        "site_id",
        "site_no",
        "site_description",
        "latitude",
        "longitude",
        "elevation_ft",
        "vertical_datum",
        "environment",
        "quality",
        "mark_type",
        "height_above_ground_ft",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise UsgsValidationError(f"USGS artifact is missing columns: {missing}")
    valid = frame.dropna(subset=["latitude", "longitude", "elevation_ft"]).copy()
    if valid.empty:
        raise UsgsValidationError("USGS artifact has no mappable elevation records.")
    datum_counts = _counts(valid["vertical_datum"])
    comparable_datum = next(iter(datum_counts)) if len(datum_counts) == 1 else None
    ranges_by_datum = {
        str(datum): {
            "minimum": round(float(group["elevation_ft"].min()), 3),
            "median": round(float(group["elevation_ft"].median()), 3),
            "maximum": round(float(group["elevation_ft"].max()), 3),
        }
        for datum, group in valid.groupby("vertical_datum", dropna=False)
    }
    comparable_range = (
        ranges_by_datum[comparable_datum] if comparable_datum is not None else None
    )
    minimum = (
        valid.loc[valid["elevation_ft"].idxmin()]
        if comparable_datum is not None
        else None
    )
    maximum = (
        valid.loc[valid["elevation_ft"].idxmax()]
        if comparable_datum is not None
        else None
    )
    return {
        "event": "2022 Hurricane Ian",
        "county": "Lee County, Florida",
        "record_count": int(len(valid)),
        "unique_site_count": int(valid["site_id"].nunique()),
        "coordinate_bounds": {
            "west": round(float(valid["longitude"].min()), 6),
            "east": round(float(valid["longitude"].max()), 6),
            "south": round(float(valid["latitude"].min()), 6),
            "north": round(float(valid["latitude"].max()), 6),
        },
        "vertical_datum_counts": datum_counts,
        "comparable_vertical_datum": comparable_datum,
        "elevation_range_ft": comparable_range,
        "elevation_ranges_by_datum_ft": ranges_by_datum,
        "minimum_record": _record_summary(minimum) if minimum is not None else None,
        "maximum_record": _record_summary(maximum) if maximum is not None else None,
        "environment_counts": _counts(valid["environment"]),
        "quality_counts": _counts(valid["quality"]),
        "mark_type_counts": _counts(valid["mark_type"]),
        "height_above_ground_available_count": int(
            valid["height_above_ground_ft"].notna().sum()
        ),
        "scientific_constraints": {
            "direct_comparison_to_noaa_mhhw_allowed": False,
            "reason": (
                "USGS elevations are referenced to NAVD88 while the NOAA time series "
                "is referenced to MHHW; a location-specific datum transformation is "
                "required before numerical comparison."
            ),
            "inundation_depth_warning": (
                "A high-water-mark elevation is not automatically flood depth at a "
                "road or building. Local ground or infrastructure elevation and the "
                "measurement location are required."
            ),
            "spatial_interpretation_warning": (
                "Point observations describe surveyed locations, not a continuous "
                "flood surface or every community in Lee County."
            ),
        },
        "calculation_method": "Deterministic pandas summary; no LLM calculation.",
        "source_url": USGS_HWM_URL,
        "event_url": USGS_EVENT_URL,
    }


def create_spatial_hwm_map(
    frame: pd.DataFrame,
    summary: Mapping[str, Any],
    output_path: Path,
) -> Path:
    """Create a static point map of USGS surveyed high-water-mark elevations."""

    datum = summary.get("comparable_vertical_datum")
    if not datum:
        raise UsgsValidationError(
            "A single elevation color scale cannot be generated for mixed vertical datums."
        )
    valid = frame.dropna(subset=["latitude", "longitude", "elevation_ft"]).copy()
    figure, axis = plt.subplots(figsize=(10.5, 7.2))
    figure.patch.set_facecolor("white")
    axis.set_facecolor("#f5f7f8")
    points = axis.scatter(
        valid["longitude"],
        valid["latitude"],
        c=valid["elevation_ft"],
        cmap="viridis",
        s=32,
        alpha=0.86,
        edgecolors="white",
        linewidths=0.35,
    )
    colorbar = figure.colorbar(points, ax=axis, pad=0.02)
    colorbar.set_label(f"Surveyed HWM elevation (ft, {datum})")
    axis.set_title(
        "USGS Hurricane Ian High-Water Marks · Lee County, Florida",
        loc="left",
        fontsize=15,
        fontweight="bold",
        color="#183247",
        pad=12,
    )
    axis.set_xlabel("Longitude (WGS84 coordinates)")
    axis.set_ylabel("Latitude (WGS84 coordinates)")
    axis.grid(color="#dfe5e8", linewidth=0.7, alpha=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    figure.text(
        0.01,
        0.01,
        "Source: USGS Short-Term Network / Flood Event Viewer, event 2022 Ian. "
        "Points are surveyed marks, not a continuous flood surface. Do not compare "
        "NAVD88 elevations directly with NOAA MHHW values.",
        fontsize=8.4,
        color="#4d596a",
    )
    figure.tight_layout(rect=(0, 0.055, 1, 1))
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return destination


def _record_summary(row: pd.Series) -> dict[str, Any]:
    return {
        "hwm_id": int(row["hwm_id"]),
        "site_no": str(row["site_no"]),
        "site_description": str(row["site_description"]),
        "latitude": round(float(row["latitude"]), 6),
        "longitude": round(float(row["longitude"]), 6),
        "elevation_ft": round(float(row["elevation_ft"]), 3),
        "vertical_datum": str(row["vertical_datum"]),
        "environment": str(row["environment"]),
        "quality": str(row["quality"]),
    }


def _counts(series: pd.Series) -> dict[str, int]:
    return {
        str(name): int(count)
        for name, count in series.fillna("Unknown").value_counts().items()
    }


def _required_float(record: Mapping[str, Any], key: str) -> float:
    value = pd.to_numeric(record.get(key), errors="coerce")
    if pd.isna(value):
        raise UsgsValidationError(f"USGS record is missing numeric {key}.")
    return float(value)


def _required_int(record: Mapping[str, Any], key: str) -> int:
    value = pd.to_numeric(record.get(key), errors="coerce")
    if pd.isna(value):
        raise UsgsValidationError(f"USGS record is missing integer {key}.")
    return int(value)


def _optional_float(value: Any) -> float | None:
    parsed = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(parsed) else float(parsed)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()
