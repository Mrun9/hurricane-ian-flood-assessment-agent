"""Small typed data structures shared by retrieval and artifact modules."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any


ALLOWED_DATUMS = {"MHHW", "MHW", "MTL", "MSL", "MLW", "MLLW", "NAVD", "STND"}
ALLOWED_UNITS = {"english", "metric"}


@dataclass(frozen=True)
class NoaaDataRequest:
    """Validated parameters for matching NOAA observations and predictions."""

    station_id: str
    begin_date: date
    end_date: date
    datum: str = "MHHW"
    units: str = "english"
    time_zone: str = "gmt"
    interval_minutes: int = 6

    def __post_init__(self) -> None:
        if not re.fullmatch(r"\d{7}", self.station_id):
            raise ValueError("station_id must contain exactly seven digits.")
        if self.begin_date > self.end_date:
            raise ValueError("begin_date must be on or before end_date.")
        inclusive_days = (self.end_date - self.begin_date).days + 1
        if inclusive_days > 31:
            raise ValueError("Six-minute NOAA requests are limited to 31 days.")
        if self.datum not in ALLOWED_DATUMS:
            raise ValueError(f"Unsupported NOAA datum: {self.datum}")
        if self.units not in ALLOWED_UNITS:
            raise ValueError(f"Unsupported NOAA units: {self.units}")
        if self.time_zone != "gmt":
            raise ValueError("This project requires time_zone='gmt' for UTC consistency.")
        if self.interval_minutes != 6:
            raise ValueError("This project currently requires six-minute data.")

    @property
    def unit_label(self) -> str:
        return "feet" if self.units == "english" else "meters"

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["begin_date"] = self.begin_date.isoformat()
        result["end_date"] = self.end_date.isoformat()
        result["unit_label"] = self.unit_label
        return result


@dataclass(frozen=True)
class FloodThresholdSet:
    """NOAA/NOS flood thresholds expressed relative to one vertical datum."""

    station_id: str
    station_name: str
    datum: str
    units: str
    minor: float
    moderate: float
    major: float
    retrieved_at_utc: str
    source_urls: tuple[str, ...]
    source_name: str = "NOAA/NOS/CO-OPS Metadata API"
    threshold_type: str = "NOAA/NOS derived coastal flood thresholds"
    definition: str = (
        "Station-specific minor, moderate, and major elevations supporting a "
        "consistent national definition of coastal flooding. These derived NOS "
        "thresholds may differ from local NWS impact thresholds."
    )
    retrospective_caveat: str = (
        "These are current NOAA/NOS thresholds retrieved after Hurricane Ian. "
        "Applying them to September 2022 is a retrospective classification."
    )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["source_urls"] = list(self.source_urls)
        return result
