"""LangChain tool adapters for the deterministic flood-assessment modules."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from hurricane_flood_agent.analysis import (
    build_analysis_document,
    calculate_flood_indicators as calculate_indicators,
    derive_thresholds_from_provenance,
    load_milestone2_artifacts,
    save_analysis_document,
)
from hurricane_flood_agent.artifacts import ArtifactStore
from hurricane_flood_agent.retrieval import (
    retrieve_and_save_thresholds,
    retrieve_noaa_data as retrieve_and_save_noaa_data,
)
from hurricane_flood_agent.schemas import NoaaDataRequest
from hurricane_flood_agent.visualization import create_water_level_plot


ARTIFACT_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


class RetrieveNoaaDataInput(BaseModel):
    """Input schema presented to the model for NOAA water-level retrieval."""

    station_id: str = Field(description="Seven-digit NOAA station identifier.")
    begin_date: str = Field(description="Inclusive start date in YYYY-MM-DD format.")
    end_date: str = Field(description="Inclusive end date in YYYY-MM-DD format.")
    datum: str = Field(default="MHHW", description="NOAA vertical datum.")
    units: str = Field(default="english", description="NOAA units: english or metric.")
    use_cached: bool = Field(
        default=True,
        description="Reuse an exact validated local artifact when available.",
    )


class FloodThresholdInput(BaseModel):
    """Input schema for authoritative station-specific flood thresholds."""

    station_id: str = Field(description="Seven-digit NOAA station identifier.")
    use_cached: bool = Field(
        default=True,
        description="Reuse the newest validated local threshold artifact when available.",
    )


class FloodIndicatorInput(BaseModel):
    """Optional explicit artifact selection for flood calculations."""

    data_artifact_id: str = Field(
        default="",
        description="Water-level artifact ID; omit to use the active session artifact.",
    )
    threshold_artifact_id: str = Field(
        default="",
        description="Threshold artifact ID; omit to use the active session artifact.",
    )


class VisualizationInput(BaseModel):
    """Optional artifact selection for visualization generation."""

    analysis_artifact_id: str = Field(
        default="",
        description="Analysis artifact ID; omit to use the active session analysis.",
    )


@dataclass
class AgentSessionState:
    """Artifact-backed state shared by tools across model/tool iterations and turns."""

    artifact_root: Path = field(default_factory=lambda: Path("artifacts").resolve())
    active_data_artifact_id: str | None = None
    active_threshold_artifact_id: str | None = None
    active_analysis_artifact_id: str | None = None
    active_visualization_path: Path | None = None
    analysis_document: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.artifact_root = Path(self.artifact_root).resolve()

    def data_directory(self, artifact_id: str | None = None) -> Path | None:
        selected = artifact_id or self.active_data_artifact_id
        return _artifact_directory(self.artifact_root, "noaa", selected)

    def threshold_directory(self, artifact_id: str | None = None) -> Path | None:
        selected = artifact_id or self.active_threshold_artifact_id
        return _artifact_directory(self.artifact_root, "thresholds", selected)

    def analysis_directory(self, artifact_id: str | None = None) -> Path | None:
        selected = artifact_id or self.active_analysis_artifact_id
        return _artifact_directory(self.artifact_root, "analysis", selected)

    def snapshot(self) -> dict[str, Any]:
        return {
            "active_data_artifact_id": self.active_data_artifact_id,
            "active_threshold_artifact_id": self.active_threshold_artifact_id,
            "active_analysis_artifact_id": self.active_analysis_artifact_id,
            "active_visualization_path": (
                str(self.active_visualization_path)
                if self.active_visualization_path is not None
                else None
            ),
        }


def create_flood_tools(state: AgentSessionState) -> list[BaseTool]:
    """Create the four state-aware LangChain tools required by the project."""

    def retrieve_noaa_data(
        station_id: str,
        begin_date: str,
        end_date: str,
        datum: str = "MHHW",
        units: str = "english",
        use_cached: bool = True,
    ) -> str:
        try:
            request = NoaaDataRequest(
                station_id=station_id,
                begin_date=date.fromisoformat(begin_date),
                end_date=date.fromisoformat(end_date),
                datum=datum,
                units=units,
                time_zone="gmt",
            )
            expected_id = _data_artifact_id(request)
            directory = state.artifact_root / "noaa" / expected_id
            cache_used = use_cached and directory.is_dir()
            if cache_used:
                metadata = _validate_cached_data_artifact(directory, request)
            else:
                retrieval = retrieve_and_save_noaa_data(
                    request,
                    artifact_store=ArtifactStore(state.artifact_root),
                )
                directory = Path(retrieval["artifact_path"])
                metadata = _read_json(directory / "metadata.json")

            artifact_id = str(metadata["artifact_id"])
            if artifact_id != state.active_data_artifact_id:
                state.active_analysis_artifact_id = None
                state.analysis_document = None
                state.active_visualization_path = None
            state.active_data_artifact_id = artifact_id
            validation = metadata["validation"]
            warnings = [
                *validation["observations"].get("warnings", []),
                *validation["predictions"].get("warnings", []),
            ]
            return _json_result(
                status="ok",
                tool="retrieve_noaa_data",
                cache_used=cache_used,
                artifact_id=state.active_data_artifact_id,
                artifact_path=str(directory.resolve()),
                validated_request=metadata["request"],
                station=metadata["station"],
                observations={
                    "valid_records": validation["observations"]["valid_value_count"],
                    "expected_records": validation["observations"]["expected_record_count"],
                    "missing_or_invalid": validation["observations"][
                        "missing_or_invalid_count"
                    ],
                    "quality_codes": validation["observations"][
                        "quality_code_counts"
                    ],
                },
                predictions={
                    "valid_records": validation["predictions"]["valid_value_count"],
                    "expected_records": validation["predictions"]["expected_record_count"],
                    "missing_or_invalid": validation["predictions"][
                        "missing_or_invalid_count"
                    ],
                },
                warnings=warnings,
                source_url=metadata["source_url"],
            )
        except Exception as error:
            return _error_result("retrieve_noaa_data", error)

    def get_flood_thresholds(
        station_id: str,
        use_cached: bool = True,
    ) -> str:
        try:
            _validate_station_id(station_id)
            directory = _newest_threshold_directory(state.artifact_root, station_id)
            cache_used = use_cached and directory is not None
            if cache_used and directory is not None:
                thresholds, provenance = _validate_cached_threshold_artifact(
                    directory, station_id
                )
                artifact_id = directory.name
            else:
                retrieval = retrieve_and_save_thresholds(
                    station_id,
                    artifact_store=ArtifactStore(state.artifact_root),
                )
                directory = Path(retrieval["artifact_path"])
                artifact_id = str(retrieval["artifact_id"])
                thresholds = _read_json(directory / "thresholds.json")
                provenance = _read_json(directory / "source_responses.json")
                _validate_threshold_conversion(thresholds, provenance)

            existing_analysis = state.analysis_document
            if artifact_id != state.active_threshold_artifact_id:
                state.active_analysis_artifact_id = None
                state.analysis_document = None
                state.active_visualization_path = None
            state.active_threshold_artifact_id = artifact_id
            result: dict[str, Any] = {
                "status": "ok",
                "tool": "get_flood_thresholds",
                "cache_used": cache_used,
                "artifact_id": artifact_id,
                "artifact_path": str(directory.resolve()),
                "station_id": thresholds["station_id"],
                "station_name": thresholds["station_name"],
                "datum": thresholds["datum"],
                "units": thresholds["units"],
                "thresholds": {
                    name: thresholds[name] for name in ("minor", "moderate", "major")
                },
                "threshold_type": thresholds["threshold_type"],
                "retrospective_caveat": thresholds["retrospective_caveat"],
                "source_urls": thresholds["source_urls"],
                "warnings": [thresholds["retrospective_caveat"]],
            }
            if existing_analysis is not None and state.analysis_document is not None:
                result["active_peak_classification"] = existing_analysis[
                    "calculations"
                ]["peak_observed"]["flood_category"]
            return json.dumps(result, sort_keys=True, allow_nan=False)
        except Exception as error:
            return _error_result("get_flood_thresholds", error)

    def calculate_flood_indicators(
        data_artifact_id: str = "",
        threshold_artifact_id: str = "",
    ) -> str:
        try:
            selected_data_id = data_artifact_id or state.active_data_artifact_id
            selected_threshold_id = (
                threshold_artifact_id or state.active_threshold_artifact_id
            )
            missing: list[str] = []
            suggested: list[str] = []
            if selected_data_id is None:
                missing.append("validated NOAA water-level artifact")
                suggested.append("retrieve_noaa_data")
            if selected_threshold_id is None:
                missing.append("verified NOAA/NOS threshold artifact")
                suggested.append("get_flood_thresholds")
            if missing:
                return _dependency_error(
                    "calculate_flood_indicators", missing, suggested
                )

            data_directory = state.data_directory(selected_data_id)
            threshold_directory = state.threshold_directory(selected_threshold_id)
            if data_directory is None or not data_directory.is_dir():
                return _dependency_error(
                    "calculate_flood_indicators",
                    [f"water-level artifact {selected_data_id!r}"],
                    ["retrieve_noaa_data"],
                )
            if threshold_directory is None or not threshold_directory.is_dir():
                return _dependency_error(
                    "calculate_flood_indicators",
                    [f"threshold artifact {selected_threshold_id!r}"],
                    ["get_flood_thresholds"],
                )

            inputs = load_milestone2_artifacts(data_directory, threshold_directory)
            calculations = calculate_indicators(
                inputs.observations,
                inputs.predictions,
                inputs.thresholds,
                interval_minutes=inputs.metadata["request"]["interval_minutes"],
            )
            document = build_analysis_document(inputs, calculations)
            analysis_id = f"analysis_{selected_data_id}"
            output_directory = state.artifact_root / "analysis" / analysis_id
            output_path = save_analysis_document(
                document, output_directory / "analysis_results.json"
            )

            state.active_data_artifact_id = selected_data_id
            state.active_threshold_artifact_id = selected_threshold_id
            state.active_analysis_artifact_id = analysis_id
            state.analysis_document = document
            state.active_visualization_path = None

            alignment = calculations["data_alignment"]
            quality = calculations["observation_quality"]
            warnings = [calculations["thresholds"]["retrospective_caveat"]]
            if alignment["observation_only_timestamp_count"]:
                warnings.append("Some observations have no matching prediction.")
            if alignment["prediction_only_timestamp_count"]:
                warnings.append("Some predictions have no matching observation.")
            if quality["missing_observed_value_count"]:
                warnings.append("Some observed water levels are missing or invalid.")

            return _json_result(
                status="ok",
                tool="calculate_flood_indicators",
                analysis_artifact_id=analysis_id,
                analysis_path=str(output_path),
                data_artifact_id=selected_data_id,
                threshold_artifact_id=selected_threshold_id,
                peak_observed=calculations["peak_observed"],
                maximum_residual=calculations["maximum_residual"],
                thresholds=calculations["thresholds"],
                threshold_exceedances=calculations["threshold_exceedances"],
                data_alignment=alignment,
                observation_quality=quality,
                sigma=calculations["sigma"],
                warnings=warnings,
                calculation_method="Deterministic pandas calculations; no LLM involved.",
            )
        except Exception as error:
            return _error_result("calculate_flood_indicators", error)

    def generate_visualization(
        analysis_artifact_id: str = "",
    ) -> str:
        try:
            selected_analysis_id = (
                analysis_artifact_id or state.active_analysis_artifact_id
            )
            if selected_analysis_id is None:
                return _dependency_error(
                    "generate_visualization",
                    ["completed flood analysis artifact"],
                    ["calculate_flood_indicators"],
                )
            analysis_directory = state.analysis_directory(selected_analysis_id)
            if analysis_directory is None or not analysis_directory.is_dir():
                return _dependency_error(
                    "generate_visualization",
                    [f"analysis artifact {selected_analysis_id!r}"],
                    ["calculate_flood_indicators"],
                )
            if state.active_data_artifact_id is None:
                return _dependency_error(
                    "generate_visualization",
                    ["active validated NOAA water-level artifact"],
                    ["retrieve_noaa_data", "calculate_flood_indicators"],
                )
            if state.active_threshold_artifact_id is None:
                return _dependency_error(
                    "generate_visualization",
                    ["active verified threshold artifact"],
                    ["get_flood_thresholds", "calculate_flood_indicators"],
                )

            document = state.analysis_document
            if document is None or selected_analysis_id != state.active_analysis_artifact_id:
                document = _read_json(analysis_directory / "analysis_results.json")
            inputs = load_milestone2_artifacts(
                state.data_directory(), state.threshold_directory()
            )
            station = inputs.metadata["station"]
            output_path = create_water_level_plot(
                inputs.observations,
                inputs.predictions,
                document["calculations"],
                station_name=station["name"],
                station_id=station["id"],
                output_path=analysis_directory / "water_levels.png",
            )
            state.active_analysis_artifact_id = selected_analysis_id
            state.analysis_document = document
            state.active_visualization_path = output_path
            return _json_result(
                status="ok",
                tool="generate_visualization",
                analysis_artifact_id=selected_analysis_id,
                visualization_path=str(output_path),
                format="PNG",
                station=station,
                datum=inputs.metadata["request"]["datum"],
                units=inputs.metadata["request"]["unit_label"],
                time_zone="UTC",
                peak_observed=document["calculations"]["peak_observed"],
                warnings=[],
            )
        except Exception as error:
            return _error_result("generate_visualization", error)

    return [
        StructuredTool.from_function(
            func=retrieve_noaa_data,
            name="retrieve_noaa_data",
            description=(
                "Retrieve and validate NOAA six-minute observed water levels and "
                "predicted astronomical tides, or reuse an exact validated local "
                "artifact. MUST be used before making event-specific claims about "
                "historical station measurements, peaks, residuals, or flood severity. "
                "Returns compact artifact metadata, record counts, and quality details, "
                "not the oversized raw time series."
            ),
            args_schema=RetrieveNoaaDataInput,
        ),
        StructuredTool.from_function(
            func=calculate_flood_indicators,
            name="calculate_flood_indicators",
            description=(
                "Calculate peak observed level, exact-time observed-minus-predicted "
                "residual, threshold exceedances, data gaps, and sigma summaries. "
                "MUST be used for quantitative peak, residual, threshold, duration, "
                "or flood-severity answers. Requires active validated water-level and "
                "threshold artifacts; follow suggested_tools if either is missing."
            ),
            args_schema=FloodIndicatorInput,
        ),
        StructuredTool.from_function(
            func=get_flood_thresholds,
            name="get_flood_thresholds",
            description=(
                "Load or retrieve authoritative station-specific NOAA/NOS minor, "
                "moderate, and major coastal flood thresholds with datum conversion "
                "and source URLs. MUST be used before classifying actual recorded "
                "water levels by flood severity."
            ),
            args_schema=FloodThresholdInput,
        ),
        StructuredTool.from_function(
            func=generate_visualization,
            name="generate_visualization",
            description=(
                "Generate the academic PNG time-series visualization from an active "
                "completed analysis. Use whenever the user requests a visualization, "
                "plot, graph, chart, or figure. Requires the analysis and its "
                "validated inputs; follow suggested_tools if analysis is missing."
            ),
            args_schema=VisualizationInput,
        ),
    ]


def _validate_cached_data_artifact(
    directory: Path,
    request: NoaaDataRequest,
) -> dict[str, Any]:
    metadata = _read_json(directory / "metadata.json")
    if metadata.get("artifact_id") != directory.name:
        raise ValueError("Cached NOAA artifact ID does not match its directory.")
    saved_request = metadata.get("request", {})
    expected_request = request.to_dict()
    for field_name in (
        "station_id",
        "begin_date",
        "end_date",
        "datum",
        "units",
        "unit_label",
        "time_zone",
        "interval_minutes",
    ):
        if saved_request.get(field_name) != expected_request.get(field_name):
            raise ValueError(
                f"Cached NOAA artifact request mismatch for {field_name}."
            )
    for filename, expected_hash in metadata.get("sha256", {}).items():
        path = directory / filename
        if not path.is_file() or _sha256(path) != expected_hash:
            raise ValueError(f"Cached NOAA artifact failed integrity check: {filename}")
    return metadata


def _validate_cached_threshold_artifact(
    directory: Path,
    station_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    thresholds = _read_json(directory / "thresholds.json")
    provenance = _read_json(directory / "source_responses.json")
    if str(thresholds.get("station_id")) != station_id:
        raise ValueError("Cached threshold station does not match the request.")
    _validate_threshold_conversion(thresholds, provenance)
    return thresholds, provenance


def _validate_threshold_conversion(
    thresholds: dict[str, Any],
    provenance: dict[str, Any],
) -> None:
    derived = derive_thresholds_from_provenance(provenance)
    for name in ("minor", "moderate", "major"):
        if float(thresholds[name]) != derived[name]:
            raise ValueError(f"Cached {name} threshold does not match its source.")
    if thresholds.get("datum") != "MHHW" or thresholds.get("units") != "feet":
        raise ValueError("Threshold artifact must use feet relative to MHHW.")


def _newest_threshold_directory(root: Path, station_id: str) -> Path | None:
    candidates = sorted(
        (root / "thresholds").glob(f"noaa_nos_thresholds_{station_id}_*"),
        reverse=True,
    )
    return next((path for path in candidates if path.is_dir()), None)


def _data_artifact_id(request: NoaaDataRequest) -> str:
    return (
        f"noaa_{request.station_id}_{request.begin_date:%Y%m%d}_"
        f"{request.end_date:%Y%m%d}_{request.datum.lower()}_"
        f"{request.units}_{request.time_zone}"
    )


def _validate_station_id(station_id: str) -> None:
    NoaaDataRequest(station_id, date(2000, 1, 1), date(2000, 1, 1))


def _artifact_directory(root: Path, category: str, artifact_id: str | None) -> Path | None:
    if artifact_id is None:
        return None
    if not ARTIFACT_ID_PATTERN.fullmatch(artifact_id):
        raise ValueError(f"Invalid artifact identifier: {artifact_id!r}")
    return root / category / artifact_id


def _dependency_error(
    tool_name: str,
    missing: list[str],
    suggested_tools: list[str],
) -> str:
    return _json_result(
        status="error",
        tool=tool_name,
        error_type="missing_prerequisite",
        message="Required artifact state is not available.",
        missing=missing,
        suggested_tools=list(dict.fromkeys(suggested_tools)),
        retryable=True,
        warnings=[],
    )


def _error_result(tool_name: str, error: Exception) -> str:
    return _json_result(
        status="error",
        tool=tool_name,
        error_type=type(error).__name__,
        message=str(error),
        retryable=False,
        warnings=[],
    )


def _json_result(**values: Any) -> str:
    return json.dumps(values, sort_keys=True, allow_nan=False)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()
