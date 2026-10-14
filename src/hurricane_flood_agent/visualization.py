"""Publication-ready deterministic plots for the Hurricane Ian case study."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

_matplotlib_cache = Path(tempfile.gettempdir()) / "hurricane-flood-agent-matplotlib"
_matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_matplotlib_cache))

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


THRESHOLD_STYLES = {
    "minor": ("#e6a700", "Minor"),
    "moderate": ("#e66b00", "Moderate"),
    "major": ("#b51f2e", "Major"),
}


def create_water_level_plot(
    observations: pd.DataFrame,
    predictions: pd.DataFrame,
    calculations: Mapping[str, Any],
    *,
    station_name: str,
    station_id: str,
    output_path: str | Path,
) -> Path:
    """Plot observations, predicted tide, thresholds, and the recorded peak."""

    observed = observations.copy()
    predicted = predictions.copy()
    observed["timestamp"] = pd.to_datetime(observed["timestamp"], utc=True)
    predicted["timestamp"] = pd.to_datetime(predicted["timestamp"], utc=True)

    figure, axis = plt.subplots(figsize=(12, 6.75))
    figure.patch.set_facecolor("white")
    axis.set_facecolor("#fbfcfe")

    axis.plot(
        predicted["timestamp"],
        predicted["value"],
        color="#5e81ac",
        linewidth=1.35,
        label="Predicted astronomical tide",
        zorder=2,
    )
    axis.plot(
        observed["timestamp"],
        observed["value"],
        color="#14213d",
        linewidth=2.0,
        label="Observed water level",
        zorder=4,
    )

    thresholds = calculations["thresholds"]
    for name, (color, label) in THRESHOLD_STYLES.items():
        value = float(thresholds[name])
        axis.axhline(
            value,
            color=color,
            linewidth=1.15,
            linestyle="--",
            alpha=0.9,
            label=f"{label} threshold ({value:.2f} ft)",
            zorder=1,
        )

    peak = calculations["peak_observed"]
    peak_time = pd.Timestamp(peak["timestamp_utc"])
    peak_value = float(peak["water_level_ft_mhhw"])
    axis.scatter(
        [peak_time],
        [peak_value],
        s=70,
        color="#d90429",
        edgecolor="white",
        linewidth=1.0,
        zorder=6,
        label=f"Recorded peak ({peak_value:.3f} ft)",
    )
    axis.annotate(
        f"Peak {peak_value:.3f} ft\n{peak_time:%Y-%m-%d %H:%M UTC}",
        xy=(peak_time, peak_value),
        xytext=(18, -12),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=9.5,
        color="#8f1023",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#d90429",
            "alpha": 0.95,
        },
        arrowprops={"arrowstyle": "->", "color": "#d90429", "linewidth": 1.0},
    )

    axis.set_title(
        f"Hurricane Ian Water Levels at {station_name} ({station_id})\n"
        "September 27–30, 2022 · six-minute samples · UTC",
        loc="left",
        fontsize=15,
        fontweight="bold",
        color="#14213d",
        pad=14,
    )
    axis.set_ylabel("Water level (ft above MHHW)")
    axis.set_xlabel("Time (UTC)")
    axis.set_xlim(observed["timestamp"].min(), observed["timestamp"].max())
    axis.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12], tz=mdates.UTC))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M", tz=mdates.UTC))
    axis.grid(axis="y", color="#d6dbe3", linewidth=0.7, alpha=0.8)
    axis.grid(axis="x", color="#e5e8ed", linewidth=0.5, alpha=0.5)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(
        loc="upper left",
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#d6dbe3",
        ncol=2,
        fontsize=9,
    )
    figure.text(
        0.01,
        0.01,
        "Source: NOAA/NOS/CO-OPS saved Milestone 2 artifacts. "
        "Current NOS thresholds are applied retrospectively.",
        fontsize=8.5,
        color="#4d596a",
    )
    figure.tight_layout(rect=(0, 0.045, 1, 1))

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return destination.resolve()
