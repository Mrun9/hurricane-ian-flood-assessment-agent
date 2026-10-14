from __future__ import annotations

import pandas as pd

from hurricane_flood_agent.analysis import calculate_flood_indicators
from hurricane_flood_agent.visualization import create_water_level_plot


def test_visualization_writes_nonempty_png(tmp_path) -> None:
    times = pd.date_range("2022-09-28", periods=4, freq="6min", tz="UTC")
    observations = pd.DataFrame({"timestamp": times, "value": [1.0, 2.0, 4.0, 3.0]})
    predictions = pd.DataFrame({"timestamp": times, "value": [0.5, 1.0, 1.5, 2.0]})
    thresholds = {
        "minor": 1.5,
        "moderate": 2.5,
        "major": 3.5,
        "datum": "MHHW",
        "units": "feet",
    }
    calculations = calculate_flood_indicators(
        observations, predictions, thresholds, interval_minutes=6
    )

    output = create_water_level_plot(
        observations,
        predictions,
        calculations,
        station_name="Synthetic Station",
        station_id="0000000",
        output_path=tmp_path / "plot.png",
    )

    assert output.is_file()
    assert output.stat().st_size > 10_000
