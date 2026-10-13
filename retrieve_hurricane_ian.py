"""Retrieve and validate the Hurricane Ian case-study data from NOAA."""

from datetime import date
import json

from hurricane_flood_agent.retrieval import (
    retrieve_and_save_thresholds,
    retrieve_noaa_data,
)
from hurricane_flood_agent.schemas import NoaaDataRequest


def main() -> None:
    request = NoaaDataRequest(
        station_id="8725520",
        begin_date=date(2022, 9, 27),
        end_date=date(2022, 9, 30),
        datum="MHHW",
        units="english",
        time_zone="gmt",
    )
    data_result = retrieve_noaa_data(request)
    threshold_result = retrieve_and_save_thresholds(request.station_id)
    summary = {
        "water_level_artifact": data_result,
        "threshold_artifact": threshold_result,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
