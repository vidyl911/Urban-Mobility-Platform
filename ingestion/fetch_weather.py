from __future__ import annotations

import json
from pathlib import Path

import requests


# Milan
LATITUDE = 45.4642
LONGITUDE = 9.1900

START_DATE = "2016-04-01"
END_DATE = "2016-06-30"

API_URL = "https://archive-api.open-meteo.com/v1/archive"

OUTPUT_DIR = Path("data/raw/weather")
OUTPUT_FILE = OUTPUT_DIR / f"milan_weather_{START_DATE}_{END_DATE}.json"


def fetch_weather() -> None:
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ",".join(
            [
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation",
                "pressure_msl",
                "wind_speed_10m",
                "wind_direction_10m",
            ]
        ),
        "timezone": "Europe/Rome",
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
    }

    response = requests.get(
        API_URL,
        params=params,
        timeout=60,
    )
    response.raise_for_status()

    data = response.json()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)

    print(f"Weather data saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    fetch_weather()