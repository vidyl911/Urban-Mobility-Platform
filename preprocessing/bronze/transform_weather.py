import json
import os
from datetime import datetime, timezone

import pandas as pd


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

INPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "weather",
    "milan_weather_2016-04-01_2016-06-30.json"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "bronze", "weather"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "weather.parquet"
)

INGESTION_TIMESTAMP = datetime.now(timezone.utc).isoformat()


def main():
    print("=" * 60)
    print("WEATHER BRONZE TRANSFORMATION")
    print("=" * 60)

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Weather JSON not found: {INPUT_FILE}"
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Reading: {INPUT_FILE}")

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    hourly = data["hourly"]

    df = pd.DataFrame(hourly)

    # Standardize column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    # Parse timestamp
    df["time"] = pd.to_datetime(
        df["time"],
        errors="coerce"
    )

    # Explicit numeric types
    numeric_columns = [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "pressure_msl",
        "wind_speed_10m",
        "wind_direction_10m",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    # Add location metadata
    df["latitude"] = data["latitude"]
    df["longitude"] = data["longitude"]
    df["elevation"] = data["elevation"]
    df["timezone"] = data["timezone"]

    # Add source metadata
    df["source"] = "Open-Meteo"
    df["ingestion_timestamp"] = INGESTION_TIMESTAMP

    # Write Parquet
    df.to_parquet(
        OUTPUT_FILE,
        engine="pyarrow",
        compression="snappy",
        index=False
    )

    print(f"Written: {OUTPUT_FILE}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {df.columns.tolist()}")

    print("=" * 60)
    print("WEATHER BRONZE TRANSFORMATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()