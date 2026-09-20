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
    "pois",
    "milan_pois.json"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "bronze", "pois"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "pois.parquet"
)

INGESTION_TIMESTAMP = datetime.now(timezone.utc).isoformat()


def main():
    print("=" * 60)
    print("POI BRONZE TRANSFORMATION")
    print("=" * 60)

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"POI JSON not found: {INPUT_FILE}"
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Reading: {INPUT_FILE}")

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    elements = data.get("elements", [])

    if not elements:
        raise ValueError("No POI elements found in the JSON file.")

    records = []

    for element in elements:
        tags = element.get("tags", {})

        records.append({
            "osm_id": element.get("id"),
            "element_type": element.get("type"),
            "latitude": element.get("lat"),
            "longitude": element.get("lon"),
            "amenity": tags.get("amenity"),
            "shop": tags.get("shop"),
            "tourism": tags.get("tourism"),
            "leisure": tags.get("leisure"),
            "office": tags.get("office"),
            "name": tags.get("name"),
            "brand": tags.get("brand"),
            "operator": tags.get("operator"),
            "religion": tags.get("religion"),
            "source": "OpenStreetMap Overpass API",
            "ingestion_timestamp": INGESTION_TIMESTAMP,
        })

    df = pd.DataFrame(records)

    # Explicit numeric types
    df["osm_id"] = pd.to_numeric(
        df["osm_id"],
        errors="coerce"
    ).astype("Int64")

    df["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce"
    )

    df["longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce"
    )

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
    print("POI BRONZE TRANSFORMATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()