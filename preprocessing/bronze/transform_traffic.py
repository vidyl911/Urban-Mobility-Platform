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
    "traffic",
    "CITTA_METROPOLITANA_MILANO_-Historical_Traffic_Observations.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "bronze","traffic"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "traffic.parquet"
)

INGESTION_TIMESTAMP = datetime.now(timezone.utc).isoformat()


def main():
    print("=" * 60)
    print("TRAFFIC BRONZE TRANSFORMATION")
    print("=" * 60)

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Traffic CSV not found: {INPUT_FILE}"
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Reading: {INPUT_FILE}")

    df = pd.read_csv(
        INPUT_FILE,
        encoding="utf-8-sig",
        dtype=str
    )

    # Standardize column names
    df.columns = [
        "varco",
        "data",
        "passaggi_24h"
    ]

    # Parse observation date
    df["data"] = pd.to_datetime(
        df["data"],
        format="%d-%m-%Y",
        errors="coerce"
    )

    # Convert vehicle counts such as "11,033" -> 11033
    df["passaggi_24h"] = (
        df["passaggi_24h"]
        .str.replace(",", "", regex=False)
    )

    df["passaggi_24h"] = pd.to_numeric(
        df["passaggi_24h"],
        errors="coerce"
    ).astype("Int64")

    # Add Bronze metadata
    df["source"] = "Traffic_Count_Data"
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
    print("TRAFFIC BRONZE TRANSFORMATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()