import os
from datetime import datetime, timezone

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

INPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "gtfs"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "bronze",
    "gtfs"
)

INGESTION_TIMESTAMP = datetime.now(timezone.utc).isoformat()


def standardize_columns(df):
    """Convert column names to lowercase snake_case."""
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    return df


def add_metadata(df):
    """Add Bronze metadata columns."""
    df["source"] = "ATM_Milan_GTFS"
    df["ingestion_timestamp"] = INGESTION_TIMESTAMP
    return df


def transform_routes():
    print("Processing routes.txt...")

    file_path = os.path.join(INPUT_DIR, "routes.txt")
    output_path = os.path.join(OUTPUT_DIR, "routes.parquet")

    df = pd.read_csv(
        file_path,
        encoding="utf-8-sig",
        dtype=str
    )

    df = standardize_columns(df)

    df["route_type"] = pd.to_numeric(
        df["route_type"],
        errors="coerce"
    ).astype("Int64")

    df = add_metadata(df)

    df.to_parquet(
        output_path,
        engine="pyarrow",
        compression="snappy",
        index=False
    )

    print(f"  Written: {output_path}")
    print(f"  Rows: {len(df):,}")


def transform_stops():
    print("Processing stops.txt...")

    file_path = os.path.join(INPUT_DIR, "stops.txt")
    output_path = os.path.join(OUTPUT_DIR, "stops.parquet")

    df = pd.read_csv(
        file_path,
        encoding="utf-8-sig",
        dtype=str
    )

    df = standardize_columns(df)

    df["stop_lat"] = pd.to_numeric(
        df["stop_lat"],
        errors="coerce"
    )

    df["stop_lon"] = pd.to_numeric(
        df["stop_lon"],
        errors="coerce"
    )

    df["location_type"] = pd.to_numeric(
        df["location_type"],
        errors="coerce"
    ).astype("Int64")

    df["wheelchair_boarding"] = pd.to_numeric(
        df["wheelchair_boarding"],
        errors="coerce"
    ).astype("Int64")

    df = add_metadata(df)

    df.to_parquet(
        output_path,
        engine="pyarrow",
        compression="snappy",
        index=False
    )

    print(f"  Written: {output_path}")
    print(f"  Rows: {len(df):,}")


def transform_trips():
    print("Processing trips.txt...")

    file_path = os.path.join(INPUT_DIR, "trips.txt")
    output_path = os.path.join(OUTPUT_DIR, "trips.parquet")

    df = pd.read_csv(
        file_path,
        encoding="utf-8-sig",
        dtype=str
    )

    df = standardize_columns(df)

    df["direction_id"] = pd.to_numeric(
        df["direction_id"],
        errors="coerce"
    ).astype("Int64")

    df["wheelchair_accessible"] = pd.to_numeric(
        df["wheelchair_accessible"],
        errors="coerce"
    ).astype("Int64")

    df["x_shape_id_order"] = pd.to_numeric(
        df["x_shape_id_order"],
        errors="coerce"
    ).astype("Int64")

    df = add_metadata(df)

    df.to_parquet(
        output_path,
        engine="pyarrow",
        compression="snappy",
        index=False
    )

    print(f"  Written: {output_path}")
    print(f"  Rows: {len(df):,}")


def transform_stop_times():
    print("Processing stop_times.txt in chunks...")

    file_path = os.path.join(INPUT_DIR, "stop_times.txt")
    output_path = os.path.join(OUTPUT_DIR, "stop_times.parquet")

    chunk_size = 100_000
    writer = None
    total_rows = 0

    string_columns = [
        "trip_id",
        "arrival_time",
        "departure_time",
        "stop_id",
        "stop_headsign",
    ]

    try:
        for chunk in pd.read_csv(
            file_path,
            encoding="utf-8-sig",
            dtype=str,
            chunksize=chunk_size
        ):
            chunk = standardize_columns(chunk)

            # Explicitly keep string columns as strings.
            for column in string_columns:
                if column in chunk.columns:
                    chunk[column] = chunk[column].fillna("").astype(str)

            chunk["stop_sequence"] = pd.to_numeric(
                chunk["stop_sequence"],
                errors="coerce"
            ).astype("Int64")

            chunk["pickup_type"] = pd.to_numeric(
                chunk["pickup_type"],
                errors="coerce"
            ).astype("Int64")

            chunk["drop_off_type"] = pd.to_numeric(
                chunk["drop_off_type"],
                errors="coerce"
            ).astype("Int64")

            chunk["shape_dist_traveled"] = pd.to_numeric(
                chunk["shape_dist_traveled"],
                errors="coerce"
            )

            chunk = add_metadata(chunk)

            table = pa.Table.from_pandas(
                chunk,
                preserve_index=False
            )

            if writer is None:
                writer = pq.ParquetWriter(
                    output_path,
                    table.schema,
                    compression="snappy"
                )

            writer.write_table(table)

            total_rows += len(chunk)

            print(f"  Processed: {total_rows:,} rows")

    finally:
        if writer is not None:
            writer.close()

    print(f"  Written: {output_path}")
    print(f"  Rows: {total_rows:,}")

def transform_shapes():
    print("Processing shapes.txt...")

    file_path = os.path.join(INPUT_DIR, "shapes.txt")
    output_path = os.path.join(OUTPUT_DIR, "shapes.parquet")

    df = pd.read_csv(
        file_path,
        encoding="utf-8-sig",
        dtype=str
    )

    df = standardize_columns(df)

    df["shape_pt_lat"] = pd.to_numeric(
        df["shape_pt_lat"],
        errors="coerce"
    )

    df["shape_pt_lon"] = pd.to_numeric(
        df["shape_pt_lon"],
        errors="coerce"
    )

    df["shape_pt_sequence"] = pd.to_numeric(
        df["shape_pt_sequence"],
        errors="coerce"
    ).astype("Int64")

    df["shape_dist_traveled"] = pd.to_numeric(
        df["shape_dist_traveled"],
        errors="coerce"
    )

    df = add_metadata(df)

    df.to_parquet(
        output_path,
        engine="pyarrow",
        compression="snappy",
        index=False
    )

    print(f"  Written: {output_path}")
    print(f"  Rows: {len(df):,}")


def main():
    print("=" * 60)
    print("GTFS BRONZE TRANSFORMATION")
    print("=" * 60)

    if not os.path.exists(INPUT_DIR):
        raise FileNotFoundError(
            f"GTFS input directory not found: {INPUT_DIR}"
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    transform_routes()
    transform_stops()
    transform_trips()
    transform_stop_times()
    transform_shapes()

    print("=" * 60)
    print("GTFS BRONZE TRANSFORMATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()