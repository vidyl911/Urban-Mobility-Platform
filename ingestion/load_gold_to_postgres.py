import os
from pathlib import Path

import pyarrow.parquet as pq
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL


# ============================================================
# PostgreSQL Configuration
# ============================================================

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "Urban_Mobility",
    "username": "postgres",
    "password": os.getenv("PGPASSWORD"),
}

if not DB_CONFIG["password"]:
    raise ValueError(
        "PGPASSWORD environment variable is not set."
    )


# ============================================================
# Database Connection
# ============================================================

connection_url = URL.create(
    drivername="postgresql+psycopg2",
    username=DB_CONFIG["username"],
    password=DB_CONFIG["password"],
    host=DB_CONFIG["host"],
    port=DB_CONFIG["port"],
    database=DB_CONFIG["database"],
)

engine = create_engine(
    connection_url,
    pool_pre_ping=True,
)


# ============================================================
# Configuration
# ============================================================

BASE_PATH = Path("data/gold")

DATASETS = {
    "roads": BASE_PATH / "roads",
    "gtfs": BASE_PATH / "gtfs",
    "weather": BASE_PATH / "weather",
    "pois": BASE_PATH / "pois",
    "traffic": BASE_PATH / "traffic",
}

SCHEMA = "mobility"
BATCH_SIZE = 50_000


# ============================================================
# Create Schema
# ============================================================

with engine.begin() as connection:

    connection.execute(
        text(
            f"""
            CREATE SCHEMA IF NOT EXISTS {SCHEMA};
            """
        )
    )

print("PostgreSQL connection successful.")
print(f"Schema ready: {SCHEMA}")


# ============================================================
# Load Dataset
# ============================================================

def load_dataset(table_name, dataset_path):

    print("\n" + "=" * 60)
    print(f"Loading: {table_name}")
    print(f"Source: {dataset_path}")
    print("=" * 60)

    parquet_files = list(
        dataset_path.glob("*.parquet")
    )

    if not parquet_files:
        raise FileNotFoundError(
            f"No Parquet files found in {dataset_path}"
        )

    first_batch = True
    total_rows = 0

    for parquet_path in parquet_files:

        parquet_file = pq.ParquetFile(parquet_path)

        for batch in parquet_file.iter_batches(
            batch_size=BATCH_SIZE
        ):

            df = batch.to_pandas()

            # ------------------------------------------------
            # Roads geometry
            # ------------------------------------------------

            if (
                table_name == "roads"
                and "geometry" in df.columns
            ):

                df = df.rename(
                    columns={
                        "geometry": "geometry_wkb"
                    }
                )

                df["geometry_wkb"] = df[
                    "geometry_wkb"
                ].apply(
                    lambda value:
                    value.hex()
                    if isinstance(
                        value,
                        (bytes, bytearray, memoryview)
                    )
                    else None
                )

            # ------------------------------------------------
            # Write batch
            # ------------------------------------------------

            df.to_sql(
                name=table_name,
                con=engine,
                schema=SCHEMA,
                if_exists=(
                    "replace"
                    if first_batch
                    else "append"
                ),
                index=False,
                chunksize=1_000,
                method="multi",
            )

            total_rows += len(df)
            first_batch = False

            print(
                f"{table_name}: "
                f"{total_rows:,} rows loaded"
            )

    print(
        f"Completed {table_name}: "
        f"{total_rows:,} rows"
    )


# ============================================================
# Load All Gold Datasets
# ============================================================

try:

    for table_name, dataset_path in DATASETS.items():

        load_dataset(
            table_name,
            dataset_path
        )

    print("\n" + "=" * 60)
    print("ALL GOLD DATASETS LOADED SUCCESSFULLY")
    print("=" * 60)

finally:

    engine.dispose()