from pathlib import Path
import pandas as pd


SILVER_DIR = Path("data/silver")


def inspect_parquet(path):
    print("\n" + "=" * 80)
    print(f"FILE: {path}")
    print("=" * 80)

    df = pd.read_parquet(path)

    print("\nSHAPE")
    print(df.shape)

    print("\nCOLUMNS")
    print(df.columns.tolist())

    print("\nDTYPES")
    print(df.dtypes)

    print("\nMISSING VALUES")
    print(df.isnull().sum())

    print("\nSAMPLE")
    print(df.head(3))


files = [
    SILVER_DIR / "osm" / "roads.parquet",

    SILVER_DIR / "gtfs" / "routes.parquet",
    SILVER_DIR / "gtfs" / "stops.parquet",
    SILVER_DIR / "gtfs" / "trips.parquet",
    SILVER_DIR / "gtfs" / "stop_times.parquet",
    SILVER_DIR / "gtfs" / "shapes.parquet",

    SILVER_DIR / "traffic" / "traffic.parquet",

    SILVER_DIR / "weather" / "weather.parquet",

    SILVER_DIR / "pois" / "pois.parquet",
]


for file in files:
    if file.exists():
        inspect_parquet(file)
    else:
        print(f"\nMISSING FILE: {file}")