import os
import pandas as pd
from sympy import python

# ============================================================
# Paths
# ============================================================

INPUT_DIR = "data/bronze/gtfs"
OUTPUT_DIR = "data/silver/gtfs"


# ============================================================
# Load Bronze data
# ============================================================

routes = pd.read_parquet(
    os.path.join(INPUT_DIR, "routes.parquet")
)

stops = pd.read_parquet(
    os.path.join(INPUT_DIR, "stops.parquet")
)

trips = pd.read_parquet(
    os.path.join(INPUT_DIR, "trips.parquet")
)

stop_times = pd.read_parquet(
    os.path.join(INPUT_DIR, "stop_times.parquet")
)

shapes = pd.read_parquet(
    os.path.join(INPUT_DIR, "shapes.parquet")
)


# ============================================================
# INSPECTION
# ============================================================

datasets = {
    "routes": routes,
    "stops": stops,
    "trips": trips,
    "stop_times": stop_times,
    "shapes": shapes,
}

for name, df in datasets.items():

    print("\n" + "=" * 60)
    print(f"{name.upper()} INSPECTION")
    print("=" * 60)

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nFirst 5 rows:")
    print(df.head().to_string())

    print("\nMissing values:")
    print(df.isnull().sum())


# ============================================================
# SILVER TRANSFORMATION [RUN AFTER INSPECTION]
# ============================================================


# ============================================================
# 1. ROUTES
# ============================================================

routes = routes.drop_duplicates(
    subset="route_id"
).copy()

route_text_columns = [
    "route_id",
    "agency_id",
    "route_short_name",
    "route_long_name",
    "route_desc",
    "route_url",
    "route_color",
    "route_text_color",
    "x_accessibilita_linea",
    "x_ordinamento_linea",
]

for column in route_text_columns:

    if column in routes.columns:
        routes[column] = (
            routes[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        routes[column] = routes[column].replace(
            "",
            pd.NA
        )


# GTFS route_type is numeric.
routes["route_type"] = pd.to_numeric(
    routes["route_type"],
    errors="coerce"
).astype("Int64")


# ============================================================
# 2. STOPS
# ============================================================

stops = stops.drop_duplicates(
    subset="stop_id"
).copy()

stop_text_columns = [
    "stop_id",
    "stop_code",
    "stop_name",
    "stop_desc",
    "zone_id",
    "stop_url",
    "parent_station",
    "stop_timezone",
]

for column in stop_text_columns:

    if column in stops.columns:
        stops[column] = (
            stops[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        stops[column] = stops[column].replace(
            "",
            pd.NA
        )


# Convert coordinates to numeric
stops["stop_lat"] = pd.to_numeric(
    stops["stop_lat"],
    errors="coerce"
)

stops["stop_lon"] = pd.to_numeric(
    stops["stop_lon"],
    errors="coerce"
)


# Validate coordinate ranges
stops.loc[
    ~stops["stop_lat"].between(-90, 90),
    "stop_lat"
] = pd.NA

stops.loc[
    ~stops["stop_lon"].between(-180, 180),
    "stop_lon"
] = pd.NA


# GTFS location_type
if "location_type" in stops.columns:

    stops["location_type"] = pd.to_numeric(
        stops["location_type"],
        errors="coerce"
    ).astype("Int64")


# GTFS wheelchair_boarding
if "wheelchair_boarding" in stops.columns:

    stops["wheelchair_boarding"] = pd.to_numeric(
        stops["wheelchair_boarding"],
        errors="coerce"
    ).astype("Int64")


# ============================================================
# 3. TRIPS
# ============================================================

trips = trips.drop_duplicates(
    subset="trip_id"
).copy()


# Text columns
trip_text_columns = [
    "route_id",
    "service_id",
    "trip_id",
    "trip_headsign",
    "trip_short_name",
    "block_id",
    "shape_id",
    "x_trip_desc",
]

for column in trip_text_columns:

    if column in trips.columns:

        trips[column] = (
            trips[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        trips[column] = trips[column].replace(
            "",
            pd.NA
        )


# Numeric columns
if "direction_id" in trips.columns:

    trips["direction_id"] = pd.to_numeric(
        trips["direction_id"],
        errors="coerce"
    ).astype("Int64")


if "wheelchair_accessible" in trips.columns:

    trips["wheelchair_accessible"] = pd.to_numeric(
        trips["wheelchair_accessible"],
        errors="coerce"
    ).astype("Int64")


if "x_shape_id_order" in trips.columns:

    trips["x_shape_id_order"] = pd.to_numeric(
        trips["x_shape_id_order"],
        errors="coerce"
    ).astype("Int64")


# ============================================================
# 4. STOP TIMES
# ============================================================

stop_times = stop_times.drop_duplicates(
    subset=[
        "trip_id",
        "stop_sequence"
    ]
).copy()

stop_time_text_columns = [
    "trip_id",
    "arrival_time",
    "departure_time",
    "stop_id",
    "stop_headsign",
]

for column in stop_time_text_columns:

    if column in stop_times.columns:

        stop_times[column] = (
            stop_times[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        stop_times[column] = stop_times[column].replace(
            "",
            pd.NA
        )


# Stop sequence is numeric
stop_times["stop_sequence"] = pd.to_numeric(
    stop_times["stop_sequence"],
    errors="coerce"
).astype("Int64")


# Pickup/drop-off types are numeric
for column in [
    "pickup_type",
    "drop_off_type",
]:

    if column in stop_times.columns:

        stop_times[column] = pd.to_numeric(
            stop_times[column],
            errors="coerce"
        ).astype("Int64")


# Distance traveled
if "shape_dist_traveled" in stop_times.columns:

    stop_times["shape_dist_traveled"] = pd.to_numeric(
        stop_times["shape_dist_traveled"],
        errors="coerce"
    )


# ============================================================
# 5. SHAPES
# ============================================================

shapes = shapes.drop_duplicates(
    subset=[
        "shape_id",
        "shape_pt_sequence"
    ]
).copy()

shapes["shape_id"] = (
    shapes["shape_id"]
    .fillna("")
    .astype(str)
    .str.strip()
)

shapes["shape_pt_lat"] = pd.to_numeric(
    shapes["shape_pt_lat"],
    errors="coerce"
)

shapes["shape_pt_lon"] = pd.to_numeric(
    shapes["shape_pt_lon"],
    errors="coerce"
)

shapes["shape_pt_sequence"] = pd.to_numeric(
    shapes["shape_pt_sequence"],
    errors="coerce"
).astype("Int64")


if "shape_dist_traveled" in shapes.columns:

    shapes["shape_dist_traveled"] = pd.to_numeric(
        shapes["shape_dist_traveled"],
        errors="coerce"
    )


# Validate shape coordinates
shapes.loc[
    ~shapes["shape_pt_lat"].between(-90, 90),
    "shape_pt_lat"
] = pd.NA

shapes.loc[
    ~shapes["shape_pt_lon"].between(-180, 180),
    "shape_pt_lon"
] = pd.NA


# ============================================================
# SILVER VALIDATION
# ============================================================

silver_datasets = {
    "routes": routes,
    "stops": stops,
    "trips": trips,
    "stop_times": stop_times,
    "shapes": shapes,
}

print("\n\n" + "=" * 60)
print("SILVER VALIDATION")
print("=" * 60)

for name, df in silver_datasets.items():

    print("\n" + "-" * 60)
    print(name.upper())

    print(f"\nShape: {df.shape}")

    print("\nMissing values:")
    print(df.isnull().sum())


# ============================================================
# WRITE SILVER DATA
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

for name, df in silver_datasets.items():

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{name}.parquet"
    )

    df.to_parquet(
        output_path,
        index=False,
        engine="pyarrow"
    )

    print(
        f"\nWritten: {output_path}"
    )
