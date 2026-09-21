import os
import pandas as pd


# ============================================================
# Paths
# ============================================================

INPUT_PATH = "data/bronze/weather/weather.parquet"
OUTPUT_DIR = "data/silver/weather"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "weather.parquet")


# ============================================================
# Load Bronze data
# ============================================================

df = pd.read_parquet(INPUT_PATH)


# ============================================================
# INSPECTION
# ============================================================

print("Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nFirst 5 rows:")
print(df.head().to_string())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDate range:")
print(df["time"].min(), "to", df["time"].max())

print("\nWeather statistics:")
print(
    df[
        [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "pressure_msl",
            "wind_speed_10m",
            "wind_direction_10m",
        ]
    ].describe()
)


# ============================================================
# SILVER TRANSFORMATION [RUN AFTER INSPECTION]
# ============================================================

# ------------------------------------------------------------
# 1. Remove duplicate timestamps
# ------------------------------------------------------------

df = df.drop_duplicates(
    subset=["time"]
).copy()


# ------------------------------------------------------------
# 2. Normalize timestamp
# ------------------------------------------------------------

df["time"] = pd.to_datetime(
    df["time"],
    errors="coerce"
)


# ------------------------------------------------------------
# 3. Normalize numeric columns
# ------------------------------------------------------------

numeric_columns = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "pressure_msl",
    "wind_speed_10m",
    "wind_direction_10m",
    "latitude",
    "longitude",
    "elevation",
]

for column in numeric_columns:

    if column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


# ------------------------------------------------------------
# 4. Validate weather ranges
# ------------------------------------------------------------

# Relative humidity must be between 0 and 100%.
df.loc[
    ~df["relative_humidity_2m"].between(0, 100),
    "relative_humidity_2m"
] = pd.NA


# Precipitation cannot be negative.
df.loc[
    df["precipitation"] < 0,
    "precipitation"
] = pd.NA


# Wind speed cannot be negative.
df.loc[
    df["wind_speed_10m"] < 0,
    "wind_speed_10m"
] = pd.NA


# Wind direction should be between 0 and 360 degrees.
df.loc[
    ~df["wind_direction_10m"].between(0, 360),
    "wind_direction_10m"
] = pd.NA


# ------------------------------------------------------------
# 5. Validate coordinates
# ------------------------------------------------------------

df.loc[
    ~df["latitude"].between(-90, 90),
    "latitude"
] = pd.NA

df.loc[
    ~df["longitude"].between(-180, 180),
    "longitude"
] = pd.NA


# ============================================================
# SILVER VALIDATION
# ============================================================

print("\n\n" + "=" * 60)
print("SILVER VALIDATION")
print("=" * 60)

print("\nShape after transformation:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDate range:")
print(
    df["time"].min(),
    "to",
    df["time"].max()
)

print("\nWeather statistics:")
print(
    df[
        [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "pressure_msl",
            "wind_speed_10m",
            "wind_direction_10m",
        ]
    ].describe()
)


# ============================================================
# WRITE SILVER DATA
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

df.to_parquet(
    OUTPUT_PATH,
    index=False,
    engine="pyarrow"
)

print(
    f"\nSilver weather data written to: {OUTPUT_PATH}"
)