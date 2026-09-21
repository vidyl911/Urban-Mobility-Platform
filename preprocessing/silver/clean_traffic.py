import os
import pandas as pd


# ============================================================
# Paths
# ============================================================

INPUT_PATH = "data/bronze/traffic/traffic.parquet"
OUTPUT_DIR = "data/silver/traffic"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "traffic.parquet")


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

print("\nUnique sources:")
print(df["source"].value_counts(dropna=False))

print("\nTraffic statistics:")
print(df["passaggi_24h"].describe())


# ============================================================
# SILVER TRANSFORMATION [RUN AFTER INSPECTION]
# ============================================================


# ------------------------------------------------------------
# 1. Remove duplicate observations
# ------------------------------------------------------------

df = df.drop_duplicates(
    subset=["varco", "data"]
).copy()


# ------------------------------------------------------------
# 2. Clean road/location names
# ------------------------------------------------------------

df["varco"] = (
    df["varco"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df["varco"] = df["varco"].replace(
    "",
    pd.NA
)


# ------------------------------------------------------------
# 3. Normalize date
# ------------------------------------------------------------

df["data"] = pd.to_datetime(
    df["data"],
    errors="coerce"
)


# ------------------------------------------------------------
# 4. Normalize traffic counts
# ------------------------------------------------------------

df["passaggi_24h"] = pd.to_numeric(
    df["passaggi_24h"],
    errors="coerce"
).astype("Int64")


# ------------------------------------------------------------
# 5. Validate traffic counts
# ------------------------------------------------------------

# Negative traffic counts are invalid.
df.loc[
    df["passaggi_24h"] < 0,
    "passaggi_24h"
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

print("\nTraffic statistics:")
print(df["passaggi_24h"].describe())

print("\nDate range:")

if df["data"].notna().any():
    print(
        df["data"].min(),
        "to",
        df["data"].max()
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
    f"\nSilver traffic data written to: {OUTPUT_PATH}"
)
