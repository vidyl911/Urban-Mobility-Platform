import os
import pandas as pd


# ============================================================
# Paths
# ============================================================

INPUT_PATH = "data/bronze/pois/pois.parquet"
OUTPUT_DIR = "data/silver/pois"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "pois.parquet")


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

print("\nElement types:")
print(df["element_type"].value_counts(dropna=False))

print("\nAmenity types:")
print(df["amenity"].value_counts(dropna=False).head(30))

print("\nShop types:")
print(df["shop"].value_counts(dropna=False).head(30))

print("\nTourism types:")
print(df["tourism"].value_counts(dropna=False).head(30))

print("\nLeisure types:")
print(df["leisure"].value_counts(dropna=False).head(30))

print("\nOffice types:")
print(df["office"].value_counts(dropna=False).head(30))


# ============================================================
# SILVER TRANSFORMATION
# ============================================================

# ------------------------------------------------------------
# 1. Remove duplicate OSM elements
# ------------------------------------------------------------

df = df.drop_duplicates(
    subset=["osm_id", "element_type"]
).copy()


# ------------------------------------------------------------
# 2. Normalize text columns
# ------------------------------------------------------------

text_columns = [
    "element_type",
    "amenity",
    "shop",
    "tourism",
    "leisure",
    "office",
    "name",
    "brand",
    "operator",
    "religion",
]

for column in text_columns:

    if column in df.columns:

        df[column] = (
            df[column]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )

        df[column] = df[column].replace(
            "",
            pd.NA
        )


# ------------------------------------------------------------
# 3. Normalize numeric columns
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 4. Validate coordinates
# ------------------------------------------------------------

df.loc[
    ~df["latitude"].between(-90, 90),
    "latitude"
] = pd.NA

df.loc[
    ~df["longitude"].between(-180, 180),
    "longitude"
] = pd.NA


# ------------------------------------------------------------
# 5. Remove records without valid coordinates
# ------------------------------------------------------------

df = df[
    df["latitude"].notna()
    & df["longitude"].notna()
].copy()


# ------------------------------------------------------------
# 6. Create a primary POI category
# ------------------------------------------------------------

# OSM POIs can have different classification tags.
# Use the first available category in a consistent order.

df["poi_category"] = (
    df["amenity"]
    .fillna(
        df["shop"]
    )
    .fillna(
        df["tourism"]
    )
    .fillna(
        df["leisure"]
    )
    .fillna(
        df["office"]
    )
)


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

print("\nPOI categories:")
print(
    df["poi_category"]
    .value_counts(dropna=False)
    .head(30)
)

print("\nCoordinate range:")

print(
    "Latitude:",
    df["latitude"].min(),
    "to",
    df["latitude"].max()
)

print(
    "Longitude:",
    df["longitude"].min(),
    "to",
    df["longitude"].max()
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
    f"\nSilver POI data written to: {OUTPUT_PATH}"
)
