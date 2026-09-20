import os
import pandas as pd
import geopandas as gpd


# ============================================================
# Paths
# ============================================================

INPUT_PATH = "data/bronze/osm/roads.parquet"
OUTPUT_DIR = "data/silver/osm"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "roads.parquet")


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

print("\nUnique highway types:")
print(df["highway"].value_counts(dropna=False).head(30))

print("\nUnique maxspeed values:")
print(df["maxspeed"].value_counts(dropna=False).head(30))

print("\nUnique lanes values:")
print(df["lanes"].value_counts(dropna=False).head(30))

print("\nUnique oneway values:")
print(df["oneway"].value_counts(dropna=False))

print("\nUnique surface values:")
print(df["surface"].value_counts(dropna=False).head(30))

print("\nUnique bridge values:")
print(df["bridge"].value_counts(dropna=False))

print("\nUnique tunnel values:")
print(df["tunnel"].value_counts(dropna=False))


# ============================================================
# SILVER TRANSFORMATION [RUN AFTER INSPECTION]
# ============================================================

# ------------------------------------------------------------
# 1. Remove duplicate road records
# ------------------------------------------------------------

df = df.drop_duplicates(subset="osm_id").copy()


# ------------------------------------------------------------
# 2. Clean text columns
# ------------------------------------------------------------

text_columns = [
    "highway",
    "name",
    "ref",
    "maxspeed",
    "lanes",
    "oneway",
    "surface",
    "bridge",
    "tunnel",
]

for column in text_columns:
    df[column] = df[column].fillna("").astype(str).str.strip()

    # Convert empty strings back to missing values
    df[column] = df[column].replace("", pd.NA)


# ------------------------------------------------------------
# 3. Convert maxspeed to numeric
# ------------------------------------------------------------

df["maxspeed"] = (
    df["maxspeed"]
    .str.extract(r"(\d+(?:\.\d+)?)", expand=False)
)

df["maxspeed"] = pd.to_numeric(
    df["maxspeed"],
    errors="coerce"
)


# ------------------------------------------------------------
# 4. Convert lanes to numeric
# ------------------------------------------------------------

df["lanes"] = (
    df["lanes"]
    .str.extract(r"(\d+(?:\.\d+)?)", expand=False)
)

df["lanes"] = pd.to_numeric(
    df["lanes"],
    errors="coerce"
)


# ------------------------------------------------------------
# 5. Normalize boolean fields
# ------------------------------------------------------------

def normalize_boolean(value):

    if pd.isna(value):
        return pd.NA

    value = str(value).lower().strip()

    if value in ["yes", "true", "1"]:
        return True

    if value in ["no", "false", "0"]:
        return False

    return pd.NA


for column in ["oneway", "bridge", "tunnel"]:
    df[column] = (
        df[column]
        .apply(normalize_boolean)
        .astype("boolean")
    )


# ------------------------------------------------------------
# 6. Convert WKT geometry to GeoDataFrame
# ------------------------------------------------------------

geometry = gpd.GeoSeries.from_wkt(
    df["geometry"],
    crs="EPSG:4326"
)

gdf = gpd.GeoDataFrame(
    df,
    geometry=geometry
)


# ------------------------------------------------------------
# 7. Remove invalid geometries
# ------------------------------------------------------------

gdf = gdf[gdf.geometry.notna()].copy()

gdf = gdf[
    gdf.geometry.is_valid
].copy()


# ------------------------------------------------------------
# 8. Keep only road LineStrings
# ------------------------------------------------------------

gdf = gdf[
    gdf.geometry.geom_type == "LineString"
].copy()


# ------------------------------------------------------------
# 9. Calculate road length
# ------------------------------------------------------------

# Project to a metric CRS before calculating length.
metric_gdf = gdf.to_crs("EPSG:32632")

gdf["length_m"] = metric_gdf.geometry.length


# ------------------------------------------------------------
# 10. Restore WGS84
# ------------------------------------------------------------

gdf = gdf.to_crs("EPSG:4326")


# ============================================================
# SILVER VALIDATION
# ============================================================

print("\n\n================ SILVER VALIDATION ================")

print("\nShape after transformation:")
print(gdf.shape)

print("\nColumns:")
print(gdf.columns.tolist())

print("\nData types:")
print(gdf.dtypes)

print("\nMissing values:")
print(gdf.isnull().sum())

print("\nGeometry types:")
print(gdf.geometry.geom_type.value_counts())

print("\nRoad length statistics:")
print(gdf["length_m"].describe())


# ============================================================
# WRITE SILVER DATA
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

gdf.to_parquet(
    OUTPUT_PATH,
    index=False,
    engine="pyarrow"
)

print(
    f"\nSilver OSM data written to: {OUTPUT_PATH}"
)


