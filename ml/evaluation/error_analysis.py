import os
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, URL
from xgboost import XGBRegressor


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DB_USER = "postgres"
DB_PASSWORD = os.getenv("PGPASSWORD")
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "Urban_Mobility"

TABLE_NAME = "mobility.int_ml_features"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "ml" / "evaluation"

PREDICTIONS_PATH = OUTPUT_DIR / "baseline_predictions.csv"
SENSOR_ERROR_PATH = OUTPUT_DIR / "error_by_sensor.csv"
ROAD_ERROR_PATH = OUTPUT_DIR / "error_by_road.csv"
DAYTYPE_ERROR_PATH = OUTPUT_DIR / "error_by_day_type.csv"


# --------------------------------------------------
# Database connection
# --------------------------------------------------

if not DB_PASSWORD:
    raise ValueError("PGPASSWORD environment variable is not set.")

engine = create_engine(
    URL.create(
        "postgresql+psycopg2",
        username=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=int(DB_PORT),
        database=DB_NAME,
    )
)


# --------------------------------------------------
# Load ML features
# --------------------------------------------------

query = f"""
SELECT *
FROM {TABLE_NAME}
ORDER BY traffic_timestamp
"""

df = pd.read_sql(query, engine)

print(f"Loaded {len(df):,} rows")


# --------------------------------------------------
# Feature preparation
# --------------------------------------------------

target = "traffic_count_24h"

categorical_features = [
    "road_ref",
    "direction",
]

model_df = pd.get_dummies(
    df,
    columns=categorical_features,
    dtype=int,
)

X = model_df.drop(
    columns=[
        target,
        "traffic_location",
        "traffic_timestamp",
    ]
)

y = model_df[target]


# --------------------------------------------------
# Chronological split
# --------------------------------------------------

train_mask = (
    df["traffic_timestamp"].dt.date
    <= pd.Timestamp("2016-05-31").date()
)

validation_mask = (
    (df["traffic_timestamp"].dt.date
     >= pd.Timestamp("2016-06-01").date())
    &
    (df["traffic_timestamp"].dt.date
     <= pd.Timestamp("2016-06-15").date())
)

test_mask = (
    df["traffic_timestamp"].dt.date
    >= pd.Timestamp("2016-06-16").date()
)


X_train = X.loc[train_mask]
y_train = y.loc[train_mask]

X_validation = X.loc[validation_mask]
y_validation = y.loc[validation_mask]

X_test = X.loc[test_mask]
y_test = y.loc[test_mask]


print(f"Train rows:      {len(X_train)}")
print(f"Validation rows: {len(X_validation)}")
print(f"Test rows:       {len(X_test)}")


# --------------------------------------------------
# Train baseline XGBoost model
# --------------------------------------------------

model = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_validation, y_validation)],
    verbose=False,
)

print("\nBaseline model trained successfully.")


# --------------------------------------------------
# Generate predictions
# --------------------------------------------------

validation_predictions = model.predict(X_validation)
test_predictions = model.predict(X_test)

validation_results = df.loc[validation_mask].copy()
test_results = df.loc[test_mask].copy()

validation_results["prediction"] = validation_predictions
test_results["prediction"] = test_predictions

results = pd.concat(
    [validation_results, test_results],
    ignore_index=True,
)

results["error"] = (
    results[target] - results["prediction"]
)

results["absolute_error"] = (
    results["error"].abs()
)

results["percentage_error"] = (
    results["absolute_error"]
    / results[target].replace(0, np.nan)
    * 100
)

results["split"] = np.where(
    results["traffic_timestamp"].dt.date
    <= pd.Timestamp("2016-06-15").date(),
    "validation",
    "test",
)


# --------------------------------------------------
# Overall error summary
# --------------------------------------------------

print("\nOverall Error Summary")
print("-" * 50)

print(
    results.groupby("split")
    .agg(
        observations=(target, "size"),
        mae=("absolute_error", "mean"),
        rmse=("error", lambda x: np.sqrt(np.mean(x ** 2))),
        mean_percentage_error=("percentage_error", "mean"),
    )
    .round(2)
)


# --------------------------------------------------
# Error by sensor
# --------------------------------------------------

sensor_error = (
    results
    .groupby(["split", "traffic_location"])
    .agg(
        observations=(target, "size"),
        actual_mean=(target, "mean"),
        predicted_mean=("prediction", "mean"),
        mae=("absolute_error", "mean"),
        rmse=("error", lambda x: np.sqrt(np.mean(x ** 2))),
        mean_percentage_error=("percentage_error", "mean"),
    )
    .reset_index()
    .sort_values(
        ["split", "mae"],
        ascending=[True, False],
    )
)

print("\nError by Sensor")
print("-" * 50)

print(
    sensor_error.to_string(
        index=False,
        formatters={
            "actual_mean": "{:,.2f}".format,
            "predicted_mean": "{:,.2f}".format,
            "mae": "{:,.2f}".format,
            "rmse": "{:,.2f}".format,
            "mean_percentage_error": "{:.2f}".format,
        },
    )
)


# --------------------------------------------------
# Error by road
# --------------------------------------------------

road_error = (
    results
    .groupby(["split", "road_ref"])
    .agg(
        observations=(target, "size"),
        actual_mean=(target, "mean"),
        predicted_mean=("prediction", "mean"),
        mae=("absolute_error", "mean"),
        rmse=("error", lambda x: np.sqrt(np.mean(x ** 2))),
        mean_percentage_error=("percentage_error", "mean"),
    )
    .reset_index()
    .sort_values(
        ["split", "mae"],
        ascending=[True, False],
    )
)

print("\nError by Road")
print("-" * 50)

print(
    road_error.to_string(
        index=False,
        formatters={
            "actual_mean": "{:,.2f}".format,
            "predicted_mean": "{:,.2f}".format,
            "mae": "{:,.2f}".format,
            "rmse": "{:,.2f}".format,
            "mean_percentage_error": "{:.2f}".format,
        },
    )
)


# --------------------------------------------------
# Error by day type
# --------------------------------------------------

results["day_type"] = np.where(
    results["is_weekend"] == 1,
    "weekend",
    "weekday",
)

day_type_error = (
    results
    .groupby(["split", "day_type"])
    .agg(
        observations=(target, "size"),
        actual_mean=(target, "mean"),
        predicted_mean=("prediction", "mean"),
        mae=("absolute_error", "mean"),
        rmse=("error", lambda x: np.sqrt(np.mean(x ** 2))),
        mean_percentage_error=("percentage_error", "mean"),
    )
    .reset_index()
)

print("\nError by Day Type")
print("-" * 50)

print(
    day_type_error.to_string(
        index=False,
        formatters={
            "actual_mean": "{:,.2f}".format,
            "predicted_mean": "{:,.2f}".format,
            "mae": "{:,.2f}".format,
            "rmse": "{:,.2f}".format,
            "mean_percentage_error": "{:.2f}".format,
        },
    )
)


# --------------------------------------------------
# Save results
# --------------------------------------------------

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

results[
    [
        "traffic_location",
        "traffic_timestamp",
        target,
        "prediction",
        "error",
        "absolute_error",
        "percentage_error",
        "road_ref",
        "kilometer_marker",
        "direction",
        "is_weekend",
        "split",
    ]
].to_csv(
    PREDICTIONS_PATH,
    index=False,
)

sensor_error.to_csv(
    SENSOR_ERROR_PATH,
    index=False,
)

road_error.to_csv(
    ROAD_ERROR_PATH,
    index=False,
)

day_type_error.to_csv(
    DAYTYPE_ERROR_PATH,
    index=False,
)


print("\nSaved:")
print(PREDICTIONS_PATH)
print(SENSOR_ERROR_PATH)
print(ROAD_ERROR_PATH)
print(DAYTYPE_ERROR_PATH)


engine.dispose()