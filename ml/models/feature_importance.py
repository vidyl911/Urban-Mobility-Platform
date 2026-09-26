
import os
from pathlib import Path

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
OUTPUT_PATH = OUTPUT_DIR / "feature_importance.csv"


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

df = pd.get_dummies(
    df,
    columns=categorical_features,
    dtype=int,
)

X = df.drop(
    columns=[
        target,
        "traffic_location",
        "traffic_timestamp",
    ]
)

y = df[target]


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

X_train = X.loc[train_mask]
y_train = y.loc[train_mask]

X_validation = X.loc[validation_mask]
y_validation = y.loc[validation_mask]

print(f"Train rows:      {len(X_train)}")
print(f"Validation rows: {len(X_validation)}")


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
# Feature importance
# --------------------------------------------------

# Use average gain: the average improvement in
# the model's loss from splits using each feature.

booster = model.get_booster()

importance_scores = booster.get_score(
    importance_type="gain"
)

importance_df = pd.DataFrame({
    "feature": X_train.columns,
    "importance_gain": [
        importance_scores.get(feature, 0.0)
        for feature in X_train.columns
    ],
})

importance_df = importance_df.sort_values(
    by="importance_gain",
    ascending=False,
).reset_index(drop=True)

importance_df["rank"] = (
    importance_df.index + 1
)

importance_df = importance_df[
    ["rank", "feature", "importance_gain"]
]


# --------------------------------------------------
# Display top 15 features
# --------------------------------------------------

print("\nTop 15 Features by Gain")
print("-" * 50)

print(
    importance_df.head(15).to_string(
        index=False,
        formatters={
            "importance_gain": "{:,.4f}".format
        },
    )
)


# --------------------------------------------------
# Save feature importance
# --------------------------------------------------

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

importance_df.to_csv(
    OUTPUT_PATH,
    index=False,
)

print(f"\nFeature importance saved to:")
print(OUTPUT_PATH)

engine.dispose()