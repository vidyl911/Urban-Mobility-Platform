import os

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, URL
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DB_USER = "postgres"
DB_PASSWORD = os.getenv("PGPASSWORD")
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "Urban_Mobility"

TABLE_NAME = "mobility.int_ml_temporal_features"


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
print(
    f"Date range: {df['traffic_timestamp'].min()} → "
    f"{df['traffic_timestamp'].max()}"
)


# --------------------------------------------------
# Match the Lag-7 experiment
# --------------------------------------------------

df = df.dropna(subset=["traffic_lag_7"]).copy()

print(f"Rows after matching Lag-7 dataset: {len(df):,}")


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
        "traffic_lag_7",
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
    (df["traffic_timestamp"].dt.date >= pd.Timestamp("2016-06-01").date())
    & (df["traffic_timestamp"].dt.date <= pd.Timestamp("2016-06-15").date())
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
# XGBoost model
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


# --------------------------------------------------
# Model training
# --------------------------------------------------

model.fit(
    X_train,
    y_train,
    eval_set=[(X_validation, y_validation)],
    verbose=False,
)

print("\nMatched baseline model trained successfully.")


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

def evaluate(name, X_data, y_data):
    predictions = model.predict(X_data)

    mae = mean_absolute_error(y_data, predictions)
    rmse = np.sqrt(mean_squared_error(y_data, predictions))
    r2 = r2_score(y_data, predictions)

    print(f"\n{name}")
    print("-" * 40)
    print(f"MAE:  {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"R²:   {r2:.4f}")


evaluate("Validation", X_validation, y_validation)
evaluate("Test", X_test, y_test)