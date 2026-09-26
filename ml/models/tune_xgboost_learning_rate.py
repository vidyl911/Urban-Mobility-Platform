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

TABLE_NAME = "mobility.int_ml_features"


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
# Learning rate / estimator experiment
# --------------------------------------------------

configurations = [
    {"learning_rate": 0.03, "n_estimators": 300},
    {"learning_rate": 0.05, "n_estimators": 300},
    {"learning_rate": 0.05, "n_estimators": 500},
    {"learning_rate": 0.10, "n_estimators": 200},
    {"learning_rate": 0.10, "n_estimators": 300},
]

results = []


for config in configurations:

    learning_rate = config["learning_rate"]
    n_estimators = config["n_estimators"]

    print(
        f"\nTraining learning_rate={learning_rate}, "
        f"n_estimators={n_estimators}..."
    )

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=n_estimators,
        max_depth=3,
        learning_rate=learning_rate,
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

    validation_predictions = model.predict(X_validation)
    test_predictions = model.predict(X_test)

    validation_mae = mean_absolute_error(
        y_validation,
        validation_predictions,
    )

    validation_rmse = np.sqrt(
        mean_squared_error(
            y_validation,
            validation_predictions,
        )
    )

    validation_r2 = r2_score(
        y_validation,
        validation_predictions,
    )

    test_mae = mean_absolute_error(
        y_test,
        test_predictions,
    )

    test_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            test_predictions,
        )
    )

    test_r2 = r2_score(
        y_test,
        test_predictions,
    )

    results.append(
        {
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "validation_mae": validation_mae,
            "validation_rmse": validation_rmse,
            "validation_r2": validation_r2,
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "test_r2": test_r2,
        }
    )


# --------------------------------------------------
# Results
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("\nXGBoost Learning Rate / Estimator Results")
print("-" * 100)

print(
    results_df.to_string(
        index=False,
        formatters={
            "learning_rate": "{:.2f}".format,
            "validation_mae": "{:,.2f}".format,
            "validation_rmse": "{:,.2f}".format,
            "validation_r2": "{:.4f}".format,
            "test_mae": "{:,.2f}".format,
            "test_rmse": "{:,.2f}".format,
            "test_r2": "{:.4f}".format,
        },
    )
)