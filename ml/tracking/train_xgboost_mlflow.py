import os
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from sqlalchemy import URL, create_engine
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


# --------------------------------------------------
# Database configuration
# --------------------------------------------------

DB_USER = "postgres"
DB_PASSWORD = os.getenv("PGPASSWORD")
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "Urban_Mobility"
TABLE_NAME = "mobility.int_ml_features"

if not DB_PASSWORD:
    raise ValueError("PGPASSWORD environment variable is not set.")


# --------------------------------------------------
# Database connection
# --------------------------------------------------

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
# Load ML dataset
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
# MLflow configuration
# --------------------------------------------------

mlflow.set_experiment("urban-mobility-xgboost")

run_name = "xgboost_final_candidate"


# --------------------------------------------------
# Model training and tracking
# --------------------------------------------------

with mlflow.start_run(run_name=run_name):

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
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

    # --------------------------------------------------
    # Predictions
    # --------------------------------------------------

    validation_predictions = model.predict(X_validation)
    test_predictions = model.predict(X_test)

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Log parameters
    # --------------------------------------------------

    mlflow.log_params(
        {
            "objective": "reg:squarederror",
            "n_estimators": 300,
            "max_depth": 3,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
        }
    )

    # --------------------------------------------------
    # Log dataset metadata
    # --------------------------------------------------

    mlflow.log_params(
        {
            "dataset": TABLE_NAME,
            "total_observations": len(df),
            "train_observations": len(X_train),
            "validation_observations": len(X_validation),
            "test_observations": len(X_test),
            "target": target,
            "feature_count": X.shape[1],
        }
    )

    mlflow.log_param(
        "feature_engineering",
        "traffic + weather + temporal + road-reference features",
    )

    # --------------------------------------------------
    # Log metrics
    # --------------------------------------------------

    mlflow.log_metrics(
        {
            "validation_mae": validation_mae,
            "validation_rmse": validation_rmse,
            "validation_r2": validation_r2,
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "test_r2": test_r2,
        }
    )

    # --------------------------------------------------
    # Log model
    # --------------------------------------------------

    mlflow.xgboost.log_model(
        model,
        artifact_path="xgboost_model",
    )

    # --------------------------------------------------
    # Output
    # --------------------------------------------------

    print("\nMLflow run completed.")

    print("\nValidation")
    print(f"MAE:  {validation_mae:,.2f}")
    print(f"RMSE: {validation_rmse:,.2f}")
    print(f"R²:   {validation_r2:.4f}")

    print("\nTest")
    print(f"MAE:  {test_mae:,.2f}")
    print(f"RMSE: {test_rmse:,.2f}")
    print(f"R²:   {test_r2:.4f}")

    print(f"\nRun ID: {mlflow.active_run().info.run_id}")