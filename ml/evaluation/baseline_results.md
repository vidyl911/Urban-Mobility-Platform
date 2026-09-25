# XGBoost Baseline Results

## Model

**Algorithm:** XGBoost Regressor
**Task:** Traffic volume prediction
**Target:** `traffic_count_24h`

### Configuration

* `n_estimators`: 300
* `max_depth`: 6
* `learning_rate`: 0.05
* `subsample`: 0.8
* `colsample_bytree`: 0.8
* `random_state`: 42
* `n_jobs`: -1
* `objective`: `reg:squarederror`

## Dataset

* Total observations: 910
* Sensors: 10
* Time period: 2016-04-01 → 2016-06-30

### Chronological Split

| Dataset    | Period                  | Observations |
| ---------- | ----------------------- | -----------: |
| Train      | 2016-04-01 → 2016-05-31 |          610 |
| Validation | 2016-06-01 → 2016-06-15 |          150 |
| Test       | 2016-06-16 → 2016-06-30 |          150 |

A chronological split is used to preserve the temporal structure of traffic observations and avoid using future observations during training.

## Baseline Performance

| Split      |      MAE |     RMSE |     R² |
| ---------- | -------: | -------: | -----: |
| Validation | 1,612.81 | 2,930.49 | 0.9227 |
| Test       | 1,048.12 | 1,972.53 | 0.9598 |

## Interpretation

The baseline model achieves a test MAE of **1,048.12 vehicles per 24 hours**, meaning the average absolute prediction error is approximately 1,048 vehicles.

The test R² of **0.9598** indicates that the model explains approximately 96% of the variance in the held-out test observations.

The test period performs better than the validation period. This is possible because the two periods represent different temporal traffic patterns; the difference should be investigated during subsequent error analysis rather than interpreted as evidence of model improvement.

## Next Steps

1. Analyze XGBoost feature importance.
2. Examine prediction errors by sensor and time period.
3. Identify systematic sources of error.
4. Tune the XGBoost model based on the findings.
5. Compare the tuned model against this baseline.
