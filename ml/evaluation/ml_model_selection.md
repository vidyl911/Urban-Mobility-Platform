# ML Dataset and Hyperparameter Selection

## 1. Purpose

This document records the comparison and selection process used to define the canonical machine-learning dataset and XGBoost hyperparameters for the Urban Mobility & Traffic Congestion Prediction Platform.

The selection process is based on validation performance while keeping the test set isolated for final evaluation.

---

## 2. Candidate Feature Datasets

Two feature datasets were evaluated.

### Canonical Dataset — `mobility.int_ml_features`

* Observations: **910**
* Sensors: **10**
* Observation period: **2016-04-01 → 2016-06-30**
* Observations per sensor: **91**
* Features:

  * Traffic sensor/location features
  * Road-reference features
  * Temporal features
  * Weather features
  * OSM-derived road features
* Target: `traffic_count_24h`

Chronological split:

| Split      | Date range              | Observations |
| ---------- | ----------------------- | -----------: |
| Train      | 2016-04-01 → 2016-05-31 |          610 |
| Validation | 2016-06-01 → 2016-06-15 |          150 |
| Test       | 2016-06-16 → 2016-06-30 |          150 |

### Experimental Dataset — `mobility.int_ml_temporal_features`

This dataset extends the canonical feature set with:

`traffic_lag_7`

The feature represents the traffic count observed seven days earlier for the same traffic sensor.

Because the first seven observations for each of the 10 sensors have no lag value:

* Original observations: **910**
* Usable observations: **840**
* Train: **540**
* Validation: **150**
* Test: **150**

---

## 3. Feature Dataset Comparison

To make the comparison fair, the baseline model was retrained on the same 840 observations available to the lag-7 model.

### Matched Baseline — 840 observations

| Metric | Validation |     Test |
| ------ | ---------: | -------: |
| MAE    |   1,634.55 | 1,006.00 |
| RMSE   |   2,818.40 | 1,866.35 |
| R²     |     0.9285 |   0.9640 |

### Lag-7 XGBoost — 840 observations

| Metric | Validation |     Test |
| ------ | ---------: | -------: |
| MAE    |   1,914.30 |   945.38 |
| RMSE   |   3,067.82 | 1,790.25 |
| R²     |     0.9153 |   0.9669 |

### Comparison

The lag-7 feature produced slightly better test performance:

* Test MAE improved by approximately **6.0%**
* Test RMSE improved by approximately **4.1%**
* Test R² increased from **0.9640 → 0.9669**

However, validation performance deteriorated:

* Validation MAE increased by approximately **17.1%**
* Validation RMSE increased by approximately **8.9%**
* Validation R² decreased from **0.9285 → 0.9153**

The lag-7 feature therefore did not demonstrate consistent improvement across the validation and test periods.

---

## 4. Selected Feature Dataset

The canonical feature dataset is:

`mobility.int_ml_features`

### Selection rationale

The 910-observation dataset was retained because it:

1. Preserves all available observations.
2. Provides the largest training set.
3. Avoids unnecessary feature complexity.
4. Does not depend on seven-day historical target availability.
5. Produced stronger validation performance than the matched lag-7 experiment.
6. Provides a clean and reproducible baseline for productionization.

The lag-7 dataset remains an **experimental feature-engineering result**, not part of the canonical ML pipeline.

---

# 5. XGBoost Hyperparameter Tuning

After selecting the canonical 910-observation feature dataset, XGBoost hyperparameters were evaluated using the chronological validation set.

The test set was not used for hyperparameter selection.

---

## 6. Maximum Depth Experiment

The following values were evaluated:

`max_depth = 3, 4, 6, 8`

| max_depth | Validation MAE | Validation RMSE | Validation R² |
| --------: | -------------: | --------------: | ------------: |
|         3 |   **1,587.97** |    **2,798.54** |    **0.9295** |
|         4 |       1,657.82 |        2,946.68 |        0.9219 |
|         6 |       1,612.81 |        2,930.49 |        0.9227 |
|         8 |       1,593.10 |        2,936.45 |        0.9224 |

### Selection

`max_depth = 3`

Depth 3 produced the strongest validation performance across the primary validation metrics.

The better test result at deeper trees was not used to select the hyperparameter because the test set must remain an unbiased final evaluation set.

---

# 7. Learning Rate and Number of Estimators

Using `max_depth = 3`, combinations of learning rate and number of estimators were evaluated.

| Learning Rate | Estimators | Validation MAE | Validation RMSE | Validation R² |
| ------------: | ---------: | -------------: | --------------: | ------------: |
|          0.03 |        300 |   **1,573.67** |    **2,722.46** |    **0.9333** |
|          0.05 |        300 |       1,587.97 |        2,798.54 |        0.9295 |
|          0.05 |        500 |       1,617.53 |        2,841.04 |        0.9274 |
|          0.10 |        200 |       1,566.66 |        2,765.24 |        0.9312 |
|          0.10 |        300 |       1,598.44 |        2,791.17 |        0.9299 |

### Selection

Selected configuration:

```text
learning_rate = 0.03
n_estimators = 300
```

This configuration produced:

* Lowest validation RMSE
* Highest validation R²
* Strong validation MAE

The `0.10 / 200` configuration produced a slightly lower validation MAE, but the `0.03 / 300` configuration provided stronger overall validation performance across RMSE and R².

---

# 8. Selected XGBoost Configuration

The candidate canonical configuration is:

```python
XGBRegressor(
    objective="reg:squarederror",
    n_estimators=300,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)
```

The remaining parameters were kept fixed during the tuning experiments.

---

# 9. Selection Hierarchy

The ML selection process follows this hierarchy:

```text
Gold PostgreSQL data
        ↓
dbt staging
        ↓
dbt intermediate feature engineering
        ↓
mobility.int_ml_features
        ↓
910-observation canonical dataset
        ↓
Baseline XGBoost
        ↓
Feature importance
        ↓
Error analysis
        ↓
Targeted temporal experiment
        ↓
Lag-7 rejected for canonical pipeline
        ↓
Hyperparameter tuning
        ↓
Validation-based parameter selection
        ↓
Final test evaluation
        ↓
MLflow experiment tracking
```

---

# 10. Validation vs Test Policy

The chronological split is intentional.

### Training set

Used to fit the XGBoost model.

### Validation set

Used for:

* Feature comparison
* Hyperparameter selection
* Model configuration decisions

### Test set

Used only for:

* Final model evaluation
* Reporting final generalization performance

The test set is therefore not used to choose the feature dataset or hyperparameters.

---

# 11. Current ML Decision

### Canonical feature dataset

```text
mobility.int_ml_features
```

### Observations

```text
910
```

### Canonical split

```text
Train:      610
Validation: 150
Test:       150
```

### Selected XGBoost candidate

```text
max_depth       = 3
learning_rate   = 0.03
n_estimators    = 300
subsample       = 0.8
colsample_bytree = 0.8
random_state    = 42
n_jobs           = -1
```

### Next step

The selected configuration should now be evaluated once on the untouched test set and then registered/tracked through **MLflow**.

No further tuning should be performed against the test set.