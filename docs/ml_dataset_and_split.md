# ML Dataset and Temporal Split

## 1. Purpose

The `int_ml_features` dbt model is the final ML-ready dataset used as input for the traffic prediction model.

It combines:

* Traffic observations
* Traffic sensor metadata
* Temporal features
* Weather conditions
* Road-network features derived from OpenStreetMap (OSM)

The target variable is:

`traffic_count_24h`

The objective is to predict daily traffic volume for a traffic sensor using the available temporal, weather, and road-network features.

---

## 2. Final Dataset

The final dataset is stored as:

`mobility.int_ml_features`

It contains **910 observations** covering **10 traffic sensors** over **91 days**.

### Target

| Column              | Description                    |
| ------------------- | ------------------------------ |
| `traffic_count_24h` | Observed 24-hour traffic count |

### Traffic / Sensor Features

| Column             | Description                                              |
| ------------------ | -------------------------------------------------------- |
| `traffic_location` | Original traffic sensor description                      |
| `road_ref`         | Normalized road reference                                |
| `kilometer_marker` | Kilometer position extracted from the sensor description |
| `direction`        | Direction associated with the traffic sensor             |

### Temporal Features

| Column         | Description       |
| -------------- | ----------------- |
| `year`         | Calendar year     |
| `month`        | Calendar month    |
| `day`          | Day of month      |
| `day_of_week`  | Day of week       |
| `week_of_year` | Week of year      |
| `is_weekend`   | Weekend indicator |

### Weather Features

| Column                 | Description             |
| ---------------------- | ----------------------- |
| `temperature_2m`       | Temperature at 2 m      |
| `relative_humidity_2m` | Relative humidity       |
| `precipitation`        | Precipitation           |
| `pressure_msl`         | Mean sea-level pressure |
| `wind_speed_10m`       | Wind speed at 10 m      |
| `wind_direction_10m`   | Wind direction          |
| `is_raining`           | Rain indicator          |

### Road-Network Features

Road attributes are aggregated at the road-reference level because the traffic dataset does not contain geographic coordinates for individual sensors.

| Column                     | Description                                                 |
| -------------------------- | ----------------------------------------------------------- |
| `road_segment_count`       | Number of OSM road segments belonging to the road reference |
| `avg_segment_length_m`     | Average OSM segment length                                  |
| `avg_maxspeed`             | Average available maximum speed                             |
| `avg_lanes`                | Average available lane count                                |
| `major_road_segment_count` | Number of major OSM road segments                           |
| `major_road_share`         | Share of segments classified as major roads                 |

This approach avoids fabricating an exact geographic relationship between a traffic sensor and an individual OSM road segment.

---

## 3. Data Quality Validation

The final ML dataset was validated before model training.

| Check                     |    Result |
| ------------------------- | --------: |
| Total observations        |       910 |
| Unique observations       |       910 |
| Null target values        |         0 |
| Null average speed values |         0 |
| Null average lane values  |         0 |
| Null temperature values   |         0 |
| Null precipitation values |         0 |
| Road features matched     | 910 / 910 |
| Road features unmatched   |         0 |

The final table therefore contains a complete set of observations for the selected features.

---

## 4. Available Time Period

The dataset covers:

**1 April 2016 → 30 June 2016**

Total coverage:

**91 days**

Each of the 10 traffic sensors contributes 91 daily observations, resulting in:

**10 × 91 = 910 observations**

---

## 5. Train / Validation / Test Strategy

A chronological split is used instead of a random train/test split.

The reason is that traffic observations are time-dependent. A random split could place observations from later dates into the training dataset while earlier observations from the same temporal period appear in validation or test data.

This can produce temporal leakage and an overly optimistic estimate of model performance.

The dataset is therefore divided chronologically.

### Training Set

**1 April 2016 → 31 May 2016**

* 61 days
* 610 observations
* Used to fit the XGBoost model

### Validation Set

**1 June 2016 → 15 June 2016**

* 15 days
* 150 observations
* Used for model selection and hyperparameter tuning

### Test Set

**16 June 2016 → 30 June 2016**

* 15 days
* 150 observations
* Reserved for final model evaluation

### Split Summary

| Dataset    | Date range                  |   Days | Observations | Purpose          |
| ---------- | --------------------------- | -----: | -----------: | ---------------- |
| Train      | 2016-04-01 → 2016-05-31     |     61 |          610 | Model training   |
| Validation | 2016-06-01 → 2016-06-15     |     15 |          150 | Model selection  |
| Test       | 2016-06-16 → 2016-06-30     |     15 |          150 | Final evaluation |
| **Total**  | **2016-04-01 → 2016-06-30** | **91** |      **910** |                  |

---

## 6. Why the Test Set Is Kept Separate

The test period represents the latest portion of the available historical data.

The model is not tuned against this period.

The intended workflow is:

```text
Training data
     ↓
Train XGBoost
     ↓
Validation data
     ↓
Select / tune model
     ↓
Final model
     ↓
Test data
     ↓
Final performance evaluation
```

This allows the test set to provide an estimate of how the trained model performs on observations occurring after the training and validation periods.

---

## 7. ML Pipeline Position

The final dataset is produced through the following transformation chain:

```text
Gold PostgreSQL tables
        ↓
    dbt staging
        ↓
Traffic + Weather
        ↓
Traffic + Sensor Features
        ↓
OSM Road Features
        ↓
Traffic + Road Features
        ↓
   int_ml_features
        ↓
Chronological Split
        ↓
   XGBoost Model
```

The resulting `int_ml_features` table is therefore the interface between the **data engineering/dbt layer** and the **machine-learning layer**.
