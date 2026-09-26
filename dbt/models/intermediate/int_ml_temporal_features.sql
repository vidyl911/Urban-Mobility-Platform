{{ config(materialized='view') }}

SELECT
    *,
    
    LAG(traffic_count_24h, 7) OVER (
        PARTITION BY traffic_location
        ORDER BY traffic_timestamp
    ) AS traffic_lag_7

FROM {{ ref('int_ml_features') }}