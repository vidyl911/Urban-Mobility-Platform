{{ config(materialized='view') }}

SELECT
    traffic_location,
    date,
    traffic_count_24h,
    year,
    month,
    day,
    day_of_week,
    week_of_year,
    is_weekend
FROM {{ source('mobility', 'traffic') }}
