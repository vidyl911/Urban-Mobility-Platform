{{ config(materialized='view') }}

SELECT
    timestamp,
    temperature_2m,
    relative_humidity_2m,
    precipitation,
    pressure_msl,
    wind_speed_10m,
    wind_direction_10m,
    latitude,
    longitude,
    date,
    hour,
    year,
    month,
    day_of_week,
    is_weekend,
    is_raining
FROM {{ source('mobility', 'weather') }}