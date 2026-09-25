{{ config(materialized='view') }}

SELECT
    t.traffic_location,
    t.date AS traffic_timestamp,
    t.traffic_count_24h,

    t.year,
    t.month,
    t.day,
    t.day_of_week,
    t.week_of_year,
    t.is_weekend,

    w.temperature_2m,
    w.relative_humidity_2m,
    w.precipitation,
    w.pressure_msl,
    w.wind_speed_10m,
    w.wind_direction_10m,
    w.is_raining

FROM {{ ref('stg_traffic') }} AS t

LEFT JOIN {{ ref('stg_weather') }} AS w
    ON DATE(t.date) = w.date
    AND EXTRACT(HOUR FROM t.date) = w.hour
