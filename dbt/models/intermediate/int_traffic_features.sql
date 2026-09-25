{{ config(materialized='view') }}

SELECT
    s.traffic_location,
    s.traffic_timestamp,
    s.traffic_count_24h,

    s.road_ref,
    s.kilometer_marker,
    s.direction,

    s.year,
    s.month,
    s.day,
    s.day_of_week,
    s.week_of_year,
    s.is_weekend,

    w.temperature_2m,
    w.relative_humidity_2m,
    w.precipitation,
    w.pressure_msl,
    w.wind_speed_10m,
    w.wind_direction_10m,
    w.is_raining

FROM {{ ref('int_traffic_sensors') }} AS s

LEFT JOIN {{ ref('int_traffic_weather') }} AS w
    ON s.traffic_timestamp = w.traffic_timestamp
    AND s.traffic_location = w.traffic_location
