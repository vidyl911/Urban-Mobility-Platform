{{ config(materialized='view') }}

SELECT
    t.traffic_location,
    t.traffic_timestamp,
    t.traffic_count_24h,

    -- Sensor / road identity
    t.road_ref,
    t.kilometer_marker,
    t.direction,

    -- Temporal features
    t.year,
    t.month,
    t.day,
    t.day_of_week,
    t.week_of_year,
    t.is_weekend,

    -- Weather features
    t.temperature_2m,
    t.relative_humidity_2m,
    t.precipitation,
    t.pressure_msl,
    t.wind_speed_10m,
    t.wind_direction_10m,
    t.is_raining,

    -- OSM road features
    r.road_segment_count,
    r.avg_segment_length_m,
    r.avg_maxspeed,
    r.avg_lanes,
    r.major_road_segment_count,
    r.major_road_share

FROM {{ ref('int_traffic_features') }} AS t

LEFT JOIN {{ ref('int_road_features') }} AS r
    ON t.road_ref = r.road_ref