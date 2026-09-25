{{ config(materialized='table') }}

SELECT
    traffic_location,
    traffic_timestamp,

    -- Target
    traffic_count_24h,

    -- Road / sensor features
    road_ref,
    kilometer_marker,
    direction,

    -- Temporal features
    year,
    month,
    day,
    day_of_week,
    week_of_year,
    is_weekend,

    -- Weather features
    temperature_2m,
    relative_humidity_2m,
    precipitation,
    pressure_msl,
    wind_speed_10m,
    wind_direction_10m,
    is_raining,

    -- Road-network features
    road_segment_count,
    avg_segment_length_m,
    avg_maxspeed,
    avg_lanes,
    major_road_segment_count,
    major_road_share

FROM {{ ref('int_traffic_road_features') }}