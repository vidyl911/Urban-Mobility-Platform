{{ config(materialized='view') }}

SELECT
    traffic_location,
    date AS traffic_timestamp,
    traffic_count_24h,

    CASE
        WHEN traffic_location LIKE 'S.P. 103%' THEN 'SP103'
        WHEN traffic_location LIKE 'S.P. 14%' THEN 'SP14'
        WHEN traffic_location LIKE 'S.P. 40%' THEN 'SP40'
        WHEN traffic_location LIKE 'S.P. ex S.S. 35%' THEN 'SPexSS35'
        WHEN traffic_location LIKE 'S.P. ex S.S. 415%' THEN 'SPexSS415'
    END AS road_ref,

    ROUND(
        (
            CAST(
                (REGEXP_MATCH(traffic_location, 'Km ([0-9]+)\+([0-9]+)'))[1]
                AS DOUBLE PRECISION
            )
            +
            CAST(
                (REGEXP_MATCH(traffic_location, 'Km ([0-9]+)\+([0-9]+)'))[2]
                AS DOUBLE PRECISION
            ) / 1000.0
        )::NUMERIC,
        3
    ) AS kilometer_marker,

    TRIM(
        SPLIT_PART(traffic_location, 'Dir. ', 2)
    ) AS direction,

    year,
    month,
    day,
    day_of_week,
    week_of_year,
    is_weekend

FROM {{ ref('stg_traffic') }}
