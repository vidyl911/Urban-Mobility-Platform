{{ config(materialized='view') }}

SELECT
    ref AS road_ref,

    COUNT(*) AS road_segment_count,

    ROUND(
        AVG(length_m)::numeric,
        2
    ) AS avg_segment_length_m,

    ROUND(
        AVG(
            CASE
                WHEN maxspeed_missing = 0
                THEN maxspeed::numeric
            END
        ),
        1
    ) AS avg_maxspeed,

    ROUND(
        AVG(
            CASE
                WHEN lanes_missing = 0
                THEN lanes::numeric
            END
        ),
        1
    ) AS avg_lanes,

    SUM(
        CASE
            WHEN is_major_road = 1
            THEN 1
            ELSE 0
        END
    ) AS major_road_segment_count,

    ROUND(
        (
            SUM(
                CASE
                    WHEN is_major_road = 1
                    THEN 1
                    ELSE 0
                END
            )::numeric
            / COUNT(*)
        ),
        3
    ) AS major_road_share

FROM {{ ref('stg_roads') }}

WHERE ref IN (
    'SP103',
    'SP14',
    'SP40',
    'SPexSS35',
    'SPexSS415'
)

GROUP BY ref