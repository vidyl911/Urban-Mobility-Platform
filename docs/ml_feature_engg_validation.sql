-- ============================================================
-- Urban Mobility Platform
-- Feature Engineering & ML Data Validation
-- ============================================================
--
-- Purpose:
-- Document the SQL validation and inspection queries used
-- during the dbt feature-engineering stage.
--
-- Source:
-- mobility.int_ml_features
--
-- ============================================================


-- ============================================================
-- 1. Validate traffic sensor distribution
-- ============================================================

SELECT
    road_ref,
    COUNT(*) AS observations,
    COUNT(DISTINCT traffic_location) AS sensors
FROM mobility.int_traffic_sensors
GROUP BY road_ref
ORDER BY road_ref;


-- ============================================================
-- 2. Inspect OSM road coverage for traffic road references
-- ============================================================

SELECT
    ref,
    COUNT(*) AS segments,
    ROUND(AVG(length_m)::numeric, 2) AS avg_segment_length_m,
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
    ) AS major_road_segments
FROM mobility.roads
WHERE ref IN (
    'SP103',
    'SP14',
    'SP40',
    'SPexSS35',
    'SPexSS415'
)
GROUP BY ref
ORDER BY ref;


-- ============================================================
-- 3. Validate traffic + weather integration
-- ============================================================

SELECT
    COUNT(*) AS total_rows,
    COUNT(w.temperature_2m) AS weather_matched,
    COUNT(*) - COUNT(w.temperature_2m) AS weather_unmatched
FROM mobility.int_traffic_features AS t
LEFT JOIN mobility.stg_weather AS w
    ON t.traffic_timestamp = w.timestamp;


-- ============================================================
-- 4. Validate traffic + road feature integration
-- ============================================================

SELECT
    COUNT(*) AS total_rows,
    COUNT(road_segment_count) AS road_features_matched,
    COUNT(*) - COUNT(road_segment_count) AS road_features_unmatched,
    COUNT(DISTINCT road_ref) AS roads
FROM mobility.int_traffic_road_features;


-- ============================================================
-- 5. Validate final ML feature table
-- ============================================================

SELECT
    COUNT(*) AS total_rows,
    COUNT(
        DISTINCT traffic_location || '|' ||
        traffic_timestamp::text
    ) AS unique_observations,
    SUM(
        CASE
            WHEN traffic_count_24h IS NULL
            THEN 1
            ELSE 0
        END
    ) AS null_target,
    SUM(
        CASE
            WHEN avg_maxspeed IS NULL
            THEN 1
            ELSE 0
        END
    ) AS null_speed,
    SUM(
        CASE
            WHEN avg_lanes IS NULL
            THEN 1
            ELSE 0
        END
    ) AS null_lanes,
    SUM(
        CASE
            WHEN temperature_2m IS NULL
            THEN 1
            ELSE 0
        END
    ) AS null_temperature,
    SUM(
        CASE
            WHEN precipitation IS NULL
            THEN 1
            ELSE 0
        END
    ) AS null_precipitation
FROM mobility.int_ml_features;


-- ============================================================
-- 6. Inspect target distribution
-- ============================================================

SELECT
    MIN(traffic_count_24h) AS min_traffic,
    ROUND(AVG(traffic_count_24h)::numeric, 0) AS avg_traffic,
    PERCENTILE_CONT(0.50)
        WITHIN GROUP (ORDER BY traffic_count_24h)
        AS median_traffic,
    PERCENTILE_CONT(0.95)
        WITHIN GROUP (ORDER BY traffic_count_24h)
        AS p95_traffic,
    MAX(traffic_count_24h) AS max_traffic
FROM mobility.int_ml_features;


-- ============================================================
-- 7. Inspect complete temporal coverage
-- ============================================================

SELECT
    MIN(traffic_timestamp)::date AS start_date,
    MAX(traffic_timestamp)::date AS end_date,
    COUNT(DISTINCT traffic_timestamp::date) AS days
FROM mobility.int_ml_features;


-- ============================================================
-- 8. Define and validate chronological ML split
-- ============================================================
--
-- Train      : 2016-04-01 → 2016-05-31
-- Validation : 2016-06-01 → 2016-06-15
-- Test       : 2016-06-16 → 2016-06-30
--
-- Random splitting is intentionally avoided because this is
-- time-dependent traffic data.
-- ============================================================

SELECT
    CASE
        WHEN traffic_timestamp::date <= DATE '2016-05-31'
            THEN 'train'
        WHEN traffic_timestamp::date <= DATE '2016-06-15'
            THEN 'validation'
        ELSE 'test'
    END AS dataset_split,

    MIN(traffic_timestamp)::date AS start_date,
    MAX(traffic_timestamp)::date AS end_date,
    COUNT(*) AS rows

FROM mobility.int_ml_features

GROUP BY dataset_split

ORDER BY MIN(traffic_timestamp);