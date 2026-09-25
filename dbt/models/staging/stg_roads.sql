{{ config(materialized='view') }}

SELECT
    road_segment_id,
    geom,
    highway,
    name,
    ref,
    maxspeed,
    lanes,
    oneway,
    surface,
    bridge,
    tunnel,
    length_m,
    road_class,
    maxspeed_missing,
    lanes_missing,
    is_major_road
FROM {{ source('mobility', 'roads') }}