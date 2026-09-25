{{ config(materialized='view') }}

SELECT
    osm_id,
    element_type,
    latitude,
    longitude,
    poi_category,
    amenity,
    shop,
    tourism,
    leisure,
    office,
    name,
    poi_group
FROM {{ source('mobility', 'pois') }}
