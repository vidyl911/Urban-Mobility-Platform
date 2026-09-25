{{ config(materialized='view') }}

SELECT
    stop_id,
    stop_name,
    latitude,
    longitude,
    wheelchair_boarding,
    trip_count,
    route_count,
    shape_count,
    first_service_time,
    last_service_time
FROM {{ source('mobility', 'gtfs') }}