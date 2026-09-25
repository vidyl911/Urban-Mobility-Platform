SELECT
    COUNT(*) AS total_roads,
    COUNT(geom) AS roads_with_geometry,
    ST_GeometryType(geom) AS geometry_type
FROM mobility.roads
GROUP BY ST_GeometryType(geom);
