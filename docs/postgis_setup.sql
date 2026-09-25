ALTER TABLE mobility.roads
ADD COLUMN IF NOT EXISTS geom geometry(Geometry, 4326);

UPDATE mobility.roads
SET geom = ST_GeomFromWKB(
    decode(geometry_wkb, 'hex'),
    4326
)
WHERE geometry_wkb IS NOT NULL;

CREATE INDEX IF NOT EXISTS roads_geom_idx
ON mobility.roads
USING GIST (geom);

ANALYZE mobility.roads;