import os
from datetime import datetime, timezone

import osmium
import pyarrow as pa
import pyarrow.parquet as pq


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

INPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "osm",
    "OSM_North_West_Italy.pbf"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "bronze",
    "osm"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "roads.parquet"
)


# --------------------------------------------------
# OSM Road Handler
# --------------------------------------------------

class RoadHandler(osmium.SimpleHandler):

    def __init__(self):
        super().__init__()
        self.records = []

    def way(self, w):

        # Keep only road-related OSM ways
        highway = w.tags.get("highway")

        if not highway:
            return

        # Exclude non-road highway features
        excluded_types = {
            "footway",
            "path",
            "cycleway",
            "bridleway",
            "steps",
            "pedestrian",
            "corridor",
            "proposed",
            "construction",
            "platform",
            "raceway",
        }

        if highway in excluded_types:
            return

        # Extract geometry
        geometry = []

        if w.is_closed():
            return

        for node in w.nodes:
            try:
                geometry.append(
                    (node.lon, node.lat)
                )
            except Exception:
                continue

        if len(geometry) < 2:
            return

        # Convert geometry to WKT
        coordinates = ", ".join(
            f"{lon} {lat}" for lon, lat in geometry
        )

        linestring_wkt = (
            f"LINESTRING ({coordinates})"
        )

        # Store Bronze record
        self.records.append(
            {
                "osm_id": int(w.id),
                "geometry": linestring_wkt,
                "highway": highway,
                "name": w.tags.get("name"),
                "ref": w.tags.get("ref"),
                "maxspeed": w.tags.get("maxspeed"),
                "lanes": w.tags.get("lanes"),
                "oneway": w.tags.get("oneway"),
                "surface": w.tags.get("surface"),
                "bridge": w.tags.get("bridge"),
                "tunnel": w.tags.get("tunnel"),
                "source": "OpenStreetMap",
                "ingestion_timestamp": datetime.now(
                    timezone.utc
                ).isoformat(),
            }
        )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("Starting OSM Bronze transformation...")
    print(f"Input: {INPUT_FILE}")

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"OSM PBF file not found: {INPUT_FILE}"
        )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    handler = RoadHandler()

    print("Reading OSM PBF file...")

    handler.apply_file(
        INPUT_FILE,
        locations=True
    )

    print(
        f"Extracted {len(handler.records):,} road features."
    )

    if not handler.records:
        raise ValueError(
            "No road features were extracted from the OSM file."
        )

    # --------------------------------------------------
    # Convert records to Arrow table
    # --------------------------------------------------

    table = pa.Table.from_pylist(
        handler.records
    )

    # --------------------------------------------------
    # Write Parquet
    # --------------------------------------------------

    pq.write_table(
        table,
        OUTPUT_FILE,
        compression="snappy"
    )

    print(
        f"Bronze OSM dataset written to:\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()