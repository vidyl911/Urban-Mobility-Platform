from __future__ import annotations

import json
from pathlib import Path

import requests


# Bounding box around Milan:
# south, west, north, east
SOUTH = 45.38
WEST = 9.02
NORTH = 45.55
EAST = 9.30

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

OUTPUT_DIR = Path("data/raw/pois")
OUTPUT_FILE = OUTPUT_DIR / "milan_pois.json"


def build_query() -> str:
    bbox = f"{SOUTH},{WEST},{NORTH},{EAST}"

    return f"""
    [out:json][timeout:180];

    (
      node["amenity"]({bbox});
      way["amenity"]({bbox});
      relation["amenity"]({bbox});

      node["shop"]({bbox});
      way["shop"]({bbox});
      relation["shop"]({bbox});

      node["office"]({bbox});
      way["office"]({bbox});
      relation["office"]({bbox});

      node["tourism"]({bbox});
      way["tourism"]({bbox});
      relation["tourism"]({bbox});
    );

    out center tags;
    """


def fetch_pois() -> None:
    query = build_query()

    response = requests.post(
        OVERPASS_URL,
        data=query,
        headers={
            "Accept": "application/json",
            "User-Agent": "UrbanMobilityPlatform/1.0 (+https://github.com/your-org/urban-mobility-platform)",
        },
        timeout=240,
    )
    response.raise_for_status()

    data = response.json()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)

    print(f"POI data saved to: {OUTPUT_FILE}")
    print(f"Elements returned: {len(data.get('elements', []))}")


if __name__ == "__main__":
    fetch_pois()