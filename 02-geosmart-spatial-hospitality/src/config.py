from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("DATA_DIR", str(BASE_DIR / "data")))
LISTINGS_CSV = DATA_DIR / "listings.csv.gz"
NEIGHBOURHOODS_GEOJSON = DATA_DIR / "neighbourhoods.geojson"

ARTIFACTS_DIR = BASE_DIR / "artifacts"
ENRICHED_PARQUET = ARTIFACTS_DIR / "enriched_listings.parquet"
ANALYTICS_DB = ARTIFACTS_DIR / "geosmart_analytics.duckdb"
MANIFEST_PATH = BASE_DIR / "index_manifest.json"

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "geosmart_berlin_stays")

DENSE_DIM = 384

DEFAULT_BUDGET = 95.0
DEFAULT_DIST_SCALE_M = 2000.0
DEFAULT_PRICE_SCALE = 35.0

BERLIN_BBOX = {"lat_min": 52.30, "lat_max": 52.70, "lon_min": 13.00, "lon_max": 13.80}
PRICE_MIN, PRICE_MAX = 10.0, 1500.0
