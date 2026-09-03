#!/usr/bin/env python3
"""Download Berlin Inside Airbnb data - reproducible, no private paths.

Usage:
  uv run python scripts/download_data.py
  uv run python scripts/download_data.py --force
  DATA_DIR=/tmp/data uv run python scripts/download_data.py

The script populates ./data/listings.csv.gz + ./data/neighbourhoods.geojson
which is the default DATA_DIR consumed by src/config.py (no env needed).

Source: Inside Airbnb - Berlin (http://insideairbnb.com/get-the-data/).
We scrape the site's Berlin row to discover the latest dated URLs, so the
script keeps working even as Inside Airbnb publishes new snapshots.
A hard-coded fallback date is kept for offline/CI.
"""

from __future__ import annotations

import argparse
import re
import urllib.error
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = BASE_DIR / "data"

# Fallback snapshot (known-good; used if scraping fails or site blocks)
FALLBACK_LISTINGS_URL = (
    "https://data.insideairbnb.com/germany/be/berlin/2024-06-22/data/listings.csv.gz"
)
FALLBACK_GEOJSON_URL = "https://data.insideairbnb.com/germany/be/berlin/2024-06-22/visualisations/neighbourhoods.geojson"

GET_THE_DATA_URL = "https://insideairbnb.com/get-the-data/"

HEADERS = {"User-Agent": "Mozilla/5.0 (Qdrant-Tutorial; +https://qdrant.tech)"}


def _discover_berlin_urls() -> tuple[str, str]:
    """Scrape Inside Airbnb's get-the-data page for Berlin's latest URLs."""
    try:
        req = urllib.request.Request(GET_THE_DATA_URL, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=30) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
        # Find the Berlin section and extract listings + geojson
        # The page lists Berlin under <h3>Berlin, ...</h3> then table rows.
        berlin_idx = html.lower().find("berlin")
        if berlin_idx == -1:
            raise ValueError("Berlin not found in page")
        snippet = html[berlin_idx : berlin_idx + 8000]
        listings = re.findall(
            r"https://data\.insideairbnb\.com/germany/be/berlin/[^\"']+/data/listings\.csv\.gz",
            snippet,
        )
        geojson = re.findall(
            r"https://data\.insideairbnb\.com/germany/be/berlin/[^\"']+/visualisations/neighbourhoods\.geojson",
            snippet,
        )
        if listings and geojson:
            return listings[0], geojson[0]
    except (OSError, TimeoutError, ValueError, urllib.error.URLError) as e:
        print(f"[warn] scrape failed ({e}), using fallback snapshot")
    return FALLBACK_LISTINGS_URL, FALLBACK_GEOJSON_URL


def _download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f" → {url}")
    print(f"   → {dest} ({dest.exists() and 'exists' or 'new'})")
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as out:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk = 1024 * 256
        while True:
            buf = resp.read(chunk)
            if not buf:
                break
            out.write(buf)
            downloaded += len(buf)
            if total and downloaded % (1024 * 1024 * 2) < chunk:
                print(
                    f"     {downloaded / 1024 / 1024:.1f} / {total / 1024 / 1024:.1f} MB",
                    end="\r",
                )
        print(f"\n   ✓ {downloaded / 1024 / 1024:.2f} MB")
    if dest.stat().st_size < 1024:
        raise RuntimeError(
            f"Download too small ({dest.stat().st_size} bytes) - likely blocked HTML: {url}"
        )


def download_berlin(data_dir: Path = DEFAULT_DATA_DIR, force: bool = False) -> Path:
    listings = data_dir / "listings.csv.gz"
    geojson = data_dir / "neighbourhoods.geojson"
    if listings.exists() and geojson.exists() and not force:
        print(f"✓ Data already present in {data_dir} (use --force to re-download)")
        print(f"  {listings} ({listings.stat().st_size / 1024 / 1024:.1f} MB)")
        print(f"  {geojson} ({geojson.stat().st_size / 1024:.1f} KB)")
        return data_dir

    listings_url, geojson_url = _discover_berlin_urls()
    print(f"Downloading Berlin snapshot → {data_dir}")
    print(f"  listings:    {listings_url}")
    print(f"  neighbourhoods: {geojson_url}")
    _download(listings_url, listings)
    _download(geojson_url, geojson)

    print(f"\n✓ Data ready in {data_dir}")
    print("  Next: uv run python main.py")
    return data_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Download Berlin Inside Airbnb data")
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="output dir (default: ./data)",
    )
    ap.add_argument(
        "--force", action="store_true", help="re-download even if files exist"
    )
    args = ap.parse_args()
    download_berlin(args.data_dir, force=args.force)
