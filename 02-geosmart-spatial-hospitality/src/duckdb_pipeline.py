"""DuckDB helper: enrichment + analytics (kept OUT of search path, like 01)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from .config import (
    ANALYTICS_DB,
    ARTIFACTS_DIR,
    BERLIN_BBOX,
    ENRICHED_PARQUET,
    LISTINGS_CSV,
    NEIGHBOURHOODS_GEOJSON,
    PRICE_MAX,
    PRICE_MIN,
)

logger = logging.getLogger("geosmart.duckdb")


def get_analytics_connection() -> duckdb.DuckDBPyConnection:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(ANALYTICS_DB))
    con.execute("INSTALL spatial; LOAD spatial; SET geometry_always_xy = true;")
    con.execute("INSTALL h3 FROM community; LOAD h3;")
    return con


def _path_arg(p: Path) -> str:
    return str(p.resolve())


def run_duckdb_spatial_pipeline(
    limit: int | None = None, refresh: bool = False
) -> pd.DataFrame:
    """Enrich listings: bbox prune, price clean, ST_Contains district, H3, search_text."""
    if not LISTINGS_CSV.exists() or not NEIGHBOURHOODS_GEOJSON.exists():
        raise FileNotFoundError(
            f"Berlin listing files missing under {LISTINGS_CSV.parent}.\n"
            "From this folder run:  uv run python scripts/download_data.py"
        )

    if ENRICHED_PARQUET.exists() and limit is None and not refresh:
        logger.info(f"Loading cached enrichment from {ENRICHED_PARQUET}")
        return pd.read_parquet(ENRICHED_PARQUET)

    con = get_analytics_connection()
    limit_clause = "LIMIT ?" if limit is not None else ""
    sql = f"""
        WITH raw AS (
            SELECT
                CAST(id AS BIGINT) AS id, name,
                COALESCE(description,'') AS description,
                COALESCE(neighborhood_overview,'') AS neighborhood_overview,
                COALESCE(room_type,'Entire home') AS room_type,
                TRY_CAST(accommodates AS INTEGER) AS accommodates,
                TRY_CAST(replace(replace(price,'$',''),',','') AS DOUBLE) AS price,
                TRY_CAST(latitude AS DOUBLE) AS latitude,
                TRY_CAST(longitude AS DOUBLE) AS longitude,
                COALESCE(TRY_CAST(review_scores_rating AS DOUBLE),0.0) AS rating,
                COALESCE(TRY_CAST(number_of_reviews AS INTEGER),0) AS number_of_reviews,
                COALESCE((host_is_superhost='t'),false) AS is_superhost,
                COALESCE(amenities,'') AS amenities
            FROM read_csv(?)
            WHERE latitude IS NOT NULL AND longitude IS NOT NULL
              AND latitude BETWEEN ? AND ? AND longitude BETWEEN ? AND ?
        ),
        valid AS (SELECT * FROM raw WHERE price IS NOT NULL AND price BETWEEN ? AND ?),
        enriched_one AS (
            SELECT l.*, h3_latlng_to_cell(l.latitude,l.longitude,8) AS h3_res8,
                   COALESCE(n.neighbourhood,'Berlin Central') AS neighbourhood,
                   COALESCE(n.neighbourhood_group,'Mitte') AS neighbourhood_group
            FROM valid l LEFT JOIN (SELECT neighbourhood, neighbourhood_group, geom FROM ST_Read(?))
                 n ON ST_Contains(n.geom, ST_Point(l.longitude,l.latitude))
        )
        SELECT e.*,
               CONCAT_WS(' ', COALESCE(name,''), COALESCE(room_type,''), COALESCE(neighbourhood,''),
                    COALESCE(neighbourhood_group,''), left(COALESCE(description,''),600),
                    left(COALESCE(neighborhood_overview,''),250), left(COALESCE(amenities,''),400)) AS search_text
        FROM enriched_one e ORDER BY id {limit_clause};
    """
    params = [
        _path_arg(LISTINGS_CSV),
        BERLIN_BBOX["lat_min"],
        BERLIN_BBOX["lat_max"],
        BERLIN_BBOX["lon_min"],
        BERLIN_BBOX["lon_max"],
        PRICE_MIN,
        PRICE_MAX,
        _path_arg(NEIGHBOURHOODS_GEOJSON),
    ]
    if limit is not None:
        params.append(int(limit))
    logger.info("DuckDB enrichment (ST_Contains + H3)…")
    df = con.execute(sql, params).df()
    if limit is None:
        df.to_parquet(ENRICHED_PARQUET, index=False)
        logger.info(f"Enriched {len(df)} -> {ENRICHED_PARQUET}")
    _refresh_district_stats(con, df)
    return df


def _refresh_district_stats(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> None:
    if NEIGHBOURHOODS_GEOJSON.exists():
        con.execute(f"""
            CREATE OR REPLACE TABLE district_centroids AS
            SELECT lower(replace(replace(neighbourhood,' ','_'),'-','_')) AS key,
                   neighbourhood AS name, neighbourhood_group AS grp,
                   ST_Y(ST_Centroid(geom)) AS lat, ST_X(ST_Centroid(geom)) AS lon,
                   ST_Area_Spheroid(geom)/1e6 AS area_sq_km,
                   ST_AsText(geom) AS wkt
            FROM ST_Read('{_path_arg(NEIGHBOURHOODS_GEOJSON)}');
        """)
    con.register("listings_tmp", df)
    con.execute("""
        CREATE OR REPLACE TABLE district_price_stats AS
        SELECT neighbourhood_group AS grp, COUNT(*) AS n,
               quantile_cont(price,0.5) AS median_price,
               median(number_of_reviews) AS median_reviews,
               avg(CASE WHEN is_superhost THEN 1 ELSE 0 END) AS superhost_ratio
        FROM listings_tmp GROUP BY 1;
    """)
    con.unregister("listings_tmp")


def _ring_area(coords: list[list[float]]) -> float:
    """Absolute shoelace area - proxy for ring size when picking exteriors."""
    area = 0.0
    for i in range(len(coords)):
        lat1, lon1 = coords[i]
        lat2, lon2 = coords[(i + 1) % len(coords)]
        area += lon1 * lat2 - lon2 * lat1
    return abs(area) / 2.0


def _parse_coords(ring: str) -> list[list[float]]:
    coords: list[list[float]] = []
    for pair in ring.split(","):
        parts = pair.strip().split()
        if len(parts) >= 2:
            try:
                lon, lat = float(parts[0]), float(parts[1])
                coords.append([lat, lon])
            except ValueError:
                continue
    return coords


def _parse_wkt_exterior(wkt: str) -> list[list[float]]:
    """Extract exterior ring as [[lat,lon],…] from WKT POLYGON/MULTIPOLYGON.

    Picks the largest ring by shoelace area: MULTIPOLYGON districts can have
    several parts and the first is not always the main boundary.
    """
    import re

    rings = re.findall(r"\(\(([^()]+)\)\)", wkt, re.DOTALL)
    parsed = [_parse_coords(r) for r in rings]
    parsed = [c for c in parsed if len(c) >= 4]
    if not parsed:
        return []
    return max(parsed, key=_ring_area)


def district_centroids() -> dict[str, dict[str, Any]]:
    con = get_analytics_connection()
    if not con.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_name='district_centroids'"
    ).fetchone():
        raise RuntimeError("district_centroids not materialized - run pipeline first")
    rows = con.execute(
        "SELECT key,name,grp,lat,lon,area_sq_km,wkt FROM district_centroids ORDER BY name"
    ).fetchall()
    out = {}
    for k, n, g, la, lo, a, wkt in rows:
        if k is None or n is None:
            continue
        coords = _parse_wkt_exterior(str(wkt))
        out[str(k)] = {
            "name": str(n),
            "group": str(g) if g else "Berlin",
            "lat": float(la),
            "lon": float(lo),
            "area_sq_km": float(a),
            "wkt": str(wkt),
            "polygon": coords,
        }
    return out


def district_price_stats() -> dict[str, dict[str, float]]:
    con = get_analytics_connection()
    rows = con.execute(
        "SELECT grp,n,median_price,median_reviews,superhost_ratio FROM district_price_stats ORDER BY grp"
    ).fetchall()
    return {
        g: {
            "n": int(n),
            "median_price": float(mp),
            "median_reviews": float(mr),
            "superhost_ratio": float(sh),
        }
        for g, n, mp, mr, sh in rows
    }


def boundary_loss_analysis(
    center_lat: float,
    center_lon: float,
    radius_m: float,
    max_price: float,
    slack_ratio: float = 1.25,
) -> dict[str, float]:
    """Measured brittleness: viable stays rejected at hard boundary."""
    con = get_analytics_connection()
    row = con.execute(
        """
        SELECT COUNT(*) FILTER (WHERE d <= ?) AS hits,
               COUNT(*) FILTER (WHERE d>? AND d<=?) AS near_miss,
               COUNT(*) AS total
        FROM (SELECT ST_Distance_Spheroid(ST_Point(?::DOUBLE,?::DOUBLE)::POINT_2D, ST_Point(longitude,latitude)::POINT_2D) AS d
              FROM read_parquet(?) WHERE price<=?)
    """,
        [
            radius_m,
            radius_m,
            radius_m * slack_ratio,
            center_lon,
            center_lat,
            _path_arg(ENRICHED_PARQUET),
            max_price,
        ],
    ).fetchone()
    hits, near, tot = (float(v or 0) for v in row)
    near_area = hits + near
    return {
        "hits": hits,
        "near_miss": near,
        # Of stays inside the slack radius, the share sitting in the ring
        # (hard filter drops these; GaussDecay keeps them).
        "dropout_fraction": (near / near_area if near_area else 0),
        "corpus_share": (near / tot if tot else 0),
        "radius_m": radius_m,
        "max_price": max_price,
    }


def geodistance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    con = get_analytics_connection()
    return float(
        con.execute(
            "SELECT ST_Distance_Spheroid(ST_Point(?,?)::POINT_2D, ST_Point(?,?)::POINT_2D)",
            [lon1, lat1, lon2, lat2],
        ).fetchone()[0]
    )


def distance_matrix_m(
    lats: list[float], lons: list[float], lat: float, lon: float
) -> list[float]:
    con = get_analytics_connection()
    df = pd.DataFrame({"lat": lats, "lon": lons})
    con.register("pts_tmp", df)
    out = con.execute(
        "SELECT ST_Distance_Spheroid(ST_Point(pts_tmp.lon,pts_tmp.lat)::POINT_2D, ST_Point(?::DOUBLE,?::DOUBLE)::POINT_2D) FROM pts_tmp",
        [lon, lat],
    ).fetchall()
    con.unregister("pts_tmp")
    return [float(r[0]) for r in out]


# ── premium spatial analytics (DuckDB complements Qdrant where Qdrant cannot) ────────────


def h3_hotspots(res: int = 8, top_n: int = 10) -> list[dict[str, Any]]:
    """
    H3 hex density - Qdrant has geo_radius/polygon but no hex binning.
    This is the DuckDB complement: aggregate 8,317 listings per H3 cell for heatmap.
    """
    con = get_analytics_connection()
    rows = con.execute(
        """
        SELECT h3_res8 AS cell, COUNT(*) AS n,
               h3_cell_to_parent(h3_res8, ?) AS parent,
               avg(price) AS avg_price, avg(rating) AS avg_rating
        FROM read_parquet(?)
        GROUP BY 1 ORDER BY n DESC LIMIT ?
    """,
        [max(5, res - 2), _path_arg(ENRICHED_PARQUET), top_n],
    ).fetchall()
    out = []
    for cell, n, parent, ap, ar in rows:
        latlon = con.execute("SELECT h3_cell_to_latlng(?)", [cell]).fetchone()[0]
        if isinstance(latlon, dict):
            lat, lon = latlon.get("lat", 0), latlon.get("lng", latlon.get("lon", 0))
        elif isinstance(latlon, (list, tuple)) and len(latlon) >= 2:
            lat, lon = float(latlon[0]), float(latlon[1])
        else:
            lat, lon = 0, 0
        out.append(
            {
                "cell": str(cell),
                "count": int(n),
                "parent": str(parent),
                "avg_price": round(float(ap or 0), 1),
                "avg_rating": round(float(ar or 0), 2),
                "center": {"lat": float(lat), "lon": float(lon)},
            }
        )
    return out
