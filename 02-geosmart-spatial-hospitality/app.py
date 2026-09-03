"""GeoSmart - Real-Time Spatial Vector Search & Continuous Decay Web Demo.

FastAPI backend serving Leaflet.js interactive map with Qdrant 1.19 client.facet,
multi-decay FormulaQuery, and GeoPolygon pushdown.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models
from src.duckdb_pipeline import (
    boundary_loss_analysis,
    district_centroids,
    h3_hotspots,
)
from src.embedder import DenseEmbedder
from src.qdrant_ops import (
    DecayWeights,
    get_qdrant_client,
    search_geo_polygon,
    search_grouped_districts,
    search_hard_radius,
    search_multi_decay,
    search_viewport_with_facets,
)

STATIC_DIR = Path(__file__).parent / "static"

# Global Singletons (Pre-warmed on server startup)
app_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app_state["client"] = get_qdrant_client()
    except ConnectionError as exc:
        raise RuntimeError(str(exc)) from exc
    app_state["dense"] = DenseEmbedder()
    _ = app_state["dense"].embed_queries("warmup")[0]
    yield
    app_state.clear()


app = FastAPI(title="GeoSmart Qdrant 1.19 Demo", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ViewportSearchRequest(BaseModel):
    query: str = Field(default="modern stylish apartment with fast wifi")
    top_left: list[float] = Field(description="[lat, lon] top left")
    bottom_right: list[float] = Field(description="[lat, lon] bottom right")
    target_price: float = Field(default=95.0)
    price_scale: float = Field(default=35.0)
    limit: int = Field(default=15)


class DecaySearchRequest(BaseModel):
    query: str = Field(default="quiet peaceful loft near cafes")
    center_lat: float = Field(default=52.5219)
    center_lon: float = Field(default=13.4132)
    dist_scale_m: float = Field(default=2000.0)
    target_price: float = Field(default=95.0)
    price_scale: float = Field(default=35.0)
    weight_geo: float = Field(default=1.2)
    weight_price: float = Field(default=0.8)
    weight_score: float = Field(default=0.8)
    weight_rating: float = Field(default=0.5)
    weight_superhost: float = Field(default=0.25)
    limit: int = Field(default=20)


class PolygonSearchRequest(BaseModel):
    query: str = Field(default="central stay")
    polygon: list[dict[str, float]] = Field(description="list of {lat, lon}")
    limit: int = Field(default=20)


class GroupedSearchRequest(BaseModel):
    query: str = Field(default="quiet sunny loft near cafes")
    center_lat: float = Field(default=52.5219)
    center_lon: float = Field(default=13.4132)
    dist_scale_m: float = Field(default=2000.0)
    target_price: float = Field(default=95.0)
    price_scale: float = Field(default=35.0)
    group_size: int = Field(default=1)
    limit: int = Field(default=6)


def _serialize_hit(h: models.ScoredPoint) -> dict[str, Any]:
    p = h.payload or {}
    loc = p.get("location", {})
    return {
        "id": h.id,
        "score": round(float(h.score), 4),
        "name": p.get("name", "Listing"),
        "price": float(p.get("price", 0.0)),
        "rating": float(p.get("rating", 0.0)),
        "number_of_reviews": int(p.get("number_of_reviews", 0)),
        "is_superhost": bool(p.get("is_superhost", False)),
        "room_type": p.get("room_type", "Entire home"),
        "neighbourhood": p.get("neighbourhood", ""),
        "neighbourhood_group": p.get("neighbourhood_group", ""),
        "lat": float(loc.get("lat", 0.0)),
        "lon": float(loc.get("lon", 0.0)),
    }


@app.post("/api/search/viewport")
async def search_viewport_api(req: ViewportSearchRequest):
    client: QdrantClient = app_state["client"]
    dense: DenseEmbedder = app_state["dense"]

    t0 = time.perf_counter()
    qvec = dense.embed_queries(req.query)[0]
    embed_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    res = search_viewport_with_facets(
        client=client,
        qvec=qvec,
        qtext=req.query,
        top_left=(req.top_left[0], req.top_left[1]),
        bottom_right=(req.bottom_right[0], req.bottom_right[1]),
        target_price=req.target_price,
        price_scale=req.price_scale,
        limit=req.limit,
    )
    qdrant_ms = (time.perf_counter() - t1) * 1000
    total_ms = (time.perf_counter() - t0) * 1000

    hits = [_serialize_hit(h) for h in res["hits"]]
    return {
        "timing": {
            "embed_ms": round(embed_ms, 2),
            "qdrant_ms": round(qdrant_ms, 2),
            "total_ms": round(total_ms, 2),
        },
        "hits": hits,
        "district_facets": res["district_facets"],
        "room_facets": res["room_facets"],
        "viewport": res["viewport"],
        "total_hits": len(hits),
    }


@app.post("/api/search/decay")
async def search_decay_api(req: DecaySearchRequest):
    client: QdrantClient = app_state["client"]
    dense: DenseEmbedder = app_state["dense"]

    t0 = time.perf_counter()
    qvec = dense.embed_queries(req.query)[0]
    embed_ms = (time.perf_counter() - t0) * 1000

    weights = DecayWeights(
        geo=req.weight_geo,
        budget=req.weight_price,
        semantic=req.weight_score,
        rating=req.weight_rating,
        superhost=req.weight_superhost,
    )

    t1 = time.perf_counter()
    hits_raw = search_multi_decay(
        client=client,
        qvec=qvec,
        qtext=req.query,
        lat=req.center_lat,
        lon=req.center_lon,
        dist_scale_m=req.dist_scale_m,
        target_price=req.target_price,
        price_scale=req.price_scale,
        weights=weights,
        limit=req.limit,
    )
    qdrant_ms = (time.perf_counter() - t1) * 1000

    # Also compute hard circle comparison
    hard_hits_raw = search_hard_radius(
        client=client,
        qvec=qvec,
        lat=req.center_lat,
        lon=req.center_lon,
        radius_m=req.dist_scale_m * 0.75,
        max_price=req.target_price * 1.1,
        limit=req.limit,
    )

    # DuckDB boundary cliff proof
    boundary_info = boundary_loss_analysis(
        center_lat=req.center_lat,
        center_lon=req.center_lon,
        radius_m=req.dist_scale_m * 0.75,
        max_price=req.target_price * 1.1,
        slack_ratio=1.25,
    )

    total_ms = (time.perf_counter() - t0) * 1000

    return {
        "timing": {
            "embed_ms": round(embed_ms, 2),
            "qdrant_ms": round(qdrant_ms, 2),
            "total_ms": round(total_ms, 2),
        },
        "hits": [_serialize_hit(h) for h in hits_raw],
        "hard_hits": [_serialize_hit(h) for h in hard_hits_raw],
        "boundary_analysis": boundary_info,
    }


@app.post("/api/search/polygon")
async def search_polygon_api(req: PolygonSearchRequest):
    client: QdrantClient = app_state["client"]
    dense: DenseEmbedder = app_state["dense"]

    t0 = time.perf_counter()
    qvec = dense.embed_queries(req.query)[0]
    embed_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    hits_raw = search_geo_polygon(
        client=client,
        qvec=qvec,
        exterior=req.polygon,
        limit=req.limit,
    )
    qdrant_ms = (time.perf_counter() - t1) * 1000
    total_ms = (time.perf_counter() - t0) * 1000

    return {
        "timing": {
            "embed_ms": round(embed_ms, 2),
            "qdrant_ms": round(qdrant_ms, 2),
            "total_ms": round(total_ms, 2),
        },
        "vertex_count": len(req.polygon),
        "hits": [_serialize_hit(h) for h in hits_raw],
    }


@app.post("/api/search/grouped")
async def search_grouped_api(req: GroupedSearchRequest):
    client: QdrantClient = app_state["client"]
    dense: DenseEmbedder = app_state["dense"]

    t0 = time.perf_counter()
    qvec = dense.embed_queries(req.query)[0]
    embed_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    grouped = search_grouped_districts(
        client=client,
        qvec=qvec,
        qtext=req.query,
        lat=req.center_lat,
        lon=req.center_lon,
        dist_scale_m=req.dist_scale_m,
        target_price=req.target_price,
        price_scale=req.price_scale,
        group_size=req.group_size,
        limit=req.limit,
    )
    qdrant_ms = (time.perf_counter() - t1) * 1000
    total_ms = (time.perf_counter() - t0) * 1000

    groups_out = []
    for g in grouped.groups:
        groups_out.append(
            {
                "district": g.id,
                "hits": [_serialize_hit(h) for h in g.hits],
            }
        )
    return {
        "timing": {
            "embed_ms": round(embed_ms, 2),
            "qdrant_ms": round(qdrant_ms, 2),
            "total_ms": round(total_ms, 2),
        },
        "groups": groups_out,
        "total_groups": len(groups_out),
    }


@app.get("/api/h3_hotspots")
async def h3_hotspots_api(res: int = 8, top_n: int = 10):
    """DuckDB H3 hex density - complements Qdrant (no hex binning in Qdrant)."""
    hotspots = h3_hotspots(res=res, top_n=top_n)
    return {"hotspots": hotspots, "res": res}


@app.get("/api/districts")
async def get_districts_api():
    cents = district_centroids()
    return {
        k: {
            "name": v.get("name", k),
            "lat": v.get("lat"),
            "lon": v.get("lon"),
            "has_polygon": bool(v.get("polygon")),
        }
        for k, v in cents.items()
    }


@app.get("/api/district_polygon/{district_key}")
async def get_district_polygon_api(district_key: str):
    info = district_centroids().get(district_key)
    if not info or not info.get("polygon"):
        return {"error": "Polygon not found"}
    points = [{"lat": lat, "lon": lon} for lat, lon in info["polygon"]]
    return {
        "district": district_key,
        "points": points,
        "vertex_count": len(points),
    }


# Serve the static frontend using FastAPI 0.141 app.frontend()
app.frontend("/", directory=STATIC_DIR)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
