"""GeoSmart - Spatial Vector Search & Continuous Decay Engine (Qdrant 1.19)."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from rich.progress import track

_QDRANT_CONNECT_ERRORS = (
    OSError,
    TimeoutError,
    ResponseHandlingException,
    UnexpectedResponse,
)

from .config import (
    COLLECTION_NAME,
    DENSE_DIM,
    MANIFEST_PATH,
    QDRANT_API_KEY,
    QDRANT_URL,
)
from .embedder import DenseEmbedder

logger = logging.getLogger("geosmart.qdrant")

BM25_MODEL = "Qdrant/bm25"


@dataclass
class DecayWeights:
    semantic: float = 0.80
    geo: float = 1.20
    budget: float = 0.80
    reviews: float = 0.15
    rating: float = 0.50
    superhost: float = 0.25


def default_weights() -> DecayWeights:
    return DecayWeights()


# ── Client & Collection Lifecycle ──────────────────────────────────


def get_qdrant_client(
    url: str = QDRANT_URL, api_key: str | None = None
) -> QdrantClient:
    """Connect to Qdrant, retrying briefly so `docker compose up -d` can finish booting."""
    client = QdrantClient(url=url, api_key=api_key or QDRANT_API_KEY, timeout=60)
    last_error: Exception | None = None
    for _ in range(15):
        try:
            client.get_collections()
            return client
        except _QDRANT_CONNECT_ERRORS as exc:
            last_error = exc
            time.sleep(1)
    raise ConnectionError(
        f"Cannot reach Qdrant at {url}. "
        "From this folder run `docker compose up -d`, then retry. "
        "If JurisBoost already started Qdrant on :6333, reuse it - "
        "collection names do not collide."
    ) from last_error


def setup_collection(
    client: QdrantClient, collection_name: str = COLLECTION_NAME, recreate: bool = True
) -> None:
    """
    Qdrant 1.19 Spatial Collection (unified `memory` tiers, explicit):
    - Dense (384-d Cosine): CACHED on disk, BinaryQuantization PINNED in RAM.
    - Sparse BM25: CACHED inverted index (the 1.19 sparse cached tier) with Modifier.IDF.
    - Payload Indexes: PINNED in RAM - 'location' (GEO) plus price/rating/reviews
      and keyword/bool fields, so decay terms evaluate without a disk hop.
    """
    if client.collection_exists(collection_name):
        if not recreate:
            return
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": models.VectorParams(
                size=DENSE_DIM,
                distance=models.Distance.COSINE,
                memory=models.Memory.CACHED,
                quantization_config=models.BinaryQuantization(
                    binary=models.BinaryQuantizationConfig(memory=models.Memory.PINNED)
                ),
            ),
        },
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(
                index=models.SparseIndexParams(memory=models.Memory.CACHED),
                modifier=models.Modifier.IDF,
            )
        },
    )

    _PINNED = models.Memory.PINNED
    for key, schema in [
        ("location", models.GeoIndexParams(type=models.GeoIndexType.GEO, memory=_PINNED)),
        ("price", models.FloatIndexParams(type=models.FloatIndexType.FLOAT, memory=_PINNED)),
        ("rating", models.FloatIndexParams(type=models.FloatIndexType.FLOAT, memory=_PINNED)),
        ("number_of_reviews", models.IntegerIndexParams(type=models.IntegerIndexType.INTEGER, memory=_PINNED)),
        ("is_superhost", models.BoolIndexParams(type=models.BoolIndexType.BOOL, memory=_PINNED)),
        ("room_type", models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD, memory=_PINNED)),
        ("neighbourhood", models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD, memory=_PINNED)),
        ("neighbourhood_group", models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD, memory=_PINNED)),
        ("accommodates", models.IntegerIndexParams(type=models.IntegerIndexType.INTEGER, memory=_PINNED)),
        ("h3_res8", models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD, memory=_PINNED)),
    ]:
        client.create_payload_index(collection_name, key, schema)

    logger.info(
        f"Collection '{collection_name}' ready: dense(CACHED/BQ PINNED) + bm25(IDF) + GEO(PINNED)."
    )


def collection_point_count(
    client: QdrantClient, collection_name: str = COLLECTION_NAME
) -> int:
    if not client.collection_exists(collection_name):
        return 0
    return int(client.get_collection(collection_name).points_count or 0)


def is_collection_schema_ready(
    client: QdrantClient, collection_name: str = COLLECTION_NAME
) -> bool:
    if not client.collection_exists(collection_name):
        return False
    params = client.get_collection(collection_name).config.params
    v = params.vectors
    sparse = params.sparse_vectors or {}
    return isinstance(v, dict) and "dense" in v and "bm25" in sparse


# ── DataFrame Ingestion (with dense embedding cache) ─────────────────


def index_corpus(
    client: QdrantClient,
    df: pd.DataFrame,
    dense: DenseEmbedder,
    collection_name: str = COLLECTION_NAME,
    batch_size: int = 256,
) -> None:
    """Standard DataFrame ingestion with cached dense embeddings."""
    from .config import ARTIFACTS_DIR as _AD

    ids = np.asarray(df["id"].astype("int64").to_numpy())
    texts = (
        df["name"].fillna("Berlin Listing").astype(str)
        + ". "
        + df["room_type"].fillna("Entire home").astype(str)
        + " in "
        + df["neighbourhood"].astype(str)
        + ","
        + df["neighbourhood_group"].astype(str)
        + ". "
        + df["description"].fillna("").astype(str).str.slice(0, 400)
    ).tolist()

    _AD.mkdir(parents=True, exist_ok=True)
    dcache = _AD / "dense_embeddings.npz"

    dense_vecs = None
    if dcache.exists():
        try:
            data = np.load(dcache)
            if np.array_equal(data["ids"], ids):
                dense_vecs = data["vecs"]
                logger.info("Dense cache hit (%s vectors).", len(dense_vecs))
        except (OSError, KeyError, ValueError) as e:
            logger.debug("Dense cache load error: %s", e)
    if dense_vecs is None:
        logger.info("Embedding dense (%s) for %s listings…", dense.model, len(texts))
        dense_vecs = dense.embed_passages(texts)
        np.savez_compressed(dcache, ids=ids, vecs=dense_vecs)

    logger.info("Upserting %s points into '%s'…", f"{len(df):,}", collection_name)
    points: list[models.PointStruct] = []
    t0 = time.perf_counter()
    # to_dict("records") avoids itertuples' clash with a column named `name`.
    for idx, row in enumerate(
        track(df.to_dict("records"), description="Upserting listings")
    ):
        name = row["name"] if pd.notna(row["name"]) else "Berlin Listing"
        description = (
            str(row["description"])[:400] if pd.notna(row["description"]) else ""
        )
        room_type = row["room_type"] if pd.notna(row["room_type"]) else "Entire home"
        accommodates = int(row["accommodates"]) if pd.notna(row["accommodates"]) else 2
        pt = models.PointStruct(
            id=int(row["id"]),
            vector={
                "dense": dense_vecs[idx].tolist(),
                "bm25": models.Document(text=str(row["search_text"]), model=BM25_MODEL),
            },
            payload={
                "listing_id": int(row["id"]),
                "name": str(name),
                "description": description,
                "room_type": str(room_type),
                "price": float(row["price"]),
                "rating": float(row["rating"]),
                "number_of_reviews": int(row["number_of_reviews"]),
                "is_superhost": bool(row["is_superhost"]),
                "neighbourhood": str(row["neighbourhood"]),
                "neighbourhood_group": str(row["neighbourhood_group"]),
                "accommodates": accommodates,
                "h3_res8": str(row["h3_res8"]),
                "location": {
                    "lat": float(row["latitude"]),
                    "lon": float(row["longitude"]),
                },
            },
        )
        points.append(pt)
        if len(points) >= batch_size:
            client.upsert(collection_name=collection_name, points=points)
            points = []
    if points:
        client.upsert(collection_name=collection_name, points=points)
    logger.info("Indexed %s in %.2fs.", f"{len(df):,}", time.perf_counter() - t0)


def write_manifest(
    n: int, dense_backend: str, dense_model: str, collection_name: str = COLLECTION_NAME
):
    m = {
        "collection": collection_name,
        "dense": {"backend": dense_backend, "model": dense_model, "dim": DENSE_DIM},
        "bm25": {"model": BM25_MODEL},
        "points": int(n),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    MANIFEST_PATH.write_text(json.dumps(m, indent=2))
    return m


# ── Mathematical Decay Formula Construction ────────────────────────


def build_formula(
    target_lat: float,
    target_lon: float,
    dist_scale_m: float,
    target_price: float,
    price_scale: float,
    weights: DecayWeights | None = None,
) -> models.FormulaQuery:
    """
    Constructs Qdrant FormulaQuery combining:
    1. Semantic similarity ($score)
    2. Continuous GaussDecay on GeoDistance
    3. Continuous ExpDecay on Price
    4. Logarithmic review popularity (log10(reviews + 1))
    5. Normalized star rating (0.2 * rating)
    6. Superhost binary bonus
    """
    w = weights or default_weights()
    return models.FormulaQuery(
        formula=models.SumExpression(
            sum=[
                models.MultExpression(mult=[w.semantic, "$score"]),
                models.MultExpression(
                    mult=[
                        w.geo,
                        models.GaussDecayExpression(
                            gauss_decay=models.DecayParamsExpression(
                                x=models.GeoDistance(
                                    geo_distance=models.GeoDistanceParams(
                                        origin=models.GeoPoint(
                                            lat=target_lat, lon=target_lon
                                        ),
                                        to="location",
                                    )
                                ),
                                target=0.0,
                                scale=dist_scale_m,
                                midpoint=0.5,
                            )
                        ),
                    ]
                ),
                models.MultExpression(
                    mult=[
                        w.budget,
                        models.ExpDecayExpression(
                            exp_decay=models.DecayParamsExpression(
                                x="price",
                                target=target_price,
                                scale=price_scale,
                                midpoint=0.5,
                            )
                        ),
                    ]
                ),
                models.MultExpression(
                    mult=[
                        w.reviews,
                        models.Log10Expression(
                            log10=models.SumExpression(sum=["number_of_reviews", 1.0])
                        ),
                    ]
                ),
                models.MultExpression(
                    mult=[w.rating, models.MultExpression(mult=[0.2, "rating"])]
                ),
                models.MultExpression(
                    mult=[
                        w.superhost,
                        models.FieldCondition(
                            key="is_superhost", match=models.MatchValue(value=True)
                        ),
                    ]
                ),
            ]
        ),
        defaults={"price": target_price, "number_of_reviews": 0.0, "rating": 4.0},
    )


def _hybrid_prefetch(
    qvec: Sequence[float],
    qtext: str,
    limit: int,
    fusion: models.Fusion = models.Fusion.DBSF,
    filter: models.Filter | None = None,
) -> models.Prefetch:
    """
    Hybrid Prefetch combining Dense BQ + Sparse BM25 via Qdrant DBSF.
    DBSF (Distribution-Based Score Fusion) normalizes candidate score distributions (mean +- 3σ)
    into [0.0, 1.0] before feeding into FormulaQuery, ensuring mathematically balanced weights.
    When a viewport filter is provided it is pushed to both prefetches for filterable HNSW.
    """
    return models.Prefetch(
        query=models.FusionQuery(fusion=fusion),
        prefetch=[
            models.Prefetch(
                query=list(qvec),
                using="dense",
                limit=limit,
                filter=filter,
                params=models.SearchParams(
                    quantization=models.QuantizationSearchParams(rescore=True)
                ),
            ),
            models.Prefetch(
                query=models.Document(text=qtext, model=BM25_MODEL),
                using="bm25",
                limit=limit,
                filter=filter,
            ),
        ],
        limit=limit,
    )


# ── The 4 Pure Spatial Modes ────────────────────────────────────────


def search_dense(
    client: QdrantClient,
    qvec: Sequence[float],
    limit: int = 6,
    collection_name: str = COLLECTION_NAME,
):
    """Mode 1: Pure Semantic Search (Dense BQ). Shows spatial and budget blindspots."""
    return client.query_points(
        collection_name=collection_name,
        query=list(qvec),
        using="dense",
        limit=limit,
        with_payload=True,
    ).points


def search_hard_radius(
    client: QdrantClient,
    qvec: Sequence[float],
    lat: float,
    lon: float,
    radius_m: float,
    max_price: float,
    limit: int = 6,
    collection_name: str = COLLECTION_NAME,
):
    """Mode 2: Hard Binary Filter (geo_radius + price <= max). Demonstrates the Boundary Cliff."""
    return client.query_points(
        collection_name=collection_name,
        query=list(qvec),
        using="dense",
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="location",
                    geo_radius=models.GeoRadius(
                        center=models.GeoPoint(lat=lat, lon=lon), radius=radius_m
                    ),
                ),
                models.FieldCondition(key="price", range=models.Range(lte=max_price)),
            ]
        ),
        limit=limit,
        with_payload=True,
    ).points


def search_multi_decay(
    client: QdrantClient,
    qvec: Sequence[float],
    qtext: str,
    lat: float,
    lon: float,
    dist_scale_m: float = 2000.0,
    target_price: float = 95.0,
    price_scale: float = 35.0,
    weights: DecayWeights | None = None,
    prefetch_limit: int = 100,
    limit: int = 6,
    collection_name: str = COLLECTION_NAME,
):
    """Mode 3: Multi-Variable Continuous Mathematical Decay (FormulaQuery). Eliminates boundary dropouts."""
    prefetch = _hybrid_prefetch(qvec, qtext, prefetch_limit)
    formula = build_formula(lat, lon, dist_scale_m, target_price, price_scale, weights)
    return client.query_points(
        collection_name=collection_name,
        prefetch=[prefetch],
        query=formula,
        limit=limit,
        with_payload=True,
    ).points


def search_viewport_with_facets(
    client: QdrantClient,
    qvec: Sequence[float],
    qtext: str,
    top_left: tuple[float, float],
    bottom_right: tuple[float, float],
    target_price: float = 95.0,
    price_scale: float = 35.0,
    weights: DecayWeights | None = None,
    limit: int = 6,
    collection_name: str = COLLECTION_NAME,
) -> dict[str, Any]:
    """
    Mode 4: Dynamic Map Viewport Bounding Box + Multi-Decay Ranking + In-Database Faceting.
    Simulates production map UIs (Airbnb/Booking):
    1. Filters candidates within visible map bounding box (geo_bounding_box).
    2. Ranks results using continuous decay (GeoDistance + price) + review trust - dense BQ with rescore for map latency.
    3. Aggregates live listing distributions across neighborhoods and room types via client.facet.
    Ranking uses 1 query_points; facets are 2 additional facet calls (total 3 Qdrant ops).
    Viewport stays dense-only (qtext is accepted for call-site symmetry with
    search_multi_decay but intentionally unused here) so map pans stay
    sub-10ms; full hybrid DBSF lives in search_multi_decay (M3).
    """
    _ = qtext  # intentionally dense-only - see docstring
    tl_lat, tl_lon = top_left
    br_lat, br_lon = bottom_right
    center_lat = (tl_lat + br_lat) / 2.0
    center_lon = (tl_lon + br_lon) / 2.0

    bbox_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="location",
                geo_bounding_box=models.GeoBoundingBox(
                    top_left=models.GeoPoint(lat=tl_lat, lon=tl_lon),
                    bottom_right=models.GeoPoint(lat=br_lat, lon=br_lon),
                ),
            )
        ]
    )

    hits = client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                query=list(qvec),
                using="dense",
                filter=bbox_filter,
                limit=100,
                params=models.SearchParams(
                    quantization=models.QuantizationSearchParams(rescore=True)
                ),
            )
        ],
        query=build_formula(
            center_lat, center_lon, 3000.0, target_price, price_scale, weights
        ),
        limit=limit,
        with_payload=True,
    ).points

    district_facets = client.facet(
        collection_name=collection_name,
        key="neighbourhood_group",
        facet_filter=bbox_filter,
        limit=6,
    ).hits

    room_facets = client.facet(
        collection_name=collection_name,
        key="room_type",
        facet_filter=bbox_filter,
        limit=4,
    ).hits

    return {
        "hits": hits,
        "district_facets": [
            {"name": f.value, "count": f.count} for f in district_facets
        ],
        "room_facets": [{"room_type": f.value, "count": f.count} for f in room_facets],
        "viewport": {
            "top_left": top_left,
            "bottom_right": bottom_right,
            "center": (center_lat, center_lon),
        },
    }


def search_geo_polygon(
    client: QdrantClient,
    qvec: Sequence[float],
    exterior: list[dict[str, float]],
    limit: int = 6,
    collection_name: str = COLLECTION_NAME,
):
    """Native GeoPolygon Filter - Strict point-in-polygon containment evaluated inside Qdrant."""
    pts = [models.GeoPoint(lat=p["lat"], lon=p["lon"]) for p in exterior]
    if pts and (pts[0].lat != pts[-1].lat or pts[0].lon != pts[-1].lon):
        pts.append(pts[0])
    return client.query_points(
        collection_name=collection_name,
        query=list(qvec),
        using="dense",
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="location",
                    geo_polygon=models.GeoPolygon(
                        exterior=models.GeoLineString(points=pts)
                    ),
                )
            ]
        ),
        limit=limit,
        with_payload=True,
    ).points


def search_grouped_districts(
    client: QdrantClient,
    qvec: Sequence[float],
    qtext: str,
    lat: float,
    lon: float,
    dist_scale_m: float = 2000.0,
    target_price: float = 95.0,
    price_scale: float = 35.0,
    weights: DecayWeights | None = None,
    group_size: int = 2,
    limit: int = 5,
    collection_name: str = COLLECTION_NAME,
):
    """
    Grouped Diversity: hybrid DBSF + Formula decay, then group_by neighbourhood_group.
    Returns one curated set per district - avoids viewport homogeneity where Mitte dominates.
    Each group contains group_size best stays for that district.
    """
    prefetch = _hybrid_prefetch(qvec, qtext, 100)
    formula = build_formula(lat, lon, dist_scale_m, target_price, price_scale, weights)
    return client.query_points_groups(
        collection_name=collection_name,
        prefetch=[prefetch],
        query=formula,
        group_by="neighbourhood_group",
        group_size=group_size,
        limit=limit,
        with_payload=True,
    )
