from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from typing import Any

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
    BM25_SPARSE_MODEL,
    COLBERT_VECTOR_DIM,
    COLLECTION_NAME,
    DENSE_VECTOR_DIM,
    QDRANT_URL,
)
from .data import resolve_query_statutory_bridges
from .embedder import ColbertLateInteractionEmbedder, DensePrecedentEmbedder


def get_qdrant_client(server_url: str = QDRANT_URL) -> QdrantClient:
    """Connect to Qdrant, retrying briefly so `docker compose up -d` can finish booting."""
    client = QdrantClient(url=server_url, timeout=120)
    last_error: Exception | None = None
    for _ in range(15):
        try:
            client.get_collections()
            return client
        except _QDRANT_CONNECT_ERRORS as exc:
            last_error = exc
            time.sleep(1)
    raise ConnectionError(
        f"Cannot reach Qdrant at {server_url}. "
        "From this folder run `docker compose up -d`, then retry. "
        "If another demo already started Qdrant on :6333, reuse it - "
        "collection names do not collide."
    ) from last_error


def setup_collection(
    client: QdrantClient,
    collection_name: str = COLLECTION_NAME,
    recreate: bool = False,
) -> None:
    """
    Configure Qdrant 1.19 collection with unified storage and memory tiers:
    - Dense (384-d): CACHED on-disk with 1-bit Binary Quantization (BQ) PINNED in RAM.
    - Sparse BM25: CACHED inverted index (the 1.19 sparse cached tier) with Modifier.IDF.
    - ColBERT (128-d): COLD on-disk with m=0 (zero HNSW graph memory tax).
    - Payload Indexes: PINNED in RAM with prefix=True for instant statutory lookups.
    """
    if client.collection_exists(collection_name):
        if not recreate:
            return
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": models.VectorParams(
                size=DENSE_VECTOR_DIM,
                distance=models.Distance.COSINE,
                memory=models.Memory.CACHED,
                quantization_config=models.BinaryQuantization(
                    binary=models.BinaryQuantizationConfig(memory=models.Memory.PINNED)
                ),
            ),
            "colbert": models.VectorParams(
                size=COLBERT_VECTOR_DIM,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
                memory=models.Memory.COLD,
                hnsw_config=models.HnswConfigDiff(m=0),
            ),
        },
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(
                index=models.SparseIndexParams(memory=models.Memory.CACHED),
                modifier=models.Modifier.IDF,
            ),
        },
    )

    # Qdrant 1.19 payload index schema (unified `memory` tiers, explicit)
    for payload_field in ("legal_references", "mapped_references"):
        client.create_payload_index(
            collection_name,
            payload_field,
            models.KeywordIndexParams(
                type=models.KeywordIndexType.KEYWORD,
                prefix=True,
                memory=models.Memory.PINNED,
            ),
        )
    client.create_payload_index(
        collection_name,
        "precedent_key",
        models.KeywordIndexParams(
            type=models.KeywordIndexType.KEYWORD, memory=models.Memory.PINNED
        ),
    )
    client.create_payload_index(
        collection_name,
        "judgment_date",
        models.DatetimeIndexParams(
            type=models.DatetimeIndexType.DATETIME, memory=models.Memory.PINNED
        ),
    )
    client.create_payload_index(
        collection_name,
        "source_case_count",
        models.IntegerIndexParams(
            type=models.IntegerIndexType.INTEGER, memory=models.Memory.PINNED
        ),
    )


def generate_qdrant_point_structs(
    precedent_records: list[dict[str, Any]],
    dense_embedder: DensePrecedentEmbedder,
    colbert_embedder: ColbertLateInteractionEmbedder,
    batch_size: int = 16,
) -> Iterable[models.PointStruct]:
    """Batch embed precedent passages and yield Qdrant PointStruct instances."""
    payload_keys = (
        "precedent_key",
        "title",
        "text",
        "ratio",
        "statute_labels",
        "legal_references",
        "mapped_references",
        "source_case_count",
        "source_case_ids",
        "word_count",
    )
    for start_idx in track(
        range(0, len(precedent_records), batch_size),
        description="Embedding + indexing precedents",
    ):
        batch_slice = precedent_records[start_idx : start_idx + batch_size]
        passages_for_embedding = [
            f"{r['title']}. {r['ratio'][:800]}" for r in batch_slice
        ]
        dense_vectors = dense_embedder.embed_precedent_passages(
            passages_for_embedding, batch_size=16
        )
        colbert_vectors = colbert_embedder.embed_precedent_passages(
            passages_for_embedding, batch_size=4
        )

        for offset, (record, dense_vec, colbert_vec) in enumerate(
            zip(batch_slice, dense_vectors, colbert_vectors)
        ):
            payload_dict = {k: record[k] for k in payload_keys}
            if record["judgment_date"]:
                payload_dict["judgment_date"] = record["judgment_date"]

            yield models.PointStruct(
                id=start_idx + offset,
                vector={
                    "dense": dense_vec.tolist(),
                    "colbert": colbert_vec.tolist(),
                    "bm25": models.Document(
                        text=record["search_text"], model=BM25_SPARSE_MODEL
                    ),
                },
                payload=payload_dict,
            )


def index_precedents(
    client: QdrantClient,
    precedent_records: list[dict[str, Any]],
    dense_embedder: DensePrecedentEmbedder,
    colbert_embedder: ColbertLateInteractionEmbedder,
    collection_name: str = COLLECTION_NAME,
) -> None:
    """Stream and index precedent points into Qdrant."""
    client.upload_points(
        collection_name=collection_name,
        points=generate_qdrant_point_structs(
            precedent_records, dense_embedder, colbert_embedder
        ),
        batch_size=16,
        parallel=1,
        wait=True,
    )


def collection_point_count(
    client: QdrantClient, collection_name: str = COLLECTION_NAME
) -> int:
    """Return count of precedent points indexed in the collection."""
    if not client.collection_exists(collection_name):
        return 0
    return int(client.get_collection(collection_name).points_count or 0)


def is_collection_schema_ready(
    client: QdrantClient, collection_name: str = COLLECTION_NAME
) -> bool:
    """True when dense, ColBERT, and BM25 named vectors are all present."""
    if not client.collection_exists(collection_name):
        return False
    params = client.get_collection(collection_name).config.params
    configured_vectors = params.vectors
    sparse = params.sparse_vectors or {}
    return (
        isinstance(configured_vectors, dict)
        and {"dense", "colbert"}.issubset(configured_vectors)
        and "bm25" in sparse
    )


# =============================================================================
# SEARCH LADDER RETRIEVAL MODES (M1 -> M4)
# =============================================================================


def search_dense(
    client: QdrantClient,
    query_dense_vector: Sequence[float],
    limit: int = 5,
    collection_name: str = COLLECTION_NAME,
):
    """Mode 1: Baseline Dense Vector Search with 1-bit Binary Quantization (sub-3ms)."""
    return client.query_points(
        collection_name=collection_name,
        query=list(query_dense_vector),
        using="dense",
        limit=limit,
        with_payload=True,
    ).points


def _rrf_query(
    rrf_k: int | None = None,
    rrf_weights: Sequence[float] | None = None,
) -> models.FusionQuery | models.RrfQuery:
    """RRF fusion. Default k=2, equal weights - the honest tutorial default."""
    if rrf_weights is not None:
        return models.RrfQuery(rrf=models.Rrf(weights=list(rrf_weights)))
    if rrf_k is not None:
        return models.RrfQuery(rrf=models.Rrf(k=rrf_k))
    return models.FusionQuery(fusion=models.Fusion.RRF)


def dense_and_bm25_prefetches(
    query_dense_vector: Sequence[float],
    query_raw_text: str,
    candidate_limit: int = 100,
    rescore: bool = False,
) -> list[models.Prefetch]:
    """The two candidate lists RRF (or a later stage) will fuse."""
    return [
        models.Prefetch(
            query=list(query_dense_vector),
            using="dense",
            limit=candidate_limit,
            params=models.SearchParams(
                quantization=models.QuantizationSearchParams(rescore=rescore)
            ),
        ),
        models.Prefetch(
            query=models.Document(text=query_raw_text, model=BM25_SPARSE_MODEL),
            using="bm25",
            limit=candidate_limit,
        ),
    ]


def build_hybrid_rrf_prefetch(
    query_dense_vector: Sequence[float],
    query_raw_text: str,
    candidate_limit: int = 100,
    rescore: bool = False,
    rrf_k: int | None = None,
    rrf_weights: Sequence[float] | None = None,
) -> models.Prefetch:
    """Nested hybrid prefetch: Dense BQ + BM25, fused with RRF.

    Used as an inner stage (M3 formula, M4 ColBERT). Standalone hybrid is
    `search_hybrid`, which fuses the same two prefetches at the top-level query
    - the canonical Qdrant hybrid pattern, not a second RRF wrap.
    """
    return models.Prefetch(
        query=_rrf_query(rrf_k, rrf_weights),
        prefetch=dense_and_bm25_prefetches(
            query_dense_vector, query_raw_text, candidate_limit, rescore
        ),
        limit=candidate_limit,
    )


def search_hybrid(
    client: QdrantClient,
    query_dense_vector: Sequence[float],
    query_raw_text: str,
    limit: int = 5,
    prefetch_limit: int = 100,
    rescore: bool = True,
    rrf_k: int | None = None,
    rrf_weights: Sequence[float] | None = None,
    collection_name: str = COLLECTION_NAME,
):
    """Mode 2: Dense BQ + server-side BM25, fused with RRF in one query_points call."""
    return client.query_points(
        collection_name=collection_name,
        prefetch=dense_and_bm25_prefetches(
            query_dense_vector, query_raw_text, prefetch_limit, rescore
        ),
        query=_rrf_query(rrf_k, rrf_weights),
        limit=limit,
        with_payload=True,
    ).points


def build_statutory_formula_query(
    query_raw_text: str,
    base_score_scale: float = 1.0,
    domain_boost_scale: float = 6.0,
    target_evaluation_date: str | None = None,
) -> models.FormulaQuery:
    """
    Server-side FormulaQuery (same expression for M3 and M4):
    1. Base similarity ($score)
    2. 20-year half-life ExpDecay on judgment_date
    3. log10(citation in-degree)
    4. Exact statute match (+0.25 × domain)
    5. Concordance bridge match (+0.50 × domain)

    domain_boost_scale=6.0 is the validated peak on this 22-query set.
    """
    direct_statutes, concordance_bridges = resolve_query_statutory_bridges(
        query_raw_text
    )
    anchor_timestamp_iso = target_evaluation_date or (
        datetime.now(UTC).strftime("%Y-%m-%d") + "T00:00:00Z"
    )

    formula_terms: list[Any] = [
        models.MultExpression(mult=[base_score_scale, "$score"]),
        # 1. Temporal Recency Exponential Decay (20-Year Half-Life)
        models.MultExpression(
            mult=[
                0.20 * domain_boost_scale,
                models.ExpDecayExpression(
                    exp_decay=models.DecayParamsExpression(
                        x=models.DatetimeKeyExpression(datetime_key="judgment_date"),
                        target=models.DatetimeExpression(datetime=anchor_timestamp_iso),
                        scale=86400 * 365 * 20,
                        midpoint=0.5,
                    )
                ),
            ]
        ),
        # 2. Citation Graph In-Degree Authority
        models.MultExpression(
            mult=[
                0.15 * domain_boost_scale,
                models.Log10Expression(
                    log10=models.SumExpression(sum=["source_case_count", 1.0])
                ),
            ]
        ),
    ]

    for direct_sec in direct_statutes:
        formula_terms.append(
            models.MultExpression(
                mult=[
                    models.FieldCondition(
                        key="legal_references",
                        match=models.MatchValue(value=direct_sec),
                    ),
                    0.25 * domain_boost_scale,
                ]
            )
        )

    for bridge_sec in concordance_bridges:
        formula_terms.append(
            models.MultExpression(
                mult=[
                    models.FieldCondition(
                        key="mapped_references",
                        match=models.MatchValue(value=bridge_sec),
                    ),
                    0.50 * domain_boost_scale,
                ]
            )
        )

    return models.FormulaQuery(
        formula=models.SumExpression(sum=formula_terms),
        defaults={"source_case_count": 0.0, "judgment_date": "2000-01-01T00:00:00Z"},
    )


def search_formula(
    client: QdrantClient,
    query_dense_vector: Sequence[float],
    query_raw_text: str,
    limit: int = 5,
    prefetch_limit: int = 100,
    rescore: bool = True,
    collection_name: str = COLLECTION_NAME,
):
    """Mode 3: Hybrid RRF prefetch, then the same FormulaQuery M4 uses (no ColBERT)."""
    return client.query_points(
        collection_name=collection_name,
        prefetch=build_hybrid_rrf_prefetch(
            query_dense_vector,
            query_raw_text,
            prefetch_limit,
            rescore=rescore,
        ),
        query=build_statutory_formula_query(query_raw_text),
        limit=limit,
        with_payload=True,
    ).points


def search_universal_pipeline(
    client: QdrantClient,
    query_dense_vector: Sequence[float],
    query_colbert_matrix: Sequence[Sequence[float]],
    query_raw_text: str,
    limit: int = 5,
    prefetch_limit: int = 100,
    colbert_rescore_limit: int = 40,
    rescore: bool = False,
    collection_name: str = COLLECTION_NAME,
):
    """
    Mode 4: one query_points round-trip.
      Stage 1 - Hybrid RRF prefetch (dense BQ + BM25)
      Stage 2 - ColBERT MAX_SIM on the top-40 (COLD disk, m=0)
      Stage 3 - the same FormulaQuery as M3 (concordance + recency + authority)
    """
    return client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                query=[list(v) for v in query_colbert_matrix],
                using="colbert",
                prefetch=build_hybrid_rrf_prefetch(
                    query_dense_vector,
                    query_raw_text,
                    prefetch_limit,
                    rescore=rescore,
                ),
                limit=colbert_rescore_limit,
            )
        ],
        query=build_statutory_formula_query(query_raw_text),
        limit=limit,
        with_payload=True,
    ).points


# =============================================================================
# QDRANT 1.19 ANALYTICS: FACETS & GROUPING (clean - Discover/Recommend stripped)
# =============================================================================


def facet_statutory_landscape(
    client: QdrantClient,
    key: str = "legal_references",
    limit: int = 12,
    collection_name: str = COLLECTION_NAME,
):
    """Qdrant 1.19 in-database faceting on indexed statutory references."""
    return client.facet(
        collection_name=collection_name,
        key=key,
        limit=limit,
    ).hits


def search_grouped_statutes(
    client: QdrantClient,
    query_dense_vector: Sequence[float],
    group_by_field: str = "legal_references",
    group_size: int = 2,
    limit: int = 4,
    collection_name: str = COLLECTION_NAME,
):
    """
    Multi-statute diversity via server-side grouping (dense query).
    legal_references is an array keyword; groups may include noisy 'section:*' keys
    where act extraction missed. Showcase filters to canonical act:section keys.
    """
    return client.query_points_groups(
        collection_name=collection_name,
        query=list(query_dense_vector),
        using="dense",
        group_by=group_by_field,
        group_size=group_size,
        limit=limit,
        with_payload=True,
    )
