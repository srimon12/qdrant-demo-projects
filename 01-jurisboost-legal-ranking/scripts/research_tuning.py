"""Research: RRF k/weights + BM25 avg_len + Formula scaling on 22 queries."""

from __future__ import annotations

from qdrant_client import models

# Inline benchmark queries to avoid import cycle
from src.benchmark import EDGE_QUERIES, EXPLICIT_QUERIES, PARAPHRASED_QUERIES
from src.config import COLLECTION_NAME
from src.embedder import ColbertLateInteractionEmbedder, DensePrecedentEmbedder
from src.qdrant_ops import (
    build_statutory_formula_query,
    get_qdrant_client,
)

ALL = EXPLICIT_QUERIES + PARAPHRASED_QUERIES + EDGE_QUERIES  # 22


def has_expected(payload, expects):
    refs = set(payload.get("legal_references", [])) | set(
        payload.get("mapped_references", [])
    )
    return bool(refs & set(expects))


def eval_with_params(
    rrf_k=None,
    rrf_weights=None,
    avg_len=None,
    formula_scales=None,  # dict base=, domain=
):
    client = get_qdrant_client()
    dense = DensePrecedentEmbedder()
    colbert = ColbertLateInteractionEmbedder()

    # build custom search fns capturing params
    def hybrid_search(qvec, qtext, limit=5):
        # custom prefetch with avg_len and rrf params
        # Inner RRF fuses dense+sparse (2 prefetches) - weights/k apply here
        dense_pref = models.Prefetch(
            query=list(qvec),
            using="dense",
            limit=100,
            params=models.SearchParams(
                quantization=models.QuantizationSearchParams(rescore=True)
            ),
        )
        bm25_doc = models.Document(text=qtext, model="Qdrant/bm25")
        if avg_len is not None:
            bm25_doc = models.Document(
                text=qtext, model="Qdrant/bm25", options={"avg_len": float(avg_len)}
            )
        sparse_pref = models.Prefetch(query=bm25_doc, using="bm25", limit=100)

        if rrf_weights is not None:
            fusion = models.RrfQuery(rrf=models.Rrf(weights=rrf_weights))
        elif rrf_k is not None:
            fusion = models.RrfQuery(rrf=models.Rrf(k=rrf_k))
        else:
            fusion = models.FusionQuery(fusion=models.Fusion.RRF)

        return client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[dense_pref, sparse_pref],
            query=fusion,
            limit=limit,
            with_payload=True,
        ).points

    def universal_search(qvec, colvec, qtext, limit=5):
        dense_pref = models.Prefetch(
            query=list(qvec),
            using="dense",
            limit=100,
            params=models.SearchParams(
                quantization=models.QuantizationSearchParams(rescore=False)
            ),
        )
        bm25_doc = models.Document(text=qtext, model="Qdrant/bm25")
        if avg_len is not None:
            bm25_doc = models.Document(
                text=qtext, model="Qdrant/bm25", options={"avg_len": float(avg_len)}
            )
        sparse_pref = models.Prefetch(query=bm25_doc, using="bm25", limit=100)
        if rrf_weights is not None:
            inner_fusion = models.RrfQuery(rrf=models.Rrf(weights=rrf_weights))
        elif rrf_k is not None:
            inner_fusion = models.RrfQuery(rrf=models.Rrf(k=rrf_k))
        else:
            inner_fusion = models.FusionQuery(fusion=models.Fusion.RRF)
        hybrid_pref = models.Prefetch(
            query=inner_fusion,
            prefetch=[dense_pref, sparse_pref],
            limit=100,
        )
        # formula scales
        base = formula_scales.get("base", 1.0) if formula_scales else 1.0
        domain = formula_scales.get("domain", 6.0) if formula_scales else 6.0
        formula_q = build_statutory_formula_query(
            qtext, base_score_scale=base, domain_boost_scale=domain
        )
        return client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=[list(v) for v in colvec],
                    using="colbert",
                    prefetch=hybrid_pref,
                    limit=40,
                )
            ],
            query=formula_q,
            limit=limit,
            with_payload=True,
        ).points

    hits = {"M2": 0, "M4": 0}
    for item in ALL:
        qtext = item["query"]
        expects = item["expects"]
        qvec = dense.embed_legal_queries(qtext)[0]
        colvec = colbert.embed_legal_queries(qtext)[0].tolist()
        # M2 hybrid
        h = hybrid_search(qvec, qtext, limit=3)
        if h and has_expected(h[0].payload or {}, expects):
            hits["M2"] += 1
        # M4 universal
        u = universal_search(qvec, colvec, qtext, limit=3)
        if u and has_expected(u[0].payload or {}, expects):
            hits["M4"] += 1
    return hits


def main():
    print("=== RRF k sweep (avg_len=256 default, formula domain=6.0) ===")
    for k in [2, 10, 30, 60]:
        h = eval_with_params(rrf_k=k)
        print(
            f"k={k:2d} -> M2 Top1 {h['M2']}/22 ({h['M2'] / 22 * 100:.1f}%) | M4 Top1 {h['M4']}/22 ({h['M4'] / 22 * 100:.1f}%)"
        )

    print("\n=== RRF weights sweep (k=2) ===")
    for w in [[1.0, 1.0], [0.6, 1.4], [1.4, 0.6], [2.0, 1.0], [1.0, 2.0]]:
        h = eval_with_params(rrf_weights=w)
        print(f"weights={w} -> M2 {h['M2']}/22 | M4 {h['M4']}/22")

    print("\n=== BM25 avg_len sweep ===")
    for al in [256, 366, 150, 40]:
        h = eval_with_params(avg_len=al)
        print(f"avg_len={al} -> M2 {h['M2']}/22 | M4 {h['M4']}/22")

    print("\n=== Formula domain_boost sweep (base 1.0) ===")
    for d in [1.0, 3.0, 6.0, 9.0]:
        h = eval_with_params(formula_scales={"base": 1.0, "domain": d})
        print(f"domain={d} -> M2 {h['M2']}/22 | M4 {h['M4']}/22")

    print("\n=== Combined best guess ===")
    # From sweeps, pick best: try k=2 default, avg_len=366 (true avg), domain=6.0 already best
    h = eval_with_params(
        rrf_k=2, avg_len=366, formula_scales={"base": 1.0, "domain": 6.0}
    )
    print(f"RRF k=2 avg_len=366 domain=6.0 -> M2 {h['M2']}/22 M4 {h['M4']}/22")
    h = eval_with_params(rrf_weights=[1.0, 1.0], avg_len=366)
    print(f"equal weights avg_len=366 -> M2 {h['M2']}/22 M4 {h['M4']}/22")


if __name__ == "__main__":
    main()
