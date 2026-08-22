"""Industry-standard BEIR evaluation of global vs per-tenant IDF.

CQADupStack is a BEIR duplicate-question retrieval task: 12 StackExchange
forums, 13,145 test queries, 23,703 qrels. Official protocol (Thakur et al.
2021) reports nDCG@10, MAP, and Recall@100, then macro-averages the 12 forums.

Both modes retrieve from the same tenant (payload filter on forum_tag). The
only difference is the IDF corpus: shard-wide statistics vs params.idf.corpus
scoped to that forum. Labels are the official duplicate-question qrels, not
the BM25 scores themselves.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    ART_DIR,
    FORUMS,
    TENANT_FIELD,
    average_precision,
    embed_texts,
    get_client,
    load_id_maps,
    load_qrels,
    load_queries,
    mrr_score,
    ndcg_labeled,
    qdrant_search,
    recall_at_k,
)

LIMIT = 100  # BEIR reports Recall@100; nDCG@10 is computed from the same list


def main() -> None:
    client = get_client()
    qrels = load_qrels()
    _, point_to_beir = load_id_maps()
    qdf = load_queries()
    print(f"Loaded {len(qdf)} BEIR queries, {sum(len(v) for v in qrels.values())} topics with qrels")
    print(f"id map: {len(point_to_beir)} point ids\n")

    rows = []
    forum_rows: dict[str, list[dict]] = {f: [] for f in FORUMS}
    t0 = time.time()

    for forum in FORUMS:
        fq = qdf[qdf[TENANT_FIELD] == forum].reset_index(drop=True)
        rels = qrels[forum]
        texts = fq["q"].tolist()
        qids = fq["_id"].astype(str).tolist()
        embs = embed_texts(texts)
        print(f"  {forum:<14} {len(fq)} queries embedded")

        n_ok = 0
        for qid, (q_ind, q_val) in zip(qids, embs):
            rel = rels.get(qid)
            if not rel:
                continue
            n_ok += 1
            hits_g = qdrant_search(client, q_ind, q_val, forum, scoped=False, limit=LIMIT)
            hits_t = qdrant_search(client, q_ind, q_val, forum, scoped=True, limit=LIMIT)
            ranked_g = [point_to_beir[pid] for pid, _, _ in hits_g if pid in point_to_beir]
            ranked_t = [point_to_beir[pid] for pid, _, _ in hits_t if pid in point_to_beir]
            rec = {
                "forum": forum,
                "query_id": qid,
                "n_rel": len(rel),
                "ndcg10_global": ndcg_labeled(ranked_g, rel, 10),
                "ndcg10_tenant": ndcg_labeled(ranked_t, rel, 10),
                "map_global": average_precision(ranked_g, rel),
                "map_tenant": average_precision(ranked_t, rel),
                "recall10_global": recall_at_k(ranked_g, rel, 10),
                "recall10_tenant": recall_at_k(ranked_t, rel, 10),
                "recall100_global": recall_at_k(ranked_g, rel, 100),
                "recall100_tenant": recall_at_k(ranked_t, rel, 100),
                "mrr_global": mrr_score(ranked_g, rel),
                "mrr_tenant": mrr_score(ranked_t, rel),
                "top1_changed": bool(ranked_g and ranked_t and ranked_g[0] != ranked_t[0]),
            }
            rows.append(rec)
            forum_rows[forum].append(rec)

        print(f"  {forum:<14} evaluated {n_ok} queries  ({time.time() - t0:.0f}s)")

    metrics = [
        "ndcg10", "map", "recall10", "recall100", "mrr",
    ]
    print("\n" + "=" * 88)
    print("BEIR CQADupStack  —  labeled duplicate-question retrieval")
    print("nDCG@10 / MAP / R@10 / R@100 / MRR    (macro-avg over forums, then micro-avg)")
    print("=" * 88)
    hdr = f"{'forum':<14}{'n':>6}"
    for met in metrics:
        hdr += f"{met+' G':>10}{met+' T':>10}"
    print(hdr)
    macro_g = {m: [] for m in metrics}
    macro_t = {m: [] for m in metrics}
    for forum in FORUMS:
        fr = forum_rows[forum]
        if not fr:
            continue
        line = f"{forum:<14}{len(fr):>6}"
        for met in metrics:
            ga = sum(r[f"{met}_global"] for r in fr) / len(fr)
            ta = sum(r[f"{met}_tenant"] for r in fr) / len(fr)
            macro_g[met].append(ga)
            macro_t[met].append(ta)
            line += f"{ga:>10.4f}{ta:>10.4f}"
        print(line)
    print("-" * 88)
    line = f"{'MACRO':<14}{len(rows):>6}"
    for met in metrics:
        line += f"{sum(macro_g[met])/len(macro_g[met]):>10.4f}{sum(macro_t[met])/len(macro_t[met]):>10.4f}"
    print(line)
    line = f"{'MICRO':<14}{len(rows):>6}"
    for met in metrics:
        ga = sum(r[f"{met}_global"] for r in rows) / len(rows)
        ta = sum(r[f"{met}_tenant"] for r in rows) / len(rows)
        line += f"{ga:>10.4f}{ta:>10.4f}"
    print(line)

    n_better = sum(r["ndcg10_tenant"] > r["ndcg10_global"] + 1e-12 for r in rows)
    n_worse = sum(r["ndcg10_global"] > r["ndcg10_tenant"] + 1e-12 for r in rows)
    n_t1 = sum(r["top1_changed"] for r in rows)
    print(f"\nQueries where tenant nDCG@10 > global : {n_better}/{len(rows)}")
    print(f"Queries where global nDCG@10 > tenant : {n_worse}/{len(rows)}")
    print(f"Queries where top-1 result changed    : {n_t1}/{len(rows)} ({n_t1/len(rows)*100:.1f}%)")

    out = ART_DIR / "beir_eval.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nPer-query BEIR results -> {out}")


if __name__ == "__main__":
    main()
