"""Accuracy benchmark (full scale): global IDF vs per-tenant IDF vs exact tenant-local BM25.

Methodology
-----------
Ground truth ("ideal") = exact BM25 computed independently in Python over ONE
tenant's documents using tenant-local statistics (N_tenant, df_tenant), from the
same sparse vectors that were uploaded to Qdrant (vectorized via scipy CSR).

Mode A (default):  Qdrant query, tenant filter, shard-wide (global) IDF stats.
Mode B (new 1.19): same query + params.idf.corpus = tenant filter.
"""

import csv
import time

import numpy as np

from common import (
    ART_DIR,
    TENANT_FIELD,
    TenantIndex,
    embed_texts,
    get_client,
    load_embeddings,
    load_queries,
    ndcg_at_k,
    okapi_idf,
    qdrant_search as _qdrant_search,
    sample_per_forum,
)

QUERIES_PER_FORUM = 250
LIMIT = 10


def qdrant_search(client, q_ind, q_val, forum: str, scoped: bool):
    hits = _qdrant_search(client, q_ind, q_val, forum, scoped, limit=LIMIT)
    return [(pid, score) for pid, score, _ in hits]


def main() -> None:
    client = get_client()
    point_ids, forums, embs = load_embeddings()
    n_global = len(point_ids)
    print(f"Loaded {n_global} stored document vectors")

    # ---- global document frequency (replicates Qdrant's default shard-wide stats)
    t0 = time.time()
    df_global: dict[int, int] = {}
    for ind, _ in embs:
        for t in set(ind.tolist()):
            df_global[t] = df_global.get(t, 0) + 1
    print(f"Global df computed over {len(df_global)} distinct terms in {time.time() - t0:.1f}s")

    # ---- queries
    qdf = load_queries()
    sample = sample_per_forum(qdf, QUERIES_PER_FORUM, seed=7)
    q_texts = sample["q"].tolist()
    t0 = time.time()
    q_embs = embed_texts(q_texts)
    print(f"{len(sample)} test queries embedded in {time.time() - t0:.1f}s\n")

    rows = []
    probe_done = False
    diag = {"deviations": 0, "score_mismatch": 0, "tie_explained": 0}
    forum_order = sorted(sample[TENANT_FIELD].unique(), key=lambda f: -int((forums == f).sum()))
    total_q = 0

    for forum in forum_order:
        mask_row = np.where(forums == forum)[0]
        items = [(int(point_ids[i]), embs[i][0], embs[i][1]) for i in mask_row]
        index = TenantIndex.build(forum, items)
        pid_to_row = {int(p): r for r, p in enumerate(index.doc_ids)}

        # ---------- one-time formula probe on the largest forum ----------
        if not probe_done:
            probe_done = True
            cand = max(index.term_cols, key=lambda t: abs(index.df(t) - df_global.get(t, 0)))
            probe_doc, probe_dv = index.probe_doc(cand)
            qi_probe = np.array([cand], dtype=np.int32)
            qv_probe = np.array([1.0], dtype=np.float32)
            sa = dict(qdrant_search(client, qi_probe, qv_probe, forum, scoped=False)).get(probe_doc)
            sb = dict(qdrant_search(client, qi_probe, qv_probe, forum, scoped=True)).get(probe_doc)
            print("=" * 74)
            print("FORMULA PROBE (single-term query, weight=1.0)")
            print(f"  term={cand}  doc={probe_doc}  stored_val(dv)={probe_dv:.6f}")
            dg, dt = df_global.get(cand, 0), index.df(cand)
            print(f"  global : N={n_global} df={dg}   tenant '{forum}': N={index.n_docs} df={dt}")
            if sa and sb:
                imp_g, imp_f = sa / probe_dv, sb / probe_dv
                pg, pt = okapi_idf(n_global, dg), okapi_idf(index.n_docs, dt)
                print(f"  Qdrant-implied idf  global={imp_g:.6f}  tenant={imp_f:.6f}")
                print(f"  Okapi prediction    global={pg:.6f}  tenant={pt:.6f}")
                print(f"  match: global={'YES' if abs(imp_g - pg) < 1e-4 else 'NO'} "
                      f"tenant={'YES' if abs(imp_f - pt) < 1e-4 else 'NO'}")
            print("=" * 74 + "\n")

        f_rows = [i for i, f in enumerate(sample[TENANT_FIELD].tolist()) if f == forum]
        for qi_idx in f_rows:
            q_ind, q_val = q_embs[qi_idx]
            rank_rows, ideal_scores = index.bm25_scores(q_ind, q_val)
            if len(rank_rows) == 0:
                continue
            ideal_by_row = np.zeros(index.n_docs)
            for r, s in zip(rank_rows, ideal_scores):
                ideal_by_row[r] = s

            rankA = qdrant_search(client, q_ind, q_val, forum, scoped=False)
            rankB = qdrant_search(client, q_ind, q_val, forum, scoped=True)
            rowsA = [pid_to_row[p] for p, _ in rankA if p in pid_to_row]
            rowsB = [pid_to_row[p] for p, _ in rankB if p in pid_to_row]
            ndcgA = ndcg_at_k(np.array(rowsA), ideal_by_row, LIMIT)
            ndcgB = ndcg_at_k(np.array(rowsB), ideal_by_row, LIMIT)
            idsA = {p for p, _ in rankA}
            idsB = {p for p, _ in rankB}

            # ---- tie diagnostics: why would tenant-mode ever differ from ideal?
            if ndcgB < 0.99999:
                diag["deviations"] += 1
                myscore = {int(index.doc_ids[r]): s for r, s in zip(rank_rows, ideal_scores)}
                score_ok = all(
                    abs(myscore.get(p, float("-inf")) - s) <= max(1e-3, abs(s) * 1e-4)
                    for p, s in rankB
                )
                if not score_ok:
                    diag["score_mismatch"] += 1
                else:
                    cutoff = ideal_scores[rank_rows[LIMIT - 1]] if len(rank_rows) >= LIMIT else 0.0
                    my_top = {int(index.doc_ids[r]) for r in rank_rows[:LIMIT]}
                    extra = idsB - my_top
                    tied = all(
                        abs(myscore.get(p, float("-inf")) - cutoff) <= 1e-3 for p in extra
                    )
                    if tied:
                        diag["tie_explained"] += 1

            rows.append({
                "forum": forum,
                "query": sample["q"].iat[qi_idx],
                "ndcg_global": ndcgA,
                "ndcg_tenant": ndcgB,
                "top10_overlap": len(idsA & idsB) / LIMIT,
                "top1_changed": rankA[0][0] != rankB[0][0],
                "n_candidates": int((ideal_scores > 0).sum()),
            })
        total_q += len(f_rows)
        print(f"  {forum:<14} done ({total_q}/{len(sample)} queries)")

    # ---------------- aggregate ----------------
    gA = np.array([r["ndcg_global"] for r in rows])
    gB = np.array([r["ndcg_tenant"] for r in rows])
    ov = np.array([r["top10_overlap"] for r in rows])
    ch = np.array([r["top1_changed"] for r in rows])

    print("\n" + "=" * 74)
    print(f"RESULTS  ({len(rows)} queries, top-{LIMIT}, ground truth = exact tenant-local BM25)")
    print("=" * 74)
    print(f"{'forum':<14}{'queries':>8}{'NDCG global':>13}{'NDCG tenant':>13}{'overlap@10':>12}")
    for forum in forum_order:
        fr = [r for r in rows if r["forum"] == forum]
        if not fr:
            continue
        fa = np.mean([r["ndcg_global"] for r in fr])
        fb = np.mean([r["ndcg_tenant"] for r in fr])
        fo = np.mean([r["top10_overlap"] for r in fr])
        print(f"{forum:<14}{len(fr):>8}{fa:>13.4f}{fb:>13.4f}{fo:>12.3f}")
    print("-" * 74)
    print(f"{'OVERALL':<14}{len(rows):>8}{gA.mean():>13.4f}{gB.mean():>13.4f}{ov.mean():>12.3f}")
    print(f"\nMean NDCG@10 loss of global-IDF mode : {(gB - gA).mean():+.4f} "
          f"({(gB.mean() - gA.mean()) / gB.mean() * 100:+.2f}% relative)")
    print(f"P95 NDCG of global-IDF mode          : {np.percentile(gA, 5):.4f} (worst 5% tail)")
    print(f"Queries where top-1 result changed   : {ch.sum()}/{len(rows)} ({ch.mean() * 100:.1f}%)")
    print(f"Queries where >=half top-10 differs  : {(ov <= 0.5).sum()} ({(ov <= 0.5).mean() * 100:.1f}%)")
    print(f"Queries where tenant-mode is better  : {(gB > gA + 1e-9).sum()}")
    print(f"Queries where global-mode is better  : {(gA > gB + 1e-9).sum()}")
    print(f"\nTie diagnostics (tenant-mode vs ideal): {diag['deviations']} deviations | "
          f"{diag['tie_explained']} explained by equal-score ties | "
          f"{diag['score_mismatch']} real score mismatches")

    worst = sorted(rows, key=lambda r: r["ndcg_global"] - r["ndcg_tenant"])[:3]
    print("\nWorst distortions (global-IDF mode):")
    for r in worst:
        print(f"  [{r['forum']}] dNDCG={r['ndcg_tenant'] - r['ndcg_global']:+.3f}  \"{r['query'][:60]}\"")

    out = ART_DIR / "accuracy_results.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nPer-query results -> {out}")


if __name__ == "__main__":
    main()
