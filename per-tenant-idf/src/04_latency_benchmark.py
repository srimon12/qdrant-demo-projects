"""Latency benchmark: global IDF vs per-tenant (corpus-filtered) IDF.

Same queries, same filter, only `params.idf` differs. Warmup, then timed runs.
"""

import csv
import time

import numpy as np

from common import (
    ART_DIR,
    TENANT_FIELD,
    embed_texts,
    get_client,
    load_queries,
    qdrant_search,
    sample_per_forum,
)

QUERIES_PER_FORUM = 100
REPS = 20
WARMUP = 5


def search(client, q_ind, q_val, forum: str, scoped: bool) -> None:
    qdrant_search(client, q_ind, q_val, forum, scoped, limit=10)


def stats(xs: list[float]) -> tuple[float, float, float, float]:
    a = np.array(xs)
    return (
        float(np.mean(a)),
        float(np.percentile(a, 50)),
        float(np.percentile(a, 95)),
        float(np.percentile(a, 99)),
    )


def main() -> None:
    client = get_client()
    qdf = load_queries()
    sample = sample_per_forum(qdf, QUERIES_PER_FORUM, seed=7)
    q_embs = embed_texts(sample["q"].tolist())

    t_global: list[float] = []
    t_tenant: list[float] = []

    for i, forum in enumerate(sample[TENANT_FIELD].tolist()):
        q_ind, q_val = q_embs[i]
        for _ in range(WARMUP):
            search(client, q_ind, q_val, forum, scoped=False)
            search(client, q_ind, q_val, forum, scoped=True)
        for _ in range(REPS):
            t0 = time.perf_counter()
            search(client, q_ind, q_val, forum, scoped=False)
            t_global.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            search(client, q_ind, q_val, forum, scoped=True)
            t_tenant.append(time.perf_counter() - t0)

    g = stats(t_global)
    t = stats(t_tenant)
    print(f"\n{len(t_global) // REPS} queries x {REPS} reps = {len(t_global)} timed measurements (client-side wall time)")
    print("-" * 64)
    print(f"{'mode':<22}{'mean':>9}{'p50':>9}{'p95':>9}{'p99':>9}")
    print(f"{'global IDF':<22}{g[0] * 1000:>8.2f}ms{g[1] * 1000:>8.2f}ms{g[2] * 1000:>8.2f}ms{g[3] * 1000:>8.2f}ms")
    print(f"{'per-tenant IDF':<22}{t[0] * 1000:>8.2f}ms{t[1] * 1000:>8.2f}ms{t[2] * 1000:>8.2f}ms{t[3] * 1000:>8.2f}ms")
    print(f"\noverhead of tenant-scoped stats: {(t[1] - g[1]) * 1000:+.2f}ms p50 "
          f"({(t[1] / g[1] - 1) * 100:+.1f}%), p95 {(t[2] - g[2]) * 1000:+.2f}ms "
          f"({(t[2] / g[2] - 1) * 100:+.1f}%)")

    out = ART_DIR / "latency_results.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["mode", "mean_ms", "p50_ms", "p95_ms", "p99_ms", "n"])
        w.writeheader()
        w.writerow({"mode": "global", "mean_ms": g[0] * 1000, "p50_ms": g[1] * 1000,
                    "p95_ms": g[2] * 1000, "p99_ms": g[3] * 1000, "n": len(t_global)})
        w.writerow({"mode": "tenant", "mean_ms": t[0] * 1000, "p50_ms": t[1] * 1000,
                    "p95_ms": t[2] * 1000, "p99_ms": t[3] * 1000, "n": len(t_tenant)})
    per_q = ART_DIR / "latency_per_query.csv"
    with open(per_q, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["i", "forum", "global_ms", "tenant_ms"])
        w.writeheader()
        # t_global/t_tenant are REPS consecutive samples per query
        n_q = len(sample)
        for qi, forum in enumerate(sample[TENANT_FIELD].tolist()):
            for r in range(REPS):
                idx = qi * REPS + r
                w.writerow({
                    "i": qi, "forum": forum,
                    "global_ms": t_global[idx] * 1000,
                    "tenant_ms": t_tenant[idx] * 1000,
                })
    print(f"latency tables -> {out} , {per_q}")


if __name__ == "__main__":
    main()
