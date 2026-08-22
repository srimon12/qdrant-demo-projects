"""Replay the single worst-distortion query (auto-detected from the accuracy run)."""

import csv

import numpy as np
from qdrant_client import models as m

from common import (
    ART_DIR,
    COLLECTION,
    SPARSE_NAME,
    TENANT_FIELD,
    embed_texts,
    get_client,
)


def search(client, q_ind, q_val, forum: str, scoped: bool):
    params = None
    if scoped:
        params = m.SearchParams(
            idf=m.IdfCorpusParams(
                corpus=m.Filter(
                    must=[m.FieldCondition(key=TENANT_FIELD, match=m.MatchValue(value=forum))]
                )
            )
        )
    res = client.query_points(
        COLLECTION,
        query=m.SparseVector(indices=q_ind.tolist(), values=q_val.tolist()),
        using=SPARSE_NAME,
        query_filter=m.Filter(
            must=[m.FieldCondition(key=TENANT_FIELD, match=m.MatchValue(value=forum))]
        ),
        search_params=params,
        limit=5,
        with_payload=True,
    )
    return [(int(p.id), float(p.score), p.payload.get("title", "")) for p in res.points]


def main() -> None:
    with open(ART_DIR / "accuracy_results.csv") as f:
        rows = list(csv.DictReader(f))
    worst = max(rows, key=lambda r: float(r["ndcg_tenant"]) - float(r["ndcg_global"]))
    query, forum = worst["query"], worst["forum"]

    client = get_client()
    q_ind, q_val = embed_texts([query])[0]
    g = search(client, q_ind, q_val, forum, scoped=False)
    t = search(client, q_ind, q_val, forum, scoped=True)
    print(f'Query: "{query}"   (tenant filter: {forum})')
    print(f"NDCG@10  global={float(worst['ndcg_global']):.3f}  tenant={float(worst['ndcg_tenant']):.3f}\n")
    print(f"{'GLOBAL IDF (default)':<46}| {'PER-TENANT IDF (1.19)':<46}")
    print("-" * 94)
    for i in range(5):
        gs = f"{g[i][1]:7.3f}  {g[i][2][:36]:<38}"
        ts = f"{t[i][1]:7.3f}  {t[i][2][:36]:<38}"
        print(f"{gs:<47}| {ts:<47}")
    overlap = {x[0] for x in g} & {x[0] for x in t}
    print(f"\ntop-5 overlap: {len(overlap)}/5")


if __name__ == "__main__":
    main()
