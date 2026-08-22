"""Edge cases for the per-tenant IDF parameter (documented behaviors).

1. idf param on a sparse vector WITHOUT the IDF modifier -> must error.
2. corpus filter matching zero points -> constant weight per term
   (rarity signal gone, ranking degenerates to plain TF).
"""

import numpy as np
from qdrant_client import models as m
from qdrant_client import QdrantClient

from common import COLLECTION, SPARSE_NAME, TENANT_FIELD, load_embeddings

COLL = "edge_case_noidf"
FORUM = "gaming"


def search(client, q_ind, q_val, corpus_filter=None):
    params = None
    if corpus_filter is not None:
        params = m.SearchParams(idf=m.IdfCorpusParams(corpus=corpus_filter))
    res = client.query_points(
        COLLECTION,
        query=m.SparseVector(indices=q_ind.tolist(), values=q_val.tolist()),
        using=SPARSE_NAME,
        query_filter=m.Filter(
            must=[m.FieldCondition(key=TENANT_FIELD, match=m.MatchValue(value=FORUM))]
        ),
        search_params=params,
        limit=5,
        with_payload=True,
    )
    return [(int(p.id), round(float(p.score), 6)) for p in res.points]


def main() -> None:
    client = QdrantClient(url="http://localhost:6333", timeout=60)

    # ---- 1. no IDF modifier -------------------------------------------------
    if client.collection_exists(COLL):
        client.delete_collection(COLL)
    client.create_collection(
        COLL,
        vectors_config={},
        sparse_vectors_config={SPARSE_NAME: m.SparseVectorParams(modifier=m.Modifier.NONE)},
    )
    client.upsert(
        COLL,
        points=[
            m.PointStruct(id=1, vector={SPARSE_NAME: m.SparseVector(indices=[10, 20], values=[1.0, 2.0])}),
            m.PointStruct(id=2, vector={SPARSE_NAME: m.SparseVector(indices=[10, 30], values=[1.0, 1.0])}),
        ],
    )
    try:
        client.query_points(
            COLL,
            query=m.SparseVector(indices=[10], values=[1.0]),
            using=SPARSE_NAME,
            search_params=m.SearchParams(idf=m.IdfCorpusParams(corpus=m.Filter(must=[]))),
            limit=5,
        )
        print("1. idf on non-IDF vector : NO ERROR (unexpected!)")
    except Exception as e:
        msg = str(e).split("\n")[0][:100]
        print(f"1. idf on non-IDF vector : rejected as documented -> {msg}")
    client.delete_collection(COLL)

    # ---- 2. empty corpus -> constant weights --------------------------------
    _, forums, embs = load_embeddings()
    fmask = forums == FORUM
    f_embs = [embs[i] for i in np.where(fmask)[0]]

    # rare-in-gaming term (df==1) and a common one (df>30% of forum)
    from collections import Counter

    df = Counter()
    for ind, _ in f_embs:
        df.update(set(ind.tolist()))
    n_f = int(fmask.sum())
    rare = min(df, key=lambda t: (df[t], t))
    common = max(df, key=lambda t: df[t])
    print(f"\n2. forum '{FORUM}' N={n_f}: rare term df={df[rare]}, common term df={df[common]}")

    q_ind = np.array([rare, common], dtype=np.int32)
    q_val = np.array([1.0, 1.0], dtype=np.float32)
    tenant_f = m.Filter(must=[m.FieldCondition(key=TENANT_FIELD, match=m.MatchValue(value=FORUM))])
    empty_f = m.Filter(must=[m.FieldCondition(key=TENANT_FIELD, match=m.MatchValue(value="no_such_tenant"))])
    empty_f2 = m.Filter(must=[m.FieldCondition(key=TENANT_FIELD, match=m.MatchValue(value="also_nonexistent"))])

    ok = search(client, q_ind, q_val, corpus_filter=tenant_f)
    dead = search(client, q_ind, q_val, corpus_filter=empty_f)
    dead2 = search(client, q_ind, q_val, corpus_filter=empty_f2)

    print(f"   valid corpus   top5: {ok}")
    print(f"   empty corpus   top5: {dead}")
    same_const = dict(dead) == dict(dead2)
    print(f"   two different empty corpora give identical scores (constant weight): {same_const}")


if __name__ == "__main__":
    main()
