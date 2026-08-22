"""Create the Qdrant 1.19 collection and index the CQADupStack corpus.

- sparse vector 'bm25' with the IDF modifier (required for per-tenant IDF)
- keyword index on forum_tag with is_tenant=True (tenant-aware layout)
"""

import time

from qdrant_client import models as m

from common import (
    COLLECTION,
    SPARSE_NAME,
    TENANT_FIELD,
    embed_texts,
    get_client,
    load_corpus,
    save_embeddings,
)

BATCH = 2048


def main() -> None:
    client = get_client()
    df = load_corpus()
    print(f"Corpus: {len(df)} docs, {df[TENANT_FIELD].nunique()} tenants")

    if client.collection_exists(COLLECTION):
        cnt = client.count(COLLECTION, exact=True).count
        if cnt == len(df):
            print(
                f"Collection '{COLLECTION}' already has {cnt} points — "
                "leaving it untouched. Delete it first if you really want a rebuild."
            )
            return
        print(f"Existing collection has {cnt} points (expected {len(df)}); rebuilding.")
        client.delete_collection(COLLECTION)

    client.create_collection(
        COLLECTION,
        vectors_config={},
        sparse_vectors_config={
            SPARSE_NAME: m.SparseVectorParams(modifier=m.Modifier.IDF),
        },
    )
    client.create_payload_index(
        COLLECTION,
        field_name=TENANT_FIELD,
        field_schema=m.KeywordIndexParams(type=m.KeywordIndexType.KEYWORD, is_tenant=True),
    )
    print("Collection + tenant keyword index created.")

    t0 = time.time()
    texts = df["doc_text"].tolist()
    embs = embed_texts(texts)
    print(f"Embedded {len(embs)} docs in {time.time() - t0:.1f}s")

    t0 = time.time()
    buf: list[m.PointStruct] = []
    for i, (ind, val) in enumerate(embs):
        buf.append(
            m.PointStruct(
                id=i,
                payload={TENANT_FIELD: df[TENANT_FIELD].iat[i], "title": df["title"].iat[i]},
                vector={SPARSE_NAME: m.SparseVector(indices=ind.tolist(), values=val.tolist())},
            )
        )
        if len(buf) == BATCH:
            client.upsert(COLLECTION, points=buf, wait=False)
            buf.clear()
            done = i + 1
            if done % (BATCH * 10) == 0:
                print(f"  upserted {done}/{len(embs)} ({time.time() - t0:.0f}s)")
    if buf:
        client.upsert(COLLECTION, points=buf, wait=True)
    print(f"Upsert done in {time.time() - t0:.1f}s")

    # wait until all points are indexed
    while True:
        cnt = client.count(COLLECTION, exact=True).count
        info = client.get_collection(COLLECTION)
        status = info.status
        print(f"  count={cnt} status={status}")
        if cnt == len(df) and status == m.CollectionStatus.GREEN:
            break
        time.sleep(2)

    save_embeddings(df, embs)
    print(f"Saved embeddings to artifacts/ for ground-truth computation.")


if __name__ == "__main__":
    main()
