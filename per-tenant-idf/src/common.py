"""Shared config & helpers for the per-tenant IDF benchmark (full-scale edition)."""

from __future__ import annotations

import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client import models as m
from scipy import sparse

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
ART_DIR = ROOT / "artifacts"
ART_DIR.mkdir(exist_ok=True)

COLLECTION = "cqadupstack_bm25"
SPARSE_NAME = "bm25"
TENANT_FIELD = "forum_tag"
QDRANT_URL = "http://localhost:6333"

EMB_PICKLE = ART_DIR / "doc_embeddings.pkl"

# Alphabetical — this is the order the live collection was upserted in
# (point 0 = first android doc, point 457198 = last wordpress doc).
FORUMS = [
    "android", "english", "gaming", "gis", "mathematica", "physics",
    "programmers", "stats", "tex", "unix", "webmasters", "wordpress",
]


def get_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, timeout=300)


def load_corpus() -> pd.DataFrame:
    """Full BeIR cqadupstack corpus (~457k docs, 12 forums = tenants)."""
    frames = []
    for forum in FORUMS:
        df = pd.read_parquet(DATA_DIR / f"beir_{forum}_corpus.parquet")
        df[TENANT_FIELD] = forum
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")
    df = df[(df["text"].str.len() + df["title"].str.len()) > 0].reset_index(drop=True)
    df["doc_text"] = (df["title"] + "\n" + df["text"]).str.slice(0, 4000)
    return df


def load_queries() -> pd.DataFrame:
    """Held-out BeIR duplicate-question queries per forum."""
    frames = []
    for forum in FORUMS:
        df = pd.read_parquet(DATA_DIR / f"beir_{forum}_queries.parquet")
        df[TENANT_FIELD] = forum
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["title"] = df["title"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    df["query_text"] = (df["title"] + " " + df["text"]).str.strip().str.slice(0, 2000)
    # realistic search-box input: prefer the short title, fall back to full text
    df["q"] = np.where(df["title"].str.strip().str.len() >= 8, df["title"], df["query_text"])
    return df


_EMBEDDER: SparseTextEmbedding | None = None


def get_embedder() -> SparseTextEmbedding:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SparseTextEmbedding(model_name="Qdrant/bm25")
    return _EMBEDDER


def embed_texts(texts: list[str]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return list of (indices int32[], values float32[]) sparse embeddings."""
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for emb in get_embedder().embed(texts, batch_size=1024):
        out.append((emb.indices.astype(np.int32), emb.values.astype(np.float32)))
    return out


# --------------------------------------------------------------------------
# Ground-truth machinery: exact BM25 scoring over one tenant's corpus using
# tenant-local statistics, computed with a scipy CSR matrix (vectorized).
# --------------------------------------------------------------------------


def okapi_idf(n: int, df: int | np.ndarray) -> float | np.ndarray:
    return np.log(1.0 + (n - df + 0.5) / (df + 0.5))


class TenantIndex:
    """CSR doc-term matrix over ONE tenant's stored sparse vectors."""

    def __init__(self, name: str):
        self.name = name
        self.doc_ids: np.ndarray | None = None  # point ids aligned with rows
        self.matrix: sparse.csr_matrix | None = None  # (n_docs x n_terms)
        self.term_cols: dict[int, int] | None = None
        self.df_arr: np.ndarray | None = None
        self.idf_arr: np.ndarray | None = None
        self._csc = None
        self.n_docs = 0

    @classmethod
    def build(cls, name: str, items: list[tuple[int, np.ndarray, np.ndarray]]) -> "TenantIndex":
        """items: (qdrant_point_id, indices, values)"""
        idx = cls(name)
        idx.doc_ids = np.array([p for p, _, _ in items], dtype=np.int64)
        idx.n_docs = len(items)

        cols_by_row: list[np.ndarray] = []
        vals_by_row: list[np.ndarray] = []
        term_cols: dict[int, int] = {}
        for _, ind, val in items:
            local = np.fromiter(
                (term_cols.setdefault(int(t), len(term_cols)) for t in ind.tolist()),
                dtype=np.int32,
                count=len(ind),
            )
            cols_by_row.append(local)
            vals_by_row.append(val.astype(np.float64))

        indptr = np.zeros(len(items) + 1, dtype=np.int64)
        np.cumsum([len(c) for c in cols_by_row], out=indptr[1:])
        indices = np.concatenate(cols_by_row) if cols_by_row else np.array([], dtype=np.int32)
        data = np.concatenate(vals_by_row) if vals_by_row else np.array([], dtype=np.float64)
        idx.matrix = sparse.csr_matrix(
            (data, indices.astype(np.int32), indptr), shape=(idx.n_docs, len(term_cols))
        )
        idx.term_cols = term_cols
        csc = idx.matrix.tocsc()
        idx.df_arr = csc.getnnz(axis=0).astype(np.float64)
        idx.idf_arr = okapi_idf(idx.n_docs, idx.df_arr)
        idx._csc = csc
        return idx

    def df(self, term: int) -> int:
        col = self.term_cols.get(int(term))
        return -1 if col is None else int(self.df_arr[col])

    def probe_doc(self, term: int) -> tuple[int, float]:
        """(point_id, stored TF weight) of the strongest posting for `term`."""
        col = self.term_cols[int(term)]
        post = self._csc.getcol(col)
        j = int(np.argmax(post.data))
        return int(self.doc_ids[int(post.indices[j])]), float(post.data[j])

    def bm25_scores(self, q_ind: np.ndarray, q_val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Exact BM25 dot-product scoring of all docs containing >=1 query term.

        Returns (row_indices, scores), sorted by score descending.
        """
        cols, weights = [], []
        for t, qw in zip(q_ind.tolist(), q_val.tolist()):
            col = self.term_cols.get(int(t))
            if col is not None:
                cols.append(col)
                weights.append(self.idf_arr[col] * qw)
        if not cols:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
        q = sparse.csr_matrix(
            (np.array(weights), (np.zeros(len(cols)), np.array(cols))),
            shape=(1, self.matrix.shape[1]),
        )
        scores = (q @ self.matrix.T).toarray().ravel()
        nz = np.nonzero(scores)[0]
        order = nz[np.argsort(-scores[nz], kind="stable")]
        return order, scores[order]


def ndcg_at_k(ranked_ids: np.ndarray, ideal_scores: np.ndarray, k: int = 10) -> float:
    """NDCG@k where ranked_ids indexes into ideal_scores (aligned arrays)."""
    dcg = 0.0
    for i, r in enumerate(ranked_ids[:k]):
        dcg += ideal_scores[r] / math.log2(i + 2)
    ideal = np.sort(ideal_scores)[::-1][:k]
    idcg = sum(s / math.log2(i + 2) for i, s in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def save_embeddings(df: pd.DataFrame, embs: list[tuple[np.ndarray, np.ndarray]]) -> None:
    with open(EMB_PICKLE, "wb") as f:
        pickle.dump({"point_ids": df.index.to_numpy(), "forums": df[TENANT_FIELD].to_numpy(), "embs": embs}, f)


def load_embeddings() -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    with open(EMB_PICKLE, "rb") as f:
        d = pickle.load(f)
    return d["point_ids"], d["forums"], d["embs"]


def tenant_filter(forum: str) -> m.Filter:
    return m.Filter(
        must=[m.FieldCondition(key=TENANT_FIELD, match=m.MatchValue(value=forum))]
    )


def idf_search_params(forum: str | None) -> m.SearchParams | None:
    """None → shard-wide (global) IDF; forum → params.idf.corpus scoped to that tenant."""
    if forum is None:
        return None
    return m.SearchParams(idf=m.IdfCorpusParams(corpus=tenant_filter(forum)))


def qdrant_search(
    client: QdrantClient,
    q_ind: np.ndarray,
    q_val: np.ndarray,
    forum: str,
    scoped: bool,
    limit: int = 10,
    with_payload: bool = False,
) -> list[tuple[int, float, dict | None]]:
    res = client.query_points(
        COLLECTION,
        query=m.SparseVector(indices=q_ind.tolist(), values=q_val.tolist()),
        using=SPARSE_NAME,
        query_filter=tenant_filter(forum),
        search_params=idf_search_params(forum if scoped else None),
        limit=limit,
        with_payload=with_payload,
    )
    out = []
    for p in res.points:
        out.append((int(p.id), float(p.score), p.payload if with_payload else None))
    return out


def sample_per_forum(df: pd.DataFrame, n: int, seed: int = 7) -> pd.DataFrame:
    parts = []
    for _, g in df.groupby(TENANT_FIELD, sort=False):
        parts.append(g.sample(n=min(n, len(g)), random_state=seed))
    return pd.concat(parts, ignore_index=True)


def load_id_maps() -> tuple[dict[str, dict[str, int]], dict[int, str]]:
    """forum → {beir_id → point_id} and point_id → beir_id, matching the live index order."""
    beir_to_point: dict[str, dict[str, int]] = {f: {} for f in FORUMS}
    point_to_beir: dict[int, str] = {}
    pid = 0
    for forum in FORUMS:
        ids = pd.read_parquet(
            DATA_DIR / f"beir_{forum}_corpus.parquet", columns=["_id"]
        )["_id"].astype(str)
        for bid in ids:
            beir_to_point[forum][bid] = pid
            point_to_beir[pid] = bid
            pid += 1
    return beir_to_point, point_to_beir


def _materialize_qrels() -> None:
    """Split BeIR/cqadupstack-qrels into per-forum files.

    Numeric ids collide across the 12 StackExchange sites. The HF dump is the
    12 official TREC qrels files concatenated in alphabetical forum order, so
    we walk that order and assign each row to the current forum.
    """
    from datasets import load_dataset

    qids = {
        f: set(
            pd.read_parquet(DATA_DIR / f"beir_{f}_queries.parquet", columns=["_id"])["_id"].astype(str)
        )
        for f in FORUMS
    }
    cids = {
        f: set(
            pd.read_parquet(DATA_DIR / f"beir_{f}_corpus.parquet", columns=["_id"])["_id"].astype(str)
        )
        for f in FORUMS
    }
    raw = load_dataset("BeIR/cqadupstack-qrels")["test"].to_pandas()
    raw["query-id"] = raw["query-id"].astype(str)
    raw["corpus-id"] = raw["corpus-id"].astype(str)

    buckets: dict[str, list[tuple[str, str, int]]] = {f: [] for f in FORUMS}
    fi = 0
    for qid, cid, score in raw.itertuples(index=False):
        while fi < len(FORUMS) and qid not in qids[FORUMS[fi]]:
            fi += 1
        if fi >= len(FORUMS) or cid not in cids[FORUMS[fi]]:
            raise RuntimeError(f"qrel ({qid}, {cid}) did not land in any forum")
        buckets[FORUMS[fi]].append((qid, cid, int(score)))

    for forum, rows in buckets.items():
        pd.DataFrame(rows, columns=["query-id", "corpus-id", "score"]).to_parquet(
            DATA_DIR / f"beir_{forum}_qrels.parquet"
        )


def load_qrels() -> dict[str, dict[str, dict[str, int]]]:
    """forum → query_id → {corpus_id: relevance}."""
    if not all((DATA_DIR / f"beir_{f}_qrels.parquet").exists() for f in FORUMS):
        _materialize_qrels()
    out: dict[str, dict[str, dict[str, int]]] = {}
    for forum in FORUMS:
        df = pd.read_parquet(DATA_DIR / f"beir_{forum}_qrels.parquet")
        d: dict[str, dict[str, int]] = {}
        for qid, cid, score in df.itertuples(index=False):
            d.setdefault(str(qid), {})[str(cid)] = int(score)
        out[forum] = d
    return out


def ndcg_labeled(ranked: list[str], rel: dict[str, int], k: int = 10) -> float:
    """Binary/graded nDCG@k against TREC-style qrels (BEIR uses gain = 2^rel - 1)."""
    dcg = 0.0
    for i, doc in enumerate(ranked[:k]):
        r = rel.get(doc, 0)
        if r:
            dcg += (2**r - 1) / math.log2(i + 2)
    gains = sorted(rel.values(), reverse=True)[:k]
    idcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(gains) if r)
    return dcg / idcg if idcg > 0 else 0.0


def average_precision(ranked: list[str], rel: dict[str, int]) -> float:
    n_rel = sum(1 for v in rel.values() if v > 0)
    if n_rel == 0:
        return 0.0
    hit = 0
    s = 0.0
    for i, doc in enumerate(ranked, 1):
        if rel.get(doc, 0) > 0:
            hit += 1
            s += hit / i
    return s / n_rel


def recall_at_k(ranked: list[str], rel: dict[str, int], k: int) -> float:
    n_rel = sum(1 for v in rel.values() if v > 0)
    if n_rel == 0:
        return 0.0
    return sum(1 for d in ranked[:k] if rel.get(d, 0) > 0) / n_rel


def mrr_score(ranked: list[str], rel: dict[str, int]) -> float:
    for i, doc in enumerate(ranked, 1):
        if rel.get(doc, 0) > 0:
            return 1.0 / i
    return 0.0
