from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "nyayarag_legal_precedents")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

DENSE_EMBEDDING_MODEL = os.getenv("DENSE_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
COLBERT_EMBEDDING_MODEL = os.getenv("COLBERT_EMBEDDING_MODEL", "colbert-ir/colbertv2.0")
BM25_SPARSE_MODEL = "Qdrant/bm25"

DENSE_VECTOR_DIM = 384
COLBERT_VECTOR_DIM = 128

DEFAULT_NYAYARAG_DATA_FILE = (
    BASE_DIR / "data" / "nyayarag" / "nyayarag_single_cited_precedents.json"
)
