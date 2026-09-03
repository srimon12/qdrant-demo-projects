from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from fastembed import LateInteractionTextEmbedding, TextEmbedding

from .config import (
    COLBERT_EMBEDDING_MODEL,
    DENSE_EMBEDDING_MODEL,
    DENSE_VECTOR_DIM,
)


def onnx_providers() -> list:
    """CUDA when the GPU runtime is installed, otherwise CPU. Safe for laptops."""
    cuda = (
        "CUDAExecutionProvider",
        {
            "arena_extend_strategy": "kSameAsRequested",
            "cudnn_conv_algo_search": "HEURISTIC",
        },
    )
    try:
        import onnxruntime as ort
    except ImportError:
        return ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers():
        return [cuda, "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class DensePrecedentEmbedder:
    """384-dimensional dense text embedder for Qdrant binary-quantized prefetch."""

    def __init__(self, model_identifier: str = DENSE_EMBEDDING_MODEL):
        self.embedding_model = TextEmbedding(
            model_name=model_identifier, providers=onnx_providers()
        )

    def embed_legal_queries(self, legal_queries: str | Iterable[str]) -> np.ndarray:
        if isinstance(legal_queries, str):
            legal_queries = [legal_queries]
        return np.asarray(
            list(self.embedding_model.query_embed(legal_queries)), dtype=np.float32
        )

    def embed_precedent_passages(
        self, precedent_passages: Iterable[str], batch_size: int = 16
    ) -> np.ndarray:
        # Cap passage length to fit standard transformer context window
        truncated_passages = [passage[:1500] for passage in precedent_passages]
        passage_vectors = list(
            self.embedding_model.embed(truncated_passages, batch_size=batch_size)
        )
        if not passage_vectors:
            return np.empty((0, DENSE_VECTOR_DIM), dtype=np.float32)
        return np.asarray(passage_vectors, dtype=np.float32)


class ColbertLateInteractionEmbedder:
    """ColBERTv2 multi-vector token embedder for Qdrant MAX_SIM late interaction."""

    def __init__(self, model_identifier: str = COLBERT_EMBEDDING_MODEL):
        self.embedding_model = LateInteractionTextEmbedding(
            model_name=model_identifier, providers=onnx_providers()
        )

    def embed_legal_queries(
        self, legal_queries: str | Iterable[str]
    ) -> list[np.ndarray]:
        if isinstance(legal_queries, str):
            legal_queries = [legal_queries]
        return [
            np.asarray(token_matrix, dtype=np.float32)
            for token_matrix in self.embedding_model.query_embed(legal_queries)
        ]

    def embed_precedent_passages(
        self, precedent_passages: Iterable[str], batch_size: int = 8
    ) -> list[np.ndarray]:
        # Cap passage length to fit standard transformer context window
        truncated_passages = [passage[:1500] for passage in precedent_passages]
        return [
            np.asarray(token_matrix, dtype=np.float32)
            for token_matrix in self.embedding_model.embed(
                truncated_passages, batch_size=batch_size
            )
        ]
