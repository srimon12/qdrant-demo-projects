"""BGE-small via FastEmbed. CUDA if the GPU runtime is installed, otherwise CPU."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("geosmart.embedder")


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


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(arr, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return (arr / n).astype(np.float32)


class DenseEmbedder:
    """BGE-small-en-v1.5 384-d via FastEmbed."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from fastembed import TextEmbedding

        providers = onnx_providers()
        self.model_name = model_name
        self.model = model_name
        self.backend = (
            "fastembed-gpu" if providers[0] != "CPUExecutionProvider" else "fastembed"
        )
        self._model = TextEmbedding(model_name=model_name, providers=providers)
        logger.info("Dense embedder ready: %s (%s)", model_name, self.backend)

    def embed_queries(self, texts: str | list[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        return _l2_normalize(
            np.array(list(self._model.query_embed(texts)), dtype=np.float32)
        )

    def embed_passages(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return _l2_normalize(
            np.array(
                list(self._model.embed(texts, batch_size=batch_size)), dtype=np.float32
            )
        )
