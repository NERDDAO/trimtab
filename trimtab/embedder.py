"""Embedder protocol and adapters for MiniLM and Ollama."""

from __future__ import annotations
import logging
import numpy as np
from typing import Protocol

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    """Protocol for embedding text into vectors."""
    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts. Returns (N, dim) array."""
        ...

    @property
    def dimension(self) -> int: ...


class MiniLMEmbedder:
    """Local embedding using sentence-transformers MiniLM (~80MB model)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        self._dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(texts, normalize_embeddings=True)

    @property
    def dimension(self) -> int:
        return self._dim


class OllamaEmbedder:
    """Embedding via local Ollama server."""

    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        import requests
        self._model = model
        self._base_url = base_url
        self._dim: int | None = None
        # Test connection
        try:
            resp = requests.get(f"{base_url}/api/tags", timeout=2)
            resp.raise_for_status()
        except Exception:
            raise ConnectionError(f"Cannot connect to Ollama at {base_url}")

    def embed(self, texts: list[str]) -> np.ndarray:
        import requests
        results = []
        for text in texts:
            resp = requests.post(
                f"{self._base_url}/api/embed",
                json={"model": self._model, "input": text},
                timeout=30,
            )
            resp.raise_for_status()
            embedding = resp.json()["embeddings"][0]
            results.append(embedding)
        arr = np.array(results, dtype=np.float32)
        if self._dim is None:
            self._dim = arr.shape[1]
        return arr

    @property
    def dimension(self) -> int:
        if self._dim is None:
            # Probe with a dummy embed
            arr = self.embed(["test"])
            self._dim = arr.shape[1]
        return self._dim


def get_default_embedder() -> Embedder:
    """Try Ollama first, fall back to MiniLM."""
    try:
        emb = OllamaEmbedder()
        logger.info("Using Ollama embedder")
        return emb
    except Exception:
        logger.info("Ollama not available, using MiniLM")
        return MiniLMEmbedder()
