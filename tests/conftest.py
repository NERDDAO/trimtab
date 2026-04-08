"""Shared fixtures for TrimTab tests."""

import pytest
import numpy as np


class FakeEmbedder:
    """Deterministic embedder for testing — hash-based, normalized vectors."""

    def __init__(self, dim: int = 8):
        self._dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = []
        for t in texts:
            np.random.seed(hash(t) % (2**31))
            vecs.append(np.random.randn(self._dim).astype(np.float32))
        arr = np.array(vecs)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / (norms + 1e-8)

    @property
    def dimension(self) -> int:
        return self._dim


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def mem_db():
    """In-memory TrimTabDB for tests."""
    from trimtab.db import TrimTabDB
    return TrimTabDB(":memory:")
