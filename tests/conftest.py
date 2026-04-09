"""Shared fixtures for TrimTab tests."""

import numpy as np
import pytest


class FakeEmbedder:
    """Deterministic async embedder matching trimtab's Embedder Protocol.

    Hash-based, normalized vectors — reproducible across runs so tests that
    assert on embedding similarity are stable.
    """

    def __init__(self, dim: int = 8):
        self._dim = dim

    async def create(self, input_data: str | list[str]) -> list[float]:
        if isinstance(input_data, list):
            input_data = " ".join(input_data)
        rng = np.random.default_rng(hash(input_data) % (2**31))
        vec = rng.standard_normal(self._dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for item in input_data_list:
            out.append(await self.create(item))
        return out


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def mem_db():
    """In-memory TrimTabDB for tests."""
    from trimtab.db import TrimTabDB
    return TrimTabDB(":memory:")
