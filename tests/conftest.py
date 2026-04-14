"""Shared pytest fixtures for trimtab tests."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest


class StubEmbedder:
    """Deterministic hash-based embedder for hermetic tests.

    Always returns the same vector for the same text. Dimension is fixed
    at construction. No HTTP, no ML deps, no Ollama — safe to use in CI.
    """

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim
        self.call_count = 0
        self.batch_call_count = 0

    def _vector(self, text: str) -> list[float]:
        # SHA256 → first `dim` bytes → normalize to roughly [-1, 1].
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        # Repeat digest if dim > 32 (sha256 is 32 bytes).
        while len(digest) < self.dim:
            digest = digest + hashlib.sha256(digest).digest()
        return [(b - 128) / 128.0 for b in digest[: self.dim]]

    async def create(self, input_data):
        self.call_count += 1
        text = input_data if isinstance(input_data, str) else " ".join(input_data)
        return self._vector(text)

    async def create_batch(self, input_data_list):
        self.batch_call_count += 1
        return [self._vector(t) for t in input_data_list]


@pytest.fixture
def stub_embedder() -> StubEmbedder:
    """A fresh 16-dim stub embedder per test."""
    return StubEmbedder(dim=16)


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
