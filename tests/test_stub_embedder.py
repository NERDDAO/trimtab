"""Self-tests for the StubEmbedder fixture."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_stub_embedder_is_deterministic(stub_embedder):
    v1 = await stub_embedder.create("hello")
    v2 = await stub_embedder.create("hello")
    assert v1 == v2


@pytest.mark.asyncio
async def test_stub_embedder_distinguishes_text(stub_embedder):
    v1 = await stub_embedder.create("hello")
    v2 = await stub_embedder.create("world")
    assert v1 != v2


@pytest.mark.asyncio
async def test_stub_embedder_batch(stub_embedder):
    batch = await stub_embedder.create_batch(["a", "b", "c"])
    assert len(batch) == 3
    assert all(len(v) == 16 for v in batch)
