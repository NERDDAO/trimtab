"""Tests for TrimTab.search."""

from __future__ import annotations

import pytest

from trimtab.core import TrimTab


@pytest.fixture
def tt(stub_embedder):
    return TrimTab(path=":memory:", embedder=stub_embedder)


@pytest.mark.asyncio
async def test_search_returns_rules(tt):
    await tt.put("g1", "notes", "forest at night is dangerous")
    await tt.put("g1", "notes", "baking bread recipe")
    results = await tt.search("g1", "notes", query="forest path", top_k=2)
    assert len(results) >= 1
    # All returned objects are Rule instances with text and metadata fields.
    assert all(hasattr(r, "text") and hasattr(r, "metadata") for r in results)


@pytest.mark.asyncio
async def test_search_missing_grammar_returns_empty(tt):
    # Missing-data semantics per the spec: empty result, not raise.
    results = await tt.search("does-not-exist", "notes", query="x", top_k=5)
    assert results == []


@pytest.mark.asyncio
async def test_search_includes_metadata(tt):
    await tt.put("g1", "friends", "Alice", metadata={"rank": 1, "entityId": "ent_a"})
    results = await tt.search("g1", "friends", query="wizard", top_k=5)
    assert len(results) == 1
    assert results[0].metadata == {"rank": 1, "entityId": "ent_a"}
