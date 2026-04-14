"""Tests for TrimTab.generate."""

from __future__ import annotations

import pytest

from trimtab.core import TrimTab


@pytest.fixture
def tt(stub_embedder):
    return TrimTab(path=":memory:", embedder=stub_embedder)


@pytest.mark.asyncio
async def test_generate_flat_grammar(tt):
    await tt.put("g1", "origin", "hello world")
    result = await tt.generate("g1", context="greeting", origin="origin")
    assert result.text == "hello world"


@pytest.mark.asyncio
async def test_generate_nested_grammar(tt):
    await tt.put("g1", "origin", "Contact: #friends#")
    await tt.put("g1", "friends", "Alice — party wizard")
    result = await tt.generate("g1", context="magic help", origin="origin")
    assert "Alice" in result.text


@pytest.mark.asyncio
async def test_generate_returns_rules_used(tt):
    await tt.put("g1", "origin", "#notes#")
    await tt.put("g1", "notes", "a note")
    result = await tt.generate("g1", context="anything", origin="origin")
    assert len(result.rules_used) >= 1
    assert any(r.text == "a note" for r in result.rules_used)


@pytest.mark.asyncio
async def test_generate_cycle_raises(tt):
    from trimtab.errors import TrimTabCycleError
    await tt.put("g1", "origin", "#origin#")
    with pytest.raises(TrimTabCycleError):
        await tt.generate("g1", context="x", origin="origin")
