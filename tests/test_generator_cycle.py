"""Tests that generator raises TrimTabCycleError on cyclic grammars."""

from __future__ import annotations

import pytest

from trimtab.db import TrimTabDB
from trimtab.errors import TrimTabCycleError
from trimtab.generator import Generator
from trimtab.grammar import Rule


@pytest.fixture
def db_with_cycle():
    db = TrimTabDB(":memory:")
    # origin → #origin# forms a trivial cycle.
    db._put_rule_with_vector(
        grammar="cyc", symbol="origin",
        rule=Rule(text="#origin#"), vector=[0.1] * 16,
    )
    return db


@pytest.mark.asyncio
async def test_self_cycle_raises(db_with_cycle, stub_embedder):
    gen = Generator(db=db_with_cycle, grammar="cyc", embedder=stub_embedder)
    with pytest.raises(TrimTabCycleError) as excinfo:
        await gen.generate(context="anything", origin="origin")
    assert "origin" in str(excinfo.value)


@pytest.mark.asyncio
async def test_mutual_cycle_raises(stub_embedder):
    db = TrimTabDB(":memory:")
    db._put_rule_with_vector("cyc2", "a", Rule(text="#b#"), [0.1] * 16)
    db._put_rule_with_vector("cyc2", "b", Rule(text="#a#"), [0.2] * 16)
    gen = Generator(db=db, grammar="cyc2", embedder=stub_embedder)
    with pytest.raises(TrimTabCycleError) as excinfo:
        await gen.generate(context="anything", origin="a")
    msg = str(excinfo.value)
    assert "a" in msg and "b" in msg


@pytest.mark.asyncio
async def test_acyclic_nesting_still_works(stub_embedder):
    db = TrimTabDB(":memory:")
    db._put_rule_with_vector("ok", "origin", Rule(text="hello #name#"), [0.1] * 16)
    db._put_rule_with_vector("ok", "name", Rule(text="world"), [0.2] * 16)
    gen = Generator(db=db, grammar="ok", embedder=stub_embedder)
    result = await gen.generate(context="anything", origin="origin")
    assert result.text == "hello world"


@pytest.mark.asyncio
async def test_missing_symbol_returns_bracket_placeholder(stub_embedder):
    """Empty/missing symbols should fall through to a placeholder, not crash."""
    db = TrimTabDB(":memory:")
    db._put_rule_with_vector("g", "origin", Rule(text="#missing#"), [0.1] * 16)
    gen = Generator(db=db, grammar="g", embedder=stub_embedder)
    result = await gen.generate(context="anything", origin="origin")
    # Generator returns "[missing]" for empty symbols (preserves existing behavior).
    assert "[missing]" in result.text
