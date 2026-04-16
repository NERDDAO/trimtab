"""Tests for TrimTab (the new public façade) — sync methods only.

Async methods (put, put_many, update, search, generate, load_file, export_file)
are added in later tasks and have their own test files.
"""

from __future__ import annotations

import pytest

from trimtab.core import TrimTab
from trimtab.grammar import Rule


@pytest.fixture
def tt(stub_embedder):
    """A fresh TrimTab backed by an in-memory DB and the stub embedder."""
    return TrimTab(path=":memory:", embedder=stub_embedder)


def test_construction_accepts_path_and_embedder(tt):
    assert tt is not None


def test_list_empty_grammar_returns_empty(tt):
    assert tt.list(grammar="none", symbol="friends") == []


def test_list_grammars_initially_empty(tt):
    assert tt.list_grammars() == []


def test_list_symbols_empty(tt):
    assert tt.list_symbols(grammar="none") == []


def test_count_empty(tt):
    assert tt.count(grammar="none", symbol="friends") == 0


def test_list_returns_rules_after_db_seed(tt, stub_embedder):
    """Verify the sync read methods see data put through the internal API.

    Async TrimTab.put isn't implemented yet (Task 12), so we seed via the
    underlying TrimTabDB internal method directly — the sync READ surface
    still has to work.
    """
    import asyncio
    vec = asyncio.get_event_loop().run_until_complete(stub_embedder.create("Alice"))
    tt._db._put_rule_with_vector(
        grammar="g1",
        symbol="friends",
        rule=Rule(text="Alice", metadata={"rank": 1}),
        vector=vec,
    )
    rules = tt.list("g1", "friends")
    assert len(rules) == 1
    assert rules[0].text == "Alice"
    assert rules[0].metadata == {"rank": 1}

    assert tt.list_grammars() == ["g1"]
    assert tt.list_symbols("g1") == ["friends"]
    assert tt.count("g1", "friends") == 1


def test_remove_via_facade(tt, stub_embedder):
    import asyncio
    vec = asyncio.get_event_loop().run_until_complete(stub_embedder.create("Alice"))
    rule = Rule(text="Alice", id="r_a")
    tt._db._put_rule_with_vector("g1", "friends", rule, vec)
    tt.remove("g1", "friends", "r_a")
    assert tt.list("g1", "friends") == []


def test_clear_via_facade(tt, stub_embedder):
    import asyncio
    loop = asyncio.get_event_loop()
    vec1 = loop.run_until_complete(stub_embedder.create("Alice"))
    vec2 = loop.run_until_complete(stub_embedder.create("Bob"))
    tt._db._put_rule_with_vector("g1", "friends", Rule(text="Alice"), vec1)
    tt._db._put_rule_with_vector("g1", "friends", Rule(text="Bob"), vec2)
    tt.clear("g1", "friends")
    assert tt.list("g1", "friends") == []
    # Symbol still exists.
    assert "friends" in tt.list_symbols("g1")


def test_drop_via_facade(tt, stub_embedder):
    import asyncio
    loop = asyncio.get_event_loop()
    vec = loop.run_until_complete(stub_embedder.create("Alice"))
    tt._db._put_rule_with_vector("g1", "friends", Rule(text="Alice"), vec)
    tt._db._put_rule_with_vector("g1", "quests", Rule(text="Q"), vec)
    tt.drop("g1")
    assert tt.list_symbols("g1") == []
    assert tt.list_grammars() == []


@pytest.mark.asyncio
async def test_put_embeds_and_stores(tt):
    r = await tt.put(
        grammar="g1", symbol="friends",
        text="Alice — party wizard",
        metadata={"rank": 1},
    )
    assert isinstance(r, Rule)
    assert r.text == "Alice — party wizard"
    assert r.metadata == {"rank": 1}
    assert r.id  # auto-assigned

    listed = tt.list("g1", "friends")
    assert len(listed) == 1
    assert listed[0].text == "Alice — party wizard"


@pytest.mark.asyncio
async def test_put_accepts_caller_id(tt):
    r = await tt.put("g1", "friends", "Bob", id="ent_bob")
    assert r.id == "ent_bob"


@pytest.mark.asyncio
async def test_put_many_batches_one_embed_call(tt, stub_embedder):
    before = stub_embedder.batch_call_count
    rules = await tt.put_many(
        grammar="g1", symbol="notes",
        entries=[
            {"text": "first note"},
            {"text": "second note", "metadata": {"source": "overheard"}},
        ],
    )
    assert stub_embedder.batch_call_count == before + 1
    assert len(rules) == 2
    listed = tt.list("g1", "notes")
    assert len(listed) == 2
    # Metadata round-trip on the second entry.
    by_text = {r.text: r for r in listed}
    assert by_text["second note"].metadata == {"source": "overheard"}


@pytest.mark.asyncio
async def test_put_many_empty_returns_empty(tt, stub_embedder):
    before = stub_embedder.batch_call_count
    rules = await tt.put_many(grammar="g1", symbol="notes", entries=[])
    assert rules == []
    # No embed call for empty input.
    assert stub_embedder.batch_call_count == before


@pytest.mark.asyncio
async def test_update_metadata_does_not_reembed(tt, stub_embedder):
    r = await tt.put("g1", "friends", "Alice")
    before = stub_embedder.call_count
    await tt.update("g1", "friends", r.id, metadata={"rank": 99})
    # No new single-embed call (text unchanged).
    assert stub_embedder.call_count == before
    listed = tt.list("g1", "friends")
    assert listed[0].metadata == {"rank": 99}


@pytest.mark.asyncio
async def test_update_text_reembeds(tt, stub_embedder):
    r = await tt.put("g1", "friends", "Alice")
    before = stub_embedder.call_count
    await tt.update("g1", "friends", r.id, text="Alice the Wizard")
    assert stub_embedder.call_count == before + 1
    listed = tt.list("g1", "friends")
    assert listed[0].text == "Alice the Wizard"


@pytest.mark.asyncio
async def test_update_unknown_id_raises(tt):
    from trimtab.errors import TrimTabNotFoundError
    with pytest.raises(TrimTabNotFoundError):
        await tt.update("g1", "friends", "nope", text="x")
