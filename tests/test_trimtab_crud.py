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
