"""Tests that v0.4 TrimTabDB methods still work (with DeprecationWarning)."""

from __future__ import annotations

import warnings

import pytest

from trimtab.db import TrimTabDB
from trimtab.grammar import Grammar


@pytest.fixture
def db():
    return TrimTabDB(":memory:")


@pytest.mark.asyncio
async def test_upsert_grammar_still_works(db, stub_embedder):
    g = Grammar.from_dict({"notes": ["hello", "world"]})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        await db.upsert_grammar("g1", g, stub_embedder)
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)

    # Data is now accessible via v0.5 read path.
    rules = db._get_rules("g1", "notes")
    assert {r.text for r in rules} == {"hello", "world"}


@pytest.mark.asyncio
async def test_upsert_grammar_with_explicit_ids(db, stub_embedder):
    g = Grammar.from_dict({
        "friends": [
            {"text": "Alice", "id": "ent_alice"},
            {"text": "Bob", "id": "ent_bob"},
        ],
    })
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        await db.upsert_grammar("g1", g, stub_embedder)
    rules = db._get_rules("g1", "friends")
    ids = {r.id for r in rules}
    assert "ent_alice" in ids
    assert "ent_bob" in ids


def test_add_expansion_still_works(db):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        db.add_expansion(
            grammar="g1",
            rule="notes",
            text="hello",
            embedding=[0.1] * 16,
            id="r_a",
        )
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)

    rules = db._get_rules("g1", "notes")
    assert len(rules) == 1
    assert rules[0].text == "hello"
    assert rules[0].id == "r_a"


def test_list_entries_still_returns_text_id_tuples(db):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        db.add_expansion("g1", "notes", "hello", [0.1] * 16, id="r_a")
        db.add_expansion("g1", "notes", "world", [0.2] * 16, id="r_b")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        entries = db.list_entries("g1", "notes")
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert ("hello", "r_a") in entries
    assert ("world", "r_b") in entries


def test_get_expansions_still_returns_text_strings(db):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        db.add_expansion("g1", "notes", "first", [0.1] * 16)
        db.add_expansion("g1", "notes", "second", [0.2] * 16)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        texts = db.get_expansions("g1", "notes")
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert set(texts) == {"first", "second"}


def test_query_still_returns_tuples(db):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        db.add_expansion("g1", "notes", "alpha", [0.9] * 16)
        db.add_expansion("g1", "notes", "beta", [0.1] * 16)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = db.query("g1", "notes", [0.9] * 16, top_k=2)
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert len(results) >= 1
    # Old shape: list of (text, score, id) tuples
    text, score, rid = results[0]
    assert text == "alpha"
    assert isinstance(score, float)
    assert isinstance(rid, str)
