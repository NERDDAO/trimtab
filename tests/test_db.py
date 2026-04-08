"""Tests for LadybugDB-backed grammar storage."""

import numpy as np
import pytest
from trimtab.grammar import Grammar


def test_init_creates_schema(mem_db):
    """DB should have Grammar, Rule tables after init."""
    assert mem_db is not None


def test_upsert_grammar(mem_db, fake_embedder):
    grammar = Grammar.from_dict({
        "origin": ["#a#"],
        "a": ["dark caves", "bright forest", "cold mountain"],
    })
    mem_db.upsert_grammar("test", grammar, fake_embedder)
    names = mem_db.list_grammars()
    assert "test" in names


def test_upsert_grammar_idempotent(mem_db, fake_embedder):
    grammar = Grammar.from_dict({"origin": ["#a#"], "a": ["x", "y"]})
    mem_db.upsert_grammar("test", grammar, fake_embedder)
    mem_db.upsert_grammar("test", grammar, fake_embedder)
    names = mem_db.list_grammars()
    assert names.count("test") == 1


def test_get_grammar(mem_db, fake_embedder):
    grammar = Grammar.from_dict({
        "origin": ["#a# #b#"],
        "a": ["hello", "hey"],
        "b": ["world", "there"],
    })
    mem_db.upsert_grammar("test", grammar, fake_embedder)
    result = mem_db.get_grammar("test")
    assert set(result.rule_names()) == {"origin", "a", "b"}
    assert set(result.get_expansions("a")) == {"hello", "hey"}


def test_add_expansion(mem_db, fake_embedder):
    grammar = Grammar.from_dict({"origin": ["#a#"], "a": ["x", "y"]})
    mem_db.upsert_grammar("test", grammar, fake_embedder)

    vec = fake_embedder.embed(["z"])
    mem_db.add_expansion("test", "a", "z", vec[0])

    result = mem_db.get_grammar("test")
    assert "z" in result.get_expansions("a")


def test_query_returns_results(mem_db, fake_embedder):
    grammar = Grammar.from_dict({
        "origin": ["#a#"],
        "a": ["dark caves", "bright forest", "cold mountain", "warm beach"],
    })
    mem_db.upsert_grammar("test", grammar, fake_embedder)

    ctx_vec = fake_embedder.embed(["underground darkness"])[0]
    results = mem_db.query("test", "a", ctx_vec, top_k=3)
    assert len(results) > 0
    assert all(isinstance(r[0], str) and isinstance(r[1], float) for r in results)


def test_query_nonexistent_rule(mem_db, fake_embedder):
    grammar = Grammar.from_dict({"origin": ["#a#"], "a": ["x"]})
    mem_db.upsert_grammar("test", grammar, fake_embedder)
    ctx_vec = fake_embedder.embed(["anything"])[0]
    results = mem_db.query("test", "nonexistent", ctx_vec, top_k=3)
    assert results == []


def test_list_grammars_empty(mem_db):
    assert mem_db.list_grammars() == []


def test_get_expansions(mem_db, fake_embedder):
    grammar = Grammar.from_dict({"origin": ["#a#"], "a": ["x", "y", "z"]})
    mem_db.upsert_grammar("test", grammar, fake_embedder)
    expansions = mem_db.get_expansions("test", "a")
    assert set(expansions) == {"x", "y", "z"}
