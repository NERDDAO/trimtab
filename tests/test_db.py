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
    assert all(
        isinstance(r[0], str) and isinstance(r[1], float) and isinstance(r[2], str)
        for r in results
    )


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


def test_add_expansion_with_custom_id(mem_db, fake_embedder):
    grammar = Grammar.from_dict({"origin": ["#a#"], "a": ["x"]})
    mem_db.upsert_grammar("test", grammar, fake_embedder)

    vec = fake_embedder.embed(["The Dark Crypt"])
    mem_db.add_expansion("test", "a", "The Dark Crypt", vec[0], id="entity-uuid-123")

    results = mem_db.query("test", "a", vec[0], top_k=1)
    assert len(results) > 0
    text, score, exp_id = results[0]
    assert text == "The Dark Crypt"
    assert exp_id == "entity-uuid-123"


def test_query_returns_auto_generated_id_by_default(mem_db, fake_embedder):
    grammar = Grammar.from_dict({"origin": ["#a#"], "a": ["hello"]})
    mem_db.upsert_grammar("test", grammar, fake_embedder)

    vec = fake_embedder.embed(["greeting"])[0]
    results = mem_db.query("test", "a", vec, top_k=1)
    assert len(results) > 0
    text, score, exp_id = results[0]
    assert exp_id.startswith("test:a:")


# ---------------------------------------------------------------------------
# register_grammar tests
# ---------------------------------------------------------------------------


class _FakeEmbedder:
    """Embedder double that returns fixed-dim zero vectors."""

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.create_calls: list[str] = []

    async def create(self, text: str) -> list[float]:
        self.create_calls.append(text)
        return [0.0] * self.dim

    async def create_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.create(t) for t in texts]


@pytest.mark.asyncio
async def test_register_grammar_creates_structure_without_embedding():
    """register_grammar creates Grammar + Rule nodes + HAS_RULE edges but
    does not embed any expansions. The only embedder call is the one
    dimension probe."""
    from trimtab.db import TrimTabDB

    db = TrimTabDB(":memory:")
    embedder = _FakeEmbedder(dim=4)
    grammar = Grammar.from_dict({
        "origin": ["#greeting# #name#"],
        "greeting": ["hello", "hi"],
        "name": ["world"],
    })

    await db.register_grammar("test_grammar", grammar, embedder)

    # Only the dimension probe should have been embedded
    assert embedder.create_calls == ["dimension probe"]

    # Grammar + Rule nodes exist
    assert "test_grammar" in db.list_grammars()

    # No expansions yet
    assert db.get_expansions("test_grammar", "greeting") == []
    assert db.get_expansions("test_grammar", "name") == []


@pytest.mark.asyncio
async def test_register_grammar_idempotent_second_call_no_probe():
    """Second call on same DB doesn't re-probe embedding dim."""
    from trimtab.db import TrimTabDB

    db = TrimTabDB(":memory:")
    embedder = _FakeEmbedder(dim=4)
    grammar = Grammar.from_dict({"origin": ["hello"]})

    await db.register_grammar("g1", grammar, embedder)
    embedder.create_calls.clear()

    await db.register_grammar("g2", grammar, embedder)
    # Dim already known, no probe needed
    assert embedder.create_calls == []


@pytest.mark.asyncio
async def test_register_grammar_then_add_expansion_works():
    """After register_grammar, add_expansion with precomputed vector works
    and query finds the result."""
    from trimtab.db import TrimTabDB

    db = TrimTabDB(":memory:")
    embedder = _FakeEmbedder(dim=4)
    grammar = Grammar.from_dict({"origin": []})

    await db.register_grammar("g", grammar, embedder)
    db.add_expansion("g", "origin", "hello world", [1.0, 0.0, 0.0, 0.0])

    results = db.query("g", "origin", [1.0, 0.0, 0.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0][0] == "hello world"
