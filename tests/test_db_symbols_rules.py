"""Unit tests for TrimTabDB v0.5 schema and put_rule/get_rule."""

from __future__ import annotations

import pytest

from trimtab.db import TrimTabDB
from trimtab.grammar import Rule


@pytest.fixture
def db() -> TrimTabDB:
    return TrimTabDB(":memory:")


def test_schema_creates_symbol_and_rule_tables(db: TrimTabDB):
    # After __init__, the v0.5 Symbol table must exist. Rule may be
    # lazily created on first put.
    res = db._conn.execute("MATCH (s:Symbol) RETURN s.id LIMIT 1")
    # Should not raise.
    res.get_all()


def test_put_rule_round_trip(db: TrimTabDB):
    db._put_rule_with_vector(
        grammar="g1",
        symbol="friends",
        rule=Rule(text="Alice", metadata={"rank": 1}),
        vector=[0.1] * 16,
    )
    got = db._get_rules(grammar="g1", symbol="friends")
    assert len(got) == 1
    assert got[0].text == "Alice"
    assert got[0].metadata == {"rank": 1}


def test_put_rule_creates_symbol_node_lazily(db: TrimTabDB):
    db._put_rule_with_vector(
        grammar="g1",
        symbol="friends",
        rule=Rule(text="Alice"),
        vector=[0.1] * 16,
    )
    syms = db._conn.execute(
        "MATCH (s:Symbol) WHERE s.grammar = 'g1' RETURN s.name"
    ).get_all()
    assert syms[0][0] == "friends"


def test_put_rule_pins_dimension(db: TrimTabDB):
    db._put_rule_with_vector(
        grammar="g1", symbol="friends",
        rule=Rule(text="Alice"), vector=[0.1] * 16,
    )
    assert db._embedding_dim == 16


def test_put_rule_second_call_with_same_dim_works(db: TrimTabDB):
    db._put_rule_with_vector(
        grammar="g1", symbol="friends",
        rule=Rule(text="Alice"), vector=[0.1] * 16,
    )
    db._put_rule_with_vector(
        grammar="g1", symbol="friends",
        rule=Rule(text="Bob"), vector=[0.2] * 16,
    )
    got = db._get_rules(grammar="g1", symbol="friends")
    assert {r.text for r in got} == {"Alice", "Bob"}


def test_put_rule_preserves_insertion_order(db: TrimTabDB):
    import time
    r1 = Rule(text="first")
    db._put_rule_with_vector("g1", "notes", r1, [0.1] * 16)
    time.sleep(0.001)  # ensure distinct created_at timestamps
    r2 = Rule(text="second")
    db._put_rule_with_vector("g1", "notes", r2, [0.2] * 16)
    time.sleep(0.001)
    r3 = Rule(text="third")
    db._put_rule_with_vector("g1", "notes", r3, [0.3] * 16)

    got = db._get_rules("g1", "notes")
    assert [r.text for r in got] == ["first", "second", "third"]
