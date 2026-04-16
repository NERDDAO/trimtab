"""Tests for the Rule dataclass and Grammar auto-upgrade."""

from __future__ import annotations

import json
from pathlib import Path

from trimtab.grammar import Grammar, Rule, upgrade_entry


def test_upgrade_plain_string_to_rule():
    r = upgrade_entry("Alice — party wizard")
    assert isinstance(r, Rule)
    assert r.text == "Alice — party wizard"
    assert r.metadata == {}
    assert r.id  # auto-assigned uuid
    assert r.created_at is not None
    assert r.updated_at == r.created_at


def test_upgrade_dict_with_text_and_id():
    r = upgrade_entry({"text": "Bob", "id": "ent_bob"})
    assert r.text == "Bob"
    assert r.id == "ent_bob"
    assert r.metadata == {}


def test_upgrade_dict_with_metadata():
    r = upgrade_entry({"text": "Carol", "id": "c_1", "metadata": {"rank": 3}})
    assert r.text == "Carol"
    assert r.id == "c_1"
    assert r.metadata == {"rank": 3}


def test_grammar_from_dict_mixed_shapes():
    g = Grammar.from_dict({
        "friends": [
            "Alice",
            {"text": "Bob", "id": "ent_bob"},
            {"text": "Carol", "id": "c_1", "metadata": {"rank": 3}},
        ],
    })
    assert g.rule_names() == ["friends"]
    items = g.get_expansion_items("friends")
    assert len(items) == 3
    assert items[0][0] == "Alice"
    assert items[1][0] == "Bob" and items[1][1] == "ent_bob"
    assert items[2][0] == "Carol" and items[2][1] == "c_1"


def test_grammar_get_rule_objects():
    """New method: return list[Rule] with full metadata."""
    g = Grammar.from_dict({
        "friends": [
            {"text": "Carol", "id": "c_1", "metadata": {"rank": 3}},
        ],
    })
    rules = g.get_rules("friends")
    assert len(rules) == 1
    assert rules[0].text == "Carol"
    assert rules[0].metadata == {"rank": 3}


def test_from_file_auto_upgrades_plain_tracery(tmp_path: Path):
    p = tmp_path / "tracery.json"
    p.write_text(json.dumps({"origin": ["hello", "world"]}))
    g = Grammar.from_file(p)
    rules = g.get_rules("origin")
    assert len(rules) == 2
    assert {r.text for r in rules} == {"hello", "world"}
    assert all(r.id for r in rules)


def test_round_trip_preserves_text(tmp_path: Path):
    src = tmp_path / "src.json"
    src.write_text(json.dumps({"origin": ["one", "two"]}))
    g = Grammar.from_file(src)
    dst = tmp_path / "dst.json"
    g.save(dst)
    reloaded = json.loads(dst.read_text())
    assert set(reloaded["origin"]) >= {"one", "two"} or all(
        (e if isinstance(e, str) else e["text"]) in {"one", "two"}
        for e in reloaded["origin"]
    )
