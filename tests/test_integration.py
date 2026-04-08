"""Integration test: full pipeline with LadybugDB."""

import json
import pytest
from trimtab import SmartGrammar
from trimtab.db import TrimTabDB


def test_full_pipeline(mem_db, fake_embedder):
    rules = {
        "origin": ["#sound#. #atmosphere#."],
        "sound": ["A drip", "Wind howls", "Chains rattle"],
        "atmosphere": ["Cold air", "Damp walls", "Silence"],
    }
    sg = SmartGrammar(mem_db, "test", embedder=fake_embedder)
    sg.load_grammar(rules)

    text = sg.generate(context="underground cave", temperature=0.0, seed=42)
    assert isinstance(text, str)
    assert "." in text
    assert "#" not in text


def test_from_file(mem_db, fake_embedder, tmp_path):
    rules = {
        "origin": ["#a# #b#"],
        "a": ["hello", "hey", "hi"],
        "b": ["world", "there", "friend"],
    }
    path = tmp_path / "test.json"
    path.write_text(json.dumps(rules))

    sg = SmartGrammar.from_file(mem_db, str(path), embedder=fake_embedder)
    text = sg.generate(context="greeting", temperature=0.0, seed=42)
    assert isinstance(text, str)
    assert "#" not in text


def test_add_expansion(mem_db, fake_embedder):
    rules = {"origin": ["#a#"], "a": ["x", "y"]}
    sg = SmartGrammar(mem_db, "test", embedder=fake_embedder)
    sg.load_grammar(rules)
    sg.add("a", "z")

    grammar = mem_db.get_grammar("test")
    assert "z" in grammar.get_expansions("a")


def test_persistent_db(tmp_path, fake_embedder):
    db_path = str(tmp_path / "test.db")

    # Write
    db1 = TrimTabDB(db_path)
    sg1 = SmartGrammar(db1, "persist", embedder=fake_embedder)
    sg1.load_grammar({"origin": ["#a#"], "a": ["hello", "world"]})

    # Read in new instance
    db2 = TrimTabDB(db_path)
    sg2 = SmartGrammar(db2, "persist", embedder=fake_embedder)
    text = sg2.generate(context="greeting", temperature=0.0, seed=42)
    assert isinstance(text, str)
    assert text in ["hello", "world"]


def test_multiple_grammars(mem_db, fake_embedder):
    sg1 = SmartGrammar(mem_db, "grammar_a", embedder=fake_embedder)
    sg1.load_grammar({"origin": ["#a#"], "a": ["alpha", "beta"]})

    sg2 = SmartGrammar(mem_db, "grammar_b", embedder=fake_embedder)
    sg2.load_grammar({"origin": ["#a#"], "a": ["gamma", "delta"]})

    assert set(mem_db.list_grammars()) == {"grammar_a", "grammar_b"}

    text_a = sg1.generate(context="test", temperature=0.0, seed=42)
    text_b = sg2.generate(context="test", temperature=0.0, seed=42)
    assert text_a in ["alpha", "beta"]
    assert text_b in ["gamma", "delta"]


def test_smartgrammar_add_with_id(mem_db, fake_embedder):
    sg = SmartGrammar(mem_db, "test", embedder=fake_embedder)
    sg.load_grammar({"origin": ["#place#"], "place": ["The Threshold"]})
    sg.add("place", "The Dark Crypt", id="entity-uuid-123")

    results = mem_db.query("test", "place", fake_embedder.embed(["dark crypt"])[0], top_k=2)
    ids = [r[2] for r in results]
    assert "entity-uuid-123" in ids
