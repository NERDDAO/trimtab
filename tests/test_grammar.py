"""Tests for grammar loading and traversal."""
import json
import pytest
from pathlib import Path
from smartgrammar.grammar import Grammar


@pytest.fixture
def sample_grammar(tmp_path):
    rules = {
        "origin": ["#greeting# #subject#."],
        "greeting": ["Hello", "Hi", "Hey"],
        "subject": ["world", "there", "friend"],
    }
    path = tmp_path / "test.json"
    path.write_text(json.dumps(rules))
    return path


def test_from_file(sample_grammar):
    g = Grammar.from_file(sample_grammar)
    assert "origin" in g.rule_names()
    assert len(g.get_expansions("greeting")) == 3


def test_from_dict():
    g = Grammar.from_dict({"a": ["x", "y"]})
    assert g.get_expansions("a") == ["x", "y"]


def test_save_roundtrip(tmp_path, sample_grammar):
    g = Grammar.from_file(sample_grammar)
    out = tmp_path / "out.json"
    g.save(out)
    g2 = Grammar.from_file(out)
    assert g.rules == g2.rules


def test_add_expansion():
    g = Grammar.from_dict({"a": ["x"]})
    g.add_expansion("a", "y")
    assert g.get_expansions("a") == ["x", "y"]


def test_add_no_duplicate():
    g = Grammar.from_dict({"a": ["x"]})
    g.add_expansion("a", "x")
    assert g.get_expansions("a") == ["x"]


def test_extract_refs():
    assert Grammar.extract_refs("#sound# from #direction#") == ["sound", "direction"]
    assert Grammar.extract_refs("no refs here") == []


def test_is_terminal():
    assert Grammar.is_terminal("plain text")
    assert not Grammar.is_terminal("#has# refs")
