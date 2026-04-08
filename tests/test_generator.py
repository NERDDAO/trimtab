"""Tests for cascading context-aware generation."""

import pytest
from trimtab.grammar import Grammar
from trimtab.generator import Generator


@pytest.fixture
def gen(mem_db, fake_embedder):
    grammar = Grammar.from_dict({
        "origin": ["#mood# and #detail#."],
        "mood": ["dark and cold", "bright and warm", "eerie and still"],
        "detail": ["shadows move", "birds sing", "silence reigns"],
    })
    mem_db.upsert_grammar("test", grammar, fake_embedder)
    return Generator(mem_db, "test", fake_embedder)


def test_generate_produces_text(gen):
    text = gen.generate(context="spooky cave", temperature=0.0, seed=42)
    assert isinstance(text, str)
    assert len(text) > 5
    assert text.endswith(".")


def test_generate_deterministic(gen):
    t1 = gen.generate(context="forest", temperature=0.0, seed=42)
    t2 = gen.generate(context="forest", temperature=0.0, seed=42)
    assert t1 == t2


def test_generate_temperature_1_is_random(gen):
    results = set()
    for seed in range(20):
        text = gen.generate(context="anything", temperature=1.0, seed=seed)
        results.add(text)
    assert len(results) > 1


def test_generate_no_rule_refs_in_output(gen):
    text = gen.generate(context="test", temperature=0.3, seed=42)
    assert "#" not in text


def test_generate_empty_context(gen):
    text = gen.generate(context="", temperature=1.0, seed=42)
    assert isinstance(text, str)
    assert len(text) > 0
