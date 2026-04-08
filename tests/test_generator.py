"""Tests for cascading context-aware generation."""
import pytest
import numpy as np
from smartgrammar.grammar import Grammar
from smartgrammar.index import GrammarIndex
from smartgrammar.generator import Generator


class FakeEmbedder:
    def __init__(self, dim=8):
        self._dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = []
        for t in texts:
            np.random.seed(hash(t) % (2**31))
            vecs.append(np.random.randn(self._dim).astype(np.float32))
        arr = np.array(vecs)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / (norms + 1e-8)

    @property
    def dimension(self):
        return self._dim


@pytest.fixture
def gen():
    g = Grammar.from_dict({
        "origin": ["#mood# and #detail#."],
        "mood": ["dark and cold", "bright and warm", "eerie and still"],
        "detail": ["shadows move", "birds sing", "silence reigns"],
    })
    emb = FakeEmbedder()
    gi = GrammarIndex(g, emb)
    gi.build()
    return Generator(gi)


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
    assert len(results) > 1  # should have variety


def test_generate_no_rule_refs_in_output(gen):
    text = gen.generate(context="test", temperature=0.3, seed=42)
    assert "#" not in text


def test_generate_empty_context(gen):
    text = gen.generate(context="", temperature=1.0, seed=42)
    assert isinstance(text, str)
    assert len(text) > 0
