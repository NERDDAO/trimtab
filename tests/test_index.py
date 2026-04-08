"""Tests for FAISS indexing."""
import pytest
import numpy as np
from trimtab.grammar import Grammar
from trimtab.index import GrammarIndex


class FakeEmbedder:
    """Deterministic embedder for testing."""
    def __init__(self, dim=8):
        self._dim = dim
        self._counter = 0

    def embed(self, texts: list[str]) -> np.ndarray:
        # Hash-based: similar texts get similar vectors
        vecs = []
        for t in texts:
            np.random.seed(hash(t) % (2**31))
            vecs.append(np.random.randn(self._dim).astype(np.float32))
        arr = np.array(vecs)
        # Normalize
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / (norms + 1e-8)

    @property
    def dimension(self):
        return self._dim


@pytest.fixture
def indexed():
    g = Grammar.from_dict({
        "origin": ["#a#"],
        "a": ["dark caves", "bright forest", "cold mountain", "warm beach"],
    })
    emb = FakeEmbedder()
    gi = GrammarIndex(g, emb)
    gi.build()
    return gi, emb


def test_build_creates_indices(indexed):
    gi, _ = indexed
    assert "a" in gi._indices
    assert len(gi._expansions["a"]) == 4


def test_query_returns_results(indexed):
    gi, emb = indexed
    ctx = emb.embed(["underground darkness"])
    results = gi.query("a", ctx, top_k=3)
    assert len(results) > 0
    assert all(isinstance(r[0], str) and isinstance(r[1], float) for r in results)


def test_add_to_rule(indexed):
    gi, _ = indexed
    gi.add_to_rule("a", "foggy swamp")
    assert "foggy swamp" in gi._expansions["a"]
    assert gi._indices["a"].ntotal == 5


def test_save_load(indexed, tmp_path):
    gi, emb = indexed
    gi.save(tmp_path / "test.sg")

    gi2 = GrammarIndex.load(tmp_path / "test.sg", emb)
    assert gi2._expansions["a"] == gi._expansions["a"]
    assert gi2._indices["a"].ntotal == gi._indices["a"].ntotal
