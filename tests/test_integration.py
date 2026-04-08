"""Integration test: full pipeline with fake embedder."""
import json
import pytest
import numpy as np
from trimtab import SmartGrammar, Grammar, GrammarIndex, Generator


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


def test_full_pipeline_grammar_first(tmp_path):
    # Create grammar file
    rules = {
        "origin": ["#sound#. #atmosphere#."],
        "sound": ["A drip", "Wind howls", "Chains rattle"],
        "atmosphere": ["Cold air", "Damp walls", "Silence"],
    }
    path = tmp_path / "test.json"
    path.write_text(json.dumps(rules))

    # Grammar-first path
    sg = SmartGrammar.from_file(str(path), embedder=FakeEmbedder())
    sg.index()

    text = sg.generate(context="underground cave", temperature=0.0, seed=42)
    assert isinstance(text, str)
    assert "." in text
    assert "#" not in text


def test_save_load_roundtrip(tmp_path):
    rules = {
        "origin": ["#a# #b#"],
        "a": ["hello", "hey", "hi"],
        "b": ["world", "there", "friend"],
    }
    path = tmp_path / "test.json"
    path.write_text(json.dumps(rules))

    emb = FakeEmbedder()
    sg = SmartGrammar.from_file(str(path), embedder=emb)
    sg.index()

    sg.save(str(tmp_path / "test.sg"))

    sg2 = SmartGrammar.load(str(tmp_path / "test.sg"), embedder=emb)
    text = sg2.generate(context="greeting", temperature=0.0, seed=42)
    assert isinstance(text, str)


def test_add_expansion(tmp_path):
    rules = {"origin": ["#a#"], "a": ["x", "y"]}
    path = tmp_path / "test.json"
    path.write_text(json.dumps(rules))

    emb = FakeEmbedder()
    sg = SmartGrammar.from_file(str(path), embedder=emb)
    sg.index()
    sg.add("a", "z")

    assert "z" in sg._index.grammar.get_expansions("a")
