"""Shared pytest fixtures for trimtab tests."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest


class StubEmbedder:
    """Deterministic hash-based embedder for hermetic tests.

    Always returns the same vector for the same text. Dimension is fixed
    at construction. No HTTP, no ML deps, no Ollama — safe to use in CI.
    """

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim
        self.call_count = 0
        self.batch_call_count = 0

    def _vector(self, text: str) -> list[float]:
        # SHA256 → first `dim` bytes → normalize to roughly [-1, 1].
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        # Repeat digest if dim > 32 (sha256 is 32 bytes).
        while len(digest) < self.dim:
            digest = digest + hashlib.sha256(digest).digest()
        return [(b - 128) / 128.0 for b in digest[: self.dim]]

    async def create(self, input_data: str | list[str]) -> list[float]:
        self.call_count += 1
        text = input_data if isinstance(input_data, str) else " ".join(input_data)
        return self._vector(text)

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        self.batch_call_count += 1
        return [self._vector(t) for t in input_data_list]


@pytest.fixture
def stub_embedder() -> StubEmbedder:
    """A fresh 16-dim stub embedder per test."""
    return StubEmbedder(dim=16)


# Legacy fixture — retained for pre-existing tests that already depend on it.
# New tests should prefer StubEmbedder (above) which ships with call counters
# and a byte-range normalization suited to hermetic v0.5 tests.
class FakeEmbedder:
    """Deterministic async embedder matching trimtab's Embedder Protocol.

    Hash-based, normalized vectors — reproducible across runs so tests that
    assert on embedding similarity are stable.
    """

    def __init__(self, dim: int = 8):
        self._dim = dim

    async def create(self, input_data: str | list[str]) -> list[float]:
        if isinstance(input_data, list):
            input_data = " ".join(input_data)
        rng = np.random.default_rng(hash(input_data) % (2**31))
        vec = rng.standard_normal(self._dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for item in input_data_list:
            out.append(await self.create(item))
        return out


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def mem_db():
    """In-memory TrimTabDB for tests."""
    from trimtab.db import TrimTabDB
    return TrimTabDB(":memory:")


async def load_grammar_bulk(db, grammar_name, grammar, embedder):
    """Test helper — replaces the deprecated ``TrimTabDB.upsert_grammar``.

    Walks the Grammar dict, batch-embeds every rule's text, and inserts
    via the v0.5 ``_put_rule_with_vector`` path.
    """
    from trimtab.grammar import upgrade_entry

    all_items = []
    for symbol_name in grammar.rule_names():
        for entry in grammar.rules.get(symbol_name, []):
            all_items.append((symbol_name, upgrade_entry(entry)))
    if not all_items:
        return
    texts = [r.text for _, r in all_items]
    vectors = await embedder.create_batch(texts)
    for (symbol_name, rule_obj), vec in zip(all_items, vectors, strict=True):
        db._put_rule_with_vector(grammar=grammar_name, symbol=symbol_name, rule=rule_obj, vector=vec)


async def add_rule(db, grammar_name, symbol, text, embedder, id=None):
    """Test helper — replaces the deprecated ``TrimTabDB.add_expansion``."""
    from trimtab.grammar import Rule

    vec = await embedder.create(text)
    rule = Rule(text=text, id=id or "")
    return db._put_rule_with_vector(grammar=grammar_name, symbol=symbol, rule=rule, vector=vec)


def list_rules_text_id(db, grammar_name, symbol):
    """Test helper — replaces the deprecated ``TrimTabDB.list_entries``."""
    return [(r.text, r.id) for r in db._get_rules(grammar_name, symbol)]
