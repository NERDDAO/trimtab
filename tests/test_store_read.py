"""Round-trip reads from v0.5 Symbol/Rule schema."""

import pytest

from tests.conftest import list_rules_text_id, load_grammar_bulk
from trimtab.db import TrimTabDB
from trimtab.grammar import Grammar


class FakeEmbedder:
    async def create(self, input_data: str | list[str]) -> list[float]:
        return [0.1] * 8

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        return [[0.1] * 8 for _ in input_data_list]


@pytest.mark.asyncio
async def test_list_rules_returns_text_and_id():
    db = TrimTabDB(":memory:")
    grammar = Grammar.from_dict({
        "writers": [
            {"text": "Shakespeare — playwright", "id": "entity/uuid1"},
            {"text": "Tolkien — novelist", "id": "entity/uuid2"},
        ],
    })
    await load_grammar_bulk(db, "lit", grammar, FakeEmbedder())

    entries = list_rules_text_id(db, "lit", "writers")
    assert len(entries) == 2
    texts = {t for t, _ in entries}
    ids = {i for _, i in entries}
    assert "Shakespeare — playwright" in texts
    assert "entity/uuid1" in ids
    assert "entity/uuid2" in ids


@pytest.mark.asyncio
async def test_list_rules_empty_symbol():
    db = TrimTabDB(":memory:")
    grammar = Grammar.from_dict({"writers": [{"text": "x", "id": "y"}]})
    await load_grammar_bulk(db, "lit", grammar, FakeEmbedder())

    entries = list_rules_text_id(db, "lit", "nonexistent")
    assert entries == []
