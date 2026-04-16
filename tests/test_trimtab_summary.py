"""Tests for TrimTab.summary() — the per-grammar snapshot read."""

from __future__ import annotations

import pytest

from trimtab.core import TrimTab


@pytest.fixture
def tt(stub_embedder):
    return TrimTab(path=":memory:", embedder=stub_embedder)


def test_summary_empty_db(tt):
    assert tt.summary() == {}


@pytest.mark.asyncio
async def test_summary_groups_rules_with_metadata(tt, stub_embedder):
    await tt.put("quests", "origin", "Find the artifact",
                 id="q1", metadata={"priority": "high"})
    await tt.put("quests", "origin", "Map the dungeon",
                 id="q2", metadata={"priority": "low"})
    await tt.put("friends", "origin", "Alice — wizard",
                 id="ent_alice", metadata={"name": "Alice", "rank": 1})

    before_create = stub_embedder.call_count
    snap = tt.summary()
    # Pure read — no embedder calls.
    assert stub_embedder.call_count == before_create

    assert set(snap.keys()) == {"quests", "friends"}
    assert len(snap["quests"]) == 2
    assert len(snap["friends"]) == 1

    quests_by_id = {e["id"]: e for e in snap["quests"]}
    assert quests_by_id["q1"]["text"] == "Find the artifact"
    assert quests_by_id["q1"]["metadata"] == {"priority": "high"}
    assert quests_by_id["q1"]["symbol"] == "origin"

    alice = snap["friends"][0]
    assert alice["metadata"] == {"name": "Alice", "rank": 1}


@pytest.mark.asyncio
async def test_summary_reflects_remove(tt):
    r = await tt.put("g", "origin", "to-go", id="rx")
    assert tt.summary()["g"][0]["id"] == r.id
    tt.remove("g", "origin", r.id)
    # Grammar entry persists but list is empty.
    assert tt.summary().get("g", []) == []
