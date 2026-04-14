"""End-to-end pipeline test: put → list → search → generate → update → remove → clear → drop.

This exercises every public TrimTab method in one realistic scenario.
Uses the StubEmbedder fixture so it's hermetic (no Ollama required).
"""

from __future__ import annotations

import pytest

from trimtab.core import TrimTab


@pytest.mark.asyncio
async def test_full_pipeline(stub_embedder):
    tt = TrimTab(path=":memory:", embedder=stub_embedder)

    # 1. Accumulate notes
    r1 = await tt.put("agent_01", "notes", "forest dangerous at night")
    r2 = await tt.put("agent_01", "notes", "Alice has the map", metadata={"person": "Alice"})
    r3 = await tt.put("agent_01", "notes", "oracle lives on the mountain")

    assert r1.id and r2.id and r3.id
    assert r2.metadata == {"person": "Alice"}

    # 2. list is insertion-ordered
    listed = tt.list("agent_01", "notes")
    assert [r.text for r in listed] == [
        "forest dangerous at night",
        "Alice has the map",
        "oracle lives on the mountain",
    ]

    # 3. count matches
    assert tt.count("agent_01", "notes") == 3

    # 4. search returns at least one rule
    results = await tt.search("agent_01", "notes", query="forest", top_k=3)
    assert len(results) >= 1
    assert all(hasattr(r, "text") for r in results)

    # 5. list_symbols sees the one symbol we've populated
    symbols = tt.list_symbols("agent_01")
    assert symbols == ["notes"]

    # 6. list_grammars sees the grammar
    grammars = tt.list_grammars()
    assert "agent_01" in grammars

    # 7. Add an origin symbol for generation
    await tt.put("agent_01", "origin", "Note: #notes#")
    result = await tt.generate("agent_01", context="journey", origin="origin")
    assert result.text.startswith("Note: ")
    # rules_used should include at least the origin rule and one notes rule
    assert len(result.rules_used) >= 2

    # 8. update: change text — triggers re-embed
    await tt.update("agent_01", "notes", r1.id, text="forest is safe in daytime")
    listed = tt.list("agent_01", "notes")
    assert any(r.text == "forest is safe in daytime" for r in listed)

    # 9. update metadata-only: no re-embed, merges into existing dict
    await tt.update("agent_01", "notes", r2.id, metadata={"verified": True})
    listed = tt.list("agent_01", "notes")
    by_id = {r.id: r for r in listed}
    assert by_id[r2.id].metadata == {"person": "Alice", "verified": True}

    # 10. remove: drops from future reads
    tt.remove("agent_01", "notes", r3.id)
    assert not any(r.id == r3.id for r in tt.list("agent_01", "notes"))
    assert tt.count("agent_01", "notes") == 2

    # 11. clear leaves symbol empty but symbol node remains
    tt.clear("agent_01", "notes")
    assert tt.list("agent_01", "notes") == []
    assert "notes" in tt.list_symbols("agent_01")

    # 12. drop wipes the whole grammar
    tt.drop("agent_01")
    assert tt.list_symbols("agent_01") == []
    assert "agent_01" not in tt.list_grammars()
