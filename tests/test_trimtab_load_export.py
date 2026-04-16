"""Tests for TrimTab.load_file / export_file."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trimtab.core import TrimTab


@pytest.fixture
def tt(stub_embedder):
    return TrimTab(path=":memory:", embedder=stub_embedder)


@pytest.mark.asyncio
async def test_load_plain_tracery_file(tt, tmp_path: Path):
    p = tmp_path / "tracery.json"
    p.write_text(json.dumps({"origin": ["hello", "world"]}))
    await tt.load_file(p, grammar="dev_memory")
    rules = tt.list("dev_memory", "origin")
    assert {r.text for r in rules} == {"hello", "world"}


@pytest.mark.asyncio
async def test_load_file_embeds_each_rule(tt, tmp_path: Path, stub_embedder):
    p = tmp_path / "tracery.json"
    p.write_text(json.dumps({"notes": ["a", "b", "c"]}))
    before = stub_embedder.batch_call_count
    await tt.load_file(p, grammar="g1")
    # At least one batch call (one per symbol).
    assert stub_embedder.batch_call_count > before


@pytest.mark.asyncio
async def test_export_file_round_trip(tt, tmp_path: Path):
    await tt.put("g1", "origin", "hello")
    out = tmp_path / "out.json"
    tt.export_file(out, grammar="g1")
    data = json.loads(out.read_text())
    assert "origin" in data
    # Each entry is either a plain string or a dict with a "text" key.
    texts = [e if isinstance(e, str) else e.get("text", "") for e in data["origin"]]
    assert "hello" in texts


@pytest.mark.asyncio
async def test_load_then_export_preserves_texts(tt, tmp_path: Path):
    """Round-trip: load Tracery JSON, then export, and texts survive."""
    src = tmp_path / "src.json"
    src.write_text(json.dumps({
        "origin": ["#greeting#, #name#"],
        "greeting": ["hello", "hi"],
        "name": ["world", "friend"],
    }))
    await tt.load_file(src, grammar="g1")

    dst = tmp_path / "dst.json"
    tt.export_file(dst, grammar="g1")
    data = json.loads(dst.read_text())
    assert set(data.keys()) == {"origin", "greeting", "name"}

    def _texts(rule_list):
        return {e if isinstance(e, str) else e.get("text", "") for e in rule_list}
    assert _texts(data["origin"]) == {"#greeting#, #name#"}
    assert _texts(data["greeting"]) == {"hello", "hi"}
    assert _texts(data["name"]) == {"world", "friend"}
