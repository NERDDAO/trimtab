"""Tests for trimtab.engram.EngramIndex.

Covers:
  - build() aggregates corpus counts from rule metadata.entities.
  - Duplicate chunk-instances of the same span increase weight.
  - Duplicate spans within a single rule count once (per-rule presence).
  - lookup() returns rule-ids with relevance scores summed across entities.
  - lookup() respects min_weight threshold.
  - Invalidation listener rebuilds on rule writes.
"""

from __future__ import annotations

import pytest

from tests.conftest import add_rule
from trimtab.engram import EngramIndex


@pytest.mark.asyncio
async def test_engram_build_aggregates_corpus_counts(mem_db, stub_embedder):
    await add_rule(mem_db, "g", "messages", "m1", stub_embedder)
    await add_rule(mem_db, "g", "messages", "m2", stub_embedder)
    await add_rule(mem_db, "g", "messages", "m3", stub_embedder)
    rules = mem_db._get_rules("g", "messages")
    # Patch in metadata.entities for each rule.
    mem_db._update_rule_fields(
        "g",
        "messages",
        rules[0].id,
        metadata={"entities": {"Preference": ["painting"]}},
    )
    mem_db._update_rule_fields(
        "g",
        "messages",
        rules[1].id,
        metadata={"entities": {"Preference": ["painting", "pottery"]}},
    )
    mem_db._update_rule_fields(
        "g", "messages", rules[2].id, metadata={"entities": {"Preference": ["pottery"]}}
    )

    index = EngramIndex(mem_db, "g", "messages")
    agg = index.aggregate()
    assert agg[("Preference", "painting")] == 2
    assert agg[("Preference", "pottery")] == 2


@pytest.mark.asyncio
async def test_engram_build_counts_rule_once_even_with_duplicate_spans(
    mem_db, stub_embedder
):
    await add_rule(mem_db, "g", "messages", "m", stub_embedder)
    rule_id = mem_db._get_rules("g", "messages")[0].id
    # Same span appears twice in one rule — corpus weight is per-rule presence.
    mem_db._update_rule_fields(
        "g",
        "messages",
        rule_id,
        metadata={"entities": {"Preference": ["painting", "painting"]}},
    )

    index = EngramIndex(mem_db, "g", "messages")
    agg = index.aggregate()
    assert agg[("Preference", "painting")] == 1


@pytest.mark.asyncio
async def test_engram_lookup_returns_rules_scored_by_weight(mem_db, stub_embedder):
    await add_rule(mem_db, "g", "messages", "m1", stub_embedder)
    await add_rule(mem_db, "g", "messages", "m2", stub_embedder)
    await add_rule(mem_db, "g", "messages", "m3", stub_embedder)
    rules = mem_db._get_rules("g", "messages")
    mem_db._update_rule_fields(
        "g",
        "messages",
        rules[0].id,
        metadata={"entities": {"Preference": ["painting"]}},
    )
    mem_db._update_rule_fields(
        "g",
        "messages",
        rules[1].id,
        metadata={"entities": {"Preference": ["painting"]}},
    )
    mem_db._update_rule_fields(
        "g", "messages", rules[2].id, metadata={"entities": {"Preference": ["pottery"]}}
    )

    index = EngramIndex(mem_db, "g", "messages")
    matches = index.lookup({"Preference": ["painting"]})
    # painting weight = 2; rule 0 and 1 should both hit with score 2.
    assert set(matches.keys()) == {rules[0].id, rules[1].id}
    assert all(v == 2 for v in matches.values())


@pytest.mark.asyncio
async def test_engram_lookup_sums_scores_across_entities(mem_db, stub_embedder):
    await add_rule(mem_db, "g", "messages", "m1", stub_embedder)
    rule_id = mem_db._get_rules("g", "messages")[0].id
    mem_db._update_rule_fields(
        "g",
        "messages",
        rule_id,
        metadata={"entities": {"Preference": ["painting"], "Person": ["caroline"]}},
    )

    index = EngramIndex(mem_db, "g", "messages")
    matches = index.lookup({"Preference": ["painting"], "Person": ["caroline"]})
    # Both spans hit this rule, each with corpus weight 1 → total 2.
    assert matches[rule_id] == 2


@pytest.mark.asyncio
async def test_engram_lookup_respects_min_weight(mem_db, stub_embedder):
    await add_rule(mem_db, "g", "messages", "m1", stub_embedder)
    rule_id = mem_db._get_rules("g", "messages")[0].id
    mem_db._update_rule_fields(
        "g",
        "messages",
        rule_id,
        metadata={"entities": {"Preference": ["painting"]}},
    )

    index = EngramIndex(mem_db, "g", "messages")
    assert index.lookup({"Preference": ["painting"]}, min_weight=1) != {}
    # weight=1 bucket filtered out at threshold 2
    assert index.lookup({"Preference": ["painting"]}, min_weight=2) == {}


@pytest.mark.asyncio
async def test_engram_skips_rules_without_entities_key(mem_db, stub_embedder):
    await add_rule(mem_db, "g", "messages", "has entities", stub_embedder)
    await add_rule(mem_db, "g", "messages", "no entities", stub_embedder)
    rules = mem_db._get_rules("g", "messages")
    mem_db._update_rule_fields(
        "g",
        "messages",
        rules[0].id,
        metadata={"entities": {"Preference": ["painting"]}},
    )
    # rules[1] left with empty metadata.

    index = EngramIndex(mem_db, "g", "messages")
    matches = index.lookup({"Preference": ["painting"]})
    assert set(matches.keys()) == {rules[0].id}


@pytest.mark.asyncio
async def test_engram_lookup_normalizes_case_and_whitespace(mem_db, stub_embedder):
    await add_rule(mem_db, "g", "messages", "m", stub_embedder)
    rule_id = mem_db._get_rules("g", "messages")[0].id
    mem_db._update_rule_fields(
        "g",
        "messages",
        rule_id,
        metadata={"entities": {"Preference": ["Painting"]}},
    )

    index = EngramIndex(mem_db, "g", "messages")
    # Query with different case + surrounding whitespace.
    matches = index.lookup({"Preference": ["  painting  "]})
    assert rule_id in matches


@pytest.mark.asyncio
async def test_engram_invalidates_on_write(mem_db, stub_embedder):
    await add_rule(mem_db, "g", "messages", "m1", stub_embedder)
    rule_id = mem_db._get_rules("g", "messages")[0].id
    mem_db._update_rule_fields(
        "g",
        "messages",
        rule_id,
        metadata={"entities": {"Preference": ["painting"]}},
    )

    index = EngramIndex(mem_db, "g", "messages")
    index.build()  # prime
    assert index._built

    # Any mutation in the (grammar, symbol) slice must invalidate.
    await add_rule(mem_db, "g", "messages", "m2", stub_embedder)
    assert not index._built
