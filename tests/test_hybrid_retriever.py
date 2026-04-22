"""Tests for the Engram-inspired hybrid retriever.

Covers:
  - Reciprocal Rank Fusion helper (math only, no DB).
  - CosineRetriever parity with the pre-hybrid code path.
  - HybridRetriever surfaces BM25-only hits that pure cosine misses.
  - Cache invalidation on rule writes (put, update, remove, clear, drop).
  - Metadata passes through unchanged from LadybugDB to retrieval result.
"""

from __future__ import annotations

import pytest

from tests.conftest import add_rule
from trimtab.db import TrimTabDB
from trimtab.retriever import (
    CosineRetriever,
    HybridRetriever,
    reciprocal_rank_fusion,
)


# --- RRF math -------------------------------------------------------------


def test_rrf_single_ranker_preserves_order():
    fused = reciprocal_rank_fusion([["a", "b", "c"]])
    assert [rid for rid, _ in fused] == ["a", "b", "c"]


def test_rrf_merges_two_rankers():
    # "a" is top in both → highest combined score.
    # "b" and "c" each appear once at rank 2 → same score, order is insertion.
    fused = reciprocal_rank_fusion([["a", "b"], ["a", "c"]])
    assert fused[0][0] == "a"
    # Top score must be strictly greater than the tail.
    assert fused[0][1] > fused[1][1]


def test_rrf_k_parameter_dampens_contribution():
    small_k = reciprocal_rank_fusion([["a", "b"]], k=1)
    large_k = reciprocal_rank_fusion([["a", "b"]], k=1000)
    # a at rank 1: small_k score = 1/2; large_k = 1/1001
    assert small_k[0][1] > large_k[0][1]


def test_rrf_absent_from_ranker_gives_zero_contribution():
    fused = reciprocal_rank_fusion([["a"], ["b"]])
    scores = dict(fused)
    assert scores["a"] == scores["b"]


# --- CosineRetriever parity ----------------------------------------------


@pytest.mark.asyncio
async def test_cosine_retriever_matches_direct_db_call(mem_db, stub_embedder):
    await add_rule(mem_db, "g", "origin", "Caroline researches adoption agencies", stub_embedder)
    await add_rule(mem_db, "g", "origin", "Melanie paints every morning", stub_embedder)

    query_vec = await stub_embedder.create("what did Caroline research")
    retriever = CosineRetriever()

    via_retriever = await retriever.search(
        mem_db, "g", "origin", "what did Caroline research",
        top_k=2, query_vector=query_vec,
    )
    via_db = mem_db._search_rules("g", "origin", query_vec, top_k=2)

    assert [r.id for r in via_retriever] == [r.id for r in via_db]


@pytest.mark.asyncio
async def test_cosine_retriever_requires_query_vector(mem_db):
    retriever = CosineRetriever()
    with pytest.raises(ValueError):
        await retriever.search(
            mem_db, "g", "s", "query", top_k=5, query_vector=None
        )


# --- HybridRetriever surface ---------------------------------------------


@pytest.mark.asyncio
async def test_hybrid_retriever_requires_query_vector(mem_db):
    retriever = HybridRetriever()
    with pytest.raises(ValueError):
        await retriever.search(
            mem_db, "g", "s", "query", top_k=5, query_vector=None
        )


@pytest.mark.asyncio
async def test_hybrid_empty_grammar_returns_empty(mem_db, stub_embedder):
    retriever = HybridRetriever()
    query_vec = await stub_embedder.create("anything")
    results = await retriever.search(
        mem_db, "missing", "origin", "anything",
        top_k=5, query_vector=query_vec,
    )
    assert results == []


@pytest.mark.asyncio
async def test_hybrid_surfaces_bm25_only_hit_that_cosine_misses(mem_db, stub_embedder):
    """The hash-based StubEmbedder gives unrelated queries ~zero similarity
    to all rule texts, so pure cosine returns arbitrary order. BM25 on the
    exact query tokens should surface the matching rule regardless.

    This simulates the single-hop failure mode where a rare-term query (e.g.
    "single" for relationship status) fails dense retrieval.
    """
    # Insert 20 rules, one of which contains the discriminating token.
    await add_rule(mem_db, "g", "origin", "Caroline lives in Boston and is single", stub_embedder)
    for i in range(19):
        await add_rule(
            mem_db, "g", "origin",
            f"Melanie sketches flower number {i} in her garden",
            stub_embedder,
        )

    retriever = HybridRetriever()
    query = "single"
    query_vec = await stub_embedder.create(query)

    results = await retriever.search(
        mem_db, "g", "origin", query,
        top_k=5, query_vector=query_vec,
    )
    top_texts = [r.text for r in results]
    # The single-containing rule must be in the top-5 via BM25 contribution.
    assert any("single" in t.lower() for t in top_texts), top_texts


@pytest.mark.asyncio
async def test_hybrid_preserves_metadata(mem_db, stub_embedder):
    await add_rule(mem_db, "g", "origin", "Caroline researches adoption", stub_embedder)
    # Attach metadata by updating the rule directly.
    existing = mem_db._get_rules("g", "origin")[0]
    mem_db._update_rule_fields(
        "g", "origin", existing.id,
        metadata={"kg_uuid": "abc-123", "kg_kind": "entity", "label": "Caroline"},
    )

    retriever = HybridRetriever()
    query_vec = await stub_embedder.create("Caroline")
    results = await retriever.search(
        mem_db, "g", "origin", "Caroline",
        top_k=1, query_vector=query_vec,
    )
    assert len(results) == 1
    assert results[0].metadata["kg_uuid"] == "abc-123"
    assert results[0].metadata["label"] == "Caroline"


# --- cache invalidation ---------------------------------------------------


@pytest.mark.asyncio
async def test_hybrid_cache_invalidated_on_put(mem_db, stub_embedder):
    retriever = HybridRetriever()

    await add_rule(mem_db, "g", "origin", "initial text alpha", stub_embedder)
    query_vec = await stub_embedder.create("alpha")
    # Prime the cache.
    await retriever.search(mem_db, "g", "origin", "alpha", top_k=5, query_vector=query_vec)
    assert ("g", "origin") in retriever._cache

    # New write must evict the cached slice.
    await add_rule(mem_db, "g", "origin", "new text beta", stub_embedder)
    assert ("g", "origin") not in retriever._cache


@pytest.mark.asyncio
async def test_hybrid_cache_invalidated_on_update(mem_db, stub_embedder):
    retriever = HybridRetriever()
    await add_rule(mem_db, "g", "origin", "first text", stub_embedder)
    query_vec = await stub_embedder.create("first")
    await retriever.search(mem_db, "g", "origin", "first", top_k=5, query_vector=query_vec)
    assert ("g", "origin") in retriever._cache

    rule_id = mem_db._get_rules("g", "origin")[0].id
    mem_db._update_rule_fields("g", "origin", rule_id, text="updated text", new_vector=await stub_embedder.create("updated"))
    assert ("g", "origin") not in retriever._cache


@pytest.mark.asyncio
async def test_hybrid_cache_invalidated_on_remove(mem_db, stub_embedder):
    retriever = HybridRetriever()
    await add_rule(mem_db, "g", "origin", "to be removed", stub_embedder)
    query_vec = await stub_embedder.create("removed")
    await retriever.search(mem_db, "g", "origin", "removed", top_k=5, query_vector=query_vec)
    assert ("g", "origin") in retriever._cache

    rule_id = mem_db._get_rules("g", "origin")[0].id
    mem_db._remove_rule("g", "origin", rule_id)
    assert ("g", "origin") not in retriever._cache


@pytest.mark.asyncio
async def test_hybrid_cache_invalidated_on_clear_symbol(mem_db, stub_embedder):
    retriever = HybridRetriever()
    await add_rule(mem_db, "g", "origin", "one", stub_embedder)
    await add_rule(mem_db, "g", "other", "two", stub_embedder)
    query_vec = await stub_embedder.create("any")
    await retriever.search(mem_db, "g", "origin", "any", top_k=5, query_vector=query_vec)
    await retriever.search(mem_db, "g", "other", "any", top_k=5, query_vector=query_vec)
    assert ("g", "origin") in retriever._cache
    assert ("g", "other") in retriever._cache

    mem_db._clear_symbol("g", "origin")
    # Only the cleared symbol is invalidated.
    assert ("g", "origin") not in retriever._cache
    assert ("g", "other") in retriever._cache


@pytest.mark.asyncio
async def test_hybrid_cache_invalidated_on_drop_grammar(mem_db, stub_embedder):
    retriever = HybridRetriever()
    await add_rule(mem_db, "g", "origin", "one", stub_embedder)
    await add_rule(mem_db, "g", "other", "two", stub_embedder)
    await add_rule(mem_db, "g2", "origin", "three", stub_embedder)
    query_vec = await stub_embedder.create("any")
    await retriever.search(mem_db, "g", "origin", "any", top_k=5, query_vector=query_vec)
    await retriever.search(mem_db, "g", "other", "any", top_k=5, query_vector=query_vec)
    await retriever.search(mem_db, "g2", "origin", "any", top_k=5, query_vector=query_vec)

    mem_db._drop_grammar("g")
    # Both slices of g go; g2 survives.
    assert ("g", "origin") not in retriever._cache
    assert ("g", "other") not in retriever._cache
    assert ("g2", "origin") in retriever._cache
