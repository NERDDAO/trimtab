"""Tests for CosineRetriever's v24 P2 fusion kwargs.

CosineRetriever historically dropped ``auxiliary_rankings`` and
``candidate_subset`` via ``del``. Swap 1 of "soup up the cascade walk"
extends it to honour both, mirroring HybridRetriever's semantics:

* dense-only fast path when neither kwarg is in play
* RRF fusion (k=60) when ``auxiliary_rankings`` is non-empty
* subset filter applied to the dense candidate pool (and to aux rankings)
* empty subset short-circuits to ``[]`` — no fallback to full pool
* aux ids unknown to the DB are silently dropped

Cross-encoder rerank is intentionally absent — that stays a HybridRetriever
concern so the cascade default keeps low per-step latency.
"""

from __future__ import annotations

import pytest

from tests.conftest import add_rule
from trimtab.retriever import CosineRetriever


# --- backcompat fast path -------------------------------------------------


@pytest.mark.asyncio
async def test_cosine_aux_none_subset_none_is_dense_byte_equivalent(
    mem_db, stub_embedder
):
    """Both kwargs ``None`` (and omitted) must give pure-dense ``_search_rules``
    output — bytes-equivalence with the pre-v24 default path."""
    await add_rule(mem_db, "g", "origin", "Caroline researches adoption", stub_embedder)
    await add_rule(mem_db, "g", "origin", "Melanie paints sunrise", stub_embedder)
    await add_rule(mem_db, "g", "origin", "Caroline lives in Boston", stub_embedder)

    qv = await stub_embedder.create("Caroline")
    retriever = CosineRetriever()

    via_retriever_omitted = await retriever.search(
        mem_db, "g", "origin", "Caroline", top_k=3, query_vector=qv
    )
    via_retriever_explicit_none = await retriever.search(
        mem_db,
        "g",
        "origin",
        "Caroline",
        top_k=3,
        query_vector=qv,
        auxiliary_rankings=None,
        candidate_subset=None,
    )
    via_db = mem_db._search_rules("g", "origin", qv, 3)

    assert (
        [r.id for r in via_retriever_omitted]
        == [r.id for r in via_retriever_explicit_none]
        == [r.id for r in via_db]
    )


@pytest.mark.asyncio
async def test_cosine_aux_empty_list_is_backcompat(mem_db, stub_embedder):
    """``auxiliary_rankings=[]`` is a no-op signal lane (same as ``None``)."""
    for text in ("Alpha one", "Alpha two", "Alpha three"):
        await add_rule(mem_db, "g", "origin", text, stub_embedder)

    qv = await stub_embedder.create("Alpha")
    retriever = CosineRetriever()

    baseline = await retriever.search(
        mem_db, "g", "origin", "Alpha", top_k=3, query_vector=qv
    )
    with_empty = await retriever.search(
        mem_db,
        "g",
        "origin",
        "Alpha",
        top_k=3,
        query_vector=qv,
        auxiliary_rankings=[],
    )
    assert [r.id for r in baseline] == [r.id for r in with_empty]


# --- candidate_subset -----------------------------------------------------


@pytest.mark.asyncio
async def test_cosine_candidate_subset_restricts_output(mem_db, stub_embedder):
    """A non-empty subset limits returned rules to ids in the subset."""
    for text in ("Alpha one", "Alpha two", "Alpha three", "Alpha four", "Alpha five"):
        await add_rule(mem_db, "g", "origin", text, stub_embedder)

    all_ids = [r.id for r in mem_db._get_rules("g", "origin")]
    subset_ids = all_ids[:2]

    retriever = CosineRetriever()
    qv = await stub_embedder.create("Alpha")
    result = await retriever.search(
        mem_db,
        "g",
        "origin",
        "Alpha",
        top_k=5,
        query_vector=qv,
        candidate_subset=subset_ids,
    )
    returned = {r.id for r in result}
    assert returned, "subset should have yielded at least one hit"
    assert returned.issubset(set(subset_ids))


@pytest.mark.asyncio
async def test_cosine_candidate_subset_empty_returns_empty(mem_db, stub_embedder):
    """Empty subset short-circuits — same semantics as HybridRetriever (no
    fallback to full pool)."""
    await add_rule(mem_db, "g", "origin", "Alpha one", stub_embedder)

    retriever = CosineRetriever()
    qv = await stub_embedder.create("Alpha")
    result = await retriever.search(
        mem_db,
        "g",
        "origin",
        "Alpha",
        top_k=5,
        query_vector=qv,
        candidate_subset=[],
    )
    assert result == []


@pytest.mark.asyncio
async def test_cosine_candidate_subset_unknown_ids_dropped(mem_db, stub_embedder):
    """Unknown ids in the subset are silently skipped — no error, just absent
    from the result. Caller owns the validity contract."""
    await add_rule(mem_db, "g", "origin", "Alpha one", stub_embedder)
    await add_rule(mem_db, "g", "origin", "Alpha two", stub_embedder)

    real_id = mem_db._get_rules("g", "origin")[0].id

    retriever = CosineRetriever()
    qv = await stub_embedder.create("Alpha")
    result = await retriever.search(
        mem_db,
        "g",
        "origin",
        "Alpha",
        top_k=5,
        query_vector=qv,
        candidate_subset=[real_id, "ghost-id-not-in-db"],
    )
    returned = {r.id for r in result}
    assert returned == {real_id}


# --- auxiliary_rankings ---------------------------------------------------


@pytest.mark.asyncio
async def test_cosine_auxiliary_rankings_promotes_ranked_id(mem_db, stub_embedder):
    """An aux ranking that top-ranks a specific id should pull it up the
    fused result vs the dense-only baseline."""
    for text in ("Alpha one", "Alpha two", "Alpha three", "Alpha four", "Alpha five"):
        await add_rule(mem_db, "g", "origin", text, stub_embedder)

    retriever = CosineRetriever()
    qv = await stub_embedder.create("Alpha")
    baseline = await retriever.search(
        mem_db, "g", "origin", "Alpha", top_k=5, query_vector=qv
    )
    baseline_ids = [r.id for r in baseline]
    target_id = baseline_ids[-1]  # weakest in dense-only

    with_aux = await retriever.search(
        mem_db,
        "g",
        "origin",
        "Alpha",
        top_k=5,
        query_vector=qv,
        auxiliary_rankings=[[target_id]],
    )
    aux_ids = [r.id for r in with_aux]
    assert target_id in aux_ids
    assert aux_ids.index(target_id) < baseline_ids.index(target_id)


@pytest.mark.asyncio
async def test_cosine_multiple_auxiliary_lanes_both_contribute(mem_db, stub_embedder):
    """Two aux lanes each top-ranking a different id should pull both ids
    above peers that no lane endorses."""
    for text in ("Alpha one", "Alpha two", "Alpha three", "Alpha four", "Alpha five"):
        await add_rule(mem_db, "g", "origin", text, stub_embedder)

    retriever = CosineRetriever()
    qv = await stub_embedder.create("Alpha")
    baseline = await retriever.search(
        mem_db, "g", "origin", "Alpha", top_k=5, query_vector=qv
    )
    baseline_ids = [r.id for r in baseline]
    # Two weakest in dense — both should get promoted by their respective lanes.
    target_a = baseline_ids[-1]
    target_b = baseline_ids[-2]

    with_aux = await retriever.search(
        mem_db,
        "g",
        "origin",
        "Alpha",
        top_k=5,
        query_vector=qv,
        auxiliary_rankings=[[target_a], [target_b]],
    )
    aux_ids = [r.id for r in with_aux]
    assert aux_ids.index(target_a) < baseline_ids.index(target_a)
    assert aux_ids.index(target_b) < baseline_ids.index(target_b)


@pytest.mark.asyncio
async def test_cosine_subset_filters_aux_referenced_id(mem_db, stub_embedder):
    """When ``candidate_subset`` excludes a rule, an aux ranking referencing
    that rule must NOT bring it back into the result. Subset is authoritative."""
    for text in ("Alpha one", "Alpha two", "Alpha three"):
        await add_rule(mem_db, "g", "origin", text, stub_embedder)

    all_ids = [r.id for r in mem_db._get_rules("g", "origin")]
    keep_ids = all_ids[:2]
    excluded_id = all_ids[2]

    retriever = CosineRetriever()
    qv = await stub_embedder.create("Alpha")
    result = await retriever.search(
        mem_db,
        "g",
        "origin",
        "Alpha",
        top_k=5,
        query_vector=qv,
        candidate_subset=keep_ids,
        auxiliary_rankings=[[excluded_id]],
    )
    returned = {r.id for r in result}
    assert excluded_id not in returned
    assert returned.issubset(set(keep_ids))
