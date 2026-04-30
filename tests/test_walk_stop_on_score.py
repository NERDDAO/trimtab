"""Tests for score-based walk-stopping in the cascade Generator.

Behavior under test (option (a) of the design):

* ``Rule.score`` is a transient float attribute set by ``Retriever.search``
  for every returned rule. It does NOT round-trip through the DB.
* ``Generator.generate(descent_margin=...)`` accepts a small positive cosine
  margin. Before descending into a ``#child#`` symbol from a parent rule,
  the generator peeks the child's best score and skips the descent when
  ``best_child_score < parent_score - margin``.
* ``descent_margin=0.0`` preserves the historical "always descend" behaviour.
* Missing scores (e.g. random-fallback selection paths) are treated as
  "always descend" — back-compat semantic.
"""

from __future__ import annotations

import logging

import pytest
import pytest_asyncio

from tests.conftest import add_rule, load_grammar_bulk
from trimtab.generator import Generator
from trimtab.grammar import Grammar
from trimtab.retriever import CosineRetriever, HybridRetriever


# --- 1. Score plumbing through retriever ----------------------------------


@pytest.mark.asyncio
async def test_cosine_retriever_populates_rule_score(mem_db, stub_embedder):
    """``CosineRetriever.search`` must surface a non-None ``Rule.score`` on
    every returned rule. Score should be a real number (cosine similarity in
    the dense fast path)."""
    for text in ("Alpha one", "Alpha two", "Alpha three"):
        await add_rule(mem_db, "g", "origin", text, stub_embedder)

    qv = await stub_embedder.create("Alpha one")
    retriever = CosineRetriever()

    out = await retriever.search(
        mem_db, "g", "origin", "Alpha one", top_k=3, query_vector=qv
    )
    assert out, "expected at least one rule"
    for r in out:
        assert r.score is not None, f"rule {r.id} missing score"
        assert isinstance(r.score, float)


@pytest.mark.asyncio
async def test_cosine_retriever_scores_descending(mem_db, stub_embedder):
    """Scores must be returned in descending order (best first)."""
    for text in ("Alpha one", "Alpha two", "Alpha three", "Alpha four"):
        await add_rule(mem_db, "g", "origin", text, stub_embedder)

    qv = await stub_embedder.create("Alpha one")
    retriever = CosineRetriever()
    out = await retriever.search(
        mem_db, "g", "origin", "Alpha one", top_k=4, query_vector=qv
    )
    scores = [r.score for r in out]
    assert all(s is not None for s in scores)
    # Cast away None for the lt check (asserted above).
    raw = [s for s in scores if s is not None]
    assert raw == sorted(raw, reverse=True)


@pytest.mark.asyncio
async def test_hybrid_retriever_populates_rule_score(mem_db, stub_embedder):
    """``HybridRetriever.search`` must populate ``Rule.score`` with the fused
    RRF score (not raw cosine)."""
    for text in ("Beta one", "Beta two", "Beta three"):
        await add_rule(mem_db, "g", "origin", text, stub_embedder)

    qv = await stub_embedder.create("Beta")
    retriever = HybridRetriever()
    out = await retriever.search(
        mem_db, "g", "origin", "Beta", top_k=3, query_vector=qv
    )
    assert out
    for r in out:
        assert r.score is not None
        assert isinstance(r.score, float)


# --- 2. Walk-stop integration --------------------------------------------


@pytest_asyncio.fixture
async def walk_grammar(mem_db, token_embedder):
    """Two-level grammar with one parent rule + ``#child#`` ref.

    Parent rule embeds toward "parent topic"; child rules all embed toward
    "child topic A/B/C". A query that matches the parent text closely will
    show high cosine for the parent but low cosine for any child — the
    walk-stop condition fires.

    Uses ``TokenEmbedder`` so cosine similarity reflects token overlap
    (deterministic semantic-style scoring), not random noise. This lets us
    assert on the actual ``parent_score > child_score`` shape that the
    walk-stop guard depends on.
    """
    grammar = Grammar.from_dict(
        {
            "origin": ["parent topic about #child#"],
            "child": ["child topic alpha", "child topic beta", "child topic gamma"],
        }
    )
    await load_grammar_bulk(mem_db, "wg", grammar, token_embedder)
    return mem_db, token_embedder


@pytest.mark.asyncio
async def test_walk_stops_when_child_score_below_margin(walk_grammar, caplog):
    """A query that strongly matches the parent and weakly matches any child
    should truncate the walk at the parent symbol when ``descent_margin``
    is large enough to cover the cosine gap."""
    db, embedder = walk_grammar
    gen = Generator(db, "wg", embedder)

    caplog.set_level(logging.INFO, logger="trimtab.generator")
    # Margin huge → guaranteed to stop at root.
    result = await gen.generate(
        context="parent topic about something",
        temperature=0.0,
        seed=1,
        descent_margin=1.0,
    )
    # ``#child#`` literal stays in the text (we don't expand it).
    assert "#child#" in result.text
    # No child-symbol rules in the walk.
    walked_symbols = {r.metadata.get("symbol", "") for r in result.rules_used}
    # Only the origin rule should be walked.
    assert len(result.rules_used) == 1
    # Log line should fire.
    assert any("walk stopped" in rec.message for rec in caplog.records)
    _ = walked_symbols  # symbol metadata is optional; keep the variable for grep


@pytest.mark.asyncio
async def test_walk_continues_when_margin_zero(walk_grammar):
    """``descent_margin=0.0`` reproduces the historical always-descend
    behaviour — the walk runs to the leaf even if the child score is lower."""
    db, embedder = walk_grammar
    gen = Generator(db, "wg", embedder)

    result = await gen.generate(
        context="parent topic about something",
        temperature=0.0,
        seed=1,
        descent_margin=0.0,
    )
    # No literal #child# left in output.
    assert "#child#" not in result.text
    # Both parent + one child rule walked.
    assert len(result.rules_used) >= 2


@pytest.mark.asyncio
async def test_walk_continues_when_child_matches_query(walk_grammar):
    """When the query closely matches one of the child rules, the cosine for
    that child is high and the walk should descend even with a moderate
    margin."""
    db, embedder = walk_grammar
    gen = Generator(db, "wg", embedder)

    result = await gen.generate(
        context="child topic alpha",
        temperature=0.0,
        seed=1,
        descent_margin=0.05,
    )
    # The walk should fully descend.
    assert "#child#" not in result.text
    assert len(result.rules_used) >= 2


@pytest.mark.asyncio
async def test_walk_huge_margin_always_stops(walk_grammar):
    """``descent_margin=1.0`` makes any descent impossible (cosine ∈ [-1, 1])."""
    db, embedder = walk_grammar
    gen = Generator(db, "wg", embedder)

    # Even with a child-matching query, margin >= 2 forbids any descent
    # (cosine gap can be at most 2.0 between [-1, 1]).
    result = await gen.generate(
        context="child topic alpha",
        temperature=0.0,
        seed=1,
        descent_margin=2.0,
    )
    assert "#child#" in result.text
    assert len(result.rules_used) == 1


# --- 3. HybridRetriever variant ------------------------------------------


@pytest.mark.asyncio
async def test_walk_stops_under_hybrid_retriever(walk_grammar, caplog):
    """Same walk-stop behaviour with HybridRetriever — scores are fused-RRF
    (not raw cosine) but the comparison logic in ``_expand`` is unchanged."""
    db, embedder = walk_grammar
    gen = Generator(db, "wg", embedder, retriever=HybridRetriever())

    caplog.set_level(logging.INFO, logger="trimtab.generator")
    result = await gen.generate(
        context="parent topic about something",
        temperature=0.0,
        seed=1,
        descent_margin=1.0,
    )
    # Margin >> any plausible RRF score gap (RRF scores are in 1/(k+r), so
    # max is around 1/61 ~= 0.016 per ranking lane; gap << 1.0).
    assert "#child#" in result.text
    assert len(result.rules_used) == 1
    assert any("walk stopped" in rec.message for rec in caplog.records)


# --- 4. Default descent_margin is the documented 0.05 --------------------


@pytest.mark.asyncio
async def test_default_descent_margin_is_small(walk_grammar):
    """The default margin (0.05) is small enough that most cascades still
    descend — only steep cosine cliffs trigger a stop. We only assert that
    the call succeeds and produces a result; the actual descent decision is
    a function of the (FakeEmbedder-derived) score gap and is covered by the
    explicit-margin tests above."""
    db, embedder = walk_grammar
    gen = Generator(db, "wg", embedder)

    result = await gen.generate(
        context="anything goes here",
        temperature=0.0,
        seed=1,
    )
    assert isinstance(result.text, str)
    assert result.text  # non-empty


# --- 5. Random-fallback path treats missing score as "always descend" ----


@pytest.mark.asyncio
async def test_walk_descends_when_parent_score_missing(mem_db, fake_embedder):
    """Random / no-context selection paths produce ``Rule.score is None``.
    The walk must treat that as "always descend" — back-compat semantic."""
    grammar = Grammar.from_dict(
        {
            "origin": ["#child#."],
            "child": ["child A", "child B", "child C"],
        }
    )
    await load_grammar_bulk(mem_db, "fb", grammar, fake_embedder)
    gen = Generator(mem_db, "fb", fake_embedder)

    # No context + temp 1.0 forces the random fallback at every level → no
    # scores anywhere → walk must still descend (back-compat).
    result = await gen.generate(
        context="",
        temperature=1.0,
        seed=42,
        descent_margin=1.0,  # huge, but missing scores override
    )
    assert "#child#" not in result.text


# --- 6. Custom-id rules carry score on retrieval -------------------------


@pytest.mark.asyncio
async def test_custom_id_rules_get_scores(mem_db, fake_embedder):
    """Rules with consumer-supplied ids should still get scores populated."""
    grammar = Grammar.from_dict({"origin": ["just one #ent#."]})
    await load_grammar_bulk(mem_db, "ci", grammar, fake_embedder)
    await add_rule(
        mem_db,
        "ci",
        "ent",
        "audited defi protocols",
        fake_embedder,
        id="applicant/security/uuid-audit",
    )
    await add_rule(
        mem_db,
        "ci",
        "ent",
        "found critical bugs",
        fake_embedder,
        id="applicant/security/uuid-bugs",
    )

    qv = await fake_embedder.create("audit")
    retriever = CosineRetriever()
    out = await retriever.search(mem_db, "ci", "ent", "audit", top_k=2, query_vector=qv)
    for r in out:
        assert r.score is not None
