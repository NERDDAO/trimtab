"""Tests for the v24-Phase-2 AuxProvider hook on the cascade Generator.

Goal: prove the plumbing is byte-compatible when ``aux_provider`` is ``None``
(no new kwargs leak into the retriever call) and that the closure is wired
correctly when supplied (kwargs threaded, called with the right args at every
cascade level).

Phase 2 is pure plumbing — bench scores must be unchanged because no real
auxiliary signal is computed yet. These tests guard the contract.
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from tests.conftest import add_rule, load_grammar_bulk
from trimtab.generator import AuxProvider, Generator
from trimtab.grammar import Grammar, Rule
from trimtab.retriever import CosineRetriever


class _RecordingRetriever:
    """Minimal Retriever fake that records every ``search`` call.

    Captures positional args, kwargs (including which kwargs were *passed*
    vs simply absent), and returns whatever rules the underlying DB has so
    the generator can keep walking. Backed by ``CosineRetriever`` for the
    actual lookup so tests don't have to fabricate ranked rule lists.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self._inner = CosineRetriever()

    async def search(
        self,
        db,
        grammar: str,
        symbol: str,
        query: str,
        **kwargs,
    ) -> list[Rule]:
        self.calls.append(
            {
                "grammar": grammar,
                "symbol": symbol,
                "query": query,
                "kwargs": kwargs,  # full kwargs dict — exact set, not sentinels
            }
        )
        # Forward only the kwargs the inner retriever knows about so we get
        # real ranked results back. ``auxiliary_rankings`` and
        # ``candidate_subset`` are accepted by CosineRetriever (Phase 2) and
        # silently ignored.
        return await self._inner.search(
            db,
            grammar,
            symbol,
            query,
            top_k=kwargs["top_k"],
            query_vector=kwargs.get("query_vector"),
            auxiliary_rankings=kwargs.get("auxiliary_rankings"),
            candidate_subset=kwargs.get("candidate_subset"),
        )


@pytest_asyncio.fixture
async def two_level_grammar(mem_db, fake_embedder):
    grammar = Grammar.from_dict(
        {
            "origin": ["#mood# and #detail#."],
            "mood": ["dark and cold", "bright and warm", "eerie and still"],
            "detail": ["shadows move", "birds sing", "silence reigns"],
        }
    )
    await load_grammar_bulk(mem_db, "auxtest", grammar, fake_embedder)
    return mem_db, fake_embedder


# --- 1. No-provider path is byte-identical ---------------------------------


@pytest.mark.asyncio
async def test_generator_without_aux_provider_passes_no_aux_kwargs(two_level_grammar):
    """When ``aux_provider`` is omitted, ``Retriever.search`` must NOT receive
    ``auxiliary_rankings`` or ``candidate_subset`` kwargs at all — preserves
    byte-for-byte equivalence with the pre-v24 cascade walk."""
    db, embedder = two_level_grammar
    fake = _RecordingRetriever()

    gen = Generator(db, "auxtest", embedder, retriever=fake)
    await gen.generate(context="spooky cave", temperature=0.0, seed=42)

    assert fake.calls, "retriever.search was never called"
    for call in fake.calls:
        kwargs = call["kwargs"]
        assert isinstance(kwargs, dict)
        assert "auxiliary_rankings" not in kwargs, (
            f"unexpected auxiliary_rankings in no-provider call: {kwargs}"
        )
        assert "candidate_subset" not in kwargs, (
            f"unexpected candidate_subset in no-provider call: {kwargs}"
        )
        # Sanity: the historical kwargs are still threaded through.
        assert "top_k" in kwargs
        assert "query_vector" in kwargs


# --- 2. With-provider path threads the kwargs ------------------------------


@pytest.mark.asyncio
async def test_generator_with_aux_provider_threads_kwargs(two_level_grammar):
    """A stub ``AuxProvider`` returning ``([["r1","r2"]], None)`` must surface
    those values verbatim as ``auxiliary_rankings=`` / ``candidate_subset=``
    kwargs on every cascaded ``retriever.search`` call."""
    db, embedder = two_level_grammar
    fake = _RecordingRetriever()

    aux_call: list[list[str]] = [["r1", "r2"]]

    async def aux_provider(
        symbol: str,
        context: str,
        context_vec: list[float],
    ) -> tuple[list[list[str]] | None, list[str] | None]:
        return aux_call, None

    gen = Generator(db, "auxtest", embedder, retriever=fake, aux_provider=aux_provider)
    await gen.generate(context="spooky cave", temperature=0.0, seed=42)

    assert fake.calls, "retriever.search was never called"
    for call in fake.calls:
        kwargs = call["kwargs"]
        assert isinstance(kwargs, dict)
        # Both kwargs must be present (even when one is None).
        assert "auxiliary_rankings" in kwargs
        assert "candidate_subset" in kwargs
        assert kwargs["auxiliary_rankings"] == aux_call
        assert kwargs["candidate_subset"] is None


# --- 3. AuxProvider receives the right args at every cascade level ---------


@pytest.mark.asyncio
async def test_aux_provider_called_with_symbol_name_context_vec(two_level_grammar):
    """The closure is invoked at every cascaded symbol with that symbol's
    name, the cascaded context, and a non-None embedding vector."""
    db, embedder = two_level_grammar
    fake = _RecordingRetriever()

    seen: list[tuple[str, str, list[float]]] = []

    async def aux_provider(
        symbol: str,
        context: str,
        context_vec: list[float],
    ) -> tuple[list[list[str]] | None, list[str] | None]:
        seen.append((symbol, context, context_vec))
        return None, None

    gen = Generator(db, "auxtest", embedder, retriever=fake, aux_provider=aux_provider)
    await gen.generate(context="spooky cave", temperature=0.0, seed=42)

    # Two-level grammar walks at least origin → mood → detail (3 symbols).
    # ``_select`` only invokes the embedder/aux_provider when the symbol has
    # > 1 expansion AND a non-empty context — origin has only 1 expansion so
    # that level is skipped, but mood and detail have 3 each.
    seen_symbols = {sym for sym, _ctx, _vec in seen}
    assert "mood" in seen_symbols
    assert "detail" in seen_symbols

    for sym, ctx, vec in seen:
        assert isinstance(sym, str) and sym  # non-empty symbol name
        assert isinstance(ctx, str) and ctx  # non-empty cascaded context
        assert isinstance(vec, list)
        assert vec, "context_vec must be a non-empty list of floats"
        assert all(isinstance(x, float) for x in vec)


# --- 4. CosineRetriever silently accepts the new kwargs --------------------


@pytest.mark.asyncio
async def test_cosine_retriever_accepts_aux_kwargs_silently(mem_db, stub_embedder):
    """``CosineRetriever`` declares the new kwargs to satisfy the Protocol but
    must ignore them — output must equal the no-kwarg dense path."""
    await add_rule(mem_db, "g", "origin", "Caroline researches adoption", stub_embedder)
    await add_rule(mem_db, "g", "origin", "Melanie paints every morning", stub_embedder)
    await add_rule(mem_db, "g", "origin", "Caroline lives in Boston", stub_embedder)

    qv = await stub_embedder.create("Caroline")
    retriever = CosineRetriever()

    baseline = await retriever.search(
        mem_db,
        "g",
        "origin",
        "Caroline",
        top_k=3,
        query_vector=qv,
    )
    with_aux = await retriever.search(
        mem_db,
        "g",
        "origin",
        "Caroline",
        top_k=3,
        query_vector=qv,
        auxiliary_rankings=[["a", "b"]],
        candidate_subset=["x"],
    )

    assert [r.id for r in baseline] == [r.id for r in with_aux]


# --- 5. Type-alias is publicly importable ----------------------------------


def test_aux_provider_type_alias_is_importable():
    """``from trimtab.generator import AuxProvider`` must work for downstream
    consumers (e.g. delve's TrimtabGrammarController)."""
    assert AuxProvider is not None
