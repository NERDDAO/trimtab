"""Pluggable retrieval strategies for rule search.

A ``Retriever`` is called by ``TrimTab.search`` and by the cascade generator
to rank rules within a ``(grammar, symbol)`` slice.

Two implementations ship:

* ``CosineRetriever`` — the historical pure-dense path. Delegates to
  ``TrimTabDB._search_rules`` (HNSW + brute-force cosine fallback).
* ``HybridRetriever`` — Engram-inspired dense + sparse (BM25) + Reciprocal
  Rank Fusion. Dense candidates come from the same ``_search_rules`` call
  today; the BM25 index is built lazily from rule text stored in LadybugDB
  and invalidated when the DB mutates.

No cross-encoder rerank yet — that's a deferred phase if hybrid alone isn't
enough on the LoCoMo smokes.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from trimtab.grammar import Rule

if TYPE_CHECKING:
    from trimtab.db import TrimTabDB


class Retriever(Protocol):
    """Ranks rules under a ``(grammar, symbol)`` for a given query."""

    async def search(
        self,
        db: "TrimTabDB",
        grammar: str,
        symbol: str,
        query: str,
        *,
        top_k: int,
        query_vector: list[float] | None = None,
        auxiliary_rankings: list[list[str]] | None = None,
        candidate_subset: list[str] | None = None,
    ) -> list[Rule]:
        """Return up to ``top_k`` rules ranked by relevance (best first).

        ``query_vector`` is supplied by the caller so the embedding cost is
        paid once at the cascade level rather than per retriever call.
        Implementations that need a vector (dense, hybrid) must accept it.

        ``auxiliary_rankings`` and ``candidate_subset`` are optional inputs
        for fusion-capable retrievers (see :class:`HybridRetriever`). Dense-
        only implementations must accept the kwargs to satisfy this Protocol
        but may silently ignore them.
        """
        ...


class CosineRetriever:
    """Dense cosine + optional auxiliary RRF + optional candidate subset filter.

    Default cascade-walk retriever: lighter weight than :class:`HybridRetriever`
    (no BM25, no cross-encoder rerank) but still consumes the v24 P2 fusion
    kwargs so callers can inject auxiliary signals from the unified retrieval
    kernel without paying HybridRetriever's per-step fan-out cost.

    Behaviour matrix:

    * ``auxiliary_rankings is None`` (or empty) **and** ``candidate_subset is
      None``: byte-equivalent to the pre-v24 pure-dense path — delegates to
      ``TrimTabDB._search_rules``.
    * ``candidate_subset`` non-empty: dense candidates are filtered to the
      subset before any fusion. Unknown ids in the subset are silently dropped
      (caller owns the validity contract).
    * ``candidate_subset`` empty (``[]``): short-circuits to ``[]`` — same
      semantics as :class:`HybridRetriever` (no fallback to full pool).
    * ``auxiliary_rankings`` non-empty: RRF-fused with the dense ranking using
      the same default ``k=60`` constant as :class:`HybridRetriever`.

    Cross-encoder rerank stays a HybridRetriever-only concern by design — the
    cascade default needs to keep latency low across many ``_expand`` steps.
    """

    def __init__(self, candidate_multiplier: int = 4, rrf_k: int = 60):
        self._candidate_multiplier = candidate_multiplier
        self._rrf_k = rrf_k

    async def search(
        self,
        db: "TrimTabDB",
        grammar: str,
        symbol: str,
        query: str,
        *,
        top_k: int,
        query_vector: list[float] | None = None,
        auxiliary_rankings: list[list[str]] | None = None,
        candidate_subset: list[str] | None = None,
    ) -> list[Rule]:
        """Dense cosine top-K, optionally fused with auxiliary rankings via RRF.

        See class docstring for the full behaviour matrix. Implementation note:
        when neither fusion kwarg is in play we take the same code path the
        pre-v24 default did, so the bytes-equivalence regression bench stays
        green.
        """
        if query_vector is None:
            raise ValueError("CosineRetriever requires a precomputed query_vector")

        has_aux = bool(auxiliary_rankings)
        has_subset = candidate_subset is not None

        # Fast path: no fusion, no filter — preserve historical behaviour exactly.
        if not has_aux and not has_subset:
            return db._search_rules(grammar, symbol, query_vector, top_k)

        # Empty subset short-circuits to no results (mirrors HybridRetriever).
        if has_subset and not candidate_subset:
            return []

        # Pull a wider dense pool when we're about to fuse / filter so RRF has
        # something to work with after the subset trim.
        candidate_k = max(top_k * self._candidate_multiplier, top_k)
        dense_rules = db._search_rules(grammar, symbol, query_vector, candidate_k)
        rules_by_id: dict[str, Rule] = {r.id: r for r in dense_rules}
        dense_ids = [r.id for r in dense_rules]

        if has_subset:
            subset = set(candidate_subset or [])
            dense_ids = [i for i in dense_ids if i in subset]
            if has_aux and auxiliary_rankings is not None:
                auxiliary_rankings = [
                    [i for i in r if i in subset] for r in auxiliary_rankings
                ]

        # No auxiliary signal — dense ranking (filtered by subset) is the answer.
        # ``rules_by_id`` already carries the dense cosine score on each rule
        # from ``_search_rules``; no rewrite needed here.
        if not has_aux:
            return [rules_by_id[i] for i in dense_ids[:top_k] if i in rules_by_id]

        # Hydrate aux-only ids that dense didn't surface so the fused ranking
        # can promote them. Mirrors HybridRetriever, which hydrates from its
        # cached BM25 corpus state (the full rule pool under the slice).
        needed_ids: set[str] = set()
        for r in auxiliary_rankings or []:
            for rid in r:
                if rid not in rules_by_id:
                    needed_ids.add(rid)
        if needed_ids:
            for rule in db._get_rules(grammar, symbol):
                if rule.id in needed_ids:
                    rules_by_id[rule.id] = rule

        all_rankings: list[list[str]] = [dense_ids]
        if auxiliary_rankings:
            all_rankings.extend(r for r in auxiliary_rankings if r)

        fused = reciprocal_rank_fusion(all_rankings, k=self._rrf_k)

        out: list[Rule] = []
        for rid, fused_score in fused:
            rule = rules_by_id.get(rid)
            if rule is None:
                # Aux id that's unknown to the DB — caller owns the validity
                # contract; drop silently as the spec requires.
                continue
            # Overwrite the per-rule score with the fused RRF score so the
            # generator's walk-stop logic compares apples-to-apples across
            # the cascade (every level uses the same scoring family).
            rule.score = fused_score
            out.append(rule)
            if len(out) >= top_k:
                break
        return out


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenization.

    Matches rank-bm25's default expectation of a pre-tokenized corpus.
    Upgrade path: domain-specific tokenizer if proper nouns / multi-word
    entities need to be preserved as single tokens.
    """
    return _TOKEN_RE.findall(text.lower())


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """RRF over multiple ranked id lists. Returns (id, score) descending.

    Score contribution for an id at rank ``r`` in a given ranking is
    ``1 / (k + r)``. Missing from a ranking = zero contribution from it.
    ``k=60`` is the canonical default from Cormack et al. 2009.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, rid in enumerate(ranking, start=1):
            scores[rid] = scores.get(rid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class _BM25State:
    """Per-(grammar, symbol) cached BM25 index and rule hydration map."""

    __slots__ = ("rules_by_id", "corpus_ids", "bm25")

    def __init__(self, rules: list[Rule], bm25: Any):
        self.rules_by_id: dict[str, Rule] = {r.id: r for r in rules}
        self.corpus_ids: list[str] = [r.id for r in rules]
        self.bm25 = bm25


class CrossEncoder(Protocol):
    """Graphiti-compatible cross-encoder reranker.

    Any object exposing an async ``rank(query, passages) -> list[(passage,
    score)]`` works here. Matches ``graphiti_core.cross_encoder.client``'s
    protocol so delve's existing Voyage / BGE client instances can be passed
    in without adaptation.
    """

    async def rank(
        self, query: str, passages: list[str]
    ) -> list[tuple[str, float]]: ...


class HybridRetriever:
    """Dense + BM25 + Reciprocal Rank Fusion retrieval, optional rerank.

    Instantiate once per ``TrimTab`` / ``Generator``. The retriever registers
    a cache-invalidation listener with ``TrimTabDB`` on first search; any
    rule write through the DB's mutating helpers clears the affected slice.

    If ``cross_encoder`` is supplied, the RRF top-``rerank_pool`` is re-scored
    by the cross-encoder and the final top-K is derived from those scores.
    This rescues cases where BM25 promotes keyword-matched but semantically
    wrong rules into prominent positions — the classic "painted a sunrise"
    query pulling Melanie's painting rules when the question was about
    Caroline.
    """

    def __init__(
        self,
        candidate_multiplier: int = 4,
        rrf_k: int = 60,
        cross_encoder: CrossEncoder | None = None,
        rerank_pool: int = 20,
    ):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as e:  # pragma: no cover - dependency check
            raise ImportError(
                "HybridRetriever requires rank-bm25. "
                "Install with `pip install rank-bm25`."
            ) from e
        self._bm25_cls = BM25Okapi
        self._candidate_multiplier = candidate_multiplier
        self._rrf_k = rrf_k
        self._cross_encoder = cross_encoder
        self._rerank_pool = rerank_pool
        # Cache keyed by (grammar, symbol). Cleared by invalidate().
        self._cache: dict[tuple[str, str], _BM25State] = {}
        # Track which DBs we've already registered a listener with to avoid
        # double-registering when a single HybridRetriever is shared across
        # multiple TrimTab instances.
        self._registered_dbs: set[int] = set()

    def invalidate(self, grammar: str, symbol: str | None = None) -> None:
        """Drop the cached BM25 state for a grammar (or a specific symbol).

        ``symbol=None`` clears every cached slice under the grammar.
        """
        if symbol is None:
            self._cache = {k: v for k, v in self._cache.items() if k[0] != grammar}
        else:
            self._cache.pop((grammar, symbol), None)

    def _ensure_listener(self, db: "TrimTabDB") -> None:
        db_id = id(db)
        if db_id in self._registered_dbs:
            return
        if hasattr(db, "register_invalidation_listener"):
            db.register_invalidation_listener(self.invalidate)
            self._registered_dbs.add(db_id)

    def _state_for(
        self, db: "TrimTabDB", grammar: str, symbol: str
    ) -> _BM25State | None:
        self._ensure_listener(db)
        key = (grammar, symbol)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        # The Rule table is created lazily on the first put; until then,
        # _get_rules raises. Mirror _search_rules's early-return and treat
        # that case as empty.
        if db._embedding_dim is None:
            return None
        rules = db._get_rules(grammar, symbol)
        if not rules:
            return None
        corpus_tokens = [_tokenize(r.text) for r in rules]
        if not any(corpus_tokens):
            # All rule texts tokenize to empty — BM25 can't help here.
            return None
        bm25 = self._bm25_cls(corpus_tokens)
        state = _BM25State(rules, bm25)
        self._cache[key] = state
        return state

    async def search(
        self,
        db: "TrimTabDB",
        grammar: str,
        symbol: str,
        query: str,
        *,
        top_k: int,
        query_vector: list[float] | None = None,
        auxiliary_rankings: list[list[str]] | None = None,
        candidate_subset: list[str] | None = None,
    ) -> list[Rule]:
        """Dense + BM25 + optional auxiliary rankings, fused via RRF.

        ``auxiliary_rankings`` — caller-computed ranked rule-id lists that
        represent additional signals (e.g. person-name hits, temporal
        proximity, engram-propagation from a sibling grammar). Each ranking
        is RRF-fused alongside dense + sparse. Empty list or ``None`` falls
        back to pure dense+BM25 behaviour. Unknown ids are silently dropped
        during rule hydration — callers own the validity contract.

        ``candidate_subset`` — optional pre-flight filter. When provided,
        every input ranking (dense, sparse, auxiliary) is post-filtered to
        ids in the subset before RRF fusion. Lets callers narrow retrieval
        to a query-relevant shortlist (e.g. from :class:`~trimtab.engram.
        EngramIndex.lookup`) while keeping the full scoring stack. An empty
        subset yields an empty result — callers own fallback to full-pool.
        """
        if query_vector is None:
            raise ValueError("HybridRetriever requires a precomputed query_vector")

        candidate_k = max(top_k * self._candidate_multiplier, top_k)

        dense_rules = db._search_rules(grammar, symbol, query_vector, candidate_k)
        dense_ids = [r.id for r in dense_rules]

        state = self._state_for(db, grammar, symbol)
        if state is None:
            # Nothing under (grammar, symbol) or all-empty corpus — dense is
            # the authoritative answer (scores already populated by
            # ``_search_rules``).
            return dense_rules[:top_k]

        query_tokens = _tokenize(query)
        sparse_ids: list[str] = []
        if query_tokens:
            # BM25Okapi.get_scores is pure-Python+numpy CPU-bound — wrap in
            # to_thread so it doesn't block the asyncio event loop. Without
            # this, parallel TaskGroup lanes that need to compute BM25
            # serialize on the GIL/main thread, costing ~3s of overhead
            # per request when chunks/kg lanes contend for it.
            import asyncio

            scores = await asyncio.to_thread(state.bm25.get_scores, query_tokens)
            ranked_indices = np.argsort(-scores)[:candidate_k]
            sparse_ids = [
                state.corpus_ids[i] for i in ranked_indices if scores[i] > 0.0
            ]

        if candidate_subset is not None:
            subset = set(candidate_subset)
            if not subset:
                return []
            dense_ids = [i for i in dense_ids if i in subset]
            sparse_ids = [i for i in sparse_ids if i in subset]
            if auxiliary_rankings:
                auxiliary_rankings = [
                    [i for i in r if i in subset] for r in auxiliary_rankings
                ]

        all_rankings: list[list[str]] = [dense_ids, sparse_ids]
        if auxiliary_rankings:
            all_rankings.extend(r for r in auxiliary_rankings if r)

        fused = reciprocal_rank_fusion(all_rankings, k=self._rrf_k)

        dense_by_id = {r.id: r for r in dense_rules}
        # Hydrate fused rule ids into Rule objects, keeping fused order for now.
        # Overwrite each rule's transient ``score`` with the fused RRF score so
        # the cascade walk-stop logic can compare across cascade levels with
        # one consistent scoring family.
        fused_rules: list[Rule] = []
        for rid, fused_score in fused:
            rule = dense_by_id.get(rid) or state.rules_by_id.get(rid)
            if rule is None:
                continue
            rule.score = fused_score
            fused_rules.append(rule)
            if len(fused_rules) >= self._rerank_pool:
                break

        if self._cross_encoder is None or len(fused_rules) <= 1:
            return fused_rules[:top_k]

        # Cross-encoder rerank: score each (query, rule.text) pair jointly.
        # Rule texts can repeat across grammars in theory, so we match by
        # list-index rather than text equality after rank() returns.
        passages = [r.text for r in fused_rules]
        try:
            scored = await self._cross_encoder.rank(query, passages)
        except Exception:
            # If rerank fails (network, model, OOM), fall back to RRF order.
            return fused_rules[:top_k]
        # Graphiti's CrossEncoderClient returns (passage, score) tuples sorted
        # by score desc, keyed by the passage string. Reassociate back to
        # the rule objects, preserving first occurrence on duplicate texts.
        text_to_rule: dict[str, Rule] = {}
        for r in fused_rules:
            text_to_rule.setdefault(r.text, r)
        reranked: list[Rule] = []
        seen: set[str] = set()
        for passage, ce_score in scored:
            rule = text_to_rule.get(passage)
            if rule is None or rule.id in seen:
                continue
            # Cross-encoder score is now the authoritative final score for
            # cascade-walk decisions when rerank is enabled.
            rule.score = float(ce_score)
            seen.add(rule.id)
            reranked.append(rule)
            if len(reranked) >= top_k:
                break
        return reranked or fused_rules[:top_k]
