"""Cascading context-aware grammar expansion."""

import logging
import random
from typing import NamedTuple

import numpy as np

from trimtab.grammar import Grammar
from trimtab.embedder import Embedder

logger = logging.getLogger(__name__)


class GenerationResult(NamedTuple):
    """Result of a cascading grammar walk.

    ``text`` is the fully-expanded final string (same as the old return type).
    ``ids`` is the list of expansion ids the walk picked at each rule level,
    in walk order. Consumer code (e.g. a scoring agent) can filter these to
    extract external ids — LadybugDB auto-generates internal ids of the form
    ``{grammar}:{rule}:{hash(text)}``, so anything that doesn't match that
    shape is a consumer-supplied id (e.g. a KG entity UUID or path).
    """

    text: str
    ids: list[str]


class Generator:
    """Generate text from a grammar in TrimTabDB using cascading embedding search."""

    def __init__(self, db, grammar: str, embedder: Embedder):
        from trimtab.db import TrimTabDB

        self._db: TrimTabDB = db
        self._grammar = grammar
        self._embedder = embedder

    async def generate(
        self,
        context: str = "",
        origin: str = "origin",
        temperature: float = 0.3,
        seed: int | None = None,
        top_k: int = 5,
        min_confidence: float = 0.0,
        no_match_text: str = "",
    ) -> GenerationResult:
        """Generate text by cascading context through the grammar tree.

        Returns a ``GenerationResult`` with both the final expanded ``text``
        and the list of ``ids`` picked at each rule level. Downstream agents
        can use the ids as center-node anchors for parallel KG search, or
        as structured handles back to the external entities they point at.

        Args:
            context: Initial context string for embedding-based selection.
            origin: Starting rule name (default: "origin").
            temperature: 0.0 = always top-1, 1.0 = uniform random from top-k.
            seed: Random seed for reproducibility.
            top_k: Number of candidates to consider at each level.
            min_confidence: Minimum cosine similarity score to accept a match.
            no_match_text: Text to use when no candidate meets min_confidence.

        Returns:
            ``GenerationResult(text, ids)`` where ``ids`` are the expansion
            ids walked through during the cascade. Empty rules and no-match
            fallbacks contribute an empty id for that level.
        """
        rng = random.Random(seed)
        text, ids = await self._expand(
            origin, context, temperature, rng, top_k, min_confidence, no_match_text, depth=0
        )
        return GenerationResult(text=text, ids=ids)

    async def _expand(
        self,
        rule: str,
        context: str,
        temperature: float,
        rng: random.Random,
        top_k: int,
        min_confidence: float,
        no_match_text: str,
        depth: int,
    ) -> tuple[str, list[str]]:
        """Recursively expand a rule with cascading context.

        Returns ``(expanded_text, ids)`` where ``ids`` is the flat list of
        expansion ids picked at this level and every level below (in walk
        order).
        """
        if depth > 20:
            return f"[{rule}]", []

        expansions = self._db.get_expansions(self._grammar, rule)
        if not expansions:
            return f"[{rule}]", []

        chosen_text, chosen_id = await self._select(
            rule, context, expansions, temperature, rng, top_k, min_confidence, no_match_text
        )

        result = chosen_text
        walk_ids: list[str] = [chosen_id] if chosen_id else []
        refs = Grammar.extract_refs(chosen_text)

        for ref in refs:
            cascaded_context = f"{context} {result}" if context else result
            sub_text, sub_ids = await self._expand(
                ref, cascaded_context, temperature, rng, top_k, min_confidence, no_match_text, depth + 1
            )
            result = result.replace(f"#{ref}#", sub_text, 1)
            walk_ids.extend(sub_ids)

        return result, walk_ids

    async def _select(
        self,
        rule: str,
        context: str,
        expansions: list[str],
        temperature: float,
        rng: random.Random,
        top_k: int,
        min_confidence: float = 0.0,
        no_match_text: str = "",
    ) -> tuple[str, str]:
        """Select an expansion using embedding similarity + temperature.

        Returns ``(text, id)`` — the chosen expansion text and its id from
        the DB. For single-expansion rules the DB still owns the id (looked
        up via a small query). For random / no-context fallbacks we return
        an empty id because we never touched the DB's id mapping.
        """
        if len(expansions) == 1:
            # Single-expansion rule — we still need the id. Cheapest path is
            # a top-1 vector query; a zero vector returns the single row.
            try:
                zero_vec = [0.0] * (self._embedding_dim_hint() or 1)
                candidates = self._db.query(self._grammar, rule, zero_vec, top_k=1)
                if candidates:
                    return candidates[0][0], candidates[0][2]
            except Exception:
                pass
            return expansions[0], ""

        if temperature >= 1.0:
            # Fully random — pick an arbitrary expansion with no id tracking.
            return rng.choice(expansions), ""

        if not context:
            return rng.choice(expansions), ""

        context_vec = await self._embedder.create(context)
        candidates = self._db.query(self._grammar, rule, context_vec, top_k=top_k)

        if not candidates:
            if min_confidence > 0:
                return no_match_text, ""
            return rng.choice(expansions), ""

        best_score = candidates[0][1]
        if min_confidence > 0 and best_score < min_confidence:
            return no_match_text, ""

        if temperature <= 0.0:
            return candidates[0][0], candidates[0][2]

        if min_confidence > 0:
            candidates = [(t, s, i) for t, s, i in candidates if s >= min_confidence]
            if not candidates:
                return no_match_text, ""

        texts = [c[0] for c in candidates]
        ids = [c[2] for c in candidates]
        scores = np.array([c[1] for c in candidates])

        scores = scores - scores.min() + 1e-6
        weights = np.exp(scores / temperature)
        weights = weights / weights.sum()

        idx = rng.choices(range(len(texts)), weights=weights.tolist(), k=1)[0]
        return texts[idx], ids[idx]

    def _embedding_dim_hint(self) -> int | None:
        """Best-effort lookup of the DB's embedding dimension for zero-vector queries."""
        return getattr(self._db, "_embedding_dim", None)
