"""Cascading context-aware grammar expansion."""

import logging
import random

import numpy as np

from trimtab.grammar import Grammar
from trimtab.embedder import Embedder

logger = logging.getLogger(__name__)


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
    ) -> str:
        """Generate text by cascading context through the grammar tree.

        Args:
            context: Initial context string for embedding-based selection.
            origin: Starting rule name (default: "origin").
            temperature: 0.0 = always top-1, 1.0 = uniform random from top-k.
            seed: Random seed for reproducibility.
            top_k: Number of candidates to consider at each level.
            min_confidence: Minimum cosine similarity score to accept a match.
            no_match_text: Text to use when no candidate meets min_confidence.

        Returns:
            Generated text string.
        """
        rng = random.Random(seed)
        return await self._expand(origin, context, temperature, rng, top_k, min_confidence, no_match_text, depth=0)

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
    ) -> str:
        """Recursively expand a rule with cascading context."""
        if depth > 20:
            return f"[{rule}]"

        expansions = self._db.get_expansions(self._grammar, rule)
        if not expansions:
            return f"[{rule}]"

        chosen = await self._select(rule, context, expansions, temperature, rng, top_k, min_confidence, no_match_text)

        result = chosen
        refs = Grammar.extract_refs(chosen)

        for ref in refs:
            cascaded_context = f"{context} {result}" if context else result
            sub_expansion = await self._expand(ref, cascaded_context, temperature, rng, top_k, min_confidence, no_match_text, depth + 1)
            result = result.replace(f"#{ref}#", sub_expansion, 1)

        return result

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
    ) -> str:
        """Select an expansion using embedding similarity + temperature."""
        if len(expansions) == 1:
            return expansions[0]

        if temperature >= 1.0:
            return rng.choice(expansions)

        if not context:
            return rng.choice(expansions)

        context_vec = await self._embedder.create(context)
        candidates = self._db.query(self._grammar, rule, context_vec, top_k=top_k)

        if not candidates:
            return no_match_text if min_confidence > 0 else rng.choice(expansions)

        best_score = candidates[0][1]
        if min_confidence > 0 and best_score < min_confidence:
            return no_match_text

        if temperature <= 0.0:
            return candidates[0][0]

        if min_confidence > 0:
            candidates = [(t, s, i) for t, s, i in candidates if s >= min_confidence]
            if not candidates:
                return no_match_text

        texts = [c[0] for c in candidates]
        scores = np.array([c[1] for c in candidates])

        scores = scores - scores.min() + 1e-6
        weights = np.exp(scores / temperature)
        weights = weights / weights.sum()

        idx = rng.choices(range(len(texts)), weights=weights.tolist(), k=1)[0]
        return texts[idx]
