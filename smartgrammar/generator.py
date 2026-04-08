"""Cascading context-aware grammar expansion."""

import logging
import random

import numpy as np

from smartgrammar.grammar import Grammar
from smartgrammar.index import GrammarIndex
from smartgrammar.embedder import Embedder

logger = logging.getLogger(__name__)


class Generator:
    """Generate text from an indexed grammar using cascading embedding search."""

    def __init__(self, grammar_index: GrammarIndex):
        self.gi = grammar_index
        self.embedder = grammar_index.embedder

    def generate(
        self,
        context: str = "",
        origin: str = "origin",
        temperature: float = 0.3,
        seed: int | None = None,
        top_k: int = 5,
    ) -> str:
        """Generate text by cascading context through the grammar tree.

        Args:
            context: Initial context string for embedding-based selection.
            origin: Starting rule name (default: "origin").
            temperature: 0.0 = always top-1, 1.0 = uniform random from top-k.
            seed: Random seed for reproducibility.
            top_k: Number of candidates to consider at each level.

        Returns:
            Generated text string.
        """
        rng = random.Random(seed)
        return self._expand(origin, context, temperature, rng, top_k, depth=0)

    def _expand(
        self,
        rule: str,
        context: str,
        temperature: float,
        rng: random.Random,
        top_k: int,
        depth: int,
    ) -> str:
        """Recursively expand a rule with cascading context."""
        if depth > 20:  # safety limit
            return f"[{rule}]"

        expansions = self.gi.grammar.get_expansions(rule)
        if not expansions:
            return f"[{rule}]"

        # Select an expansion
        chosen = self._select(rule, context, expansions, temperature, rng, top_k)

        # Expand any rule references in the chosen template
        result = chosen
        refs = Grammar.extract_refs(chosen)

        for ref in refs:
            # Cascade: context grows with what we've resolved so far
            cascaded_context = f"{context} {result}" if context else result
            sub_expansion = self._expand(ref, cascaded_context, temperature, rng, top_k, depth + 1)
            result = result.replace(f"#{ref}#", sub_expansion, 1)

        return result

    def _select(
        self,
        rule: str,
        context: str,
        expansions: list[str],
        temperature: float,
        rng: random.Random,
        top_k: int,
    ) -> str:
        """Select an expansion using embedding similarity + temperature."""
        if len(expansions) == 1:
            return expansions[0]

        # Temperature 1.0 = pure random (classic Tracery)
        if temperature >= 1.0:
            return rng.choice(expansions)

        # No context = random
        if not context:
            return rng.choice(expansions)

        # Embed context and query
        context_vec = self.embedder.embed([context])
        candidates = self.gi.query(rule, context_vec, top_k=top_k)

        if not candidates:
            return rng.choice(expansions)

        # Temperature 0.0 = always top-1
        if temperature <= 0.0:
            return candidates[0][0]

        # Temperature-weighted sampling from candidates
        texts = [c[0] for c in candidates]
        scores = np.array([c[1] for c in candidates])

        # Shift scores to be positive, then apply temperature
        scores = scores - scores.min() + 1e-6
        weights = np.exp(scores / temperature)
        weights = weights / weights.sum()

        idx = rng.choices(range(len(texts)), weights=weights.tolist(), k=1)[0]
        return texts[idx]
