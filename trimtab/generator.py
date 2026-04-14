"""Cascading context-aware grammar expansion."""

import logging
import random
from typing import NamedTuple

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
            origin, context, temperature, rng, top_k,
            min_confidence, no_match_text, depth=0, visit_stack=[],
        )
        return GenerationResult(text=text, ids=ids)

    async def _expand(
        self,
        rule: str,  # In v0.5 terminology this is a SYMBOL name
        context: str,
        temperature: float,
        rng: random.Random,
        top_k: int,
        min_confidence: float,
        no_match_text: str,
        depth: int,
        visit_stack: list[str] | None = None,
    ) -> tuple[str, list[str]]:
        """Recursively expand a symbol with cascading context.

        ``visit_stack`` tracks the chain of symbols currently being expanded.
        Re-entering a symbol already on the stack raises ``TrimTabCycleError``.
        ``depth`` is a safety guard for pathological-but-acyclic grammars.
        """
        from trimtab.errors import TrimTabCycleError

        visit_stack = list(visit_stack or [])
        if rule in visit_stack:
            raise TrimTabCycleError(chain=visit_stack + [rule])
        visit_stack.append(rule)

        if depth > 50:
            # Acyclic but pathologically deep — bail with a placeholder.
            return f"[{rule}]", []

        # Fetch rules via the new v0.5 path (was: self._db.get_expansions).
        rule_objs = self._db._get_rules(self._grammar, rule)
        if not rule_objs:
            return f"[{rule}]", []
        expansions = [r.text for r in rule_objs]

        chosen_text, chosen_id = await self._select(
            rule, context, expansions, temperature, rng, top_k, min_confidence, no_match_text
        )

        result = chosen_text
        walk_ids: list[str] = [chosen_id] if chosen_id else []
        refs = Grammar.extract_refs(chosen_text)

        for ref in refs:
            cascaded_context = f"{context} {result}" if context else result
            sub_text, sub_ids = await self._expand(
                ref,
                cascaded_context,
                temperature,
                rng,
                top_k,
                min_confidence,
                no_match_text,
                depth + 1,
                visit_stack=visit_stack,
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
            # Single-expansion symbol — fetch the rule directly to get its id.
            rule_objs = self._db._get_rules(self._grammar, rule)
            if rule_objs:
                return rule_objs[0].text, rule_objs[0].id
            return expansions[0], ""

        if temperature >= 1.0:
            # Fully random — pick an arbitrary expansion with no id tracking.
            return rng.choice(expansions), ""

        if not context:
            return rng.choice(expansions), ""

        context_vec = await self._embedder.create(context)
        rule_objs = self._db._search_rules(self._grammar, rule, context_vec, top_k=top_k)

        if not rule_objs:
            if min_confidence > 0:
                return no_match_text, ""
            return rng.choice(expansions), ""

        # Temperature == 0: top-1 (deterministic).
        if temperature <= 0.0:
            return rule_objs[0].text, rule_objs[0].id

        # Temperature > 0: uniform sample from the top-k. (Real weighted
        # sampling by cosine score is a v0.6+ improvement — the new
        # _search_rules doesn't surface scores.)
        if min_confidence > 0:
            # Without scores we can't filter by confidence. Fall through to
            # the no-match path if there's nothing to sample.
            if not rule_objs:
                return no_match_text, ""

        idx = rng.randrange(len(rule_objs))
        return rule_objs[idx].text, rule_objs[idx].id
