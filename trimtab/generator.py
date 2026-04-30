"""Cascading context-aware grammar expansion."""

import logging
import random
from collections.abc import Awaitable, Callable
from typing import NamedTuple

from trimtab.grammar import Grammar, Rule
from trimtab.embedder import Embedder
from trimtab.retriever import CosineRetriever, Retriever

logger = logging.getLogger(__name__)


# ``AuxProvider`` is the cascade-walk hook that lets callers inject per-symbol
# auxiliary rankings and/or candidate subsets into the retriever call. The
# closure is invoked by ``Generator._select`` once per cascaded symbol with
# ``(symbol_name, context, context_vec)``. Either element of the returned
# tuple may be ``None`` to opt-out of that signal for the current symbol.
#
# Phase 2 of v24 wires the plumbing; Phase 3+ light up real per-symbol bridge
# propagators on top of this contract. When ``aux_provider`` is ``None`` the
# generator preserves byte-identical behaviour with the pre-v24 cascade walk
# (no new kwargs threaded into ``Retriever.search``).
AuxProvider = Callable[
    [str, str, list[float]],
    Awaitable[tuple[list[list[str]] | None, list[str] | None]],
]


class GenerationResult(NamedTuple):
    """Result of a cascading grammar walk.

    ``text`` is the fully-expanded final string (same as the old return type).
    ``ids`` is the list of expansion ids the walk picked at each rule level,
    in walk order. Consumer code (e.g. a scoring agent) can filter these to
    extract external ids — LadybugDB auto-generates internal ids of the form
    ``{grammar}:{rule}:{hash(text)}``, so anything that doesn't match that
    shape is a consumer-supplied id (e.g. a KG entity UUID or path).
    ``rules_used`` carries the full ``Rule`` objects walked during generation,
    in walk order. Rules selected via random fallback paths (no DB lookup)
    contribute nothing here.
    """

    text: str
    ids: list[str]
    rules_used: list[Rule]  # full Rule objects walked, in order


class Generator:
    """Generate text from a grammar in TrimTabDB using cascading embedding search."""

    def __init__(
        self,
        db,
        grammar: str,
        embedder: Embedder,
        retriever: Retriever | None = None,
        aux_provider: AuxProvider | None = None,
    ):
        from trimtab.db import TrimTabDB

        self._db: TrimTabDB = db
        self._grammar = grammar
        self._embedder = embedder
        self._retriever: Retriever = retriever or CosineRetriever()
        # Phase 2 of v24: optional per-symbol AuxProvider for the cascade walk.
        # When ``None`` (the default), ``_select`` calls ``retriever.search``
        # with exactly the historical kwarg set — no auxiliary_rankings, no
        # candidate_subset — preserving byte-identical pre-v24 behaviour.
        self._aux_provider: AuxProvider | None = aux_provider

    async def generate(
        self,
        context: str = "",
        origin: str = "origin",
        temperature: float = 0.3,
        seed: int | None = None,
        top_k: int = 5,
        min_confidence: float = 0.0,
        no_match_text: str = "",
        descent_margin: float = 0.05,
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
            descent_margin: Tolerance budget [0.0, 2.0] for stopping the
                walk at the parent symbol instead of descending into a
                ``#child#`` ref. Larger margin = more aggressive stopping.
                Concretely: ``stop iff parent_score - child_score > 1.0
                - descent_margin``. When the required-improvement budget
                (``1.0 - margin``) is exceeded by the actual cosine drop,
                the child is deemed "not worth descending into" and the
                literal ``#child#`` token is left in the output.

                Edge cases:
                * ``descent_margin == 0.0``: required budget is 1.0, which
                  exceeds almost any plausible gap → always descend
                  (back-compat with pre-walk-stop behaviour).
                * ``descent_margin >= 2.0``: required budget <= -1.0,
                  always stop (useful for testing).
                * Missing scores (random fallback, no-context paths) →
                  always descend, regardless of margin (back-compat).

                Default ``0.05`` is intentionally near-zero; in practice
                cosine drops of 0.95+ never happen so the default lets
                every walk descend. Tune up to 0.10–0.30 for a gentle
                cluster→entity guard with cosine retrievers. RRF scores
                from ``HybridRetriever`` are tiny (max ~0.016 per lane)
                so any non-trivial RRF gap easily clears the budget —
                the same margin is more aggressive under hybrid.

        Returns:
            ``GenerationResult(text, ids)`` where ``ids`` are the expansion
            ids walked through during the cascade. Empty rules and no-match
            fallbacks contribute an empty id for that level.
        """
        rng = random.Random(seed)
        text, ids, rules_used = await self._expand(
            origin,
            context,
            temperature,
            rng,
            top_k,
            min_confidence,
            no_match_text,
            descent_margin,
            depth=0,
            visit_stack=[],
        )
        return GenerationResult(text=text, ids=ids, rules_used=rules_used)

    async def _expand(
        self,
        rule: str,  # In v0.5 terminology this is a SYMBOL name
        context: str,
        temperature: float,
        rng: random.Random,
        top_k: int,
        min_confidence: float,
        no_match_text: str,
        descent_margin: float,
        depth: int,
        visit_stack: list[str] | None = None,
        preselected: tuple[str, str, Rule | None, float | None] | None = None,
    ) -> tuple[str, list[str], list[Rule]]:
        """Recursively expand a symbol with cascading context.

        ``visit_stack`` tracks the chain of symbols currently being expanded.
        Re-entering a symbol already on the stack raises ``TrimTabCycleError``.
        ``depth`` is a safety guard for pathological-but-acyclic grammars.

        ``preselected`` lets the caller skip the inner ``_select`` call when
        a pick was already made (e.g. by the walk-stop peek in the parent's
        recursion loop). When ``None``, ``_expand`` does its own ``_select``.
        """
        from trimtab.errors import TrimTabCycleError

        visit_stack = list(visit_stack or [])
        if rule in visit_stack:
            raise TrimTabCycleError(chain=visit_stack + [rule])
        visit_stack.append(rule)

        if depth > 50:
            # Acyclic but pathologically deep — bail with a placeholder.
            return f"[{rule}]", [], []

        if preselected is not None:
            chosen_text, chosen_id, chosen_rule, _chosen_score = preselected
        else:
            # Fetch rules via the new v0.5 path (was: self._db.get_expansions).
            rule_objs = self._db._get_rules(self._grammar, rule)
            if not rule_objs:
                return f"[{rule}]", [], []
            expansions = [r.text for r in rule_objs]

            chosen_text, chosen_id, chosen_rule, _chosen_score = await self._select(
                rule,
                context,
                expansions,
                temperature,
                rng,
                top_k,
                min_confidence,
                no_match_text,
            )

        # Score for this symbol's chosen rule (used by the walk-stop check
        # before recursing into each child ref).
        parent_score: float | None = (
            chosen_rule.score if chosen_rule is not None else None
        )

        result = chosen_text
        walk_ids: list[str] = [chosen_id] if chosen_id else []
        walk_rules: list[Rule] = [chosen_rule] if chosen_rule is not None else []
        refs = Grammar.extract_refs(chosen_text)

        for ref in refs:
            cascaded_context = f"{context} {result}" if context else result

            # Peek the child's best candidate so we can decide whether to
            # descend. If the child symbol has no rules / hits a fallback
            # path, the peek returns score=None and we fall through to the
            # historical "always descend" behaviour (back-compat).
            child_pick = await self._peek_select(
                ref,
                cascaded_context,
                temperature,
                rng,
                top_k,
                min_confidence,
                no_match_text,
            )
            child_score = (
                child_pick[2].score
                if child_pick is not None and child_pick[2] is not None
                else None
            )

            # Walk-stop guard: only fires when both parent + child have real
            # scores AND the child underperforms the parent enough that the
            # ``descent_margin`` "tolerance budget" is exceeded.
            #
            # Formula: ``stop iff parent_score - child_score > 1.0 - descent_margin``
            #
            # Equivalent restatement: ``stop iff child_score < parent_score
            # - (1.0 - descent_margin)``. The ``(1.0 - margin)`` term is the
            # required-improvement budget — child must be within that many
            # cosine points of the parent to descend.
            #
            # Edge cases:
            # * ``descent_margin == 0.0`` → required budget is 1.0, which
            #   almost always exceeds any cosine-score gap, so the walk
            #   always descends (back-compat with pre-walk-stop behaviour).
            # * ``descent_margin >= 2.0`` → required budget is <= -1.0, less
            #   than the minimum possible cosine gap, so the walk always
            #   stops (useful for testing / aggressive truncation).
            # * Either score is ``None`` (random fallback path) → falls
            #   through to "always descend" (back-compat semantic).
            if (
                parent_score is not None
                and child_score is not None
                and (parent_score - child_score) > (1.0 - descent_margin)
            ):
                logger.info(
                    "walk stopped at symbol=%s (parent=%.4f, best_child=%.4f, margin=%.4f)",
                    ref,
                    parent_score,
                    child_score,
                    descent_margin,
                )
                # Leave the literal #ref# in the output. walk_ids / walk_rules
                # are NOT extended — the truncation is observable downstream.
                continue

            if child_pick is None:
                # Child had no rules at all — recurse with no preselection so
                # we surface the f"[{ref}]" placeholder and depth-guard the same
                # way we always have.
                sub_text, sub_ids, sub_rules = await self._expand(
                    ref,
                    cascaded_context,
                    temperature,
                    rng,
                    top_k,
                    min_confidence,
                    no_match_text,
                    descent_margin,
                    depth + 1,
                    visit_stack=visit_stack,
                )
            else:
                # Reuse the peek's selection — avoid a duplicate retriever call.
                sub_text, sub_ids, sub_rules = await self._expand(
                    ref,
                    cascaded_context,
                    temperature,
                    rng,
                    top_k,
                    min_confidence,
                    no_match_text,
                    descent_margin,
                    depth + 1,
                    visit_stack=visit_stack,
                    preselected=child_pick,
                )
            result = result.replace(f"#{ref}#", sub_text, 1)
            walk_ids.extend(sub_ids)
            walk_rules.extend(sub_rules)

        return result, walk_ids, walk_rules

    async def _peek_select(
        self,
        symbol: str,
        context: str,
        temperature: float,
        rng: random.Random,
        top_k: int,
        min_confidence: float,
        no_match_text: str,
    ) -> tuple[str, str, Rule | None, float | None] | None:
        """Run a ``_select`` against ``symbol`` for the walk-stop peek.

        Returns the same 4-tuple ``_select`` does, or ``None`` if the symbol
        has no rules at all (so the caller can fall back to its historical
        placeholder path).
        """
        rule_objs = self._db._get_rules(self._grammar, symbol)
        if not rule_objs:
            return None
        expansions = [r.text for r in rule_objs]
        return await self._select(
            symbol,
            context,
            expansions,
            temperature,
            rng,
            top_k,
            min_confidence,
            no_match_text,
        )

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
    ) -> tuple[str, str, Rule | None, float | None]:
        """Select an expansion using embedding similarity + temperature.

        Returns ``(text, id, rule, score)`` — the chosen expansion text,
        its id from the DB, the full ``Rule`` object (or ``None`` for
        random / no-context fallback paths that don't go through a DB rule
        lookup), and the rule's transient retrieval score (or ``None`` for
        fallback paths). The score is the raw cosine for ``CosineRetriever``,
        the fused RRF score for ``HybridRetriever``, or the cross-encoder
        score when one is configured.
        """
        if len(expansions) == 1:
            # Single-expansion symbol — fetch the rule directly to get its id.
            rule_objs = self._db._get_rules(self._grammar, rule)
            if rule_objs:
                # Single-rule symbols normally bypass the retriever, but the
                # walk-stop guard needs a real score for the parent vs.
                # child comparison to fire. When we have a non-empty
                # ``context`` (and a non-fully-random ``temperature``),
                # route through ``_search_rules`` so we get a cosine score
                # against the lone rule. When there's no context (or
                # temperature is fully random) the historical
                # ``score=None`` sentinel stands and the walk-stop guard
                # falls through to the back-compat "always descend" path.
                if context and temperature < 1.0:
                    context_vec = await self._embedder.create(context)
                    scored = self._db._search_rules(
                        self._grammar, rule, context_vec, top_k=1
                    )
                    if scored:
                        top = scored[0]
                        return top.text, top.id, top, top.score
                return rule_objs[0].text, rule_objs[0].id, rule_objs[0], None
            return expansions[0], "", None, None

        if temperature >= 1.0:
            # Fully random — pick an arbitrary expansion with no id tracking.
            return rng.choice(expansions), "", None, None

        if not context:
            return rng.choice(expansions), "", None, None

        context_vec = await self._embedder.create(context)

        if self._aux_provider is None:
            # Pre-v24 path — do NOT pass the new kwargs at all so byte-for-byte
            # call equivalence is preserved (matters for fakes that assert on
            # exact kwarg sets, and for HybridRetriever's cache-hit path).
            rule_objs = await self._retriever.search(
                self._db,
                self._grammar,
                rule,
                context,
                top_k=top_k,
                query_vector=context_vec,
            )
        else:
            aux_rankings, cand_subset = await self._aux_provider(
                rule, context, context_vec
            )
            rule_objs = await self._retriever.search(
                self._db,
                self._grammar,
                rule,
                context,
                top_k=top_k,
                query_vector=context_vec,
                auxiliary_rankings=aux_rankings,
                candidate_subset=cand_subset,
            )

        if not rule_objs:
            if min_confidence > 0:
                return no_match_text, "", None, None
            return rng.choice(expansions), "", None, None

        # Temperature == 0: top-1 (deterministic).
        if temperature <= 0.0:
            top = rule_objs[0]
            return top.text, top.id, top, top.score

        # Temperature > 0: uniform sample from the top-k. (Real weighted
        # sampling by cosine score is a v0.6+ improvement — the new
        # _search_rules doesn't surface scores.)
        if min_confidence > 0:
            # Without scores we can't filter by confidence. Fall through to
            # the no-match path if there's nothing to sample.
            if not rule_objs:
                return no_match_text, "", None, None

        idx = rng.randrange(len(rule_objs))
        picked = rule_objs[idx]
        return picked.text, picked.id, picked, picked.score
