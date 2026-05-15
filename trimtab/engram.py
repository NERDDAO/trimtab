"""Inverse index over rule metadata entities.

Given a ``(grammar, symbol)`` pair whose rules carry
``metadata["entities"] = {label: [text, ...]}``, :class:`EngramIndex` exposes
two things:

* A **corpus-weight aggregate** — ``(label, normalized_text) → count`` — so
  recurring spans score higher than incidental mentions.
* A **query-time lookup** — ``{label: [query_text, ...]} → {rule_id: score}``
  — feeding directly into :meth:`trimtab.retriever.HybridRetriever.search`
  via the ``candidate_subset`` kwarg, or into ``auxiliary_rankings`` as a
  soft-boost ranking.

The index rebuilds on demand and caches in-process. It registers as an
invalidation listener against the underlying ``TrimTabDB`` so mutations to
the scanned ``(grammar, symbol)`` slice drop the cache on the next lookup.

Contract on rule metadata: producers write
``rule.metadata["entities"] = {label: [text, ...]}`` (e.g. via
:class:`trimtab.extract.EntityExtractor`). Rules without an ``entities``
key are silently ignored.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trimtab.db import TrimTabDB


def _normalize(text: str) -> str:
    return text.strip().lower()


class EngramIndex:
    """Inverse index + corpus aggregate for one ``(grammar, symbol)`` slice.

    Usage::

        index = EngramIndex(db, "chunks:locomo-v25a", "messages")
        index.build()  # scans rules, populates aggregate + inverse
        matches = index.lookup({"Preference": ["painting"]})
        # → {"rule-id-1": weight, "rule-id-7": weight, ...}

    Build-on-demand: calling :meth:`lookup` on a stale index rebuilds first.
    """

    def __init__(self, db: "TrimTabDB", grammar: str, symbol: str) -> None:
        self._db = db
        self._grammar = grammar
        self._symbol = symbol
        # (label, normalized_text) → corpus count
        self._aggregate: dict[tuple[str, str], int] = {}
        # (label, normalized_text) → set of rule ids containing that span
        self._inverse: dict[tuple[str, str], set[str]] = defaultdict(set)
        self._built = False
        # Register for invalidation so mutations trigger a rebuild next call.
        if hasattr(db, "register_invalidation_listener"):
            db.register_invalidation_listener(self._on_invalidate)

    def _on_invalidate(self, grammar: str, symbol: str | None) -> None:
        if grammar != self._grammar:
            return
        if symbol is not None and symbol != self._symbol:
            return
        self._built = False
        self._aggregate = {}
        self._inverse = defaultdict(set)

    def build(self) -> None:
        """Scan rules and populate aggregate + inverse. Idempotent."""
        self._aggregate = {}
        self._inverse = defaultdict(set)
        rules = self._db._get_rules(self._grammar, self._symbol)
        for rule in rules:
            entities = (rule.metadata or {}).get("entities")
            if not isinstance(entities, dict):
                continue
            seen_in_rule: set[tuple[str, str]] = set()
            for label, texts in entities.items():
                if not isinstance(texts, list):
                    continue
                for t in texts:
                    if not isinstance(t, str):
                        continue
                    key = (label, _normalize(t))
                    if not key[1]:
                        continue
                    # A span appearing multiple times in one rule counts once
                    # for that rule; corpus weight = number of rules that
                    # contain the span, not total raw occurrences.
                    if key in seen_in_rule:
                        continue
                    seen_in_rule.add(key)
                    self._aggregate[key] = self._aggregate.get(key, 0) + 1
                    self._inverse[key].add(rule.id)
        self._built = True

    def aggregate(self) -> dict[tuple[str, str], int]:
        """``{(label, normalized_text): count}`` — diagnostic view."""
        if not self._built:
            self.build()
        return dict(self._aggregate)

    def lookup(
        self,
        query_entities: dict[str, list[str]],
        *,
        min_weight: int = 1,
    ) -> dict[str, int]:
        """Return ``{rule_id: relevance_score}`` for rules that contain
        at least one query entity.

        ``relevance_score`` is the sum of corpus weights of every query
        span that matches on that rule — so a rule that matches multiple
        query entities, or a high-weight one, ranks higher.

        ``min_weight`` filters out spans whose corpus count is below the
        threshold (e.g. to suppress spans that appear in only one rule
        and therefore don't discriminate).
        """
        if not self._built:
            self.build()
        if not query_entities:
            return {}
        scores: dict[str, int] = {}
        for label, query_texts in query_entities.items():
            if not query_texts:
                continue
            for qt in query_texts:
                key = (label, _normalize(qt))
                if not key[1]:
                    continue
                weight = self._aggregate.get(key, 0)
                if weight < min_weight:
                    continue
                for rid in self._inverse.get(key, ()):
                    scores[rid] = scores.get(rid, 0) + weight
        return scores

    def ranked_rule_ids(
        self,
        query_entities: dict[str, list[str]],
        *,
        min_weight: int = 1,
    ) -> list[str]:
        """Sugar: return rule ids sorted by descending score.

        Handy when feeding ``auxiliary_rankings`` (soft-boost path) — the
        returned list is an RRF-ready ranked sequence.
        """
        matches = self.lookup(query_entities, min_weight=min_weight)
        ordered = sorted(matches.items(), key=lambda kv: -kv[1])
        return [rid for rid, _ in ordered]
