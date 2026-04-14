"""Error hierarchy for trimtab.

All errors subclass ``TrimTabError`` so callers can ``except TrimTabError``
once and recover.
"""

from __future__ import annotations


class TrimTabError(Exception):
    """Base class for all trimtab errors. Never raised directly."""


class TrimTabEmbedderError(TrimTabError):
    """Embedder unreachable, model missing, or embed call failed."""


class TrimTabNotFoundError(TrimTabError):
    """Rule id was referenced on update/remove but does not exist."""

    def __init__(self, grammar: str, symbol: str, rule_id: str) -> None:
        self.grammar = grammar
        self.symbol = symbol
        self.rule_id = rule_id
        super().__init__(
            f"Rule not found: grammar={grammar!r} symbol={symbol!r} id={rule_id!r}"
        )


class TrimTabDimensionError(TrimTabError):
    """Caller swapped embedder mid-DB (per-DB dimension pinning)."""

    def __init__(self, expected: int, got: int) -> None:
        self.expected = expected
        self.got = got
        super().__init__(
            f"Embedding dimension mismatch: DB pinned at {expected}, embedder returned {got}. "
            f"Options: (a) switch back to the original embedder, (b) use a different DB file, "
            f"or (c) run 'trimtab reembed --embedder <new>' to wipe and rebuild."
        )


class TrimTabMigrationError(TrimTabError):
    """v0.4 → v0.5 schema migration failed."""


class TrimTabGrammarError(TrimTabError):
    """Invalid grammar JSON on load (malformed shape, not just missing)."""


class TrimTabCycleError(TrimTabError):
    """Generate hit a cyclic reference chain and bailed."""

    def __init__(self, chain: list[str]) -> None:
        self.chain = list(chain)
        super().__init__(f"Cyclic symbol reference in generate walk: {' → '.join(self.chain)}")
