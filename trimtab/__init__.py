"""TrimTab — embedded memory and generation for agentic Python.

Primary entry point: ``trimtab.TrimTab``. See the README for the 30-second
model and the v0.5 design at
``docs/superpowers/specs/2026-04-14-trimtab-memory-system-design.md``
in the Bonfires workspace.
"""

from __future__ import annotations

import warnings
from pathlib import Path

from trimtab.builder import build_grammar, cluster_ngrams, extract_ngrams
from trimtab.core import TrimTab
from trimtab.db import TrimTabDB
from trimtab.embedder import Embedder
from trimtab.errors import (
    TrimTabCycleError,
    TrimTabDimensionError,
    TrimTabEmbedderError,
    TrimTabError,
    TrimTabGrammarError,
    TrimTabMigrationError,
    TrimTabNotFoundError,
)
from trimtab.generator import GenerationResult, Generator
from trimtab.grammar import Grammar, Rule

__version__ = "0.5.0"

__all__ = [
    "TrimTab",
    "Grammar",
    "Rule",
    "Embedder",
    "TrimTabDB",
    "Generator",
    "GenerationResult",
    "build_grammar",
    "extract_ngrams",
    "cluster_ngrams",
    # Errors
    "TrimTabError",
    "TrimTabEmbedderError",
    "TrimTabNotFoundError",
    "TrimTabDimensionError",
    "TrimTabMigrationError",
    "TrimTabGrammarError",
    "TrimTabCycleError",
    # Deprecated v0.4 surface
    "SmartGrammar",
]


class SmartGrammar:
    """DEPRECATED. Use ``TrimTab`` instead.

    Thin v0.4-shape wrapper retained for one release so existing code that
    constructs ``SmartGrammar(db, grammar_name, embedder)`` keeps working.
    Will be removed in v0.6. Internally delegates to the v0.5 internals.

    Construction emits ``DeprecationWarning`` at stacklevel=2 so users see
    exactly where in their code the deprecation fires.
    """

    def __init__(
        self,
        db: TrimTabDB,
        grammar_name: str,
        embedder: Embedder,
    ) -> None:
        warnings.warn(
            "SmartGrammar is deprecated; use TrimTab instead. "
            "Will be removed in v0.6.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._db = db
        self._name = grammar_name
        self._embedder = embedder
        # Lazy generator creation — keeps the attribute available for
        # callers that reach into internals.
        from trimtab.generator import Generator
        self._generator = Generator(self._db, self._name, self._embedder)

    @classmethod
    async def from_file(
        cls,
        db: TrimTabDB,
        path: str,
        embedder: Embedder,
        grammar_name: str | None = None,
    ) -> "SmartGrammar":
        """DEPRECATED. Use ``TrimTab.load_file`` instead."""
        grammar = Grammar.from_file(path)
        name = grammar_name or Path(path).stem
        await db.upsert_grammar(name, grammar, embedder)
        return cls(db, name, embedder)

    @classmethod
    async def build_from_corpus(
        cls,
        db: TrimTabDB,
        texts: list[str],
        embedder: Embedder,
        grammar_name: str = "corpus",
        min_count: int = 2,
    ) -> "SmartGrammar":
        """DEPRECATED. Use ``TrimTab`` and the builder module directly."""
        grammar = await build_grammar(texts, embedder, min_count=min_count)
        await db.upsert_grammar(grammar_name, grammar, embedder)
        return cls(db, grammar_name, embedder)

    async def load_grammar(self, rules: dict[str, list[str]]) -> None:
        """DEPRECATED. Use ``TrimTab.load_file`` or ``put_many``."""
        grammar = Grammar.from_dict(rules)
        await self._db.upsert_grammar(self._name, grammar, self._embedder)

    async def generate(
        self,
        context: str = "",
        temperature: float = 0.3,
        seed: int | None = None,
        min_confidence: float = 0.0,
        no_match_text: str = "",
    ) -> GenerationResult:
        """DEPRECATED. Use ``TrimTab.generate`` instead."""
        return await self._generator.generate(
            context=context,
            temperature=temperature,
            seed=seed,
            min_confidence=min_confidence,
            no_match_text=no_match_text,
        )

    async def add(self, rule: str, expansion: str, id: str | None = None) -> None:
        """DEPRECATED. Use ``TrimTab.put`` instead."""
        vec = await self._embedder.create(expansion)
        self._db.add_expansion(self._name, rule, expansion, vec, id=id)

    def export(self) -> Grammar:
        """DEPRECATED. Use ``TrimTab.export_file`` instead."""
        return self._db.get_grammar(self._name)

    def list_entries(self, rule: str) -> list[tuple[str, str]]:
        """DEPRECATED. Use ``TrimTab.list()`` instead."""
        return self._db.list_entries(self._name, rule)
