"""TrimTab — Context-aware grammar generation with cascading embedding search.

trimtab is BYO-embedder: pass any object that satisfies the ``Embedder``
Protocol (an async ``create(text)`` + ``create_batch(texts)`` pair matching
``graphiti_core.embedder.client.EmbedderClient``).
"""

from __future__ import annotations

from pathlib import Path

from trimtab.builder import build_grammar, cluster_ngrams, extract_ngrams
from trimtab.db import TrimTabDB
from trimtab.embedder import Embedder
from trimtab.generator import Generator
from trimtab.grammar import Grammar

__version__ = "0.3.0"

__all__ = [
    "Grammar",
    "Embedder",
    "TrimTabDB",
    "Generator",
    "build_grammar",
    "extract_ngrams",
    "cluster_ngrams",
    "SmartGrammar",
]


class SmartGrammar:
    """High-level API combining TrimTabDB, a grammar, and a generator.

    Construction and the grammar methods are async because the underlying
    embedder is async. The embedder is required — trimtab no longer ships
    a default implementation.
    """

    def __init__(
        self,
        db: TrimTabDB,
        grammar_name: str,
        embedder: Embedder,
    ):
        self._db = db
        self._name = grammar_name
        self._embedder = embedder
        self._generator = Generator(self._db, self._name, self._embedder)

    @classmethod
    async def from_file(
        cls,
        db: TrimTabDB,
        path: str,
        embedder: Embedder,
        grammar_name: str | None = None,
    ) -> SmartGrammar:
        """Import a grammar JSON file into the DB and return a SmartGrammar."""
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
    ) -> SmartGrammar:
        """Build a grammar from a text corpus and store in the DB."""
        grammar = await build_grammar(texts, embedder, min_count=min_count)
        await db.upsert_grammar(grammar_name, grammar, embedder)
        return cls(db, grammar_name, embedder)

    async def load_grammar(self, rules: dict[str, list[str]]) -> None:
        """Load a grammar from a rules dict into the DB."""
        grammar = Grammar.from_dict(rules)
        await self._db.upsert_grammar(self._name, grammar, self._embedder)

    async def generate(
        self,
        context: str = "",
        temperature: float = 0.3,
        seed: int | None = None,
        min_confidence: float = 0.0,
        no_match_text: str = "",
    ) -> str:
        """Generate text using cascading context-aware expansion."""
        return await self._generator.generate(
            context=context,
            temperature=temperature,
            seed=seed,
            min_confidence=min_confidence,
            no_match_text=no_match_text,
        )

    async def add(self, rule: str, expansion: str, id: str | None = None) -> None:
        """Add a new expansion to a rule (auto-embeds and writes to DB).

        Args:
            rule: Rule name within this grammar.
            expansion: The expansion text.
            id: Optional custom Expansion id (e.g., a KG entity UUID).
                If None, an internal id is auto-generated.
        """
        vec = await self._embedder.create(expansion)
        self._db.add_expansion(self._name, rule, expansion, vec, id=id)

    def export(self) -> Grammar:
        """Export the grammar from the DB as a Grammar object."""
        return self._db.get_grammar(self._name)
