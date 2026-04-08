"""TrimTab — Context-aware grammar generation with cascading embedding search."""

from __future__ import annotations

from pathlib import Path

from trimtab.grammar import Grammar
from trimtab.embedder import Embedder, MiniLMEmbedder, OllamaEmbedder, get_default_embedder
from trimtab.db import TrimTabDB
from trimtab.generator import Generator
from trimtab.builder import build_grammar, extract_ngrams, cluster_ngrams

__version__ = "0.2.0"

__all__ = [
    "Grammar",
    "Embedder",
    "MiniLMEmbedder",
    "OllamaEmbedder",
    "get_default_embedder",
    "TrimTabDB",
    "Generator",
    "build_grammar",
    "extract_ngrams",
    "cluster_ngrams",
    "SmartGrammar",
]


class SmartGrammar:
    """High-level API combining TrimTabDB, grammar, and generator."""

    def __init__(
        self,
        db: TrimTabDB,
        grammar_name: str,
        embedder: Embedder | None = None,
    ):
        self._db = db
        self._name = grammar_name
        self._embedder = embedder or get_default_embedder()
        self._generator = Generator(self._db, self._name, self._embedder)

    @classmethod
    def from_file(
        cls,
        db: TrimTabDB,
        path: str,
        embedder: Embedder | None = None,
        grammar_name: str | None = None,
    ) -> SmartGrammar:
        """Import a grammar JSON file into the DB and return a SmartGrammar."""
        grammar = Grammar.from_file(path)
        name = grammar_name or Path(path).stem
        emb = embedder or get_default_embedder()
        db.upsert_grammar(name, grammar, emb)
        return cls(db, name, emb)

    @classmethod
    def build_from_corpus(
        cls,
        db: TrimTabDB,
        texts: list[str],
        grammar_name: str = "corpus",
        embedder: Embedder | None = None,
        min_count: int = 2,
    ) -> SmartGrammar:
        """Build a grammar from a text corpus and store in the DB."""
        emb = embedder or get_default_embedder()
        grammar = build_grammar(texts, emb, min_count=min_count)
        db.upsert_grammar(grammar_name, grammar, emb)
        return cls(db, grammar_name, emb)

    def load_grammar(self, rules: dict[str, list[str]]) -> None:
        """Load a grammar from a rules dict into the DB."""
        grammar = Grammar.from_dict(rules)
        self._db.upsert_grammar(self._name, grammar, self._embedder)

    def generate(
        self,
        context: str = "",
        temperature: float = 0.3,
        seed: int | None = None,
        min_confidence: float = 0.0,
        no_match_text: str = "",
    ) -> str:
        """Generate text using cascading context-aware expansion."""
        return self._generator.generate(
            context=context,
            temperature=temperature,
            seed=seed,
            min_confidence=min_confidence,
            no_match_text=no_match_text,
        )

    def add(self, rule: str, expansion: str, id: str | None = None) -> None:
        """Add a new expansion to a rule (auto-embeds and writes to DB).

        Args:
            rule: Rule name within this grammar.
            expansion: The expansion text.
            id: Optional custom Expansion id (e.g., a KG entity UUID).
                If None, an internal id is auto-generated.
        """
        vec = self._embedder.embed([expansion])[0]
        self._db.add_expansion(self._name, rule, expansion, vec, id=id)

    def export(self) -> Grammar:
        """Export the grammar from the DB as a Grammar object."""
        return self._db.get_grammar(self._name)
