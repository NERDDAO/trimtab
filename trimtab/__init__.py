"""TrimTab — Context-aware grammar generation with cascading embedding search."""

from trimtab.grammar import Grammar
from trimtab.embedder import Embedder, MiniLMEmbedder, OllamaEmbedder, get_default_embedder
from trimtab.index import GrammarIndex
from trimtab.generator import Generator
from trimtab.builder import build_grammar, extract_ngrams, cluster_ngrams

__version__ = "0.1.0"

__all__ = [
    "Grammar",
    "Embedder",
    "MiniLMEmbedder",
    "OllamaEmbedder",
    "get_default_embedder",
    "GrammarIndex",
    "Generator",
    "build_grammar",
    "extract_ngrams",
    "cluster_ngrams",
    "SmartGrammar",
]


class SmartGrammar:
    """High-level API combining grammar, index, and generator."""

    def __init__(self, grammar: Grammar, embedder: Embedder | None = None):
        self._embedder = embedder or get_default_embedder()
        self._index = GrammarIndex(grammar, self._embedder)
        self._generator: Generator | None = None

    @classmethod
    def from_file(cls, path: str, embedder: Embedder | None = None) -> "SmartGrammar":
        """Load a grammar from a Tracery JSON file."""
        grammar = Grammar.from_file(path)
        return cls(grammar, embedder)

    @classmethod
    def load(cls, directory: str, embedder: Embedder | None = None) -> "SmartGrammar":
        """Load an indexed grammar from a .sg directory."""
        emb = embedder or get_default_embedder()
        sg = cls.__new__(cls)
        sg._embedder = emb
        sg._index = GrammarIndex.load(directory, emb)
        sg._generator = Generator(sg._index)
        return sg

    @classmethod
    def build_from_corpus(
        cls,
        texts: list[str],
        embedder: Embedder | None = None,
        min_count: int = 2,
    ) -> "SmartGrammar":
        """Build a grammar from a text corpus."""
        emb = embedder or get_default_embedder()
        grammar = build_grammar(texts, emb, min_count=min_count)
        return cls(grammar, emb)

    def index(self) -> None:
        """Embed all rules and build FAISS indices."""
        self._index.build()
        self._generator = Generator(self._index)

    def generate(
        self,
        context: str = "",
        temperature: float = 0.3,
        seed: int | None = None,
        min_confidence: float = 0.0,
        no_match_text: str = "",
    ) -> str:
        """Generate text using cascading context-aware expansion.

        Args:
            context: Context string for embedding-based selection.
            temperature: 0.0=deterministic, 0.3=recommended, 1.0=random.
            seed: Random seed for reproducibility.
            min_confidence: Minimum cosine similarity to accept a match (0.0-1.0).
                Slots below threshold return no_match_text instead.
            no_match_text: Text for slots with no confident match.
        """
        if self._generator is None:
            self.index()
        assert self._generator is not None
        return self._generator.generate(
            context=context, temperature=temperature, seed=seed,
            min_confidence=min_confidence, no_match_text=no_match_text,
        )

    def add(self, rule: str, expansion: str) -> None:
        """Add a new expansion to a rule (auto-embeds and indexes)."""
        self._index.add_to_rule(rule, expansion)

    def save(self, directory: str) -> None:
        """Save grammar + indices to a .sg directory."""
        self._index.save(directory)
