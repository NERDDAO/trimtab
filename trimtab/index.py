"""FAISS index per grammar rule for embedding-based lookup."""

import json
import logging
from pathlib import Path

import faiss
import numpy as np

from trimtab.embedder import Embedder
from trimtab.grammar import Grammar

logger = logging.getLogger(__name__)


class GrammarIndex:
    """Manages FAISS indices for each rule in a grammar."""

    def __init__(self, grammar: Grammar, embedder: Embedder):
        self.grammar = grammar
        self.embedder = embedder
        self._indices: dict[str, faiss.IndexFlatIP] = {}  # inner product (cosine on normalized)
        self._expansions: dict[str, list[str]] = {}  # rule -> ordered list matching index

    def build(self) -> None:
        """Embed all expansions and build FAISS indices."""
        for rule in self.grammar.rule_names():
            expansions = self.grammar.get_expansions(rule)
            if not expansions:
                continue

            vectors = self.embedder.embed(expansions)
            dim = vectors.shape[1]

            index = faiss.IndexFlatIP(dim)
            index.add(vectors.astype(np.float32))

            self._indices[rule] = index
            self._expansions[rule] = list(expansions)

        logger.info("Indexed %d rules", len(self._indices))

    def query(self, rule: str, context_vector: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
        """Query a rule's index with a context vector. Returns (expansion, score) pairs."""
        if rule not in self._indices:
            return []

        index = self._indices[rule]
        expansions = self._expansions[rule]

        k = min(top_k, len(expansions))
        context_vector = context_vector.reshape(1, -1).astype(np.float32)
        scores, ids = index.search(context_vector, k)

        results = []
        for i in range(k):
            idx = ids[0][i]
            if idx >= 0:
                results.append((expansions[idx], float(scores[0][i])))
        return results

    def add_to_rule(self, rule: str, expansion: str) -> None:
        """Add a new expansion to a rule's index."""
        self.grammar.add_expansion(rule, expansion)

        vector = self.embedder.embed([expansion])

        if rule not in self._indices:
            dim = vector.shape[1]
            self._indices[rule] = faiss.IndexFlatIP(dim)
            self._expansions[rule] = []

        self._indices[rule].add(vector.astype(np.float32))
        self._expansions[rule].append(expansion)

    def save(self, directory: str | Path) -> None:
        """Save grammar + FAISS indices to a directory."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)

        self.grammar.save(d / "grammar.json")

        indices_dir = d / "indices"
        indices_dir.mkdir(exist_ok=True)
        for rule, index in self._indices.items():
            faiss.write_index(index, str(indices_dir / f"{rule}.faiss"))

        meta = {
            "dimension": self.embedder.dimension,
            "rules": {r: len(e) for r, e in self._expansions.items()},
            "expansions": self._expansions,
        }
        (d / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    @classmethod
    def load(cls, directory: str | Path, embedder: Embedder) -> "GrammarIndex":
        """Load grammar + FAISS indices from a directory."""
        d = Path(directory)

        grammar = Grammar.from_file(d / "grammar.json")
        meta = json.loads((d / "meta.json").read_text())

        gi = cls(grammar, embedder)
        gi._expansions = meta.get("expansions", {})

        indices_dir = d / "indices"
        for rule in gi._expansions:
            index_path = indices_dir / f"{rule}.faiss"
            if index_path.exists():
                gi._indices[rule] = faiss.read_index(str(index_path))

        return gi
