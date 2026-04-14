"""TrimTab — the v0.5 public façade.

Wraps ``TrimTabDB`` plus an ``Embedder``. Provides the memory-store API:
``put`` / ``put_many`` / ``update`` (async, Task 12), ``search`` (async,
Task 13), ``generate`` (async, Task 14), ``list`` / ``list_grammars`` /
``list_symbols`` / ``count`` / ``remove`` / ``clear`` / ``drop`` (sync,
this task), and ``load_file`` / ``export_file`` (Task 15).

Users should prefer importing ``TrimTab`` over the lower-level classes
(``TrimTabDB``, ``Generator``) which remain as the implementation surface
for advanced use.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from trimtab.db import TrimTabDB
from trimtab.embedder import Embedder
from trimtab.grammar import Rule

if TYPE_CHECKING:  # avoid eager OllamaEmbedder import at module load
    pass


class TrimTab:
    """Lightweight embedded memory system with Tracery-style grammars.

    Construction:
        tt = TrimTab(path="memory.db")                  # default Ollama embedder
        tt = TrimTab(path=":memory:", embedder=...)     # bring your own
    """

    def __init__(self, path: str = ":memory:", embedder: Embedder | None = None) -> None:
        self._db = TrimTabDB(self._expand_path(path))
        if embedder is None:
            # Lazy import so TrimTab can be constructed in test environments
            # without Ollama as long as the caller supplies an embedder.
            from trimtab.embedders import OllamaEmbedder
            embedder = OllamaEmbedder()  # may raise TrimTabEmbedderError
        self._embedder = embedder

    @staticmethod
    def _expand_path(path: str) -> str:
        """Expand ~ and create parent directories for file-backed DBs."""
        if path == ":memory:":
            return path
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        return str(p)

    # --- sync read / introspection ------------------------------------------

    def list(self, grammar: str, symbol: str) -> list[Rule]:
        """Return all rules under (grammar, symbol) in insertion order.

        Returns an empty list if the grammar or symbol does not exist —
        absence is a valid answer for a memory store. Use ``count`` to
        distinguish "missing" from "empty".
        """
        try:
            return self._db._get_rules(grammar=grammar, symbol=symbol)
        except RuntimeError:
            # Rule table doesn't exist yet (no rules have been put into any
            # grammar). Treat as empty — the caller asked for zero rules.
            return []

    def list_grammars(self) -> list[str]:
        """Return the names of all grammars currently in the DB."""
        return self._db.list_grammars()

    def list_symbols(self, grammar: str) -> list[str]:
        """Return the names of all symbols under a grammar."""
        return self._db._list_symbols(grammar)

    def count(self, grammar: str, symbol: str) -> int:
        """Return the number of rules under (grammar, symbol). Zero if missing."""
        return self._db._count_rules(grammar, symbol)

    # --- sync mutation -----------------------------------------------------

    def remove(self, grammar: str, symbol: str, rule_id: str) -> None:
        """Hard-delete a rule by id. Raises TrimTabNotFoundError if missing."""
        self._db._remove_rule(grammar=grammar, symbol=symbol, rule_id=rule_id)

    def clear(self, grammar: str, symbol: str) -> None:
        """Wipe a symbol's rules. Leaves the symbol node intact for re-use."""
        self._db._clear_symbol(grammar=grammar, symbol=symbol)

    def drop(self, grammar: str) -> None:
        """Remove a whole grammar — all rules, symbols, and the Grammar node."""
        self._db._drop_grammar(grammar)
