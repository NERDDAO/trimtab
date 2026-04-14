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
from typing import TYPE_CHECKING, Any

from trimtab.db import TrimTabDB
from trimtab.embedder import Embedder
from trimtab.grammar import Rule, upgrade_entry

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
        # Resolve the embedder into a narrowed local so pyright can track its
        # type across the if-branch, then assign. Without this the attribute
        # would be typed as Embedder | None and every async write call site
        # would need a guard.
        resolved: Embedder
        if embedder is None:
            # Lazy import so TrimTab can be constructed in test environments
            # without Ollama as long as the caller supplies an embedder.
            from trimtab.embedders import OllamaEmbedder
            resolved = OllamaEmbedder()  # may raise TrimTabEmbedderError
        else:
            resolved = embedder
        self._embedder = resolved

    @staticmethod
    def _expand_path(path: str) -> str:
        """Expand ~ and create parent directories for file-backed DBs."""
        if path == ":memory:":
            return path
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        return str(p)

    # --- async writes ------------------------------------------------------

    async def put(
        self,
        grammar: str,
        symbol: str,
        text: str,
        metadata: dict[str, Any] | None = None,
        id: str | None = None,
    ) -> Rule:
        """Insert a single rule. Always embeds before writing.

        ``id`` defaults to an auto-generated ``r_<uuid>`` if not supplied.
        ``metadata`` defaults to ``{}``. Returns the inserted ``Rule`` object
        with all fields populated (including the auto-id and timestamps).
        """
        rule = Rule(text=text, id=id or "", metadata=metadata or {})
        vector = await self._embedder.create(text)
        return self._db._put_rule_with_vector(
            grammar=grammar,
            symbol=symbol,
            rule=rule,
            vector=vector,
        )

    async def put_many(
        self,
        grammar: str,
        symbol: str,
        entries: list[dict[str, Any] | str],
    ) -> list[Rule]:
        """Bulk insert — one embedder.create_batch call for the whole batch.

        ``entries`` accepts a mix of plain strings (each becomes ``Rule(text=...)``)
        and ``{"text": ..., "id"?: ..., "metadata"?: ...}`` dicts. ``upgrade_entry``
        normalizes both shapes to ``Rule`` objects.

        Empty input returns an empty list without calling the embedder.
        """
        if not entries:
            return []

        rules = [upgrade_entry(e) for e in entries]
        texts = [r.text for r in rules]
        vectors = await self._embedder.create_batch(texts)

        written: list[Rule] = []
        for rule, vec in zip(rules, vectors, strict=True):
            written.append(
                self._db._put_rule_with_vector(
                    grammar=grammar,
                    symbol=symbol,
                    rule=rule,
                    vector=vec,
                )
            )
        return written

    async def update(
        self,
        grammar: str,
        symbol: str,
        rule_id: str,
        text: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Update a rule by id. Re-embeds only if ``text`` is provided.

        ``metadata`` is merged into the existing dict (new keys added,
        existing keys overwritten). Pass ``None`` (the default) to leave
        metadata alone.

        Raises ``TrimTabNotFoundError`` if no rule with ``rule_id`` exists
        under (grammar, symbol).
        """
        from trimtab.errors import TrimTabNotFoundError

        new_vector: list[float] | None = None
        if text is not None:
            new_vector = await self._embedder.create(text)
        try:
            self._db._update_rule_fields(
                grammar=grammar,
                symbol=symbol,
                rule_id=rule_id,
                text=text,
                metadata=metadata,
                new_vector=new_vector,
            )
        except RuntimeError as exc:
            # LadybugDB raises RuntimeError("Table Rule does not exist") when
            # no rules have ever been written — treat as not-found.
            if "does not exist" in str(exc):
                raise TrimTabNotFoundError(
                    grammar=grammar, symbol=symbol, rule_id=rule_id
                ) from exc
            raise

    # --- async reads ----------------------------------------------------------

    async def search(
        self,
        grammar: str,
        symbol: str,
        query: str,
        top_k: int = 5,
    ) -> list[Rule]:
        """Semantic search within (grammar, symbol). Returns list[Rule].

        Returns an empty list if the grammar or symbol is missing or has
        no rules — absence is a valid answer for a memory store.
        """
        # Early-return on missing/empty symbol avoids a wasted embed call.
        if self._db._count_rules(grammar, symbol) == 0:
            return []
        query_vector = await self._embedder.create(query)
        return self._db._search_rules(
            grammar=grammar,
            symbol=symbol,
            query_vector=query_vector,
            top_k=top_k,
        )

    async def generate(
        self,
        grammar: str,
        context: str = "",
        origin: str = "origin",
        temperature: float = 0.3,
        seed: int | None = None,
        top_k: int = 5,
        min_confidence: float = 0.0,
        no_match_text: str = "",
    ):
        """Tracery-style walk with cascading embedding-based selection.

        Returns a ``GenerationResult`` with ``text``, ``ids`` (walk order),
        and ``rules_used`` (full Rule objects). Raises ``TrimTabCycleError``
        if the grammar has a cyclic symbol reference.
        """
        from trimtab.generator import Generator
        gen = Generator(db=self._db, grammar=grammar, embedder=self._embedder)
        return await gen.generate(
            context=context,
            origin=origin,
            temperature=temperature,
            seed=seed,
            top_k=top_k,
            min_confidence=min_confidence,
            no_match_text=no_match_text,
        )

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
