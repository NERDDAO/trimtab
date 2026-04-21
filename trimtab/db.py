"""LadybugDB-backed storage for grammars, rules, and embeddings."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import real_ladybug as lb

from trimtab.grammar import Grammar, Rule

logger = logging.getLogger(__name__)


class TrimTabDB:
    """LadybugDB-backed storage for grammars, symbols, and vector-indexed rules.

    v0.5 graph schema::

        (:Grammar {name}) -[:HAS_SYMBOL]-> (:Symbol {id, name, grammar})
            -[:HAS_RULE]-> (:Rule {
                id, text, grammar, symbol, metadata, embedding,
                embedded, created_at, updated_at
            })

    On first open, the v0.4 → v0.5 migration runs automatically and is
    idempotent. Old v0.4 methods (``upsert_grammar``, ``add_expansion``,
    ``query``, ``get_expansions``, ``list_entries``) still exist for one
    release — they will be rewritten as deprecated shims in a later task.

    Args:
        path: Database path. Use ":memory:" for in-memory (tests/ephemeral),
              or a file path for persistent storage.
    """

    def __init__(self, path: str = ":memory:"):
        self._db = lb.Database(path)
        self._conn = lb.Connection(self._db)
        self._init_vector_extension()
        self._embedding_dim: int | None = None
        # Migration MUST run before _init_schema. The migration's detector
        # checks whether a Symbol table already exists and treats its
        # presence as "DB is already v0.5". If _init_schema ran first and
        # created an empty Symbol table, the detector would skip migration
        # entirely on real v0.4 DBs. Order: migrate → init_schema → probe.
        from trimtab.migrations import run_migration
        try:
            run_migration(self._conn)
        except Exception as e:
            logger.warning("Migration attempt failed (non-fatal): %s", e)
        self._init_schema()
        # If a Rule table exists (fresh-after-migration or previously-used v0.5),
        # probe its embedding dimension so subsequent puts respect it.
        self._embedding_dim = self._probe_existing_rule_dim()

    def _init_schema(self) -> None:
        """Create v0.5 node and relationship tables if they don't exist."""
        self._conn.execute(
            "CREATE NODE TABLE IF NOT EXISTS Grammar(name STRING PRIMARY KEY)"
        )
        self._conn.execute(
            "CREATE NODE TABLE IF NOT EXISTS Symbol("
            "  id STRING PRIMARY KEY,"
            "  name STRING,"
            "  grammar STRING"
            ")"
        )
        self._conn.execute(
            "CREATE REL TABLE IF NOT EXISTS HAS_SYMBOL(FROM Grammar TO Symbol)"
        )
        # Rule node table is created lazily in _ensure_rule_table when we
        # know the embedding dimension. HAS_RULE edge is created alongside.

    def _init_vector_extension(self) -> None:
        """Load vector extension."""
        try:
            self._conn.execute("INSTALL vector")
        except Exception:
            pass
        try:
            self._conn.execute("LOAD EXTENSION vector")
        except Exception:
            pass

    def _probe_existing_rule_dim(self) -> int | None:
        """Return the embedding dim of the Rule table if it exists, else None.

        Called from __init__ after migration so we know the pinned dim
        without forcing callers to re-supply it on every put.
        """
        try:
            rows = self._conn.execute(
                "MATCH (r:Rule) RETURN r.embedding LIMIT 1"
            ).get_all()
            if rows and rows[0][0]:
                return len(rows[0][0])
        except Exception:
            pass
        return None

    def _ensure_rule_table(self, dim: int) -> None:
        """Create the v0.5 Rule node table with the given embedding dim.

        The Rule table is created lazily on first put so we can pin the
        correct FLOAT[dim] column width. If the table already exists at a
        different dim, raise TrimTabDimensionError — LadybugDB does not
        allow changing a fixed-width array column after creation.
        """
        if self._embedding_dim == dim:
            # Already created at this dim — no-op.
            return
        if self._embedding_dim is not None and self._embedding_dim != dim:
            from trimtab.errors import TrimTabDimensionError
            raise TrimTabDimensionError(expected=self._embedding_dim, got=dim)

        # First-ever creation (fresh DB with no migration, no prior put).
        # CREATE ... IF NOT EXISTS handles the idempotent case without raising,
        # so we intentionally do NOT wrap these in try/except — genuine
        # creation failures should propagate instead of leaving the DB in an
        # indeterminate state with _embedding_dim claiming the table exists.
        self._conn.execute(
            f"CREATE NODE TABLE IF NOT EXISTS Rule("
            f"  id STRING PRIMARY KEY,"
            f"  text STRING,"
            f"  grammar STRING,"
            f"  symbol STRING,"
            f"  metadata STRING,"
            f"  embedding FLOAT[{dim}],"
            f"  embedded BOOLEAN,"
            f"  created_at STRING,"
            f"  updated_at STRING"
            f")"
        )
        self._conn.execute(
            "CREATE REL TABLE IF NOT EXISTS HAS_RULE(FROM Symbol TO Rule)"
        )
        # Create HNSW index on the embedding column. The index creation can
        # legitimately raise if it already exists, so narrow the catch here.
        try:
            self._conn.execute(
                "CALL CREATE_VECTOR_INDEX("
                "  'Rule', 'rule_embedding_idx', 'embedding',"
                "  metric := 'cosine'"
                ")"
            )
        except Exception as e:
            logger.debug("HNSW index creation (likely already exists): %s", e)
        self._embedding_dim = dim

    def _put_rule_with_vector(
        self,
        grammar: str,
        symbol: str,
        rule: Rule,
        vector: list[float],
    ) -> Rule:
        """Insert a Rule into v0.5 tables. Vector must be pre-computed."""
        self._ensure_rule_table(len(vector))

        # Ensure Grammar and Symbol nodes exist.
        self._conn.execute(
            "MERGE (g:Grammar {name: $name})",
            {"name": grammar},
        )
        sym_id = f"{grammar}:{symbol}"
        self._conn.execute(
            "MERGE (s:Symbol {id: $id}) "
            "ON CREATE SET s.name = $name, s.grammar = $grammar "
            "ON MATCH SET s.name = $name",
            {"id": sym_id, "name": symbol, "grammar": grammar},
        )
        self._conn.execute(
            "MATCH (g:Grammar), (s:Symbol) "
            "WHERE g.name = $gname AND s.id = $sid "
            "MERGE (g)-[:HAS_SYMBOL]->(s)",
            {"gname": grammar, "sid": sym_id},
        )

        # Delete-then-create pattern (vector-indexed columns reject SET).
        # LadybugDB's DETACH DELETE on a non-matching node is a no-op, so
        # we do NOT wrap this in try/except — let genuine failures propagate.
        self._conn.execute(
            "MATCH (r:Rule) WHERE r.id = $id DETACH DELETE r",
            {"id": rule.id},
        )

        # Serialize updated_at (may be None on construction if caller built
        # a Rule with no updated_at override; __post_init__ normally fills it
        # but defensive handling here too).
        updated_at_iso = (
            rule.updated_at.isoformat() if rule.updated_at is not None
            else rule.created_at.isoformat()
        )

        self._conn.execute(
            "CREATE (r:Rule {"
            "  id: $id, text: $text, grammar: $grammar, symbol: $symbol,"
            "  metadata: $metadata, embedding: $embedding, embedded: $embedded,"
            "  created_at: $created_at, updated_at: $updated_at"
            "})",
            {
                "id": rule.id,
                "text": rule.text,
                "grammar": grammar,
                "symbol": symbol,
                # LadybugDB auto-parses string parameters that start with `{`
                # and re-serializes them as Cypher map literals (unquoted keys),
                # which breaks `json.loads` on read. The `"json:"` prefix
                # bypasses that heuristic — read side strips it in _get_rules.
                "metadata": "json:" + json.dumps(rule.metadata),
                "embedding": vector,
                "embedded": True,
                "created_at": rule.created_at.isoformat(),
                "updated_at": updated_at_iso,
            },
        )
        self._conn.execute(
            "MATCH (s:Symbol), (r:Rule) "
            "WHERE s.id = $sid AND r.id = $rid "
            "MERGE (s)-[:HAS_RULE]->(r)",
            {"sid": sym_id, "rid": rule.id},
        )
        return rule

    def _get_rules(self, grammar: str, symbol: str) -> list[Rule]:
        """Return all rules under (grammar, symbol), insertion-ordered by created_at."""
        result = self._conn.execute(
            "MATCH (r:Rule) "
            "WHERE r.grammar = $g AND r.symbol = $s "
            "RETURN r.id, r.text, r.metadata, r.created_at, r.updated_at "
            "ORDER BY r.created_at ASC",
            {"g": grammar, "s": symbol},
        )
        rules: list[Rule] = []
        for row in result.get_all():
            rid, rtext, rmeta, rcreated, rupdated = row
            # Strip the "json:" prefix that _put_rule_with_vector adds to
            # bypass LadybugDB's auto-parse on map-literal-shaped strings.
            meta = json.loads(rmeta[5:]) if rmeta else {}
            rules.append(
                Rule(
                    text=rtext,
                    id=rid,
                    metadata=meta,
                    created_at=datetime.fromisoformat(rcreated),
                    updated_at=datetime.fromisoformat(rupdated),
                )
            )
        return rules

    def _update_rule_fields(
        self,
        grammar: str,
        symbol: str,
        rule_id: str,
        text: str | None = None,
        metadata: dict | None = None,
        new_vector: list[float] | None = None,
    ) -> None:
        """Update a Rule by id. Replaces text and/or merges metadata.

        If ``text`` is provided, ``new_vector`` should also be supplied
        (the caller is responsible for re-embedding). If ``text`` is None,
        the existing text and embedding are preserved.

        ``metadata`` is **merged** into the existing dict, not replaced —
        new keys are added, existing keys are overwritten with new values,
        unmentioned keys remain. Pass an empty dict ``{}`` to leave
        metadata alone.

        Raises ``TrimTabNotFoundError`` if no rule with ``rule_id`` exists
        under the given (grammar, symbol).
        """
        from trimtab.errors import TrimTabNotFoundError

        # Fetch existing rule fields for merge.
        rows = self._conn.execute(
            "MATCH (r:Rule) WHERE r.id = $id AND r.grammar = $g AND r.symbol = $s "
            "RETURN r.text, r.metadata, r.embedding, r.created_at",
            {"id": rule_id, "g": grammar, "s": symbol},
        ).get_all()
        if not rows:
            raise TrimTabNotFoundError(grammar=grammar, symbol=symbol, rule_id=rule_id)

        existing_text, existing_meta_str, existing_embedding, existing_created = rows[0]

        # Strip the "json:" prefix from existing metadata, merge, re-encode.
        existing_meta = json.loads(existing_meta_str[5:]) if existing_meta_str else {}
        if metadata is not None:
            existing_meta.update(metadata)
        merged_meta_str = "json:" + json.dumps(existing_meta)

        new_text = text if text is not None else existing_text
        new_embedding_list = new_vector if new_vector is not None else list(existing_embedding)
        now = datetime.now(timezone.utc).isoformat()

        # Delete-then-create (vector-indexed columns reject SET).
        self._conn.execute(
            "MATCH (r:Rule) WHERE r.id = $id DETACH DELETE r",
            {"id": rule_id},
        )
        self._conn.execute(
            "CREATE (r:Rule {"
            "  id: $id, text: $text, grammar: $grammar, symbol: $symbol,"
            "  metadata: $metadata, embedding: $embedding, embedded: $embedded,"
            "  created_at: $created_at, updated_at: $updated_at"
            "})",
            {
                "id": rule_id,
                "text": new_text,
                "grammar": grammar,
                "symbol": symbol,
                "metadata": merged_meta_str,
                "embedding": new_embedding_list,
                "embedded": True,
                "created_at": existing_created,  # preserve original created_at
                "updated_at": now,
            },
        )
        # Re-wire the HAS_RULE edge from the symbol.
        sym_id = f"{grammar}:{symbol}"
        self._conn.execute(
            "MATCH (s:Symbol), (r:Rule) "
            "WHERE s.id = $sid AND r.id = $rid "
            "MERGE (s)-[:HAS_RULE]->(r)",
            {"sid": sym_id, "rid": rule_id},
        )

    def _remove_rule(self, grammar: str, symbol: str, rule_id: str) -> None:
        """Hard-delete a Rule by id. Raises TrimTabNotFoundError if missing."""
        from trimtab.errors import TrimTabNotFoundError

        rows = self._conn.execute(
            "MATCH (r:Rule) WHERE r.id = $id AND r.grammar = $g AND r.symbol = $s "
            "RETURN r.id",
            {"id": rule_id, "g": grammar, "s": symbol},
        ).get_all()
        if not rows:
            raise TrimTabNotFoundError(grammar=grammar, symbol=symbol, rule_id=rule_id)
        self._conn.execute(
            "MATCH (r:Rule) WHERE r.id = $id DETACH DELETE r",
            {"id": rule_id},
        )

    def _clear_symbol(self, grammar: str, symbol: str) -> None:
        """Remove all Rules under (grammar, symbol). Leaves the Symbol node intact."""
        self._conn.execute(
            "MATCH (r:Rule) WHERE r.grammar = $g AND r.symbol = $s DETACH DELETE r",
            {"g": grammar, "s": symbol},
        )

    def _drop_grammar(self, grammar: str) -> None:
        """Remove all Rules, Symbols, and the Grammar node for a grammar."""
        self._conn.execute(
            "MATCH (r:Rule) WHERE r.grammar = $g DETACH DELETE r",
            {"g": grammar},
        )
        self._conn.execute(
            "MATCH (s:Symbol) WHERE s.grammar = $g DETACH DELETE s",
            {"g": grammar},
        )
        self._conn.execute(
            "MATCH (g:Grammar) WHERE g.name = $g_name DETACH DELETE g",
            {"g_name": grammar},
        )

    def _search_rules(
        self,
        grammar: str,
        symbol: str,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[Rule]:
        """Vector similarity search within (grammar, symbol). Returns list[Rule].

        Uses LadybugDB's HNSW index when available, falls back to a
        brute-force cosine scan if HNSW returns nothing (e.g., recently
        inserted rows that haven't been indexed yet).
        """
        if self._embedding_dim is None:
            # No Rule table exists yet — nothing to search.
            return []

        rows: list[list] = []
        try:
            result = self._conn.execute(
                "CALL QUERY_VECTOR_INDEX('Rule', 'rule_embedding_idx', $vec, $k) "
                "WHERE node.grammar = $g AND node.symbol = $s "
                "RETURN node.id, node.text, node.metadata, node.created_at, node.updated_at, distance "
                "ORDER BY distance",
                {"vec": query_vector, "k": top_k * 4, "g": grammar, "s": symbol},
            )
            rows = [list(r) for r in result.get_all()]
        except Exception:
            rows = []

        if not rows:
            # Brute-force fallback over the (grammar, symbol) slice.
            rows = self._brute_force_search(grammar, symbol, query_vector, top_k)

        rules: list[Rule] = []
        for row in rows[:top_k]:
            rid, rtext, rmeta, rcreated, rupdated = row[0], row[1], row[2], row[3], row[4]
            # Strip the "json:" prefix that _put_rule_with_vector adds.
            meta = json.loads(rmeta[5:]) if rmeta else {}
            rules.append(
                Rule(
                    text=rtext,
                    id=rid,
                    metadata=meta,
                    created_at=datetime.fromisoformat(rcreated),
                    updated_at=datetime.fromisoformat(rupdated),
                )
            )
        return rules

    def _brute_force_search(
        self, grammar: str, symbol: str, query_vector: list[float], top_k: int
    ) -> list[list]:
        """Fallback cosine search when HNSW returns nothing.

        Returns rows shaped to match _search_rules's expected schema:
        [id, text, metadata, created_at, updated_at] (distance is dropped
        because the caller doesn't surface it).
        """
        try:
            result = self._conn.execute(
                "MATCH (r:Rule) WHERE r.grammar = $g AND r.symbol = $s "
                "RETURN r.id, r.text, r.metadata, r.created_at, r.updated_at, r.embedding",
                {"g": grammar, "s": symbol},
            )
            db_rows = result.get_all()
        except Exception:
            db_rows = []
        if not db_rows:
            return []
        q = np.array(query_vector, dtype=np.float32)
        q_norm = q / (np.linalg.norm(q) + 1e-8)
        scored: list[tuple[float, list]] = []
        for rid, rtext, rmeta, rcreated, rupdated, remb in db_rows:
            e = np.array(remb, dtype=np.float32)
            e_norm = e / (np.linalg.norm(e) + 1e-8)
            sim = float(np.dot(q_norm, e_norm))
            distance = 1.0 - sim
            scored.append((distance, [rid, rtext, rmeta, rcreated, rupdated]))
        scored.sort(key=lambda x: x[0])
        return [row for _, row in scored[:top_k]]

    def _list_symbols(self, grammar: str) -> list[str]:
        """Return the names of all symbols under a grammar, alphabetically sorted."""
        try:
            result = self._conn.execute(
                "MATCH (s:Symbol) WHERE s.grammar = $g RETURN s.name ORDER BY s.name",
                {"g": grammar},
            )
            return [row[0] for row in result.get_all()]
        except Exception:
            return []

    def _count_rules(self, grammar: str, symbol: str) -> int:
        """Return the number of rules under (grammar, symbol). Zero if missing."""
        try:
            result = self._conn.execute(
                "MATCH (r:Rule) WHERE r.grammar = $g AND r.symbol = $s RETURN count(r)",
                {"g": grammar, "s": symbol},
            )
            db_rows = result.get_all()
            return int(db_rows[0][0]) if db_rows else 0
        except Exception:
            return 0

    def _summary_all(self) -> dict[str, list[dict[str, Any]]]:
        """One-roundtrip snapshot of every rule grouped by grammar.

        Returns ``{grammar: [{"id", "symbol", "text", "metadata",
        "created_at", "updated_at"}, ...]}``. Empty dict if no rules
        have been written yet (Rule node table is created lazily on
        first put).

        Used by ``TrimTab.summary()`` for callers that build context
        snapshots on every turn and want to skip N round-trips.
        """
        try:
            result = self._conn.execute(
                "MATCH (r:Rule) "
                "RETURN r.grammar, r.symbol, r.id, r.text, r.metadata, "
                "r.created_at, r.updated_at "
                "ORDER BY r.grammar, r.symbol, r.created_at"
            )
            rows = result.get_all()
        except Exception:
            return {}

        snapshot: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            grammar, symbol, rid, text, meta_str, created, updated = row
            meta = json.loads(meta_str[5:]) if meta_str else {}
            snapshot.setdefault(grammar, []).append(
                {
                    "id": rid,
                    "symbol": symbol,
                    "text": text,
                    "metadata": meta,
                    "created_at": created,
                    "updated_at": updated,
                }
            )
        return snapshot

    def get_grammar(self, name: str) -> Grammar:
        """Export a grammar from the DB back to a Grammar object.

        v0.5: reads from the new Symbol → Rule schema and reconstructs a
        Grammar dict. Returns rules as plain text strings (the round-trip
        compatible shape).
        """
        from trimtab.grammar import ExpansionEntry

        rules: dict[str, list[ExpansionEntry]] = {}
        for sym_name in self._list_symbols(name):
            rule_objs = self._get_rules(name, sym_name)
            rules[sym_name] = [r.text for r in rule_objs]
        return Grammar(rules=rules)

    def list_grammars(self) -> list[str]:
        """List all grammar names in the DB."""
        result = self._conn.execute("MATCH (g:Grammar) RETURN g.name")
        return [row[0] for row in result.get_all()]
