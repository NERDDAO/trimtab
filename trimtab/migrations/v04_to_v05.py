"""Migrate LadybugDB schema from v0.4 to v0.5.

v0.4 shape: (:Grammar)-[:HAS_RULE]->(:Rule)-[:HAS_EXPANSION]->(:Expansion)
v0.5 shape: (:Grammar)-[:HAS_SYMBOL]->(:Symbol)-[:HAS_RULE_V05]->(:Rule_v05)

This is a double rename. Ordering matters — we create new tables first,
copy data, then drop old tables. The new v0.5 Rule node is named
Rule_v05 during migration because it cannot share a name with the old
v0.4 Rule table.

The migration is idempotent: running on an already-migrated DB is a no-op
because ``detect_v04_schema`` returns False when Symbol already exists.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import real_ladybug as lb

from trimtab.errors import TrimTabMigrationError

logger = logging.getLogger(__name__)

MIGRATION_VERSION = "0.5.0"


def detect_v04_schema(conn: lb.Connection) -> bool:
    """Return True if the connected DB has the v0.4 schema shape.

    Heuristic: an ``Expansion`` node table exists AND a ``Symbol`` node
    table does not. This catches v0.4 DBs without false-positiving on
    fresh v0.5 DBs (which have Symbol+Rule_v05 but no Expansion).
    """
    has_expansion = False
    has_symbol = False
    try:
        conn.execute("MATCH (e:Expansion) RETURN e.id LIMIT 1")
        has_expansion = True
    except Exception:
        has_expansion = False
    try:
        conn.execute("MATCH (s:Symbol) RETURN s.id LIMIT 1")
        has_symbol = True
    except Exception:
        has_symbol = False
    return has_expansion and not has_symbol


def _probe_old_embedding_dim(conn: lb.Connection) -> int:
    """Read a single Expansion row to learn the fixed-width dim."""
    try:
        rows = conn.execute(
            "MATCH (e:Expansion) RETURN e.embedding LIMIT 1"
        ).get_all()
        if rows:
            return len(rows[0][0])
    except Exception:
        pass
    return 0


def _ensure_v05_tables(conn: lb.Connection, dim: int) -> None:
    """Create v0.5 Symbol + Rule_v05 node tables (and their edges)."""
    try:
        conn.execute(
            "CREATE NODE TABLE IF NOT EXISTS Symbol("
            "  id STRING PRIMARY KEY,"
            "  name STRING,"
            "  grammar STRING"
            ")"
        )
    except Exception:
        pass
    try:
        conn.execute(
            f"CREATE NODE TABLE IF NOT EXISTS Rule_v05("
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
    except Exception:
        pass
    try:
        conn.execute(
            "CREATE REL TABLE IF NOT EXISTS HAS_SYMBOL(FROM Grammar TO Symbol)"
        )
    except Exception:
        pass
    try:
        conn.execute(
            "CREATE REL TABLE IF NOT EXISTS HAS_RULE_V05(FROM Symbol TO Rule_v05)"
        )
    except Exception:
        pass


def run_migration(conn: lb.Connection) -> None:
    """Run the v0.4 → v0.5 migration once, idempotently.

    If the DB is not in v0.4 shape, this is a no-op.
    """
    if not detect_v04_schema(conn):
        logger.debug("No v0.4 schema detected; migration is a no-op.")
        return

    try:
        dim = _probe_old_embedding_dim(conn)
        if dim == 0:
            # Empty v0.4 DB — use a placeholder dim; first real put will
            # either match or trigger TrimTabDimensionError.
            dim = 384
        _ensure_v05_tables(conn, dim)

        # --- Copy old Rule nodes → new Symbol nodes ---
        old_rules = conn.execute(
            "MATCH (r_old:Rule) RETURN r_old.id, r_old.name, r_old.grammar"
        ).get_all()
        for row in old_rules:
            rid, rname, rgrammar = row[0], row[1], row[2]
            conn.execute(
                "MERGE (s:Symbol {id: $id}) "
                "ON CREATE SET s.name = $name, s.grammar = $grammar",
                {"id": rid, "name": rname, "grammar": rgrammar},
            )
            conn.execute(
                "MATCH (g:Grammar), (s:Symbol) "
                "WHERE g.name = $gname AND s.id = $sid "
                "MERGE (g)-[:HAS_SYMBOL]->(s)",
                {"gname": rgrammar, "sid": rid},
            )

        # --- Copy old Expansion nodes → new Rule_v05 nodes ---
        now = datetime.now(timezone.utc).isoformat()
        old_expansions = conn.execute(
            "MATCH (e:Expansion) "
            "RETURN e.id, e.text, e.rule_id, e.grammar, e.embedding"
        ).get_all()
        for row in old_expansions:
            eid, etext, erule_id, egrammar, eembedding = row

            # Get the symbol name from the old Rule node (its id is the rule_id).
            sym_rows = conn.execute(
                "MATCH (r_old:Rule) WHERE r_old.id = $rid RETURN r_old.name",
                {"rid": erule_id},
            ).get_all()
            symbol_name = sym_rows[0][0] if sym_rows else ""

            conn.execute(
                "CREATE (r:Rule_v05 {"
                "  id: $id, text: $text, grammar: $grammar, symbol: $symbol,"
                "  metadata: $metadata, embedding: $embedding, embedded: $embedded,"
                "  created_at: $created_at, updated_at: $updated_at"
                "})",
                {
                    "id": eid,
                    "text": etext,
                    "grammar": egrammar,
                    "symbol": symbol_name,
                    "metadata": "{}",
                    "embedding": list(eembedding),
                    "embedded": True,
                    "created_at": now,
                    "updated_at": now,
                },
            )
            conn.execute(
                "MATCH (s:Symbol), (r:Rule_v05) "
                "WHERE s.id = $sid AND r.id = $rid "
                "MERGE (s)-[:HAS_RULE_V05]->(r)",
                {"sid": erule_id, "rid": eid},
            )

        # --- Drop old v0.4 tables (rels first, then nodes) ---
        try:
            conn.execute("DROP TABLE HAS_EXPANSION")
        except Exception as e:
            logger.warning("drop HAS_EXPANSION: %s", e)
        try:
            conn.execute("DROP TABLE HAS_RULE")
        except Exception as e:
            logger.warning("drop HAS_RULE: %s", e)
        try:
            conn.execute("DROP TABLE Expansion")
        except Exception as e:
            logger.warning("drop Expansion: %s", e)
        try:
            conn.execute("DROP TABLE Rule")
        except Exception as e:
            logger.warning("drop Rule (old): %s", e)

        logger.info(
            "Migrated %d symbols and %d rules to v0.5 schema",
            len(old_rules), len(old_expansions),
        )
    except Exception as e:
        raise TrimTabMigrationError(
            f"v0.4 → v0.5 migration failed: {e}. "
            f"The DB file is in an indeterminate state — restore from backup "
            f"or delete the file to start fresh."
        ) from e
