"""Tests for the v0.4 → v0.5 LadybugDB schema migration.

The migration double-renames: old Rule → new Symbol, old Expansion → new Rule.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import real_ladybug as lb

from trimtab.migrations.v04_to_v05 import (
    detect_v04_schema,
    run_migration,
    MIGRATION_VERSION,
)


def _seed_v04_schema(db_path: str) -> None:
    """Create a v0.4-shape DB with a grammar, a rule, and an expansion."""
    database = lb.Database(db_path)
    conn = lb.Connection(database)
    try:
        conn.execute("INSTALL vector")
        conn.execute("LOAD EXTENSION vector")
    except Exception:
        pass
    conn.execute("CREATE NODE TABLE Grammar(name STRING PRIMARY KEY)")
    conn.execute(
        "CREATE NODE TABLE Rule("
        "  id STRING PRIMARY KEY,"
        "  name STRING,"
        "  grammar STRING"
        ")"
    )
    conn.execute(
        "CREATE NODE TABLE Expansion("
        "  id STRING PRIMARY KEY,"
        "  text STRING,"
        "  rule_id STRING,"
        "  grammar STRING,"
        "  embedding FLOAT[4]"
        ")"
    )
    conn.execute("CREATE REL TABLE HAS_RULE(FROM Grammar TO Rule)")
    conn.execute("CREATE REL TABLE HAS_EXPANSION(FROM Rule TO Expansion)")

    conn.execute("CREATE (g:Grammar {name: 'dev_memory'})")
    conn.execute(
        "CREATE (r:Rule {id: 'dev_memory:notes', name: 'notes', grammar: 'dev_memory'})"
    )
    conn.execute(
        "CREATE (e:Expansion {"
        "id: 'e1', text: 'first note', rule_id: 'dev_memory:notes',"
        " grammar: 'dev_memory', embedding: [0.1, 0.2, 0.3, 0.4]"
        "})"
    )
    conn.execute(
        "CREATE (e:Expansion {"
        "id: 'e2', text: 'second note', rule_id: 'dev_memory:notes',"
        " grammar: 'dev_memory', embedding: [0.5, 0.6, 0.7, 0.8]"
        "})"
    )
    conn.execute(
        "MATCH (g:Grammar), (r:Rule) WHERE g.name = 'dev_memory' AND r.id = 'dev_memory:notes' "
        "CREATE (g)-[:HAS_RULE]->(r)"
    )
    conn.execute(
        "MATCH (r:Rule), (e:Expansion) WHERE r.id = 'dev_memory:notes' AND e.id = 'e1' "
        "CREATE (r)-[:HAS_EXPANSION]->(e)"
    )
    conn.execute(
        "MATCH (r:Rule), (e:Expansion) WHERE r.id = 'dev_memory:notes' AND e.id = 'e2' "
        "CREATE (r)-[:HAS_EXPANSION]->(e)"
    )
    del conn
    del database


def test_detect_v04_schema_on_seeded_db(tmp_path: Path):
    db_path = str(tmp_path / "old.db")
    _seed_v04_schema(db_path)
    database = lb.Database(db_path)
    conn = lb.Connection(database)
    assert detect_v04_schema(conn) is True


def test_detect_v04_schema_on_empty_db(tmp_path: Path):
    db_path = str(tmp_path / "fresh.db")
    database = lb.Database(db_path)
    conn = lb.Connection(database)
    # Fresh DB has no tables — not v0.4.
    assert detect_v04_schema(conn) is False


def test_run_migration_creates_symbol_and_rule_tables(tmp_path: Path):
    db_path = str(tmp_path / "old.db")
    _seed_v04_schema(db_path)
    database = lb.Database(db_path)
    conn = lb.Connection(database)

    run_migration(conn)

    # After migration, new Symbol and new Rule tables must exist with data.
    symbols = conn.execute(
        "MATCH (s:Symbol) WHERE s.grammar = 'dev_memory' RETURN s.name"
    ).get_all()
    assert len(symbols) == 1
    assert symbols[0][0] == "notes"

    rules = conn.execute(
        "MATCH (r:Rule) WHERE r.grammar = 'dev_memory' RETURN r.text"
    ).get_all()
    texts = {row[0] for row in rules}
    assert texts == {"first note", "second note"}


def test_migration_is_idempotent(tmp_path: Path):
    db_path = str(tmp_path / "old.db")
    _seed_v04_schema(db_path)
    database = lb.Database(db_path)
    conn = lb.Connection(database)

    run_migration(conn)
    # Second run should be a no-op, not an error.
    run_migration(conn)

    rules = conn.execute(
        "MATCH (r:Rule) WHERE r.grammar = 'dev_memory' RETURN r.text"
    ).get_all()
    assert len(rules) == 2  # not doubled


def test_migration_version_constant():
    assert MIGRATION_VERSION == "0.5.0"


def test_migration_wraps_failures_in_trimtab_error(tmp_path: Path):
    """A failure inside run_migration must surface as TrimTabMigrationError, not raw."""
    from unittest.mock import patch
    from trimtab.errors import TrimTabMigrationError

    db_path = str(tmp_path / "old.db")
    _seed_v04_schema(db_path)
    database = lb.Database(db_path)
    conn = lb.Connection(database)

    # Let detect_v04_schema run normally, then poison the next execute call.
    real_execute = conn.execute
    call_count = {"n": 0}

    def flaky_execute(*args, **kwargs):
        call_count["n"] += 1
        # After the detect probes (2 calls), fail the first real migration call.
        if call_count["n"] > 2:
            raise RuntimeError("simulated LadybugDB failure")
        return real_execute(*args, **kwargs)

    with patch.object(conn, "execute", side_effect=flaky_execute):
        with pytest.raises(TrimTabMigrationError) as excinfo:
            run_migration(conn)
        # Chained cause should be our RuntimeError.
        assert excinfo.value.__cause__ is not None
        assert "simulated LadybugDB failure" in str(excinfo.value.__cause__)
