"""LadybugDB-backed storage for grammars, rules, and embeddings."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import real_ladybug as lb

from trimtab.embedder import Embedder
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
        self._init_schema()
        self._init_vector_extension()
        self._embedding_dim: int | None = None
        # Run v0.4 → v0.5 migration if the old schema is detected.
        # Idempotent on fresh and already-migrated DBs.
        from trimtab.migrations import run_migration
        try:
            run_migration(self._conn)
        except Exception as e:
            logger.warning("Migration attempt failed (non-fatal): %s", e)
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

    def _ensure_expansion_table(self, dim: int) -> None:
        """Create Expansion table and HNSW index with correct embedding dimension."""
        if self._embedding_dim == dim:
            return

        if self._embedding_dim is not None and self._embedding_dim != dim:
            # Dimension changed — need to recreate
            try:
                self._conn.execute("DROP TABLE IF EXISTS HAS_EXPANSION")
            except Exception:
                pass
            try:
                self._conn.execute("DROP TABLE IF EXISTS Expansion")
            except Exception:
                pass

        try:
            self._conn.execute(
                f"CREATE NODE TABLE IF NOT EXISTS Expansion("
                f"  id STRING PRIMARY KEY,"
                f"  text STRING,"
                f"  rule_id STRING,"
                f"  grammar STRING,"
                f"  embedding FLOAT[{dim}]"
                f")"
            )
        except Exception:
            pass  # already exists

        try:
            self._conn.execute(
                "CREATE REL TABLE IF NOT EXISTS HAS_EXPANSION(FROM Rule TO Expansion)"
            )
        except Exception:
            pass

        self._embedding_dim = dim

    def _ensure_hnsw_index(self) -> None:
        """Create HNSW index if it doesn't exist yet."""
        try:
            self._conn.execute(
                "CALL CREATE_VECTOR_INDEX("
                "  'Expansion', 'expansion_embedding_idx', 'embedding',"
                "  metric := 'cosine'"
                ")"
            )
        except Exception:
            pass  # index may already exist

    def _upsert_expansion(
        self, exp_id: str, text: str, rule_id: str, grammar: str, vec_list: list
    ) -> None:
        """Insert or replace an expansion node.

        LadybugDB forbids MERGE/SET on columns covered by a vector index,
        so we DETACH DELETE the old node (if any) then CREATE a fresh one.
        """
        try:
            self._conn.execute(
                "MATCH (e:Expansion) WHERE e.id = $id DETACH DELETE e",
                {"id": exp_id},
            )
        except Exception:
            pass  # node may not exist

        self._conn.execute(
            "CREATE (e:Expansion {"
            "  id: $id, text: $text, rule_id: $rule_id,"
            "  grammar: $grammar, embedding: $embedding"
            "})",
            {
                "id": exp_id,
                "text": text,
                "rule_id": rule_id,
                "grammar": grammar,
                "embedding": vec_list,
            },
        )

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

    async def upsert_grammar(self, name: str, grammar: Grammar, embedder: Embedder) -> None:
        """Import a Grammar object into the DB. Embeds all expansions.

        Honors explicit expansion ids when the grammar supplies them (via
        ``{"text", "id"}`` dict entries in ``Grammar.rules``). Expansions
        with no explicit id fall back to the auto-generated
        ``{name}:{rule_name}:{hash(text)}`` shape.

        Args:
            name: Grammar name (e.g., "dev-memory", "narrator").
            grammar: Grammar object with rules and expansions.
            embedder: Async embedder (graphiti_core EmbedderClient or any
                Protocol-compatible object with ``create`` / ``create_batch``).
        """
        # Probe embedding dimension so we can create the fixed-width table.
        probe = await embedder.create("dimension probe")
        self._ensure_expansion_table(len(probe))

        # Create grammar node
        self._conn.execute(
            "MERGE (g:Grammar {name: $name})",
            {"name": name},
        )

        for rule_name in grammar.rule_names():
            rule_id = f"{name}:{rule_name}"

            # Create rule node
            self._conn.execute(
                "MERGE (r:Rule {id: $id}) ON CREATE SET r.name = $name, r.grammar = $grammar "
                "ON MATCH SET r.name = $name, r.grammar = $grammar",
                {"id": rule_id, "name": rule_name, "grammar": name},
            )

            # Create HAS_RULE edge
            self._conn.execute(
                "MATCH (g:Grammar), (r:Rule) "
                "WHERE g.name = $gname AND r.id = $rid "
                "MERGE (g)-[:HAS_RULE]->(r)",
                {"gname": name, "rid": rule_id},
            )

            # Insert expansions — honor consumer-supplied ids when present.
            expansion_items = grammar.get_expansion_items(rule_name)
            if not expansion_items:
                continue

            texts = [text for text, _ in expansion_items]
            vectors = await embedder.create_batch(texts)

            for (text, explicit_id), vec in zip(expansion_items, vectors, strict=True):
                exp_id = explicit_id if explicit_id else f"{name}:{rule_name}:{hash(text)}"
                self._upsert_expansion(exp_id, text, rule_id, name, vec)

                # Create HAS_EXPANSION edge
                self._conn.execute(
                    "MATCH (r:Rule), (e:Expansion) "
                    "WHERE r.id = $rid AND e.id = $eid "
                    "MERGE (r)-[:HAS_EXPANSION]->(e)",
                    {"rid": rule_id, "eid": exp_id},
                )

        # Create HNSW index after all data is inserted (avoids the MERGE/SET
        # restriction on indexed columns during bulk load).
        self._ensure_hnsw_index()

        logger.info("Upserted grammar '%s' with %d rules", name, len(grammar.rule_names()))

    async def register_grammar(self, name: str, grammar: Grammar, embedder: Embedder) -> None:
        """Register a grammar's rule structure without embedding expansions.

        Use this when expansion vectors will be added separately via
        ``add_expansion`` with precomputed embeddings — for example, when
        loading from a cache of previously-embedded text. Creates the
        Grammar node, Rule nodes, and HAS_RULE edges. Does not create
        Expansion nodes.

        The embedder is only used to probe the vector dimension if this is
        the first grammar being registered on this DB instance. Subsequent
        calls reuse the cached dimension and make zero embedder calls.

        Args:
            name: Grammar name (e.g. "bonfire123:applicant_review").
            grammar: Grammar object with rule structure.
            embedder: Async embedder (only used for the one-shot dim probe).
        """
        if self._embedding_dim is None:
            probe = await embedder.create("dimension probe")
            self._ensure_expansion_table(len(probe))

        self._conn.execute(
            "MERGE (g:Grammar {name: $name})",
            {"name": name},
        )

        for rule_name in grammar.rule_names():
            rule_id = f"{name}:{rule_name}"
            self._conn.execute(
                "MERGE (r:Rule {id: $id}) ON CREATE SET r.name = $name, r.grammar = $grammar "
                "ON MATCH SET r.name = $name, r.grammar = $grammar",
                {"id": rule_id, "name": rule_name, "grammar": name},
            )
            self._conn.execute(
                "MATCH (g:Grammar), (r:Rule) "
                "WHERE g.name = $gname AND r.id = $rid "
                "MERGE (g)-[:HAS_RULE]->(r)",
                {"gname": name, "rid": rule_id},
            )

        self._ensure_hnsw_index()

        logger.info(
            "Registered grammar '%s' with %d rules (no embeddings)",
            name,
            len(grammar.rule_names()),
        )

    def add_expansion(
        self,
        grammar: str,
        rule: str,
        text: str,
        embedding: list[float],
        id: str | None = None,
    ) -> None:
        """Add a single expansion with its embedding vector.

        Args:
            grammar: Grammar name.
            rule: Rule name within the grammar.
            text: Expansion text.
            embedding: Pre-computed embedding vector (list of floats, as
                produced by an embedder's ``create(text)`` method).
            id: Optional custom Expansion id. If None, auto-generates
                "{grammar}:{rule}:{hash(text)}". Set this to associate the
                expansion with an external entity (e.g., a KG entity UUID).
        """
        rule_id = f"{grammar}:{rule}"
        exp_id = id if id is not None else f"{grammar}:{rule}:{hash(text)}"

        # Ensure rule exists
        self._conn.execute(
            "MERGE (r:Rule {id: $id}) ON CREATE SET r.name = $name, r.grammar = $grammar "
            "ON MATCH SET r.name = $name",
            {"id": rule_id, "name": rule, "grammar": grammar},
        )

        # Insert expansion (delete+create to work with vector index)
        self._upsert_expansion(exp_id, text, rule_id, grammar, embedding)

        # Create HAS_EXPANSION edge
        self._conn.execute(
            "MATCH (r:Rule), (e:Expansion) "
            "WHERE r.id = $rid AND e.id = $eid "
            "MERGE (r)-[:HAS_EXPANSION]->(e)",
            {"rid": rule_id, "eid": exp_id},
        )

    def query(
        self, grammar: str, rule: str, vector: list[float], top_k: int = 5
    ) -> list[tuple[str, float, str]]:
        """Vector similarity search within a specific rule's expansions.

        Args:
            grammar: Grammar name.
            rule: Rule name.
            vector: Query embedding (list of floats from embedder.create()).
            top_k: Number of results to return.

        Returns list of (text, score, id) tuples sorted by relevance (highest first).
        Score is cosine similarity (higher = more similar).
        The id is the Expansion.id — auto-generated by default, or consumer-provided
        via add_expansion(id=...).
        """
        rule_id = f"{grammar}:{rule}"

        try:
            result = self._conn.execute(
                "CALL QUERY_VECTOR_INDEX('Expansion', 'expansion_embedding_idx', $vec, $k) "
                "WHERE node.rule_id = $rule_id "
                "RETURN node.text, distance, node.id "
                "ORDER BY distance",
                {"vec": vector, "k": top_k * 4, "rule_id": rule_id},
            )
            rows = result.get_all()
        except Exception:
            rows = []

        # Fall back to brute force if HNSW returned nothing.
        # New nodes added after _ensure_hnsw_index() may not be indexed yet,
        # so empty HNSW results don't necessarily mean no matching data.
        if not rows:
            rows = self._brute_force_query(rule_id, vector, top_k)

        results = []
        for row in rows[:top_k]:
            text = row[0]
            distance = float(row[1])
            score = 1.0 - distance
            exp_id = row[2] if len(row) > 2 else ""
            results.append((text, score, exp_id))

        return results

    def _brute_force_query(
        self, rule_id: str, vector: list[float], top_k: int
    ) -> list[list]:
        """Fallback: fetch all expansions for rule and compute cosine similarity."""
        result = self._conn.execute(
            "MATCH (e:Expansion) WHERE e.rule_id = $rule_id "
            "RETURN e.text, e.embedding, e.id",
            {"rule_id": rule_id},
        )
        rows = result.get_all()
        if not rows:
            return []

        query_arr = np.array(vector, dtype=np.float32)
        vec_norm = query_arr / (np.linalg.norm(query_arr) + 1e-8)
        scored = []
        for text, emb, exp_id in rows:
            emb_arr = np.array(emb, dtype=np.float32)
            emb_norm = emb_arr / (np.linalg.norm(emb_arr) + 1e-8)
            sim = float(np.dot(vec_norm, emb_norm))
            distance = 1.0 - sim
            scored.append([text, distance, exp_id or ""])

        scored.sort(key=lambda x: x[1])
        return scored[:top_k]

    def get_grammar(self, name: str) -> Grammar:
        """Export a grammar from the DB back to a Grammar object.

        Returns expansions as plain text strings (the round-trip shape).
        To also get expansion ids, use ``get_expansions_with_ids``.
        """
        from trimtab.grammar import ExpansionEntry

        result = self._conn.execute(
            "MATCH (r:Rule) WHERE r.grammar = $grammar RETURN r.name",
            {"grammar": name},
        )
        rules: dict[str, list[ExpansionEntry]] = {}
        for row in result.get_all():
            rule_name = row[0]
            rules[rule_name] = list(self.get_expansions(name, rule_name))
        return Grammar(rules=rules)

    def get_expansions(self, grammar: str, rule: str) -> list[str]:
        """Get all expansion texts for a specific rule."""
        rule_id = f"{grammar}:{rule}"
        result = self._conn.execute(
            "MATCH (e:Expansion) WHERE e.rule_id = $rule_id RETURN e.text",
            {"rule_id": rule_id},
        )
        return [row[0] for row in result.get_all()]

    def list_entries(self, grammar: str, rule: str) -> list[tuple[str, str]]:
        """List all (text, id) pairs for a rule — the store-read mode.

        Unlike ``query`` (which does embedding similarity search), this returns
        every expansion in the rule with no ranking or filtering. Use this when
        you want to treat a rule as a flat table rather than a search index.

        Returns:
            List of ``(text, id)`` tuples. The id is the consumer-supplied
            expansion id (e.g., a KG entity UUID or path-tagged string), or
            the auto-generated ``{grammar}:{rule}:{hash}`` if none was set.
        """
        rule_id = f"{grammar}:{rule}"
        result = self._conn.execute(
            "MATCH (e:Expansion) WHERE e.rule_id = $rule_id RETURN e.text, e.id",
            {"rule_id": rule_id},
        )
        return [(row[0], row[1]) for row in result.get_all()]

    def list_grammars(self) -> list[str]:
        """List all grammar names in the DB."""
        result = self._conn.execute("MATCH (g:Grammar) RETURN g.name")
        return [row[0] for row in result.get_all()]
