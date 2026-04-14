# Changelog

## 0.5.0 — 2026-04-14

**Breaking release.** TrimTab evolves from a generation library into a
lightweight embedded memory system.

### Added

- `TrimTab` public façade class with the full CRUD + read API
  (`put`, `put_many`, `update`, `remove`, `clear`, `drop`, `list`,
  `search`, `generate`, `load_file`, `export_file`, plus introspection).
- `Rule` dataclass with typed metadata, stable id, and timestamps.
- `trimtab.embedders.OllamaEmbedder` — shipped default embedder that
  calls Ollama's `/api/embed` endpoint with `nomic-embed-text` by default.
- New CLI commands: `put`, `search`, `remove`, `reembed`.
- Full error class hierarchy under `TrimTabError`, including
  `TrimTabCycleError` raised by `generate` on cyclic grammars.
- End-to-end integration test exercising the full pipeline.

### Changed

- **Terminology aligned to Tracery.** What the old code called a
  `Rule` is now a `Symbol`; what the old code called an `Expansion`
  is now a `Rule`. All public APIs use the new terms.
- **LadybugDB schema double-renamed** with an automatic migration on
  first open. Migration is idempotent. Backup any DB files you rely on
  outside of trimtab's own Python API before upgrading.
- `Generator` detects cyclic symbol references and raises
  `TrimTabCycleError` instead of silently truncating at depth 20.
  Depth guard bumped from 20 to 50 for pathological-but-acyclic grammars.
- `TrimTab.search` returns `list[Rule]` — full objects with metadata.
  The old `TrimTabDB.query` tuple shape is deprecated.
- Metadata is stored as a `"json:" + json.dumps(...)` prefixed string
  to bypass LadybugDB's auto-parse of JSON-looking values.
- Per-DB dimension pinning: one DB file = one embedder model.
  Mixing embedder models across grammars in one DB raises
  `TrimTabDimensionError`.
- Temperature sampling at `temperature > 0` now uses uniform selection
  over the top-k cosine-ranked candidates instead of softmax-weighted
  sampling. The ranking pool is still score-ordered; only the intra-pool
  weighting is uniform.

### Deprecated

- `SmartGrammar` — use `TrimTab` instead. Kept as a standalone wrapper
  for one release; construction emits `DeprecationWarning`. Will be
  removed in v0.6.
- `TrimTabDB.upsert_grammar`, `register_grammar`, `add_expansion`,
  `query`, `get_expansions`, `list_entries` — all still work, all emit
  `DeprecationWarning`. Reroute through the new v0.5 internals. Will
  be removed in v0.6.

### Migration

- On first open of a v0.4 DB, trimtab runs the schema migration
  automatically. No action required for callers using the public Python
  API. The migration drops old `Rule` and `Expansion` tables and
  finalizes canonical `Rule` and `HAS_RULE` names.
- Callers that reach into LadybugDB directly (not through trimtab's
  public API) must update their Cypher to use the new `Symbol` / `Rule`
  node labels.
- If you use multiple embedder models, use one DB file per model.

### Fixed

- v0.4 → v0.5 migration order: migration now runs BEFORE `_init_schema`
  so an empty pre-v0.5 DB with a freshly-created `Symbol` table doesn't
  confuse the migration detector.
- Migrated rules store metadata with the `"json:"` prefix so post-migration
  reads via `_get_rules` don't crash on an empty string slice.
