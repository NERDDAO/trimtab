# TrimTab

**Embedded memory and generation for agentic Python.**

TrimTab is a lightweight memory store built on Tracery-style grammars, where every rule is embedded on ingest so you can list, search, or walk it with context. One inbound route (`put`), three read modes (`list`, `search`, `generate`), and a shipped Ollama default embedder.

## Install

```bash
pip install trimtab
```

TrimTab ships with Ollama as the default embedder. If you don't already have it:

```bash
# install Ollama: https://ollama.com
ollama pull nomic-embed-text
ollama serve
```

Or bring your own embedder:

```python
from trimtab import TrimTab

tt = TrimTab(path="memory.db", embedder=my_own_embedder)
```

## The 30-second model

A **grammar** is a named namespace (e.g. `"agent_abc123"`, `"dev_memory"`). Inside it, **symbols** are categories (`"friends"`, `"notes"`, `"origin"`). Under each symbol, **rules** are individual entries — the things you'd call rows in a database. Every rule carries text, optional metadata, and an embedding computed on ingest.

One way in:

```python
from trimtab import TrimTab

tt = TrimTab(path="memory.db")

await tt.put(
    grammar="agent_01", symbol="notes",
    text="The forest is dangerous at night",
    metadata={"source": "ranger_alice"},
)
```

Three ways out:

```python
# Flat enumeration — insertion ordered, no embedding math
notes = tt.list(grammar="agent_01", symbol="notes")

# Semantic search — top-k by cosine similarity
hits = await tt.search(grammar="agent_01", symbol="notes",
                       query="safe travel?", top_k=3)

# Cascading Tracery walk — context-aware structured retrieval
result = await tt.generate(grammar="agent_01",
                           context="journey planning",
                           origin="origin")
```

![Data flow](docs/diagrams/trimtab-data-flow.svg)

## Agentic memory pattern

```python
from trimtab import TrimTab

tt = TrimTab(path="~/.trimtab/agent.db")
GRAMMAR = "agent_01"

async def agent_turn(user_message: str) -> str:
    # 1. Read relevant context from accumulated memory.
    relevant = await tt.search(GRAMMAR, "notes",
                               query=user_message, top_k=5)
    context = "\n".join(f"- {r.text}" for r in relevant)

    # 2. Call your LLM with context + user message.
    response = await call_llm(context=context, message=user_message)

    # 3. Write new observations back to memory.
    await tt.put(GRAMMAR, "notes", text=f"User asked: {user_message}")
    if response.notable:
        await tt.put(GRAMMAR, "notes", text=response.summary,
                     metadata={"turn": turn_id})

    return response.text
```

The agent reads with `search`, writes with `put`, and that's the whole loop. No semantic-vs-store-mode distinction — every write is searchable by construction.

## Use cases

### Software development memory

Three symbols — rules, patterns, preferences — retrieved in one walk:

```python
await tt.put("dev_memory", "rules",
    "Close old WebSocket before creating new one in session.ts")
await tt.put("dev_memory", "patterns",
    "Background art uses daemon threads — non-blocking crew kickoff")
await tt.put("dev_memory", "preferences",
    "Scaffold mode over replace — crews keep running with pre-filled context")

await tt.put("dev_memory", "origin",
    "Rule: #rules#\nPattern: #patterns#\nPreference: #preferences#")

result = await tt.generate("dev_memory",
                           context="debugging WebSocket connection drops")
print(result.text)
# → Rule: Close old WebSocket before creating new one in session.ts
#   Pattern: Background art uses daemon threads — non-blocking crew kickoff
#   Preference: Scaffold mode over replace — crews keep running with pre-filled context
```

### Project management

```python
await tt.put("project", "status",
    "Auth — JWT flow merged, waiting on staging deploy")
await tt.put("project", "blockers",
    "Staging down — DevOps ticket INFRA-342 open since Thursday")
await tt.put("project", "next_actions",
    "Deploy auth changes once staging is restored")

await tt.put("project", "origin",
    "Status: #status#\nBlocker: #blockers#\nNext: #next_actions#")

result = await tt.generate("project",
                           context="what's blocking the auth deploy")
```

### Research / learning

```python
await tt.put("research", "facts",
    "HDBSCAN finds clusters of varying density without requiring k")
await tt.put("research", "connections",
    "HDBSCAN + embeddings is essentially unsupervised topic modeling")
await tt.put("research", "questions",
    "How does embedding quality degrade for domain-specific jargon?")

await tt.put("research", "origin",
    "Known: #facts#\nRelated: #connections#\nGap: #questions#")

result = await tt.generate("research",
                           context="clustering text without labels")
```

## CLI

```bash
# Put a rule
trimtab put agent_01 notes "The forest is dangerous at night"

# Semantic search
trimtab search agent_01 notes "safe travel" --top-k 3

# Remove a rule by id
trimtab remove agent_01 notes r_01HXYZ...

# List grammars and symbols
trimtab list
trimtab show agent_01

# Export a grammar to JSON (Tracery-compatible)
trimtab export agent_01 > agent_01.json

# Wipe and rebuild embeddings with a new embedder (post-v0.5 scope;
# currently a no-op that prints instructions)
trimtab reembed --embedder nomic-embed-text
```

## Embedder

TrimTab ships with `trimtab.embedders.OllamaEmbedder` as the default. It calls `http://localhost:11434/api/embed` and defaults to `nomic-embed-text` (768-dim, fast, commonly available).

BYO embedder — implement the `Embedder` protocol:

```python
from trimtab import Embedder, TrimTab

class MyEmbedder:
    async def create(self, text: str) -> list[float]: ...
    async def create_batch(self, texts: list[str]) -> list[list[float]]: ...

tt = TrimTab(path="memory.db", embedder=MyEmbedder())
```

**Per-DB dimension pinning.** LadybugDB fixes embedding dimension at first write. One DB file = one embedder model. If you need multiple embedders, use multiple DB files.

## Errors

| Error | When |
|---|---|
| `TrimTabEmbedderError` | Ollama unreachable / model missing / embed call failed |
| `TrimTabNotFoundError` | `update`/`remove` on an id that doesn't exist |
| `TrimTabDimensionError` | Embedder dim doesn't match the DB's pinned dim |
| `TrimTabMigrationError` | v0.4 → v0.5 auto-migration failed |
| `TrimTabGrammarError` | Malformed JSON on `load_file` |
| `TrimTabCycleError` | `generate` hit a cyclic symbol reference |

All are subclasses of `TrimTabError` — catch once to handle any.

## Status

**v0.5.0 is a breaking release.** Terminology aligned to Tracery (what was `Rule` is now `Symbol`, what was `Expansion` is now `Rule`). LadybugDB schema migrates automatically on first open. `SmartGrammar` stays as a deprecated alias; removed in v0.6. See `CHANGELOG.md` for the full list.

**Concurrency:** Single-writer per grammar. Last-write-wins if you violate that. Locks and optimistic concurrency are out of scope for v0.5.

## See also

- `trimtab/builder.py` — build a grammar from a text corpus (HDBSCAN clustering). Power-user feature, not covered here.
- `docs/superpowers/specs/2026-04-14-trimtab-memory-system-design.md` (in the Bonfires workspace) — full v0.5 design spec.

## License

MIT
