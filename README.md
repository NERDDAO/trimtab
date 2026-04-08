# SmartGrammar

Context-aware grammar generation with cascading embedding search.

Combines Tracery-style grammars with embedding-based selection. Grammar rules define **structure** (enforced). Embeddings select the most **contextually appropriate** expansion at each level (cascading). The grammar tree is always respected — only the content within each slot is selected by semantic relevance.

## Install

```bash
pip install smartgrammar
```

## Quick Start

```python
from smartgrammar import SmartGrammar

# Load a grammar and index it
sg = SmartGrammar.from_file("narration.json")
sg.index()

# Generate with context — embedding selects the best expansion at each level
text = sg.generate(context="dark crypt, eerie mood", temperature=0.3)
# → "Chains rattle from the depths below. The stale air tastes of decay."

# Same grammar, different context
text = sg.generate(context="sunlit forest, peaceful", temperature=0.3)
# → "Wind howls through the clearing. The warm air carries a faint breeze."
```

## How It Works

### Cascading Context

At each level of the grammar tree, the chosen expansion feeds into the context for the next level:

```
Context: "dark crypt, eerie mood"

origin → "#sensory#. #atmosphere#."     (structure enforced)
  sensory → "#sound# echoes from #direction#"
    sound → "Chains rattle"              ← embedding picks this for "crypt"
    direction → "the depths below"       ← embedding picks this for "crypt + chains"
  atmosphere → "The #air# air tastes of #smell#"
    air → "stale"                        ← embedding picks this for "crypt + chains + depths"
    smell → "decay"                      ← embedding picks this for all of the above
```

Grammar structure is always enforced. Content is selected by semantic relevance cascading through the tree. This is not random generation — it's structured retrieval.

### Temperature

```python
sg.generate(context, temperature=0.0)  # always top-1 (deterministic)
sg.generate(context, temperature=0.3)  # weighted sample from top-5 (recommended)
sg.generate(context, temperature=1.0)  # uniform random (classic Tracery behavior)
```

## Use Cases

### 1. Software Development Memory

Template structure: what rule applies, what pattern to follow, what the user prefers.

```json
{
  "origin": ["Rule: #rules#\nPattern: #patterns#\nPreference: #preferences#"],
  "rules": [
    "Always close old WebSocket before creating new one in session.ts",
    "Use get_entity by UUID not text search — names are display only",
    "Run impact analysis before editing any symbol — check blast radius",
    "Don't hand-roll when a library exists — check pip and GitHub first"
  ],
  "patterns": [
    "Crews consume procgen scaffolds — scaffold param on first task description",
    "Background art uses daemon threads — non-blocking crew kickoff",
    "Config resolves via get_model_for_crew — crew name to override key",
    "NPC agents act through mm_ tools — dialogue movement inventory"
  ],
  "preferences": [
    "Scaffold mode over replace — crews keep running with pre-filled context",
    "Subagent-driven development for plan execution",
    "Leverage open source over hand-rolling",
    "Flat KG attributes not nested JSON — Neo4j friendly"
  ]
}
```

```python
sg = SmartGrammar.from_file("dev_memory.json")
sg.index()

sg.generate(context="debugging WebSocket connection drops")
# → Rule: Always close old WebSocket before creating new one in session.ts
#   Pattern: Background art uses daemon threads — non-blocking crew kickoff
#   Preference: Scaffold mode over replace — crews keep running with pre-filled context
```

Every query returns a relevant rule, a relevant architectural pattern, and a relevant preference — structured multi-faceted recall, not just "find the one best match."

### 2. Project Management

Template structure: where things stand, what's in the way, what to do next.

```json
{
  "origin": ["Status: #status#\nBlocker: #blockers#\nNext: #next_actions#"],
  "status": [
    "Auth system — JWT refresh flow merged, waiting on staging deploy",
    "Payment integration — Stripe webhook handler 80% complete, needs error handling",
    "Mobile app — v2.1 release branch cut, QA in progress",
    "Search rewrite — Elasticsearch migration planned, not started",
    "Onboarding flow — A/B test running since Monday, early results positive"
  ],
  "blockers": [
    "Staging environment is down — DevOps ticket INFRA-342 open since Thursday",
    "Stripe test keys expired — waiting on finance team to rotate",
    "iOS build failing on CI — Xcode 16 compatibility issue",
    "Search index rebuild requires 4h maintenance window — needs scheduling",
    "Design review for onboarding v2 not yet scheduled"
  ],
  "next_actions": [
    "Deploy auth changes to staging once environment is restored",
    "Write integration tests for Stripe webhook edge cases",
    "Update CI to Xcode 16 and fix Swift package resolution",
    "Draft Elasticsearch migration plan and circulate for review",
    "Schedule design review for onboarding with product team"
  ]
}
```

```python
sg = SmartGrammar.from_file("project.json")
sg.index()

sg.generate(context="payment processing, what's the state of Stripe")
# → Status: Payment integration — Stripe webhook handler 80% complete, needs error handling
#   Blocker: Stripe test keys expired — waiting on finance team to rotate
#   Next: Write integration tests for Stripe webhook edge cases
```

The cascading context means the blocker and next action are selected in light of the status — not independently. If the status is about payments, the blocker will be payment-related too.

### 3. Research / Learning

Template structure: what you know, what connects to it, what's still unknown.

```json
{
  "origin": ["Known: #facts#\nRelated: #connections#\nGap: #questions#"],
  "facts": [
    "Transformer attention is O(n^2) in sequence length",
    "Wave Function Collapse generates tilemaps from adjacency constraints",
    "HDBSCAN finds clusters of varying density without requiring k",
    "Embedding cosine similarity approximates semantic relatedness",
    "N-gram frequency follows Zipf's law — few very common, long tail of rare"
  ],
  "connections": [
    "WFC adjacency rules are similar to Markov chain transition matrices",
    "HDBSCAN clustering + embeddings is essentially unsupervised topic modeling",
    "Cascading context in grammar expansion mirrors beam search decoding",
    "N-gram extraction from LLM output is a form of knowledge distillation",
    "Tracery grammars are context-free — SmartGrammar adds context-sensitivity via embeddings"
  ],
  "questions": [
    "Can cascading embedding search approximate attention without quadratic cost?",
    "What happens when grammar rules have cyclic references?",
    "How does embedding quality degrade for domain-specific jargon?",
    "Is there a principled way to set the confidence threshold per rule?",
    "Could grammar structure itself be learned from the embedding clusters?"
  ]
}
```

```python
sg = SmartGrammar.from_file("research.json")
sg.index()

sg.generate(context="using embeddings to cluster text without labeled data")
# → Known: HDBSCAN finds clusters of varying density without requiring k
#   Related: HDBSCAN clustering + embeddings is essentially unsupervised topic modeling
#   Gap: How does embedding quality degrade for domain-specific jargon?
```

The three slots work as a learning scaffold: here's what you know, here's how it connects, here's what you don't know yet.

## Building Grammars From Text

Don't have a grammar? Build one from a corpus of text:

```python
sg = SmartGrammar.build_from_corpus(texts=[
    "The server crashed at 3am due to memory leak in the cache layer",
    "Database connection pool exhausted during peak traffic",
    "JWT validation failing silently — tokens expired but no error logged",
    # ... hundreds more lines from incident reports, standups, etc.
])
sg.index()
```

The builder extracts n-grams, clusters them by embedding similarity, and creates grammar rules from the clusters. Optionally, an LLM crew pass names the clusters and prunes garbage entries.

## Self-Improving

Grammars grow over time. When a crew or user produces good output, feed it back:

```python
sg.add("rules", "Never force-push to main without explicit user confirmation")
# Auto-embeds and indexes — available for next query
```

Combined with n-gram extraction from ongoing work, the grammar becomes a distillation of accumulated knowledge.

## CLI

```bash
# Index a grammar for embedding search
smartgrammar index grammar.json

# Generate with context
smartgrammar generate grammar.sg --context "dark crypt" --temperature 0.3

# Build grammar from text corpus
smartgrammar build --input corpus.txt --output grammar.json

# Add entry to a rule
smartgrammar add grammar.sg sound "Bones crunch underfoot"
```

## Embedder

Ships with `sentence-transformers/all-MiniLM-L6-v2` (~80MB, works offline). Auto-upgrades to Ollama if running locally for better quality.

```python
from smartgrammar import OllamaEmbedder, SmartGrammar

sg = SmartGrammar.from_file("grammar.json", embedder=OllamaEmbedder(model="nomic-embed-text"))
```

## License

MIT
