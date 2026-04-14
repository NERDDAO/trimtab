"""CLI for TrimTab: build, index, generate, add, list, show, export.

Uses the shipped OllamaEmbedder from trimtab.embedders. Any
Protocol-compatible embedder can be used programmatically.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from trimtab.builder import build_grammar
from trimtab.db import TrimTabDB
from trimtab.embedders import OllamaEmbedder
from trimtab.grammar import Grammar


DEFAULT_DB = str(Path.home() / ".trimtab" / "default.db")


def _get_db(args) -> TrimTabDB:
    db_path = getattr(args, "db", None) or DEFAULT_DB
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return TrimTabDB(db_path)


def _get_embedder() -> OllamaEmbedder:
    return OllamaEmbedder()


async def cmd_index(args):
    """Import a grammar JSON file into the DB."""
    db = _get_db(args)
    embedder = _get_embedder()
    grammar = Grammar.from_file(args.grammar)
    name = args.name or Path(args.grammar).stem
    await db.upsert_grammar(name, grammar, embedder)
    print(f"Imported '{name}' ({len(grammar.rule_names())} rules)")


async def cmd_generate(args):
    """Generate text from a grammar in the DB."""
    db = _get_db(args)
    embedder = _get_embedder()

    from trimtab.generator import Generator
    gen = Generator(db, args.grammar, embedder)
    text = await gen.generate(
        context=args.context,
        temperature=args.temperature,
        seed=args.seed,
    )
    print(text)


async def cmd_add(args):
    """Add an expansion to a rule."""
    db = _get_db(args)
    embedder = _get_embedder()
    vec = await embedder.create(args.value)
    custom_id = getattr(args, "id", None) or None
    db.add_expansion(args.grammar, args.rule, args.value, vec, id=custom_id)
    msg = f"Added '{args.value}' to {args.grammar}:{args.rule}"
    if custom_id:
        msg += f" (id={custom_id})"
    print(msg)


async def cmd_build(args):
    """Build grammar from corpus."""
    db = _get_db(args)
    embedder = _get_embedder()
    with open(args.input) as f:
        texts = [line.strip() for line in f if line.strip()]
    grammar = await build_grammar(texts, embedder, min_count=args.min_count)
    name = args.name or Path(args.input).stem
    await db.upsert_grammar(name, grammar, embedder)
    print(f"Built and imported '{name}' ({len(grammar.rule_names())} rules)")


async def cmd_list(args):
    """List all grammars in the DB."""
    db = _get_db(args)
    names = db.list_grammars()
    if not names:
        print("No grammars in database.")
        return
    for name in sorted(names):
        grammar = db.get_grammar(name)
        print(f"  {name} ({len(grammar.rule_names())} rules)")


async def cmd_show(args):
    """Show rules and expansions for a grammar."""
    db = _get_db(args)
    grammar = db.get_grammar(args.grammar)
    if not grammar.rule_names():
        print(f"Grammar '{args.grammar}' not found.")
        return
    for rule in grammar.rule_names():
        expansions = grammar.get_expansions(rule)
        print(f"  {rule}: [{len(expansions)} expansions]")
        for exp in expansions:
            print(f"    - {exp}")


async def cmd_export(args):
    """Export a grammar to JSON."""
    db = _get_db(args)
    grammar = db.get_grammar(args.grammar)
    if not grammar.rule_names():
        print(f"Grammar '{args.grammar}' not found.", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(grammar.rules, indent=2, ensure_ascii=False))


async def cmd_put(args):
    """Insert a rule into a grammar/symbol."""
    from trimtab.core import TrimTab
    tt = TrimTab(path=args.db or DEFAULT_DB, embedder=_get_embedder())
    metadata = json.loads(args.metadata) if args.metadata else None
    rule = await tt.put(
        grammar=args.grammar,
        symbol=args.symbol,
        text=args.text,
        metadata=metadata,
        id=args.id,
    )
    print(f"put {rule.id}  {rule.text}")


async def cmd_search(args):
    """Semantic search within a grammar/symbol."""
    from trimtab.core import TrimTab
    tt = TrimTab(path=args.db or DEFAULT_DB, embedder=_get_embedder())
    rules = await tt.search(
        grammar=args.grammar,
        symbol=args.symbol,
        query=args.query,
        top_k=args.top_k,
    )
    if not rules:
        print("(no results)")
        return
    for rule in rules:
        print(f"{rule.id}  {rule.text}")


async def cmd_remove(args):
    """Delete a rule by id."""
    from trimtab.core import TrimTab
    tt = TrimTab(path=args.db or DEFAULT_DB, embedder=_get_embedder())
    tt.remove(grammar=args.grammar, symbol=args.symbol, rule_id=args.id)
    print(f"removed {args.id}")


async def cmd_reembed(args):
    """Stubbed placeholder for v0.5.

    Full wipe-and-rebuild is post-v0.5 scope; for now the user must
    delete the DB file and re-ingest with the new embedder.
    """
    print(
        "reembed: v0.5 does not yet wipe-and-rebuild in place. Delete the "
        "DB file and re-ingest with the new embedder, or open an issue."
    )
    return 1


def main():
    parser = argparse.ArgumentParser(prog="trimtab", description="Context-aware grammar generation")
    parser.add_argument("--db", default=None, help=f"Database path (default: {DEFAULT_DB})")
    sub = parser.add_subparsers(dest="command")

    # index (import)
    p = sub.add_parser("index", help="Import grammar JSON into DB")
    p.add_argument("grammar", help="Grammar JSON file")
    p.add_argument("--name", "-n", default=None, help="Grammar name (default: filename stem)")

    # generate
    p = sub.add_parser("generate", help="Generate text")
    p.add_argument("grammar", help="Grammar name in DB")
    p.add_argument("--context", "-c", default="", help="Context string")
    p.add_argument("--temperature", "-t", type=float, default=0.3)
    p.add_argument("--seed", "-s", type=int, default=None)

    # add
    p = sub.add_parser("add", help="Add expansion to rule")
    p.add_argument("grammar", help="Grammar name in DB")
    p.add_argument("rule", help="Rule name")
    p.add_argument("value", help="Expansion text to add")
    p.add_argument("--id", default=None, help="Custom Expansion id (e.g., KG entity UUID)")

    # build
    p = sub.add_parser("build", help="Build grammar from corpus")
    p.add_argument("--input", "-i", required=True, help="Corpus file (one text per line)")
    p.add_argument("--name", "-n", default=None, help="Grammar name (default: filename stem)")
    p.add_argument("--min-count", type=int, default=2)

    # list
    sub.add_parser("list", help="List all grammars in DB")

    # show
    p = sub.add_parser("show", help="Show grammar rules and expansions")
    p.add_argument("grammar", help="Grammar name")

    # export
    p = sub.add_parser("export", help="Export grammar to JSON")
    p.add_argument("grammar", help="Grammar name")

    # --- put
    p_put = sub.add_parser("put", help="insert a rule into a grammar/symbol")
    p_put.add_argument("grammar")
    p_put.add_argument("symbol")
    p_put.add_argument("text")
    p_put.add_argument("--metadata", help="JSON metadata dict", default=None)
    p_put.add_argument("--id", default=None)
    p_put.set_defaults(func=cmd_put)

    # --- search
    p_search = sub.add_parser("search", help="semantic search within a symbol")
    p_search.add_argument("grammar")
    p_search.add_argument("symbol")
    p_search.add_argument("query")
    p_search.add_argument("--top-k", type=int, default=5)
    p_search.set_defaults(func=cmd_search)

    # --- remove
    p_rm = sub.add_parser("remove", help="delete a rule by id")
    p_rm.add_argument("grammar")
    p_rm.add_argument("symbol")
    p_rm.add_argument("id")
    p_rm.set_defaults(func=cmd_remove)

    # --- reembed
    p_re = sub.add_parser("reembed", help="wipe and rebuild embeddings for a DB")
    p_re.add_argument("--embedder", default=None, help="embedder selector (reserved)")
    p_re.set_defaults(func=cmd_reembed)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    cmds = {
        "index": cmd_index,
        "generate": cmd_generate,
        "add": cmd_add,
        "build": cmd_build,
        "list": cmd_list,
        "show": cmd_show,
        "export": cmd_export,
        "put": cmd_put,
        "search": cmd_search,
        "remove": cmd_remove,
        "reembed": cmd_reembed,
    }
    asyncio.run(cmds[args.command](args))


if __name__ == "__main__":
    main()
