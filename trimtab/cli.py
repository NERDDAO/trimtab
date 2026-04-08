"""CLI for TrimTab: build, index, generate, add, list, show, export."""

import argparse
import json
import sys
from pathlib import Path

from trimtab.db import TrimTabDB
from trimtab.grammar import Grammar
from trimtab.embedder import get_default_embedder
from trimtab.builder import build_grammar


DEFAULT_DB = str(Path.home() / ".trimtab" / "default.db")


def _get_db(args) -> TrimTabDB:
    db_path = getattr(args, "db", None) or DEFAULT_DB
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return TrimTabDB(db_path)


def cmd_index(args):
    """Import a grammar JSON file into the DB."""
    db = _get_db(args)
    embedder = get_default_embedder()
    grammar = Grammar.from_file(args.grammar)
    name = args.name or Path(args.grammar).stem
    db.upsert_grammar(name, grammar, embedder)
    print(f"Imported '{name}' ({len(grammar.rule_names())} rules)")


def cmd_generate(args):
    """Generate text from a grammar in the DB."""
    db = _get_db(args)
    embedder = get_default_embedder()

    from trimtab.generator import Generator
    gen = Generator(db, args.grammar, embedder)
    text = gen.generate(
        context=args.context,
        temperature=args.temperature,
        seed=args.seed,
    )
    print(text)


def cmd_add(args):
    """Add an expansion to a rule."""
    db = _get_db(args)
    embedder = get_default_embedder()
    vec = embedder.embed([args.value])[0]
    db.add_expansion(args.grammar, args.rule, args.value, vec)
    print(f"Added '{args.value}' to {args.grammar}:{args.rule}")


def cmd_build(args):
    """Build grammar from corpus."""
    db = _get_db(args)
    embedder = get_default_embedder()
    with open(args.input) as f:
        texts = [line.strip() for line in f if line.strip()]
    grammar = build_grammar(texts, embedder, min_count=args.min_count)
    name = args.name or Path(args.input).stem
    db.upsert_grammar(name, grammar, embedder)
    print(f"Built and imported '{name}' ({len(grammar.rule_names())} rules)")


def cmd_list(args):
    """List all grammars in the DB."""
    db = _get_db(args)
    names = db.list_grammars()
    if not names:
        print("No grammars in database.")
        return
    for name in sorted(names):
        grammar = db.get_grammar(name)
        print(f"  {name} ({len(grammar.rule_names())} rules)")


def cmd_show(args):
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


def cmd_export(args):
    """Export a grammar to JSON."""
    db = _get_db(args)
    grammar = db.get_grammar(args.grammar)
    if not grammar.rule_names():
        print(f"Grammar '{args.grammar}' not found.", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(grammar.rules, indent=2, ensure_ascii=False))


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
    }
    cmds[args.command](args)


if __name__ == "__main__":
    main()
