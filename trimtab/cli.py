"""CLI for TrimTab: build, index, generate."""

import argparse
import json
import sys

from trimtab.grammar import Grammar
from trimtab.embedder import get_default_embedder
from trimtab.index import GrammarIndex
from trimtab.generator import Generator
from trimtab.builder import build_grammar


def cmd_build(args):
    """Build grammar from corpus."""
    with open(args.input) as f:
        texts = [line.strip() for line in f if line.strip()]

    embedder = get_default_embedder()
    grammar = build_grammar(texts, embedder, min_count=args.min_count)
    grammar.save(args.output)
    print(f"Grammar saved to {args.output} ({len(grammar.rule_names())} rules)")


def cmd_index(args):
    """Index an existing grammar."""
    grammar = Grammar.from_file(args.grammar)
    embedder = get_default_embedder()
    gi = GrammarIndex(grammar, embedder)
    gi.build()
    gi.save(args.output or args.grammar.replace(".json", ".sg"))
    print(f"Indexed {len(grammar.rule_names())} rules")


def cmd_generate(args):
    """Generate text from indexed grammar."""
    embedder = get_default_embedder()
    gi = GrammarIndex.load(args.grammar, embedder)
    gen = Generator(gi)
    text = gen.generate(
        context=args.context,
        temperature=args.temperature,
        seed=args.seed,
    )
    print(text)


def cmd_add(args):
    """Add an expansion to a rule."""
    embedder = get_default_embedder()
    gi = GrammarIndex.load(args.grammar, embedder)
    gi.add_to_rule(args.rule, args.value)
    gi.save(args.grammar)
    print(f"Added '{args.value}' to rule '{args.rule}'")


def main():
    parser = argparse.ArgumentParser(prog="trimtab", description="Context-aware grammar generation")
    sub = parser.add_subparsers(dest="command")

    # build
    p = sub.add_parser("build", help="Build grammar from corpus")
    p.add_argument("--input", "-i", required=True, help="Input corpus file (one text per line)")
    p.add_argument("--output", "-o", default="grammar.json", help="Output grammar file")
    p.add_argument("--min-count", type=int, default=2, help="Minimum n-gram frequency")

    # index
    p = sub.add_parser("index", help="Index existing grammar")
    p.add_argument("grammar", help="Grammar JSON file")
    p.add_argument("--output", "-o", help="Output directory (default: grammar.sg)")

    # generate
    p = sub.add_parser("generate", help="Generate text")
    p.add_argument("grammar", help="Indexed grammar directory (.sg)")
    p.add_argument("--context", "-c", default="", help="Context string")
    p.add_argument("--temperature", "-t", type=float, default=0.3, help="Temperature (0=deterministic, 1=random)")
    p.add_argument("--seed", "-s", type=int, default=None, help="Random seed")

    # add
    p = sub.add_parser("add", help="Add expansion to rule")
    p.add_argument("grammar", help="Indexed grammar directory (.sg)")
    p.add_argument("rule", help="Rule name")
    p.add_argument("value", help="Expansion text to add")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    cmds = {"build": cmd_build, "index": cmd_index, "generate": cmd_generate, "add": cmd_add}
    cmds[args.command](args)


if __name__ == "__main__":
    main()
