"""TrimTab — embedded memory and generation for agentic Python.

Primary entry point: ``trimtab.TrimTab``. See the README for the 30-second
model and the v0.5 design at
``docs/superpowers/specs/2026-04-14-trimtab-memory-system-design.md``
in the Bonfires workspace.
"""

from __future__ import annotations

from trimtab.builder import build_grammar, cluster_ngrams, extract_ngrams
from trimtab.core import TrimTab
from trimtab.db import TrimTabDB
from trimtab.embedder import Embedder
from trimtab.errors import (
    TrimTabCycleError,
    TrimTabDimensionError,
    TrimTabEmbedderError,
    TrimTabError,
    TrimTabGrammarError,
    TrimTabMigrationError,
    TrimTabNotFoundError,
)
from trimtab.generator import GenerationResult, Generator
from trimtab.grammar import Grammar, Rule

__version__ = "0.6.0"

__all__ = [
    "TrimTab",
    "Grammar",
    "Rule",
    "Embedder",
    "TrimTabDB",
    "Generator",
    "GenerationResult",
    "build_grammar",
    "extract_ngrams",
    "cluster_ngrams",
    # Errors
    "TrimTabError",
    "TrimTabEmbedderError",
    "TrimTabNotFoundError",
    "TrimTabDimensionError",
    "TrimTabMigrationError",
    "TrimTabGrammarError",
    "TrimTabCycleError",
]
