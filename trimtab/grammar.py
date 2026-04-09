"""Tracery-compatible grammar loading, saving, and traversal.

An expansion is either a plain string (text only, ids get auto-generated
on upsert) or a ``{"text": str, "id": str}`` dict (consumer-provided id,
e.g. a KG entity UUID or path-tagged string). Mixing is allowed within
a single rule's expansion list.
"""

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

_RULE_PATTERN = re.compile(r"#(\w+)#")

# An expansion is either plain text or {"text": ..., "id": ...}.
ExpansionEntry = str | dict[str, str]


def _expansion_text(entry: ExpansionEntry) -> str:
    """Extract the text of an expansion entry (handles both shapes)."""
    if isinstance(entry, str):
        return entry
    return entry.get("text", "")


def _expansion_id(entry: ExpansionEntry) -> str | None:
    """Extract the explicit id of an expansion entry, or None for auto-id."""
    if isinstance(entry, str):
        return None
    return entry.get("id") or None


@dataclass
class Grammar:
    """A Tracery-compatible grammar with named rules and expansions.

    Each rule's expansion list is heterogeneous — entries can be either
    plain strings (auto-id on upsert) or ``{"text", "id"}`` dicts
    (consumer-supplied id, e.g. KG entity UUID or path-tagged string).
    """

    rules: dict[str, list[ExpansionEntry]] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str | Path) -> "Grammar":
        data = json.loads(Path(path).read_text())
        return cls(rules=data)

    @classmethod
    def from_dict(cls, rules: Mapping[str, Sequence[ExpansionEntry]]) -> "Grammar":
        """Build a Grammar from a rules mapping.

        Accepts both the plain-text shape (``dict[str, list[str]]``) and
        the typed shape (``dict[str, list[dict[str, str]]]``). Uses
        ``Mapping``/``Sequence`` so dict invariance doesn't force callers
        to annotate with the precise ``ExpansionEntry`` element type.
        """
        materialized: dict[str, list[ExpansionEntry]] = {
            rule_name: list(expansions) for rule_name, expansions in rules.items()
        }
        return cls(rules=materialized)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.rules, indent=2, ensure_ascii=False) + "\n")

    def get_expansions(self, rule: str) -> list[str]:
        """Return the expansion text strings for a rule (backward-compat)."""
        return [_expansion_text(e) for e in self.rules.get(rule, [])]

    def get_expansion_items(self, rule: str) -> list[tuple[str, str | None]]:
        """Return expansion ``(text, id)`` pairs for a rule.

        ``id`` is ``None`` when the entry didn't supply an explicit id — the
        consumer (e.g. ``TrimTabDB.upsert_grammar``) should auto-generate in
        that case. When the entry is a ``{"text", "id"}`` dict, the explicit
        id is returned so the upsert path can use it directly.
        """
        return [(_expansion_text(e), _expansion_id(e)) for e in self.rules.get(rule, [])]

    def add_expansion(self, rule: str, value: ExpansionEntry) -> None:
        if rule not in self.rules:
            self.rules[rule] = []
        # Dedup by text (preserving the shape that was passed in).
        existing_texts = {_expansion_text(e) for e in self.rules[rule]}
        if _expansion_text(value) not in existing_texts:
            self.rules[rule].append(value)

    def rule_names(self) -> list[str]:
        return list(self.rules.keys())

    @staticmethod
    def extract_refs(text: str) -> list[str]:
        """Extract rule references (#name#) from an expansion string."""
        return _RULE_PATTERN.findall(text)

    @staticmethod
    def is_terminal(text: str) -> bool:
        """Check if text has no rule references (is a leaf)."""
        return not _RULE_PATTERN.search(text)
