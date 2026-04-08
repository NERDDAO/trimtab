"""Tracery-compatible grammar loading, saving, and traversal."""

import json
import re
from pathlib import Path
from dataclasses import dataclass, field


_RULE_PATTERN = re.compile(r"#(\w+)#")


@dataclass
class Grammar:
    """A Tracery-compatible grammar with named rules and expansions."""
    rules: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str | Path) -> "Grammar":
        data = json.loads(Path(path).read_text())
        return cls(rules=data)

    @classmethod
    def from_dict(cls, rules: dict[str, list[str]]) -> "Grammar":
        return cls(rules=rules)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.rules, indent=2, ensure_ascii=False) + "\n")

    def get_expansions(self, rule: str) -> list[str]:
        return self.rules.get(rule, [])

    def add_expansion(self, rule: str, value: str) -> None:
        if rule not in self.rules:
            self.rules[rule] = []
        if value not in self.rules[rule]:
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
