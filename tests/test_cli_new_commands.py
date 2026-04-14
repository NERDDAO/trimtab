"""Tests for the v0.5 CLI commands (put, search, remove, reembed).

These tests shell out to `python -m trimtab.cli` and verify behavior
end-to-end. They require a live Ollama (the CLI uses OllamaEmbedder by
default), so they are marked integration_ollama and skipped in CI.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration_ollama


def _run(args: list[str], db_path: str) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", "trimtab.cli", "--db", db_path, *args]
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def test_put_then_search(tmp_path: Path):
    db = str(tmp_path / "cli.db")
    r = _run(["put", "g1", "notes", "the forest is dangerous at night"], db)
    assert r.returncode == 0, f"put failed: stderr={r.stderr}"
    r = _run(["search", "g1", "notes", "forest", "--top-k", "3"], db)
    assert r.returncode == 0, f"search failed: stderr={r.stderr}"
    assert "forest" in r.stdout.lower()


def test_remove(tmp_path: Path):
    db = str(tmp_path / "cli.db")
    r = _run(["put", "g1", "notes", "to be deleted", "--id", "note_1"], db)
    assert r.returncode == 0
    r = _run(["remove", "g1", "notes", "note_1"], db)
    assert r.returncode == 0
