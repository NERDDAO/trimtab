"""Verify SmartGrammar remains accessible as a deprecated alias and TrimTab is exported."""

from __future__ import annotations

import warnings


def test_trimtab_is_top_level_export():
    import trimtab
    assert hasattr(trimtab, "TrimTab")
    assert trimtab.TrimTab is not None


def test_version_is_v05():
    import trimtab
    assert trimtab.__version__ == "0.5.0"


def test_errors_exported_at_top_level():
    import trimtab
    assert hasattr(trimtab, "TrimTabError")
    assert hasattr(trimtab, "TrimTabEmbedderError")
    assert hasattr(trimtab, "TrimTabNotFoundError")
    assert hasattr(trimtab, "TrimTabCycleError")


def test_rule_and_grammar_exported():
    import trimtab
    assert hasattr(trimtab, "Rule")
    assert hasattr(trimtab, "Grammar")


def test_smartgrammar_still_importable():
    """SmartGrammar must remain accessible for one release as a deprecated alias."""
    from trimtab import SmartGrammar
    assert SmartGrammar is not None


def test_smartgrammar_construction_emits_deprecation_warning():
    """Constructing SmartGrammar the v0.4 way should emit DeprecationWarning."""
    from trimtab import SmartGrammar, TrimTabDB
    from tests.conftest import StubEmbedder

    db = TrimTabDB(":memory:")
    embedder = StubEmbedder(dim=16)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sg = SmartGrammar(db, "test_grammar", embedder)
        assert any(
            issubclass(w.category, DeprecationWarning) for w in caught
        ), "SmartGrammar construction should emit DeprecationWarning"

    # The object should still be usable — at minimum accessible.
    assert sg is not None
