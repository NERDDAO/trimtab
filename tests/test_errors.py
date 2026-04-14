"""Unit tests for trimtab.errors."""

import pytest

from trimtab.errors import (
    TrimTabError,
    TrimTabEmbedderError,
    TrimTabNotFoundError,
    TrimTabDimensionError,
    TrimTabMigrationError,
    TrimTabGrammarError,
    TrimTabCycleError,
)


def test_base_error_is_exception():
    assert issubclass(TrimTabError, Exception)


@pytest.mark.parametrize(
    "subclass",
    [
        TrimTabEmbedderError,
        TrimTabNotFoundError,
        TrimTabDimensionError,
        TrimTabMigrationError,
        TrimTabGrammarError,
        TrimTabCycleError,
    ],
)
def test_all_subclasses_inherit_from_base(subclass):
    assert issubclass(subclass, TrimTabError)


def test_not_found_error_carries_fields():
    err = TrimTabNotFoundError(grammar="g", symbol="s", rule_id="r_1")
    assert err.grammar == "g"
    assert err.symbol == "s"
    assert err.rule_id == "r_1"
    assert "g" in str(err) and "s" in str(err) and "r_1" in str(err)


def test_dimension_error_carries_dims():
    err = TrimTabDimensionError(expected=768, got=384)
    assert err.expected == 768
    assert err.got == 384
    assert "768" in str(err) and "384" in str(err)


def test_cycle_error_carries_chain():
    err = TrimTabCycleError(chain=["origin", "friends", "friends"])
    assert err.chain == ["origin", "friends", "friends"]
    assert "friends" in str(err)


def test_embedder_error_is_raisable():
    with pytest.raises(TrimTabEmbedderError):
        raise TrimTabEmbedderError("Ollama unreachable")
