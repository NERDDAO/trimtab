"""Tests for corpus-to-grammar building."""
import pytest
from smartgrammar.builder import extract_ngrams, NGram


def test_extract_ngrams_basic():
    texts = [
        "the dark cave echoes",
        "the dark cave rumbles",
        "the dark cave whispers",
        "a bright forest sings",
    ]
    ngrams = extract_ngrams(texts, min_n=2, max_n=3, min_count=2)
    # "dark cave" should appear 3 times
    dark_cave = [ng for ng in ngrams if ng.text == "dark cave"]
    assert len(dark_cave) == 1
    assert dark_cave[0].count == 3


def test_extract_ngrams_filters_stopwords():
    texts = ["the the the", "a a a", "is is is"]
    ngrams = extract_ngrams(texts, min_n=2, max_n=2, min_count=1)
    # All-stopword n-grams should be filtered
    assert len(ngrams) == 0


def test_extract_ngrams_min_count():
    texts = ["unique phrase here", "another thing entirely"]
    ngrams = extract_ngrams(texts, min_count=2)
    # Nothing repeats, so nothing should pass min_count=2
    assert len(ngrams) == 0


def test_extract_ngrams_sorted_by_count():
    texts = ["a b c"] * 5 + ["x y z"] * 3
    ngrams = extract_ngrams(texts, min_n=2, max_n=2, min_count=1)
    if len(ngrams) >= 2:
        assert ngrams[0].count >= ngrams[1].count
