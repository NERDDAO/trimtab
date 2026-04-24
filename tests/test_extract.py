"""Tests for trimtab.extract.GLiNER2Extractor.

Gated by ``pytest.mark.integration_gliner2`` — these tests load a ~200MB
model and need the optional ``trimtab[entities]`` extra. Skipped by default;
enable with ``pytest -m integration_gliner2``.

Unit-level shape tests (output schema, normalization) live inline here; the
heavy semantic test just eyeballs that painter/painting-style inputs yield
non-empty ``Preference`` and ``Person`` spans.
"""

from __future__ import annotations

import pytest

pytest.importorskip("gliner2", reason="trimtab[entities] extra not installed")

from trimtab.extract import DEFAULT_LABELS  # noqa: E402


@pytest.mark.integration_gliner2
def test_gliner2_extracts_person_and_preference():
    from trimtab.extract import GLiNER2Extractor

    extractor = GLiNER2Extractor()
    result = extractor.extract("Caroline: I love painting and pottery.")

    # Output shape: dict[label, list[str]], non-empty only for labels that hit.
    assert isinstance(result, dict)
    for label, spans in result.items():
        assert label in DEFAULT_LABELS
        assert isinstance(spans, list)
        assert all(isinstance(s, str) and s for s in spans)

    # Semantic expectations — the canonical LoCoMo-style utterance.
    assert "Person" in result, f"expected Person in {result}"
    assert any("caroline" in s.lower() for s in result["Person"])
    assert "Preference" in result, f"expected Preference in {result}"
    assert any("painting" in s.lower() for s in result["Preference"])


@pytest.mark.integration_gliner2
def test_gliner2_empty_text_yields_empty_dict():
    from trimtab.extract import GLiNER2Extractor

    extractor = GLiNER2Extractor()
    assert extractor.extract("") == {}


@pytest.mark.integration_gliner2
def test_gliner2_respects_custom_labels():
    from trimtab.extract import GLiNER2Extractor

    extractor = GLiNER2Extractor()
    result = extractor.extract(
        "Caroline paints landscapes.",
        labels={"Person": "Human names"},
    )
    # Only the requested label can appear in the result.
    assert set(result.keys()) <= {"Person"}


@pytest.mark.integration_gliner2
def test_gliner2_extract_batch_aligns_with_input_order():
    """``extract_batch`` returns one dict per text in input order; empty
    inputs yield ``{}`` so callers can zip results to source objects."""
    from trimtab.extract import GLiNER2Extractor

    extractor = GLiNER2Extractor()
    texts = [
        "Caroline loves painting.",
        "",  # empty slot
        "Melanie went running this morning.",
    ]
    results = extractor.extract_batch(texts)

    assert len(results) == 3
    assert results[1] == {}, "empty text must produce empty dict"
    # Both non-empty texts should have at least the Person label populated.
    assert "Person" in results[0]
    assert "Person" in results[2]
    assert any("caroline" in s.lower() for s in results[0]["Person"])
    assert any("melanie" in s.lower() for s in results[2]["Person"])


@pytest.mark.integration_gliner2
def test_gliner2_classify_batch_picks_top_class_per_text():
    """``classify_batch`` returns one chosen class label per input text."""
    from trimtab.extract import GLiNER2Extractor

    extractor = GLiNER2Extractor()
    texts = [
        "I just finished reading War and Peace and it was amazing.",
        "",  # empty slot — should map to None
        "We went hiking in the mountains last weekend.",
    ]
    classes = ["books", "outdoor activities", "food", "music"]
    chosen = extractor.classify_batch(texts, classes, threshold=0.3)

    assert len(chosen) == 3
    assert chosen[1] is None, "empty input must yield None"
    assert chosen[0] in classes, f"first text → {chosen[0]}"
    assert chosen[2] in classes, f"third text → {chosen[2]}"
