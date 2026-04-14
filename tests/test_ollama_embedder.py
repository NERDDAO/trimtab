"""Unit tests for OllamaEmbedder (HTTP calls mocked)."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from trimtab.embedders.ollama import OllamaEmbedder
from trimtab.errors import TrimTabEmbedderError


def _mock_tags_ok():
    m = MagicMock()
    m.raise_for_status.return_value = None
    return m


def _mock_embed_response(vectors):
    m = MagicMock()
    m.raise_for_status.return_value = None
    m.json.return_value = {"embeddings": vectors}
    return m


def test_constructor_probes_ollama():
    with patch("trimtab.embedders.ollama.requests") as req:
        req.get.return_value = _mock_tags_ok()
        emb = OllamaEmbedder(model="nomic-embed-text")
        req.get.assert_called_once()
        assert "nomic-embed-text" == emb.model


def test_constructor_raises_when_ollama_unreachable():
    with patch("trimtab.embedders.ollama.requests") as req:
        req.get.side_effect = ConnectionError("refused")
        with pytest.raises(TrimTabEmbedderError) as excinfo:
            OllamaEmbedder(model="nomic-embed-text")
        assert "ollama serve" in str(excinfo.value).lower() or "unreachable" in str(excinfo.value).lower()


@pytest.mark.asyncio
async def test_create_returns_vector():
    with patch("trimtab.embedders.ollama.requests") as req:
        req.get.return_value = _mock_tags_ok()
        req.post.return_value = _mock_embed_response([[0.1, 0.2, 0.3]])
        emb = OllamaEmbedder(model="nomic-embed-text")
        vec = await emb.create("hello")
        assert vec == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_create_batch_returns_vectors():
    with patch("trimtab.embedders.ollama.requests") as req:
        req.get.return_value = _mock_tags_ok()
        req.post.return_value = _mock_embed_response([[0.1], [0.2], [0.3]])
        emb = OllamaEmbedder(model="nomic-embed-text")
        vecs = await emb.create_batch(["a", "b", "c"])
        assert vecs == [[0.1], [0.2], [0.3]]


@pytest.mark.asyncio
async def test_create_raises_trimtab_error_on_http_failure():
    with patch("trimtab.embedders.ollama.requests") as req:
        req.get.return_value = _mock_tags_ok()
        bad = MagicMock()
        bad.raise_for_status.side_effect = RuntimeError("500")
        req.post.return_value = bad
        emb = OllamaEmbedder(model="nomic-embed-text")
        with pytest.raises(TrimTabEmbedderError):
            await emb.create("hello")


@pytest.mark.integration_ollama
@pytest.mark.asyncio
async def test_live_ollama_contract():
    """Runs against a real Ollama. Skipped unless `-m integration_ollama`."""
    emb = OllamaEmbedder(model="nomic-embed-text")
    vec = await emb.create("hello world")
    assert isinstance(vec, list)
    assert len(vec) > 0
    assert all(isinstance(x, float) for x in vec)
