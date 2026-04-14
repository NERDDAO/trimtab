"""Ollama-backed default embedder.

Implements the trimtab ``Embedder`` Protocol structurally (async ``create`` +
``create_batch``). Uses ``requests`` (already a trimtab dep) — no new HTTP
client dependency. Fails fast at construction if Ollama is unreachable.
"""

from __future__ import annotations

import os

import requests

from trimtab.errors import TrimTabEmbedderError


DEFAULT_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")


class OllamaEmbedder:
    """Default embedder for trimtab. Calls Ollama's /api/embed endpoint.

    Construction probes /api/tags and raises ``TrimTabEmbedderError`` if
    the server is unreachable. The error message points at ``ollama serve``
    and ``ollama pull`` so the fix is obvious.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=2)
            resp.raise_for_status()
        except Exception as e:
            raise TrimTabEmbedderError(
                f"Ollama not reachable at {self.base_url}: {e}. "
                f"Run 'ollama serve' to start it, then 'ollama pull {model}'."
            ) from e

    async def create(self, input_data: str | list[str]) -> list[float]:
        """Embed a single text (or a list of texts joined with spaces).

        When ``input_data`` is a list, this collapses the list into one string
        with ``" ".join(...)`` and returns ONE embedding — not one embedding
        per element. Use ``create_batch`` when you need per-element vectors.
        """
        text = input_data if isinstance(input_data, str) else " ".join(input_data)
        try:
            resp = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": text},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return [float(x) for x in resp.json()["embeddings"][0]]
        except Exception as e:
            raise TrimTabEmbedderError(
                f"Embed call failed for model {self.model!r}: {e}. "
                f"If the model is missing, run 'ollama pull {self.model}'."
            ) from e

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        try:
            # Batches can be slower under load — double the single-embed timeout.
            resp = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": input_data_list},
                timeout=self.timeout * 2,
            )
            resp.raise_for_status()
            return [[float(x) for x in vec] for vec in resp.json()["embeddings"]]
        except Exception as e:
            raise TrimTabEmbedderError(
                f"Batch embed call failed for model {self.model!r}: {e}."
            ) from e
