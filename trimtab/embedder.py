"""Embedder Protocol.

trimtab is BYO-embedder: callers pass any object that structurally matches
this Protocol. The shape matches ``graphiti_core.embedder.client.EmbedderClient``
so delve's unified embedder (the primary consumer) plugs in directly.

- ``create(text)`` returns a single vector (``list[float]``).
- ``create_batch(texts)`` returns one vector per input (``list[list[float]]``).

Both methods are async to match graphiti_core's shape and avoid sync/async
bridging at the call sites.
"""

from __future__ import annotations

from typing import Protocol


class Embedder(Protocol):
    """Structural Protocol for an async embedder.

    Any class with these two coroutine methods satisfies it — no need to
    inherit. This matches ``graphiti_core.embedder.client.EmbedderClient``
    so delve's ``create_embedder()`` factory output plugs in directly.
    """

    async def create(self, input_data: str | list[str]) -> list[float]:
        """Embed a single item. Returns one vector."""
        ...

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        """Embed a batch. Returns one vector per input, in order."""
        ...
