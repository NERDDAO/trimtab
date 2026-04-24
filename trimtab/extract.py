"""Typed-span entity extraction over rule text.

An ``EntityExtractor`` turns raw text into ``{label: [text, ...]}`` — typed
spans usable as structured metadata on rules (``metadata["entities"]``) and
as query-time routing signals via :class:`trimtab.engram.EngramIndex`.

One implementation ships:

* :class:`GLiNER2Extractor` — zero-shot NER on local CPU via the
  ``gliner2`` library. Single ``extract_entities`` call covers all labels
  in ``DEFAULT_LABELS`` (Person / Location / Event / Organization /
  Preference / Emotion / Relationship / Topic). Model load is synchronous
  at first use (~1s on CPU); construct once per process.

The extractor is an optional dependency — ``pip install trimtab[entities]``
pulls ``gliner2``. Importing :class:`GLiNER2Extractor` without the extra
raises an ``ImportError`` with a concrete install hint.
"""

from __future__ import annotations

from typing import Any, Protocol

DEFAULT_MODEL = "fastino/gliner2-base-v1"
DEFAULT_THRESHOLD = 0.5

# Default label vocabulary — validated on LoCoMo-style conversational data.
# Callers can override per-call by passing their own ``labels`` dict.
DEFAULT_LABELS: dict[str, str] = {
    "Person": "Human names",
    "Location": "Places / geographic regions",
    "Event": "Named events (races, trips, ceremonies, specific activities)",
    "Organization": "Companies, groups, institutions",
    "Preference": (
        "Things the speaker likes, prefers, owns, or practices — hobbies, "
        "possessions, preferred activities, things they enjoy or care about"
    ),
    "Emotion": "Feelings, moods, emotional states",
    "Relationship": "Family or social relationships (spouse, friend, child, parent, sibling)",
    "Topic": "Subjects of discussion — themes, domains, fields",
}


class EntityExtractor(Protocol):
    """Turns text into ``{label: [span_text, ...]}``.

    Implementations are free to cache models or batch internally; the
    ``extract`` method must be callable many times after construction.
    """

    def extract(
        self,
        text: str,
        labels: dict[str, str] | None = None,
        *,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> dict[str, list[str]]: ...

    def extract_batch(
        self,
        texts: list[str],
        labels: dict[str, str] | None = None,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        batch_size: int = 8,
    ) -> list[dict[str, list[str]]]:
        """Extract over a list of texts in one shot.

        Returns one ``{label: [span_text, ...]}`` dict per input text in
        the same order. Implementations should batch at the model layer
        when possible — pure-loop implementations are acceptable but
        defeat the point of the API.
        """
        ...

    def classify_batch(
        self,
        texts: list[str],
        classes: list[str],
        *,
        threshold: float = DEFAULT_THRESHOLD,
        batch_size: int = 8,
    ) -> list[str | None]:
        """Multi-class single-label classification over a batch of texts.

        Returns one chosen class per input text — the highest-scoring
        class above ``threshold``, or ``None`` when no class fires (or
        the input is empty). Result list aligns 1:1 with ``texts``.
        """
        ...


class GLiNER2Extractor:
    """Zero-shot NER via the ``gliner2`` library.

    Loads the model at construction time. Safe to share a single instance
    across threads for read-only ``extract`` calls.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        default_labels: dict[str, str] | None = None,
    ) -> None:
        try:
            from gliner2 import GLiNER2  # type: ignore[import-untyped]
        except ImportError as e:  # pragma: no cover - dependency check
            raise ImportError(
                "GLiNER2Extractor requires gliner2. "
                "Install with `pip install trimtab[entities]`."
            ) from e
        self._model = GLiNER2.from_pretrained(model_name)
        self._default_labels = default_labels or DEFAULT_LABELS

    def extract(
        self,
        text: str,
        labels: dict[str, str] | None = None,
        *,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> dict[str, list[str]]:
        """Return ``{label: [text, ...]}`` with blanks dropped.

        ``text`` can be empty — empty string yields ``{}``.
        ``labels=None`` falls back to the extractor's ``default_labels``.
        """
        if not text:
            return {}
        active_labels = labels or self._default_labels
        raw: dict[str, Any] = self._model.extract_entities(
            text,
            active_labels,
            threshold=threshold,
        )
        return self._format(raw, active_labels)

    def extract_batch(
        self,
        texts: list[str],
        labels: dict[str, str] | None = None,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        batch_size: int = 8,
    ) -> list[dict[str, list[str]]]:
        """Batched ``extract`` — one model call per ``batch_size`` texts.

        gliner2's ``GLiNER2.batch_extract_entities`` runs the underlying
        transformer on padded batches, amortising tokenisation and CUDA
        kernel-launch cost. Empty inputs in ``texts`` are kept (return
        ``{}`` for that slot) so the result list aligns 1:1 with the
        input list, letting callers zip results back to source objects.
        """
        if not texts:
            return []
        active_labels = labels or self._default_labels
        non_empty_indices = [i for i, t in enumerate(texts) if t]
        if not non_empty_indices:
            return [{} for _ in texts]
        raws = self._model.batch_extract_entities(
            [texts[i] for i in non_empty_indices],
            active_labels,
            threshold=threshold,
            batch_size=batch_size,
        )
        formatted = [self._format(raw or {}, active_labels) for raw in raws]
        out: list[dict[str, list[str]]] = [{} for _ in texts]
        for slot, parsed in zip(non_empty_indices, formatted, strict=True):
            out[slot] = parsed
        return out

    def classify_batch(
        self,
        texts: list[str],
        classes: list[str],
        *,
        threshold: float = DEFAULT_THRESHOLD,
        batch_size: int = 8,
    ) -> list[str | None]:
        """Batched single-label classification.

        Wraps ``GLiNER2.batch_classify_text`` with a single task name so
        callers don't have to think about the underlying multi-task
        schema. Returns ``classes[i]`` per text where i is the
        argmax-above-threshold; ``None`` for inputs that scored no class
        above threshold (or were empty). Result list aligns 1:1 with
        ``texts``.
        """
        if not texts or not classes:
            return [None for _ in texts]
        non_empty_indices = [i for i, t in enumerate(texts) if t]
        if not non_empty_indices:
            return [None for _ in texts]
        tasks = {"label": list(classes)}
        raws = self._model.batch_classify_text(
            [texts[i] for i in non_empty_indices],
            tasks,
            threshold=threshold,
            batch_size=batch_size,
            include_confidence=True,
        )
        out: list[str | None] = [None for _ in texts]
        for slot, raw in zip(non_empty_indices, raws, strict=True):
            out[slot] = self._top_class(raw or {}, "label")
        return out

    @staticmethod
    def _top_class(raw: dict[str, Any], task_name: str) -> str | None:
        """Pick the chosen class from one ``batch_classify_text`` result.

        gliner2 keys the result by task name directly (no ``classifications``
        wrapper) and returns either:

        - bare string (``include_confidence=False``): ``{"label": "books"}``
        - dict (``include_confidence=True``):
          ``{"label": {"confidence": 1.0, "label": "books"}}``
        - list (rare; multi-top-k mode): ``{"label": [{"label": ..., "confidence": ...}, ...]}``

        Below threshold, gliner2 omits the task entirely from the dict.
        """
        task_result = raw.get(task_name)
        if not task_result:
            return None
        if isinstance(task_result, str):
            return task_result
        if isinstance(task_result, dict):
            label = task_result.get("label") or task_result.get("class")
            return str(label) if label else None
        if isinstance(task_result, list):
            if not task_result:
                return None
            best = max(
                task_result,
                key=lambda item: float(item.get("confidence", 0.0)) if isinstance(item, dict) else 0.0,
            )
            if isinstance(best, dict):
                label = best.get("label") or best.get("class")
                return str(label) if label else None
            return str(best) if best else None
        return None

    @staticmethod
    def _format(
        raw: dict[str, Any],
        active_labels: dict[str, str],
    ) -> dict[str, list[str]]:
        """Common gliner2-result-dict → ``{label: [span_text, ...]}`` reshape."""
        entities_dict = raw.get("entities", {}) or {}
        out: dict[str, list[str]] = {}
        for label in active_labels:
            items = entities_dict.get(label, []) or []
            spans: list[str] = []
            for item in items:
                span_text = item["text"] if isinstance(item, dict) else str(item)
                if span_text:
                    spans.append(span_text)
            if spans:
                out[label] = spans
        return out
