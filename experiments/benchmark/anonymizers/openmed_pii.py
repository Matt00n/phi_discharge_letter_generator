"""Off-the-shelf German clinical PII detector (OpenMed).

Wraps a dedicated Hugging Face token-classification model for PII detection —
by default `OpenMed/OpenMed-PII-German-ClinicalE5-Base-109M-v1
<https://huggingface.co/OpenMed/OpenMed-PII-German-ClinicalE5-Base-109M-v1>`_,
which tags dozens of PII entity types (names, dates, IDs, contact details, …).

Unlike the general-purpose NER methods, this model is trained specifically for
PII/de-identification, so it is used **standalone** — no Presidio pattern layer
is added. The benchmark scores in text mode (detection only), so predicted
labels are irrelevant; only the character regions the model flags are redacted.
That makes the exact 54-way label taxonomy immaterial here (a case number tagged
``BANKACCOUNT`` still redacts the right span).

Segmentation
------------
The model is trained on short, templated PII sentences and — as verified on this
dataset — its recall (names/dates especially) collapses when handed a whole
multi-paragraph clinical letter, even well within its 512-token limit. Feeding it
inputs sized like its training data (one line / sentence at a time) restores
detection. We therefore split each document into line- and sentence-level
segments, run the model on each, and map the local predictions back to global
character offsets. This gives the off-the-shelf model a fair shot rather than
penalising it for an input distribution it was never trained on.

Transformers is imported lazily so the harness and config import without the
experiment dependency group installed.
"""

from __future__ import annotations

import re

from experiments.benchmark.anonymizers.base import Anonymizer
from experiments.benchmark.anonymizers.spans import merge_spans, redact

#: Default off-the-shelf model.
DEFAULT_MODEL = "OpenMed/OpenMed-PII-German-ClinicalE5-Base-109M-v1"

#: Splits a line into sentence-level chunks after terminal punctuation.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


class OpenMedPIIAnonymizer(Anonymizer):
    """Dedicated German PII token-classifier as a text anonymizer."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        score_threshold: float = 0.35,
    ) -> None:
        self.name = f"openmed_{model.split('/')[-1]}"
        self.score_threshold = score_threshold
        self._pipeline = _build_pipeline(model)

    def anonymize_text(self, text: str) -> str:
        """Return ``text`` with every detected PII span replaced by the marker."""
        return redact(text, self.detect_spans(text))

    def detect_spans(self, text: str) -> list[tuple[int, int]]:
        """Return merged ``(start, end)`` spans for the model's PII predictions.

        The document is split into line/sentence segments (see module docstring);
        each is classified independently and the local offsets are shifted back to
        the document frame before merging.
        """
        if not text.strip():
            return []

        segments = [(s, e) for s, e in _segment_bounds(text) if text[s:e].strip()]
        if not segments:
            return []

        # "first" aggregation labels each word by its first sub-token, avoiding
        # the sub-word fragmentation the default "simple" strategy produces here.
        batch = self._pipeline(
            [text[s:e] for s, e in segments], aggregation_strategy="first"
        )

        spans: list[tuple[int, int]] = []
        for (seg_start, _seg_end), entities in zip(segments, batch, strict=True):
            for entity in entities:
                if entity.get("score", 1.0) < self.score_threshold:
                    continue
                spans.append(
                    (seg_start + int(entity["start"]), seg_start + int(entity["end"]))
                )
        return merge_spans(spans)


def _segment_bounds(text: str) -> list[tuple[int, int]]:
    """Yield ``(start, end)`` offsets of line- and sentence-level segments.

    Splitting on newlines first (then on sentence boundaries within each line)
    keeps segments short enough to match the model's training distribution while
    preserving exact global character offsets.
    """
    bounds: list[tuple[int, int]] = []
    line_start = 0
    for line in text.split("\n"):
        seg_start = line_start
        for match in _SENTENCE_BOUNDARY.finditer(line):
            bounds.append((seg_start, line_start + match.start()))
            seg_start = line_start + match.end()
        bounds.append((seg_start, line_start + len(line)))
        line_start += len(line) + 1  # account for the stripped "\n"
    return bounds


def _build_pipeline(model: str):
    """Construct a Hugging Face token-classification pipeline for ``model``."""
    from transformers import pipeline

    return pipeline("token-classification", model=model)
