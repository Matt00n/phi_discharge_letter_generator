"""Anonymizer interface for the text-replacement benchmark."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Anonymizer(ABC):
    """A de-identification method that returns full PHI-replaced text.

    Concrete anonymizers implement :meth:`anonymize_text`, which takes the
    original document and returns the same text with every PII entity replaced
    by :data:`~experiments.benchmark.prompts.REDACTION_MARKER`. The benchmark
    runner aligns that output back to character offsets for scoring, so the
    non-PII text must be preserved verbatim.
    """

    #: Stable, human-readable identifier used in report/output paths.
    name: str

    @abstractmethod
    def anonymize_text(self, text: str) -> str:
        """Return ``text`` with every PII entity replaced by the marker."""
        raise NotImplementedError
