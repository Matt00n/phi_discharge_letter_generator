"""Flair German NER de-identifier, wrapped in Presidio.

Uses the off-the-shelf `flair/ner-german-large
<https://huggingface.co/flair/ner-german-large>`_ sequence tagger (an XLM-R based
NER model) as the entity backbone, embedded in Microsoft Presidio as a custom
recogniser. Presidio's language-agnostic pattern recognisers (dates, phone
numbers, e-mail, URLs, …) are kept, so this method differs from the
:class:`~experiments.benchmark.anonymizers.presidio_spacy.PresidioAnonymizer`
*only* in the NER engine — spaCy is swapped for Flair. That makes the two a
controlled comparison of NER backbones on an otherwise identical pipeline.

Flair tags ``PER``/``LOC``/``ORG``/``MISC``; only person, location, and
organisation are mapped to PII entities. ``MISC`` is not a PII category and is
ignored (this also avoids redacting drug/eponym look-alikes Flair tags as MISC).

The model is used out of the box — no fine-tuning or dataset-specific tuning.
Flair, Presidio, and spaCy are imported lazily so the harness and config import
without the experiment dependency group installed.
"""

from __future__ import annotations

import contextlib

from experiments.benchmark.anonymizers.base import Anonymizer
from experiments.benchmark.anonymizers.spans import merge_spans, redact

#: Flair NER labels (flair/ner-german-large) → Presidio entity types. ``MISC``
#: is intentionally absent — it is not a PII category.
_LABEL_MAPPING = {
    "PER": "PERSON",
    "LOC": "LOCATION",
    "ORG": "ORGANIZATION",
}


class FlairAnonymizer(Anonymizer):
    """Presidio pipeline with a Flair German NER backbone."""

    def __init__(
        self,
        flair_model: str = "flair/ner-german-large",
        spacy_model: str = "de_core_news_lg",
        language: str = "de",
        score_threshold: float = 0.35,
    ) -> None:
        self.name = f"flair_{flair_model.split('/')[-1]}"
        self.language = language
        self.score_threshold = score_threshold
        self._analyzer = _build_analyzer(flair_model, spacy_model, language)

    def anonymize_text(self, text: str) -> str:
        """Return ``text`` with every detected PII span replaced by the marker."""
        return redact(text, self.detect_spans(text))

    def detect_spans(self, text: str) -> list[tuple[int, int]]:
        """Return merged ``(start, end)`` spans for Presidio+Flair detections."""
        results = self._analyzer.analyze(
            text=text,
            language=self.language,
            score_threshold=self.score_threshold,
        )
        return merge_spans([(r.start, r.end) for r in results])


def _build_analyzer(flair_model: str, spacy_model: str, language: str):
    """Presidio AnalyzerEngine: spaCy tokeniser, Flair NER, pattern recognisers."""
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    # spaCy provides tokenisation/artifacts for the pattern recognisers; its own
    # NER is removed below so Flair is the sole entity backbone.
    configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": language, "model_name": spacy_model}],
    }
    nlp_engine = NlpEngineProvider(nlp_configuration=configuration).create_engine()
    analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=[language])
    with contextlib.suppress(ValueError):
        analyzer.registry.remove_recognizer("SpacyRecognizer")
    analyzer.registry.add_recognizer(
        _FlairRecognizer(model_path=flair_model, supported_language=language)
    )
    return analyzer


class _FlairRecognizer:
    """Deferred wrapper; the real recogniser class is built on first use.

    Defined via a factory so ``presidio_analyzer`` is imported lazily rather than
    at module import time.
    """

    def __new__(cls, *, model_path: str, supported_language: str):
        return _make_flair_recognizer(model_path, supported_language)


def _make_flair_recognizer(model_path: str, supported_language: str):
    from flair.models import SequenceTagger
    from flair.splitter import SegtokSentenceSplitter
    from presidio_analyzer import EntityRecognizer, RecognizerResult

    class FlairRecognizer(EntityRecognizer):
        """Presidio recogniser backed by a Flair SequenceTagger."""

        def __init__(self) -> None:
            self._tagger = SequenceTagger.load(model_path)
            self._splitter = SegtokSentenceSplitter()
            super().__init__(
                supported_entities=sorted(set(_LABEL_MAPPING.values())),
                supported_language=supported_language,
                name="FlairRecognizer",
            )

        def load(self) -> None:  # model already loaded in __init__
            pass

        def analyze(self, text, entities, nlp_artifacts=None):
            results: list[RecognizerResult] = []
            sentences = self._splitter.split(text)
            if not sentences:
                return results
            self._tagger.predict(sentences)
            for sentence in sentences:
                base = sentence.start_position
                for span in sentence.get_spans("ner"):
                    label = span.get_label("ner")
                    entity_type = _LABEL_MAPPING.get(label.value)
                    if entity_type is None or entity_type not in entities:
                        continue
                    results.append(
                        RecognizerResult(
                            entity_type=entity_type,
                            start=base + span.start_position,
                            end=base + span.end_position,
                            score=float(label.score),
                        )
                    )
            return results

    return FlairRecognizer()
