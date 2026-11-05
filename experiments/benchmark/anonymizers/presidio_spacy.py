"""Microsoft Presidio de-identifier with a German spaCy NLP pipeline.

`Presidio <https://microsoft.github.io/presidio/>`_ is a widely used open-source
PII detection framework. This adapter runs its :class:`AnalyzerEngine` with a
German spaCy model (``de_core_news_lg`` by default) as the NLP engine, so named
entities (persons, locations, organisations) come from spaCy NER while Presidio's
language-agnostic pattern recognisers add dates, phone numbers, e-mail, URLs,
IBANs, and similar. Every detected span above a score threshold is replaced with
:data:`REDACTION_MARKER`.

This is the framework's *out-of-the-box* German behaviour: no custom recognisers
or dataset-specific tuning are added. Its recall on person names is strong (spaCy
NER generalises beyond any gazetteer) but German's noun capitalisation leads the
NER to over-tag ordinary nouns, so precision is expected to suffer — a genuine
property of the method, reported honestly rather than patched away.

Presidio and spaCy are imported lazily so the benchmark harness and config import
without the experiment dependency group installed.
"""

from __future__ import annotations

from experiments.benchmark.anonymizers.base import Anonymizer
from experiments.benchmark.anonymizers.spans import merge_spans, redact

#: spaCy NER labels (de_core_news_*) → Presidio entity types.
_ENTITY_MAPPING = {
    "PER": "PERSON",
    "PERSON": "PERSON",
    "LOC": "LOCATION",
    "LOCATION": "LOCATION",
    "GPE": "LOCATION",
    "ORG": "ORGANIZATION",
    "ORGANIZATION": "ORGANIZATION",
    "MISC": "MISC",
}


class PresidioAnonymizer(Anonymizer):
    """Presidio AnalyzerEngine (German spaCy pipeline) as a text anonymizer."""

    def __init__(
        self,
        spacy_model: str = "de_core_news_lg",
        language: str = "de",
        score_threshold: float = 0.35,
    ) -> None:
        self.name = f"presidio_{spacy_model}"
        self.language = language
        self.score_threshold = score_threshold
        self._analyzer = _build_analyzer(spacy_model, language)

    def anonymize_text(self, text: str) -> str:
        """Return ``text`` with every detected PII span replaced by the marker."""
        return redact(text, self.detect_spans(text))

    def detect_spans(self, text: str) -> list[tuple[int, int]]:
        """Return merged ``(start, end)`` spans for Presidio's detections."""
        results = self._analyzer.analyze(
            text=text,
            language=self.language,
            score_threshold=self.score_threshold,
        )
        return merge_spans([(r.start, r.end) for r in results])


def _build_analyzer(spacy_model: str, language: str):
    """Construct a Presidio AnalyzerEngine backed by a German spaCy pipeline."""
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": language, "model_name": spacy_model}],
        "ner_model_configuration": {
            "model_to_presidio_entity_mapping": _ENTITY_MAPPING,
        },
    }
    nlp_engine = NlpEngineProvider(nlp_configuration=configuration).create_engine()
    return AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=[language])
