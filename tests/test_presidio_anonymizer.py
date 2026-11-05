"""Tests for the Presidio (German spaCy) de-identification method.

Presidio and the ``de_core_news_lg`` model are heavy optional dependencies, so
the whole module is skipped when they are unavailable. Loading the analyzer is
expensive, so it is built once per module.
"""

import importlib.util

import pytest

pytest.importorskip("presidio_analyzer")
if importlib.util.find_spec("de_core_news_lg") is None:
    pytest.skip("de_core_news_lg model not installed", allow_module_level=True)

from experiments.benchmark.anonymizers import build_anonymizer  # noqa: E402
from experiments.benchmark.anonymizers.presidio_spacy import (  # noqa: E402
    PresidioAnonymizer,
)
from experiments.benchmark.config import AnonymizerConfig  # noqa: E402
from experiments.benchmark.prompts import REDACTION_MARKER  # noqa: E402


@pytest.fixture(scope="module")
def anon() -> PresidioAnonymizer:
    return PresidioAnonymizer(score_threshold=0.35)


def test_registered_via_config():
    built = build_anonymizer(AnonymizerConfig(name="presidio"))
    assert isinstance(built, PresidioAnonymizer)
    assert built.name == "presidio_de_core_news_lg"


def test_detects_person_name(anon):
    text = "Wir berichten über den Aufenthalt von Wilfried Hecht."
    out = anon.anonymize_text(text)
    assert "Wilfried Hecht" not in out
    assert REDACTION_MARKER in out


def test_detects_email(anon):
    regions = [
        text_region
        for start, end in anon.detect_spans("Kontakt: max.mustermann@klinik.de")
        for text_region in ["Kontakt: max.mustermann@klinik.de"[start:end]]
    ]
    assert any("max.mustermann@klinik.de" in r for r in regions)


def test_preserves_plain_text(anon):
    # No PII → output identical to input.
    text = "Die Behandlung verlief komplikationslos."
    assert anon.anonymize_text(text) == text
