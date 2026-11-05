"""Tests for the Flair (German NER) de-identification method.

Flair, Presidio, spaCy, and the large NER weights are heavy optional
dependencies, so the module is skipped when any is unavailable. The model is
loaded once per module. The cached model is ~2 GB; these tests are skipped on a
cold cache to avoid triggering a multi-gigabyte download in CI.
"""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("flair")
pytest.importorskip("presidio_analyzer")
if importlib.util.find_spec("de_core_news_lg") is None:
    pytest.skip("de_core_news_lg model not installed", allow_module_level=True)
_FLAIR_CACHE = Path.home() / ".flair" / "models" / "ner-german-large"
if not _FLAIR_CACHE.exists():
    pytest.skip("flair/ner-german-large not cached", allow_module_level=True)

from experiments.benchmark.anonymizers import build_anonymizer  # noqa: E402
from experiments.benchmark.anonymizers.flair_ner import FlairAnonymizer  # noqa: E402
from experiments.benchmark.config import AnonymizerConfig  # noqa: E402
from experiments.benchmark.prompts import REDACTION_MARKER  # noqa: E402


@pytest.fixture(scope="module")
def anon() -> FlairAnonymizer:
    return FlairAnonymizer(score_threshold=0.35)


def test_registered_via_config():
    built = build_anonymizer(AnonymizerConfig(name="flair"))
    assert isinstance(built, FlairAnonymizer)
    assert built.name == "flair_ner-german-large"


def test_detects_person_name(anon):
    text = "Wir berichten über den Aufenthalt von Wilfried Hecht."
    out = anon.anonymize_text(text)
    assert "Wilfried Hecht" not in out
    assert REDACTION_MARKER in out


def test_pattern_recognizers_still_active(anon):
    # Dates/e-mail come from Presidio's pattern layer, not Flair NER.
    out = anon.anonymize_text("Termin am 20.05.2014, Kontakt a.b@klinik.de.")
    assert "20.05.2014" not in out
    assert "a.b@klinik.de" not in out


def test_offsets_valid_across_sentences(anon):
    # A multi-sentence document must yield in-bounds, non-overlapping spans.
    text = (
        "Die Aufnahme erfolgte regulär. Behandelnder Arzt war Thorben-Florian "
        "Münch. Die Entlassung verlief komplikationslos."
    )
    spans = anon.detect_spans(text)
    assert spans == sorted(spans)
    for start, end in spans:
        assert 0 <= start < end <= len(text)
