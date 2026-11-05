"""Tests for the off-the-shelf German PII detector (OpenMed).

Transformers and the model weights are heavy optional dependencies, so the
module is skipped when either is unavailable. The model is loaded once per
module. To avoid triggering a download in CI, the tests are skipped on a cold
Hugging Face cache.
"""

from pathlib import Path

import pytest

pytest.importorskip("transformers")
_HF_CACHE = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--OpenMed--OpenMed-PII-German-ClinicalE5-Base-109M-v1"
)
if not _HF_CACHE.exists():
    pytest.skip("OpenMed PII model not cached", allow_module_level=True)

from experiments.benchmark.anonymizers import build_anonymizer  # noqa: E402
from experiments.benchmark.anonymizers.openmed_pii import (  # noqa: E402
    OpenMedPIIAnonymizer,
    _segment_bounds,
)
from experiments.benchmark.config import AnonymizerConfig  # noqa: E402
from experiments.benchmark.prompts import REDACTION_MARKER  # noqa: E402


@pytest.fixture(scope="module")
def anon() -> OpenMedPIIAnonymizer:
    return OpenMedPIIAnonymizer(score_threshold=0.35)


def test_registered_via_config():
    built = build_anonymizer(AnonymizerConfig(name="openmed"))
    assert isinstance(built, OpenMedPIIAnonymizer)
    assert built.name == "openmed_OpenMed-PII-German-ClinicalE5-Base-109M-v1"


def test_segment_bounds_preserve_offsets():
    # Every segment must index back into the source text exactly, and newlines
    # (which segments never include) must fall between consecutive segments.
    text = "Name: Wilfried Hecht.\nDiagnose gestellt. Entlassung erfolgt.\n"
    bounds = _segment_bounds(text)
    assert bounds == sorted(bounds)
    for start, end in bounds:
        assert 0 <= start <= end <= len(text)
        assert "\n" not in text[start:end]


def test_detects_person_name(anon):
    # The model recovers names when the input is segmented to sentence scale.
    # It is flaky on isolated first names, so we assert on the surname only.
    text = "Name: Wilfried Hecht\nAufnahme regulär."
    out = anon.anonymize_text(text)
    assert "Hecht" not in out
    assert REDACTION_MARKER in out


def test_offsets_valid_across_document(anon):
    text = (
        "Name: Wilfried Hecht\n"
        "Geburtsdatum: 02.06.1958\n"
        "Die Aufnahme erfolgte regulär. Die Entlassung verlief komplikationslos.\n"
        "Kontakt: 030 1435353"
    )
    spans = anon.detect_spans(text)
    assert spans == sorted(spans)
    for start, end in spans:
        assert 0 <= start < end <= len(text)
