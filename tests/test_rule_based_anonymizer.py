"""Tests for the rule-based (REGEX-GAZ) de-identification baseline.

These lock in the *plumbing* and the high-precision rules that do not depend on
gazetteer coverage (regex identifiers, document-structure cues) plus the
structural safety that keeps medical eponyms unredacted. Gazetteer-name recall
is exercised with names that are known to exist in the external Faker lexica.
"""

import pytest

# The gazetteers are sourced from Faker (experiments dependency group).
pytest.importorskip("faker")

from experiments.benchmark.anonymizers import build_anonymizer  # noqa: E402
from experiments.benchmark.anonymizers.rule_based import (  # noqa: E402
    RuleBasedAnonymizer,
)
from experiments.benchmark.config import AnonymizerConfig  # noqa: E402
from experiments.benchmark.prompts import REDACTION_MARKER  # noqa: E402


@pytest.fixture(scope="module")
def anon() -> RuleBasedAnonymizer:
    return RuleBasedAnonymizer()


def _redacted_regions(anon: RuleBasedAnonymizer, text: str) -> list[str]:
    return [text[s:e] for s, e in anon.detect_spans(text)]


# --- Registry plumbing ---------------------------------------------------------


def test_registered_via_config():
    built = build_anonymizer(AnonymizerConfig(name="rule_based"))
    assert isinstance(built, RuleBasedAnonymizer)
    assert built.name == "rule_based"


# --- Regex identifiers (gazetteer-independent) ---------------------------------


def test_email_url_and_dates():
    text = "Kontakt: a.b@klinik.de, www.klinik-x.de, geb.: 20.05.2014."
    regions = _redacted_regions(anon=RuleBasedAnonymizer(), text=text)
    assert "a.b@klinik.de" in regions
    assert any(r.startswith("www.klinik-x.de") for r in regions)
    assert "20.05.2014" in regions


def test_phone_and_fax_keep_context_word(anon):
    out = anon.anonymize_text("Tel: 089 9061743 | Fax: 030 1435353")
    assert out == f"Tel: {REDACTION_MARKER} | Fax: {REDACTION_MARKER}"


def test_context_triggered_id(anon):
    out = anon.anonymize_text("Fallnr.: 7348312")
    assert out == f"Fallnr.: {REDACTION_MARKER}"


def test_insurance_number(anon):
    assert "A123456789" in _redacted_regions(anon, "KV: A123456789 gültig.")


# --- Structural name rules -----------------------------------------------------


def test_field_label_name(anon):
    assert anon.anonymize_text("Name: Wilfried Hecht") == f"Name: {REDACTION_MARKER}"


def test_signature_block_single_token_with_role(anon):
    text = "Mit freundlichen kollegialen Grüßen\n\nHecht\nStationsärztin"
    assert "Hecht" in _redacted_regions(anon, text)


# --- Gazetteer names (names known to be in Faker's German lexica) --------------


def test_gazetteer_first_name_anchors_full_name(anon):
    # "Wilfried" is a Faker given name; the surname is dragged along, and the
    # leading non-name words are dropped.
    regions = _redacted_regions(anon, "Die Einweisung von Wilfried Hecht erfolgte.")
    assert "Wilfried Hecht" in regions


def test_hyphenated_first_name_component_matches(anon):
    # "Kamil" is a Faker given name inside a hyphenated compound.
    regions = _redacted_regions(anon, "Vorgestellt durch Kamil-Maksymilian Platonow.")
    assert "Kamil-Maksymilian Platonow" in regions


# --- Negatives must be preserved (structural safety, no stoplist) --------------


def test_medical_eponyms_preserved(anon):
    text = (
        "Ausgeschlossen wurde eine Hashimoto-Thyreoiditis. Antikoagulation mit "
        "Marcumar. Erhebung eines Apgar-Score."
    )
    out = anon.anonymize_text(text)
    assert REDACTION_MARKER not in out
