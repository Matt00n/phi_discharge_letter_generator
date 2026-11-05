"""Tests for the deterministic core: morphology engine, renderer, quality gates."""

from generator.labels import Label, Tier
from generator.morphology import Case, Person, PersonForm, genitive, render_person
from generator.render import (
    AtomicEntity,
    Catalog,
    Negative,
    RenderResult,
    Span,
    SpanIntegrityError,
    render,
    scan_leaks,
    scan_negatives,
    validate_spans,
)

# --- Morphology -----------------------------------------------------------------


def _patient(nachname="Müller", vorname="Michael", gender="m", anrede="Herr"):
    return Person(
        entity_id="patient",
        label=Label.PATIENT_NAME,
        vorname=vorname,
        nachname=nachname,
        gender=gender,
        anrede=anrede,
    )


def _doctor(nachname="Wagner", vorname="Frank", titel="Dr. med."):
    return Person(
        entity_id="arzt_1",
        label=Label.STAFF_NAME,
        vorname=vorname,
        nachname=nachname,
        gender="m",
        anrede="Herr",
        titel=titel,
    )


def test_genitive_regular_and_sibilant():
    assert genitive("Müller") == "Müllers"
    assert genitive("Markus") == "Markus'"
    assert genitive("Weiß") == "Weiß'"
    assert genitive("Schulz") == "Schulz'"


def test_hon_forms_emit_no_prefix():
    """hon_* forms are functionally identical to non-hon — no honorific in output."""
    p = _patient()
    assert render_person(p, PersonForm.HON_LAST, Case.NOM).text == "Müller"
    assert render_person(p, PersonForm.HON_LAST, Case.DAT).text == "Müller"
    assert render_person(p, PersonForm.HON_LAST, Case.AKK).text == "Müller"
    assert render_person(p, PersonForm.HON_LAST, Case.NOM).prefix == ""
    assert (
        render_person(p, PersonForm.HON_FIRST_LAST, Case.NOM).text == "Michael Müller"
    )
    assert render_person(p, PersonForm.HON_FIRST_LAST, Case.NOM).prefix == ""


def test_genitive_only_inflects_final_token():
    p = _patient()
    m = render_person(p, PersonForm.HON_FIRST_LAST, Case.GEN)
    assert m.prefix == ""
    assert m.core == "Michael Müllers"


def test_prefix_is_non_pii_core_is_pii():
    p = _patient()
    m = render_person(p, PersonForm.HON_LAST, Case.GEN)
    # honorific is NEVER emitted — LLM writes it in skeleton text
    assert m.prefix == ""
    assert m.core == "Müllers"
    assert m.label is Label.PATIENT_NAME


def test_initial_forms():
    p = _patient()
    assert render_person(p, PersonForm.INITIAL_LAST, Case.NOM).text == "M. Müller"
    assert render_person(p, PersonForm.HON_INITIAL, Case.NOM).text == "M."
    assert render_person(p, PersonForm.HON_INITIAL, Case.NOM).prefix == ""


def test_salutation_gender_agreement():
    male = render_person(_patient(gender="m"), PersonForm.SALUTATION, Case.NOM)
    female = render_person(
        _patient(gender="f", anrede="Frau"), PersonForm.SALUTATION, Case.NOM
    )
    assert male.text == "Sehr geehrter Herr Müller"
    assert female.text == "Sehr geehrte Frau Müller"


def test_doctor_title_not_in_span():
    """Doctor titles are NOT part of the span — LLM writes them in skeleton."""
    d = _doctor()
    assert render_person(d, PersonForm.HON_LAST, Case.NOM).text == "Wagner"
    assert render_person(d, PersonForm.HON_LAST, Case.NOM).prefix == ""
    assert render_person(d, PersonForm.HON_FIRST_LAST, Case.NOM).text == "Frank Wagner"


# --- Renderer: spans by construction --------------------------------------------


def _catalog():
    cat = Catalog()
    cat.persons["patient"] = _patient()
    cat.persons["arzt_1"] = _doctor()
    cat.atomics["geburtsdatum"] = AtomicEntity(
        entity_id="geburtsdatum",
        label=Label.DATE_BIRTH,
        variants={"long": "14.05.1963", "short": "14.5.63"},
        default_variant="long",
    )
    cat.atomics["kvnr"] = AtomicEntity(
        entity_id="kvnr",
        label=Label.ID_INSURANCE,
        variants={"default": "A123456789"},
    )
    cat.negatives["crohn"] = Negative(
        entity_id="crohn", label=Label.EPONYM, surface="Morbus Crohn"
    )
    return cat


def test_render_records_exact_spans():
    cat = _catalog()
    skeleton = (
        "[[patient form=hon_first_last case=nom]], geb. [[geburtsdatum]], "
        "Versichertennr. [[kvnr]]. Vorstellung bei [[arzt_1 form=hon_last case=dat]]."
    )
    result = render(skeleton, cat)
    # Every recorded span must exactly cover its label's surface in the text.
    spans_by_eid = {s.entity_id: s for s in result.spans}
    assert result.text[spans_by_eid["patient"].start : spans_by_eid["patient"].end] == (
        "Michael Müller"
    )
    assert (
        result.text[
            spans_by_eid["geburtsdatum"].start : spans_by_eid["geburtsdatum"].end
        ]
        == "14.05.1963"
    )
    assert result.text[spans_by_eid["kvnr"].start : spans_by_eid["kvnr"].end] == (
        "A123456789"
    )
    # honorific excluded from the doctor span
    assert result.text[spans_by_eid["arzt_1"].start : spans_by_eid["arzt_1"].end] == (
        "Wagner"
    )
    assert spans_by_eid["geburtsdatum"].tier == Tier.DIRECT.value


def test_render_handles_shared_surname_without_ambiguity():
    """The classic failure of substring recovery: patient and doctor share a name."""
    cat = Catalog()
    cat.persons["patient"] = _patient(
        nachname="Weber", vorname="Anna", gender="f", anrede="Frau"
    )
    cat.persons["arzt_1"] = _doctor(
        nachname="Weber", vorname="Sabine", titel="Dr. med."
    )
    skeleton = "[[patient form=last]] wurde von [[arzt_1 form=hon_last]] behandelt."
    result = render(skeleton, cat)
    # Two distinct spans, same surface, correctly attributed to different entities.
    labels = {(s.entity_id, s.label) for s in result.spans}
    assert ("patient", Label.PATIENT_NAME.value) in labels
    assert ("arzt_1", Label.STAFF_NAME.value) in labels
    assert len(result.spans) == 2


def test_atomic_variant_selection():
    cat = _catalog()
    result = render("Geboren am [[geburtsdatum variant=short]].", cat)
    assert "14.5.63" in result.text
    assert result.spans[0].label == Label.DATE_BIRTH.value


# --- Quality gates --------------------------------------------------------------


def test_scan_leaks_clean_when_well_formed():
    cat = _catalog()
    result = render("[[patient form=hon_last]] mit [[kvnr]].", cat)
    assert scan_leaks(result, cat) == []


def test_scan_leaks_detects_literal_pii_in_skeleton():
    cat = _catalog()
    # The skeleton author leaked a literal value instead of using a placeholder.
    result = render("Versichertennr. A123456789 erfasst.", cat)
    findings = scan_leaks(result, cat)
    assert any(f.entity_id == "kvnr" for f in findings)


def test_scan_leaks_detects_residual_placeholder():
    cat = _catalog()
    # Malformed placeholder (lowercase ref ok, but missing closing) leaves markers.
    result = render("Patient [[ und Rest", cat)
    findings = scan_leaks(result, cat)
    assert any(f.entity_id == "__residual__" for f in findings)


def test_scan_negatives_ok_and_flags_overlap():
    cat = _catalog()
    ok = render("Bekannter [[crohn]] seit Jahren.", cat)
    assert scan_negatives(ok, cat) == []

    # Missing negative is reported.
    missing = render("Kein Eintrag.", cat)
    assert any(f.reason == "missing_in_text" for f in scan_negatives(missing, cat))


# --- Structural integrity assertions (validate_spans) ---------------------------

import pytest


def test_validate_spans_passes_on_well_formed():
    cat = _catalog()
    result = render("Wir berichten über [[patient form=hon_last case=akk]].", cat)
    validate_spans(result, cat)  # should not raise


def test_validate_spans_catches_out_of_bounds():
    cat = _catalog()
    text = "Hello world"
    bad_span = Span(start=5, end=99, label="X", tier="DIRECT", entity_id="patient")
    result = RenderResult(text=text, spans=[bad_span], negative_spans=[])
    with pytest.raises(SpanIntegrityError, match="end.*> text length"):
        validate_spans(result, cat)


def test_validate_spans_catches_inverted():
    cat = _catalog()
    text = "Hello world"
    bad_span = Span(start=5, end=3, label="X", tier="DIRECT", entity_id="patient")
    result = RenderResult(text=text, spans=[bad_span], negative_spans=[])
    with pytest.raises(SpanIntegrityError, match="start >= end"):
        validate_spans(result, cat)


def test_validate_spans_catches_whitespace():
    cat = _catalog()
    text = "Hello world "
    bad_span = Span(start=5, end=12, label="X", tier="DIRECT", entity_id="patient")
    result = RenderResult(text=text, spans=[bad_span], negative_spans=[])
    with pytest.raises(SpanIntegrityError, match="whitespace"):
        validate_spans(result, cat)


def test_validate_spans_catches_unknown_entity():
    cat = _catalog()
    text = "Hello world"
    bad_span = Span(start=0, end=5, label="X", tier="DIRECT", entity_id="ghost")
    result = RenderResult(text=text, spans=[bad_span], negative_spans=[])
    with pytest.raises(SpanIntegrityError, match="not found in catalog"):
        validate_spans(result, cat)


def test_validate_spans_catches_overlap():
    cat = _catalog()
    text = "Herr Michael Müller berichtet"
    span_a = Span(start=5, end=19, label="X", tier="DIRECT", entity_id="patient")
    span_b = Span(start=13, end=19, label="X", tier="DIRECT", entity_id="patient")
    result = RenderResult(text=text, spans=[span_a, span_b], negative_spans=[])
    with pytest.raises(SpanIntegrityError, match="Overlapping PII"):
        validate_spans(result, cat)


def test_validate_spans_catches_pii_neg_overlap():
    cat = _catalog()
    text = "Morbus Crohn ist bekannt"
    pii = Span(start=0, end=12, label="X", tier="DIRECT", entity_id="patient")
    neg = Span(start=7, end=12, label="EPONYM", tier="NONPII", entity_id="crohn")
    result = RenderResult(text=text, spans=[pii], negative_spans=[neg])
    with pytest.raises(SpanIntegrityError, match="overlaps negative"):
        validate_spans(result, cat)
