"""Integration tests for the typed-placeholder pipeline (Stages 1-5).

These exercise the deterministic, offline path: profile sampling, full record
generation via the dry-run skeleton author, splits, and the diversity audit.
No network / API access is required.
"""

from __future__ import annotations

import json

import pytest

from generator.clinical_seeds import ClinicalSeedsManager
from generator.diversity import audit
from generator.dry_run import template_skeleton
from generator.labels import Tier, tier_of
from generator.pipeline import Pipeline
from generator.profile_sampler import ProfileSampler
from generator.splits import assign_splits


@pytest.fixture(scope="module")
def seeds() -> ClinicalSeedsManager:
    return ClinicalSeedsManager(data_dir="data")


@pytest.fixture(scope="module")
def pipeline(seeds: ClinicalSeedsManager) -> Pipeline:
    return Pipeline(seeds, skeleton_provider=template_skeleton)


# --------------------------------------------------------------------------- #
# Stage 1: profile sampler
# --------------------------------------------------------------------------- #
def test_profile_sampler_is_deterministic(seeds: ClinicalSeedsManager) -> None:
    s = ProfileSampler(seeds)
    cat_a, spec_a = s.sample("doc_00000", 1000)
    cat_b, spec_b = s.sample("doc_00000", 1000)
    assert spec_a.to_metadata() == spec_b.to_metadata()
    assert cat_a.pii_surfaces() == cat_b.pii_surfaces()


def test_profile_sampler_difficulty_controls_negatives(
    seeds: ClinicalSeedsManager,
) -> None:
    s = ProfileSampler(seeds)
    _, easy = s.sample("d", 1, difficulty="easy")
    _, hard = s.sample("d", 1, difficulty="hard")
    assert easy.difficulty == "easy"
    assert hard.difficulty == "hard"


def test_profile_sampler_builds_patient(seeds: ClinicalSeedsManager) -> None:
    s = ProfileSampler(seeds)
    cat, spec = s.sample("doc", 1000)
    assert any(p.label.name == "PATIENT_NAME" for p in cat.persons.values())
    assert spec.patient_surname


# --------------------------------------------------------------------------- #
# Stage 1-4: full record generation (offline)
# --------------------------------------------------------------------------- #
def test_generate_record_spans_are_exact(pipeline: Pipeline) -> None:
    rec = pipeline.generate_record("doc_00000", 1000)
    d = rec.as_dict()
    text = d["text"]
    assert text
    assert d["spans"], "expected at least one PII span"
    for s in d["spans"]:
        assert 0 <= s["start"] < s["end"] <= len(text)
        # Tier must agree with the label taxonomy.
        assert tier_of_label(s["label"]) == s["tier"]


def test_generate_record_negatives_present_in_text(pipeline: Pipeline) -> None:
    # Hard documents reliably carry eponym / look-alike negatives.
    rec = pipeline.generate_record("doc_00001", 1001, difficulty="hard")
    d = rec.as_dict()
    text = d["text"]
    assert d["negatives"], "hard docs should contain negative spans"
    for s in d["negatives"]:
        assert text[s["start"] : s["end"]]
        assert s["tier"] == Tier.NONPII.value


def test_generate_record_is_deterministic(pipeline: Pipeline) -> None:
    a = pipeline.generate_record("doc_00000", 1000).as_dict()
    b = pipeline.generate_record("doc_00000", 1000).as_dict()
    assert a["text"] == b["text"]
    assert a["spans"] == b["spans"]


def test_generate_record_not_flagged_on_clean_render(pipeline: Pipeline) -> None:
    # The deterministic author must not leak literal PII outside gold spans.
    rec = pipeline.generate_record("doc_00002", 1002)
    assert rec.qc["leaks"] == []


def tier_of_label(label: str) -> str:
    return tier_of(label).value


# --------------------------------------------------------------------------- #
# Entity-collapse detection (planned persons must be rendered)
# --------------------------------------------------------------------------- #
def test_clean_render_has_no_missing_planned_persons(pipeline: Pipeline) -> None:
    # The dry-run author references every person in the mention plan, so no
    # planned person should be missing from the rendered spans.
    rec = pipeline.generate_record("doc_00000", 1000, difficulty="hard")
    assert rec.qc["missing_planned_persons"] == []


def test_entity_collapse_is_flagged(seeds: ClinicalSeedsManager) -> None:
    # A skeleton that references ONLY the patient — collapsing all other
    # planned persons (clinicians, relatives) — must be flagged, not dropped.
    def degenerate_skeleton(spec, catalog) -> str:
        return (
            "wir berichten über "
            "[[patient form=hon_first_last case=akk]], "
            "geb. [[geburtsdatum variant=long]]. "
            "Die stationäre Behandlung verlief komplikationslos. "
            "Wir entlassen [[patient form=hon_last case=akk]] in gutem Zustand."
        )

    pipe = Pipeline(seeds, skeleton_provider=degenerate_skeleton)
    rec = pipe.generate_record("doc_00000", 1000, difficulty="hard")

    # The record is produced (flagged, not dropped).
    assert rec.text
    # Planned clinicians/relatives never rendered -> detected.
    assert rec.qc["missing_planned_persons"]
    assert "patient" not in rec.qc["missing_planned_persons"]
    assert rec.qc["flagged"]
    assert any("entity collapse" in r for r in rec.qc["reasons"])


def test_single_planned_person_does_not_flag(seeds: ClinicalSeedsManager) -> None:
    # A document whose plan features only the patient must not flag when the
    # patient is rendered: the check is relative to the plan, not a static count.
    def patient_only_skeleton(spec, catalog) -> str:
        # Force a single-person plan to mimic a future single-entity case.
        spec.mention_plan.clear()
        spec.mention_plan["patient"] = ["hon_first_last"]
        return (
            "wir berichten über "
            "[[patient form=hon_first_last case=akk]]. "
            "Die Behandlung verlief komplikationslos."
        )

    pipe = Pipeline(seeds, skeleton_provider=patient_only_skeleton)
    rec = pipe.generate_record("doc_00000", 1000)
    assert rec.qc["missing_planned_persons"] == []


def test_incidental_omission_is_recorded_but_not_flagged(
    seeds: ClinicalSeedsManager,
) -> None:
    # Normal authoring drops the occasional peripheral person (a second
    # relative, an optional consultant). That must be *recorded* as missing
    # but must NOT flag the document — only a majority collapse flags.
    def drop_one_person_skeleton(spec, catalog) -> str:
        refs = list(spec.mention_plan)
        assert len(refs) >= 3, "need a multi-person plan for this test"
        dropped = refs[-1]  # a peripheral (non-patient) person
        assert dropped != "patient"
        parts = ["Wir berichten:"]
        for ref in refs:
            if ref == dropped:
                continue
            form = spec.mention_plan[ref][0]
            parts.append(f"[[{ref} form={form} case=nom]] war beteiligt.")
        return " ".join(parts)

    pipe = Pipeline(seeds, skeleton_provider=drop_one_person_skeleton)
    rec = pipe.generate_record("doc_00000", 1000, difficulty="hard")

    # Exactly one peripheral person missing -> recorded ...
    assert len(rec.qc["missing_planned_persons"]) == 1
    # ... but not a collapse, so no entity-collapse reason is raised.
    assert not any("entity collapse" in r for r in rec.qc["reasons"])


def test_entity_collapse_predicate_boundary(seeds: ClinicalSeedsManager) -> None:
    # The predicate is plan-relative: collapse iff more planned persons are
    # missing than rendered (2 * missing > planned). No static count.
    from generator.pipeline import _is_entity_collapse

    class _Spec:
        def __init__(self, n: int) -> None:
            self.mention_plan = {f"p{i}": ["last"] for i in range(n)}

    # 6 planned: 1 missing (incidental) does not collapse; 4 missing does.
    assert _is_entity_collapse(["p0"], _Spec(6)) is False
    assert _is_entity_collapse(["p0", "p1", "p2"], _Spec(6)) is False  # exactly half
    assert _is_entity_collapse(["p0", "p1", "p2", "p3"], _Spec(6)) is True
    # Single-entity plan: only a missing patient (100%) collapses.
    assert _is_entity_collapse([], _Spec(1)) is False
    assert _is_entity_collapse(["p0"], _Spec(1)) is True


# --------------------------------------------------------------------------- #
# Presentation diversity: ICD/OPS code citation + opening register
# --------------------------------------------------------------------------- #
def test_clinical_block_respects_code_style(seeds: ClinicalSeedsManager) -> None:
    import re

    from generator.skeleton_prompt import _clinical_block

    sampler = ProfileSampler(seeds)
    _cat, spec = sampler.sample("doc_00000", 1000, difficulty="hard")
    code_re = re.compile(r"\([A-Z]\d\d(?:\.\d+)?-?\)")

    spec.code_style = "inline"
    inline = _clinical_block(spec)
    spec.code_style = "primary_only"
    primary = _clinical_block(spec)
    spec.code_style = "none"
    none = _clinical_block(spec)

    # "none" must not show any ICD code anywhere.
    assert not code_re.search(none)
    # "primary_only" codes exactly the Hauptdiagnose line, not the others.
    haupt_line = next(
        line for line in primary.splitlines() if line.startswith("Hauptdiagnose:")
    )
    other_lines = [
        line
        for line in primary.splitlines()
        if line.startswith(("Nebendiagnosen:", "Prozeduren:"))
    ]
    assert code_re.search(haupt_line)
    assert all(not code_re.search(line) for line in other_lines)
    # "inline" codes at least as much as "primary_only".
    assert len(code_re.findall(inline)) >= len(code_re.findall(primary))


def test_user_content_carries_style_instructions(
    seeds: ClinicalSeedsManager,
) -> None:
    from generator.skeleton_prompt import build_user_content

    sampler = ProfileSampler(seeds)
    catalog, spec = sampler.sample("doc_00000", 1000, difficulty="hard")

    spec.code_style = "none"
    spec.opening_style = "frei"
    content = build_user_content(spec, catalog)
    assert "KEINE ICD-10" in content
    assert "Wir berichten über" in content  # the opening instruction forbids it

    spec.code_style = "inline"
    spec.opening_style = "bericht"
    content = build_user_content(spec, catalog)
    assert "ICD-10- bzw. OPS-Kode in Klammern" in content


def test_code_style_is_diverse_across_seeds(seeds: ClinicalSeedsManager) -> None:
    # The sampler must NOT emit codes for (almost) every document — that was the
    # original defect. Across many seeds we expect a real spread, with a
    # meaningful fraction citing no codes at all.
    sampler = ProfileSampler(seeds)
    from collections import Counter

    styles = Counter(
        sampler.sample(f"doc_{i:05d}", 1000 + i)[1].code_style for i in range(120)
    )
    assert set(styles) == {"inline", "primary_only", "none"}
    # No single style dominates everything, and "none" is well represented.
    assert styles["none"] / 120 >= 0.15
    assert styles["inline"] / 120 <= 0.70


def test_opening_style_is_diverse_across_seeds(seeds: ClinicalSeedsManager) -> None:
    sampler = ProfileSampler(seeds)
    from collections import Counter

    openings = Counter(
        sampler.sample(f"doc_{i:05d}", 1000 + i)[1].opening_style for i in range(120)
    )
    # At least 4 of the 5 registers appear, and the default never exceeds ~40%.
    assert len(openings) >= 4
    assert openings["bericht"] / 120 <= 0.45


# --------------------------------------------------------------------------- #
# Stage 5a: splits
# --------------------------------------------------------------------------- #
def test_assign_splits_attaches_buckets(pipeline: Pipeline) -> None:
    records = [
        pipeline.generate_record(f"doc_{i:05d}", 1000 + i).as_dict() for i in range(8)
    ]
    assign_splits(records)
    for r in records:
        sp = r["metadata"]["splits"]
        assert sp["by_domain"] in {"train", "test"}
        assert sp["by_difficulty"] in {"easy", "medium", "hard"}
        assert sp["by_surface"] in {"train", "test"}


# --------------------------------------------------------------------------- #
# Stage 5b: diversity audit
# --------------------------------------------------------------------------- #
def test_diversity_audit_reports_core_metrics(pipeline: Pipeline) -> None:
    records = [
        pipeline.generate_record(f"doc_{i:05d}", 1000 + i).as_dict() for i in range(8)
    ]
    report = audit(records)
    assert report["n_documents"] == 8
    assert report["unique_surnames"] >= 1
    assert "origin_distribution" in report
    assert "label_distribution" in report
