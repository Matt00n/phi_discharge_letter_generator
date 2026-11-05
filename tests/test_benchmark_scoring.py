"""Tests for the benchmark anonymizer-scoring framework."""

import json
import tempfile
from pathlib import Path

import pytest

from experiments.benchmark.scoring.alignment import (
    _MARKER_SENTINEL,
    REDACTION_MARKER,
    align_texts,
)
from experiments.benchmark.scoring.evaluate import run_evaluation
from experiments.benchmark.scoring.faithfulness import compute_faithfulness
from experiments.benchmark.scoring.matching import (
    MatchResult,
    Span,
    boundary_quality,
    match_lenient,
    match_strict,
    match_token_level,
)
from experiments.benchmark.scoring.metrics import (
    PRF,
    compute_metrics,
    get_family,
    get_tier,
)
from experiments.benchmark.scoring.negatives import compute_negatives
from experiments.benchmark.scoring.risk import compute_risk, get_risk_tier

# === matching.py tests ===


class TestMatchStrict:
    def test_exact_match(self):
        gold = [Span(0, 5, "NAME"), Span(10, 15, "DATE")]
        pred = [Span(0, 5), Span(10, 15)]
        result = match_strict(gold, pred, doc_id="d1")
        assert len(result.true_positives) == 2
        assert len(result.false_negatives) == 0
        assert len(result.false_positives) == 0

    def test_no_match(self):
        gold = [Span(0, 5, "NAME")]
        pred = [Span(1, 6)]
        result = match_strict(gold, pred, doc_id="d1")
        assert len(result.true_positives) == 0
        assert len(result.false_negatives) == 1
        assert len(result.false_positives) == 1

    def test_partial_overlap_no_strict_match(self):
        gold = [Span(0, 10, "NAME")]
        pred = [Span(0, 8)]
        result = match_strict(gold, pred, doc_id="d1")
        assert len(result.true_positives) == 0
        assert len(result.false_negatives) == 1
        assert len(result.false_positives) == 1

    def test_empty_inputs(self):
        result = match_strict([], [], doc_id="d1")
        assert len(result.true_positives) == 0
        assert len(result.false_negatives) == 0
        assert len(result.false_positives) == 0


class TestMatchLenient:
    def test_exact_match(self):
        gold = [Span(0, 10, "NAME")]
        pred = [Span(0, 10)]
        result = match_lenient(gold, pred, doc_id="d1")
        assert len(result.true_positives) == 1
        assert len(result.false_negatives) == 0

    def test_high_iou_match(self):
        # Gold: 0-10, Pred: 1-11 → overlap=9, union=11, IoU=9/11≈0.818
        gold = [Span(0, 10, "NAME")]
        pred = [Span(1, 11)]
        result = match_lenient(gold, pred, doc_id="d1", iou_threshold=0.5)
        assert len(result.true_positives) == 1

    def test_low_iou_no_match(self):
        # Gold: 0-10, Pred: 8-20 → overlap=2, union=20, IoU=2/20=0.1
        gold = [Span(0, 10, "NAME")]
        pred = [Span(8, 20)]
        result = match_lenient(gold, pred, doc_id="d1", iou_threshold=0.5)
        assert len(result.true_positives) == 0
        assert len(result.false_negatives) == 1
        assert len(result.false_positives) == 1

    def test_greedy_matching(self):
        # Two gold spans, two predictions, best match is 1-to-1
        gold = [Span(0, 10, "A"), Span(20, 30, "B")]
        pred = [Span(0, 10), Span(20, 30)]
        result = match_lenient(gold, pred, doc_id="d1")
        assert len(result.true_positives) == 2

    def test_optimal_matching_kuhn(self):
        # Kuhn invariant: pred order shouldn't affect which pairs are matched.
        # Gold A (0-10) best matches pred at 1-11 (IoU=9/11≈0.82).
        # Gold B (20-30) best matches pred at 21-31 (IoU=9/11≈0.82).
        # Preds supplied in reverse order — greedy-by-start would still work here,
        # but this validates Kuhn doesn't drop a match due to ordering.
        gold = [Span(0, 10, "A"), Span(20, 30, "B")]
        pred = [Span(21, 31), Span(1, 11)]  # reversed vs gold order
        result = match_lenient(gold, pred, doc_id="d1")
        assert len(result.true_positives) == 2
        assert len(result.false_negatives) == 0
        assert len(result.false_positives) == 0

    def test_optimal_uses_augmenting_path(self):
        # The case where greedy-best-first fails but Kuhn succeeds:
        # gold A (0-10): eligible pred X (IoU=1.0) and pred Y (IoU=9/11≈0.82)
        # gold B (1-11): eligible pred X (IoU=9/11≈0.82) and pred Y (IoU=1.0)
        # Greedy-best-first: A takes X (1.0), B takes Y (1.0) → 2 TP (fine here).
        # True suboptimal case for best-first: A has only X; B has X and Y.
        # gold A (0-10) eligible: pred X (0-10, IoU=1.0) ONLY
        # gold B (100-110) eligible: pred X IoU=0 (disjoint); pred Y (100-110, IoU=1.0)
        # That's trivially separated. The canonical case requires a shared eligible pred:
        # gold A: 0-10, gold B: 4-14
        # pred X: 0-10 → IoU(A)=1.0, IoU(B)=6/14≈0.43 < 0.5
        # pred Y: 4-14 → IoU(A)=6/14≈0.43 < 0.5, IoU(B)=1.0
        # Each gold only matches its own pred anyway. Valid 2 TP regardless.
        # Since non-overlapping gold with shared pred at IoU>=0.5 is geometrically
        # constrained, Kuhn's main value is correctness guarantee + pred-order invariance.
        gold = [Span(0, 10, "A"), Span(4, 14, "B")]
        pred = [Span(4, 14), Span(0, 10)]
        result = match_lenient(gold, pred, doc_id="d1")
        assert len(result.true_positives) == 2


class TestMatchTokenLevel:
    def test_basic(self):
        text = "Hello World Foo Bar"
        gold = [Span(0, 5, "NAME")]  # "Hello"
        pred = [Span(0, 5)]
        result = match_token_level(text, gold, pred, doc_id="d1")
        assert len(result.true_positives) == 1
        assert len(result.false_negatives) == 0
        assert len(result.false_positives) == 0

    def test_partial_overlap_counted(self):
        text = "Hello World"
        gold = [Span(0, 5, "NAME")]  # "Hello"
        pred = [Span(0, 8)]  # "Hello Wo" — covers "Hello" token fully
        result = match_token_level(text, gold, pred, doc_id="d1")
        # "Hello" is both gold and pred → TP
        # "World" is pred but not gold → FP
        assert len(result.true_positives) == 1
        assert len(result.false_positives) == 1


class TestBoundaryQuality:
    def test_perfect_boundaries(self):
        mr = MatchResult(
            doc_id="d1",
            true_positives=[(Span(0, 10, "A"), Span(0, 10))],
        )
        bq = boundary_quality(mr)
        assert bq["mean_iou"] == 1.0
        assert bq["mean_start_offset"] == 0.0
        assert bq["over_extension_rate"] == 0.0
        assert bq["under_extension_rate"] == 0.0

    def test_over_extension(self):
        mr = MatchResult(
            doc_id="d1",
            true_positives=[(Span(5, 10, "A"), Span(3, 12))],
        )
        bq = boundary_quality(mr)
        assert bq["over_extension_rate"] == 1.0
        assert bq["mean_iou"] < 1.0


# === alignment.py tests ===


class TestAlignment:
    def test_simple_redaction(self):
        original = "Patient Max Mustermann wurde aufgenommen."
        deident = "Patient [REDACTED] wurde aufgenommen."
        result = align_texts(original, deident)
        assert len(result.predicted_spans) == 1
        span = result.predicted_spans[0]
        assert original[span.start : span.end] == "Max Mustermann"

    def test_no_changes(self):
        text = "No PII here."
        result = align_texts(text, text)
        assert len(result.predicted_spans) == 0
        assert len(result.corruptions) == 0

    def test_corruption_detected(self):
        original = "Patient was seen in Berlin."
        deident = "Patient was seen in München."  # Changed non-PII
        result = align_texts(original, deident)
        assert len(result.corruptions) > 0

    def test_multi_marker_anchor_alignment(self):
        # Original has two separate PII entities with preserved text between them.
        # The LLM replaces both in a region that difflib merges into one opcode.
        # Anchor alignment should recover exact boundaries using the separator ", geb. ".
        original = "Max Mustermann, geb. 01.01.1980"
        deident = "[REDACTED], geb. [REDACTED]"
        result = align_texts(original, deident)
        assert len(result.predicted_spans) == 2
        # First span should cover "Max Mustermann" (0-14)
        spans = sorted(result.predicted_spans, key=lambda s: s.start)
        assert original[spans[0].start : spans[0].end] == "Max Mustermann"
        # Second span should cover "01.01.1980" (21-31)
        assert original[spans[1].start : spans[1].end] == "01.01.1980"

    def test_fragmented_marker_recovery(self):
        # Regression test for the difflib fragmentation bug.
        # "Reha-Zentrum Berlin" starts with 'R', which matches the 'R' in
        # "[REDACTED]". difflib emits:
        #   insert '' → '['
        #   equal  'R' == 'R'   ← shared char fragments the marker
        #   replace 'eha-Zentrum Berlin' → 'EDACTED]'
        # The sentinel-normalisation approach must still recover the full span.
        original = "Reha-Zentrum Berlin\nAbteilung"
        deident = "[REDACTED]\nAbteilung"
        result = align_texts(original, deident)
        assert len(result.predicted_spans) == 1
        assert (
            original[result.predicted_spans[0].start : result.predicted_spans[0].end]
            == "Reha-Zentrum Berlin"
        )

    def test_leading_letter_collision_no_sliver(self):
        # Real-data regression: names whose leading capital coincides with a
        # letter in "[REDACTED]" (R/E/D/A/C/T) used to leak a sliver — e.g.
        # "Emily" → spurious FP "mily", "Engelbert" → "ngelbert". With sentinel
        # normalisation each name maps to exactly one whole-name span.
        for name in ("Emily", "Engelbert", "Rehabilitationszentrum", "Daniela"):
            original = f"Frau {name} Berger"
            deident = f"Frau {REDACTION_MARKER} {REDACTION_MARKER}"
            result = align_texts(original, deident)
            surfaces = sorted(
                original[s.start : s.end] for s in result.predicted_spans
            )
            assert surfaces == sorted([name, "Berger"]), name

    def test_adjacent_spans_split_on_separator(self):
        # A ZIP+city block redacted as two markers separated by a space must
        # yield two distinct predicted spans, not one merged region.
        original = "07743 Jena"
        deident = f"{REDACTION_MARKER} {REDACTION_MARKER}"
        result = align_texts(original, deident)
        surfaces = sorted(original[s.start : s.end] for s in result.predicted_spans)
        assert surfaces == ["07743", "Jena"]

    def test_sentinel_in_input_flags_error(self):
        # If the private-use sentinel ever appears in the input, alignment must
        # not silently mismap: it flags the condition and reports failure.
        original = f"x{_MARKER_SENTINEL}y"
        deident = f"x{REDACTION_MARKER}y"
        result = align_texts(original, deident)
        assert result.success is False
        assert result.alignment_errors


# === metrics.py tests ===


class TestMetrics:
    def test_perfect_scores(self):
        mr = MatchResult(
            doc_id="d1",
            true_positives=[(Span(0, 5, "NAME"), Span(0, 5))],
            false_negatives=[],
            false_positives=[],
        )
        report = compute_metrics([mr])
        assert report.micro.precision == 1.0
        assert report.micro.recall == 1.0
        assert report.micro.f1 == 1.0

    def test_no_predictions(self):
        mr = MatchResult(
            doc_id="d1",
            true_positives=[],
            false_negatives=[Span(0, 5, "NAME")],
            false_positives=[],
        )
        report = compute_metrics([mr])
        assert report.micro.recall == 0.0
        assert report.micro.f1 == 0.0

    def test_by_label_slicing(self):
        mr = MatchResult(
            doc_id="d1",
            true_positives=[(Span(0, 5, "PATIENT_NAME"), Span(0, 5))],
            false_negatives=[Span(10, 15, "DATE_BIRTH")],
            false_positives=[],
        )
        report = compute_metrics([mr])
        assert report.by_label["PATIENT_NAME"].tp == 1
        assert report.by_label["DATE_BIRTH"].fn == 1
        assert report.by_family["NAMES"].tp == 1
        assert report.by_family["DATES"].fn == 1

    def test_prf_class(self):
        prf = PRF(tp=80, fp=20, fn=10)
        assert prf.precision == 0.8
        assert prf.recall == pytest.approx(80 / 90)


# === risk.py tests ===


class TestRisk:
    def test_all_detected_safe(self):
        mr = MatchResult(
            doc_id="d1",
            true_positives=[(Span(0, 5, "PATIENT_NAME"), Span(0, 5))],
            false_negatives=[],
            false_positives=[],
        )
        report = compute_risk([mr])
        assert report.breach_rate == 0.0
        assert report.safe_rate == 1.0
        assert report.weighted_recall == 1.0

    def test_missed_direct_high_breached(self):
        mr = MatchResult(
            doc_id="d1",
            true_positives=[],
            false_negatives=[Span(0, 5, "PATIENT_NAME")],
            false_positives=[],
        )
        report = compute_risk([mr])
        assert report.breach_rate == 1.0
        assert report.breached_count == 1
        assert len(report.critical_misses) == 1

    def test_missed_quasi_safe(self):
        mr = MatchResult(
            doc_id="d1",
            true_positives=[],
            false_negatives=[Span(0, 5, "PROFESSION")],
            false_positives=[],
        )
        report = compute_risk([mr])
        assert report.breach_rate == 0.0
        doc_risk = report.doc_risks[0]
        assert doc_risk.classification == "SAFE"


# === negatives.py tests ===


class TestNegatives:
    def test_no_over_anonymization(self):
        gold_docs = [
            {
                "doc_id": "d1",
                "negatives": [
                    {"start": 0, "end": 5, "label": "EPONYM"},
                ],
            }
        ]
        pred_spans = {"d1": [Span(10, 15)]}  # Doesn't overlap
        report = compute_negatives(gold_docs, pred_spans)
        assert report.total_negatives == 1
        assert report.redacted_negatives == 0

    def test_over_anonymization(self):
        gold_docs = [
            {
                "doc_id": "d1",
                "negatives": [
                    {"start": 0, "end": 10, "label": "EPONYM"},
                ],
            }
        ]
        pred_spans = {"d1": [Span(2, 8)]}  # Overlaps
        report = compute_negatives(gold_docs, pred_spans)
        assert report.redacted_negatives == 1
        assert report.clinical_corruptions == 1


# === faithfulness.py tests ===


class TestFaithfulness:
    def test_perfect_faithfulness(self):
        originals = {"d1": "Hello Max World"}
        deidentified = {"d1": "Hello [REDACTED] World"}
        gold_spans = {"d1": [Span(6, 9, "NAME")]}  # "Max"
        report = compute_faithfulness(originals, deidentified, gold_spans)
        assert report.mean_faithfulness == 1.0

    def test_corruption_lowers_score(self):
        originals = {"d1": "Hello Max in Berlin today"}
        deidentified = {"d1": "Hello [REDACTED] in München today"}  # Berlin→München
        gold_spans = {"d1": [Span(6, 9, "NAME")]}  # Only "Max" is PII
        report = compute_faithfulness(originals, deidentified, gold_spans)
        assert report.mean_faithfulness < 1.0


# === End-to-end test ===


class TestEndToEnd:
    def test_spans_mode(self, tmp_path):
        # Create gold file
        gold_doc = {
            "doc_id": "doc1",
            "text": "Patient Max Mustermann, geb. 01.01.1980, wohnhaft Berlin.",
            "spans": [
                {"start": 8, "end": 22, "label": "PATIENT_LAST_NAME"},
                {"start": 29, "end": 39, "label": "DATE_BIRTH"},
                {"start": 51, "end": 57, "label": "CITY"},
            ],
            "metadata": {
                "difficulty": "easy",
                "style": "formal",
                "fachabteilung": "Innere",
            },
        }
        gold_path = tmp_path / "gold.jsonl"
        gold_path.write_text(json.dumps(gold_doc) + "\n", encoding="utf-8")

        # Create predictions (one miss: CITY)
        pred_doc = {
            "doc_id": "doc1",
            "spans": [
                {"start": 8, "end": 22},
                {"start": 29, "end": 39},
            ],
        }
        pred_path = tmp_path / "pred.jsonl"
        pred_path.write_text(json.dumps(pred_doc) + "\n", encoding="utf-8")

        output_path = tmp_path / "report.json"
        report = run_evaluation(
            gold_path=gold_path,
            pred_path=pred_path,
            mode="spans",
            matching="strict",
            output_path=output_path,
        )

        assert report["detection"]["micro"]["tp"] == 2
        assert report["detection"]["micro"]["fn"] == 1
        assert report["detection"]["micro"]["fp"] == 0
        assert output_path.exists()

    def test_text_mode(self, tmp_path):
        original = "Patient Max Mustermann wurde in Berlin behandelt."
        gold_doc = {
            "doc_id": "doc1",
            "text": original,
            "spans": [
                {"start": 8, "end": 22, "label": "PATIENT_LAST_NAME"},
                {"start": 32, "end": 38, "label": "CITY"},
            ],
            "metadata": {},
        }
        gold_path = tmp_path / "gold.jsonl"
        gold_path.write_text(json.dumps(gold_doc) + "\n", encoding="utf-8")

        deident = "Patient [REDACTED] wurde in [REDACTED] behandelt."
        pred_doc = {"doc_id": "doc1", "text": deident}
        pred_path = tmp_path / "pred.jsonl"
        pred_path.write_text(json.dumps(pred_doc) + "\n", encoding="utf-8")

        report = run_evaluation(
            gold_path=gold_path,
            pred_path=pred_path,
            mode="text",
            matching="lenient",
        )

        # Both spans should be detected
        assert report["detection"]["micro"]["tp"] == 2
        assert report["detection"]["micro"]["fn"] == 0
        assert "faithfulness" in report
