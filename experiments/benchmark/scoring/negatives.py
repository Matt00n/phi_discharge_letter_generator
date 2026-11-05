"""Over-anonymization analysis: false positives on intentional negatives.

The dataset includes gold negative spans (EPONYM, LOOKALIKE, NONPII_ID)
that should NOT be anonymized. This module measures how often a system
incorrectly redacts these safe spans (over-anonymization / clinical corruption).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from experiments.benchmark.scoring.matching import Span, _overlap

# Labels that mark intentional negatives (should NOT be redacted)
NEGATIVE_LABELS = {"EPONYM", "LOOKALIKE", "NONPII_ID"}


@dataclass
class NegativeHit:
    """A negative span that was incorrectly redacted."""

    doc_id: str
    gold_span: Span
    pred_span: Span
    overlap_chars: int


@dataclass
class NegativesReport:
    """Report on over-anonymization of intentional negatives."""

    total_negatives: int = 0
    redacted_negatives: int = 0
    over_anonymization_rate: float = 0.0

    # By negative type
    by_label: dict[str, dict] = field(default_factory=dict)

    # Detailed hits
    hits: list[NegativeHit] = field(default_factory=list)

    # Clinical corruption: negatives that are clinical terms (EPONYMs)
    clinical_corruptions: int = 0

    def to_dict(self) -> dict:
        return {
            "total_negatives": self.total_negatives,
            "redacted_negatives": self.redacted_negatives,
            "over_anonymization_rate": round(self.over_anonymization_rate, 4),
            "clinical_corruptions": self.clinical_corruptions,
            "by_label": self.by_label,
        }


def compute_negatives(
    gold_docs: Sequence[dict],
    pred_spans_by_doc: dict[str, list[Span]],
    overlap_threshold: int = 1,
) -> NegativesReport:
    """Evaluate over-anonymization on gold negative spans.

    Args:
        gold_docs: List of gold documents, each with 'doc_id' and 'annotations'.
            Annotations include spans with labels in NEGATIVE_LABELS.
        pred_spans_by_doc: Mapping doc_id -> list of predicted spans.
        overlap_threshold: Minimum character overlap to count as a hit.
    """
    report = NegativesReport()
    by_label_counts: dict[str, dict[str, int]] = {}

    for doc in gold_docs:
        doc_id = doc["doc_id"]
        # Negatives live in their own top-level key, or fall back to spans
        neg_annotations = doc.get("negatives", [])
        if not neg_annotations:
            neg_annotations = [
                a for a in doc.get("spans", []) if a.get("label") in NEGATIVE_LABELS
            ]
        pred_spans = pred_spans_by_doc.get(doc_id, [])

        # Extract negative spans
        negatives = [
            Span(start=a["start"], end=a["end"], label=a["label"])
            for a in neg_annotations
            if a.get("label") in NEGATIVE_LABELS
        ]

        for neg in negatives:
            report.total_negatives += 1
            label = neg.label

            if label not in by_label_counts:
                by_label_counts[label] = {"total": 0, "redacted": 0}
            by_label_counts[label]["total"] += 1

            # Check if any prediction overlaps this negative
            for pred in pred_spans:
                ol = _overlap(neg, pred)
                if ol >= overlap_threshold:
                    report.redacted_negatives += 1
                    by_label_counts[label]["redacted"] += 1

                    hit = NegativeHit(
                        doc_id=doc_id,
                        gold_span=neg,
                        pred_span=pred,
                        overlap_chars=ol,
                    )
                    report.hits.append(hit)

                    # EPONYMs are clinical terms
                    if label == "EPONYM":
                        report.clinical_corruptions += 1
                    break  # One hit per negative span

    report.over_anonymization_rate = (
        report.redacted_negatives / report.total_negatives
        if report.total_negatives > 0
        else 0.0
    )

    # Format by_label
    for label, counts in by_label_counts.items():
        total = counts["total"]
        redacted = counts["redacted"]
        report.by_label[label] = {
            "total": total,
            "redacted": redacted,
            "rate": round(redacted / total, 4) if total > 0 else 0.0,
        }

    return report
