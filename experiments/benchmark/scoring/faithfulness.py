"""Faithfulness evaluation for Mode B (text-based) predictions.

Verifies that the de-identification system only changes PII regions and
preserves all non-PII text faithfully. Measures:
- Faithfulness score: % of non-PII characters preserved unchanged
- Corruption list: regions where non-PII text was modified
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from experiments.benchmark.scoring.alignment import AlignmentResult, align_texts
from experiments.benchmark.scoring.matching import Span, _overlap


@dataclass
class FaithfulnessResult:
    """Faithfulness metrics for a single document."""

    doc_id: str
    faithfulness_score: float  # % of non-PII chars preserved
    total_non_pii_chars: int
    corrupted_chars: int
    corruptions: list[dict] = field(default_factory=list)


@dataclass
class FaithfulnessReport:
    """Aggregate faithfulness report across all documents."""

    mean_faithfulness: float = 0.0
    min_faithfulness: float = 1.0
    total_corruptions: int = 0
    docs_with_corruptions: int = 0
    total_docs: int = 0
    per_doc: list[FaithfulnessResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "mean_faithfulness": round(self.mean_faithfulness, 4),
            "min_faithfulness": round(self.min_faithfulness, 4),
            "total_corruptions": self.total_corruptions,
            "docs_with_corruptions": self.docs_with_corruptions,
            "total_docs": self.total_docs,
            "corruption_rate": round(self.docs_with_corruptions / self.total_docs, 4)
            if self.total_docs > 0
            else 0.0,
        }


def _is_in_gold_span(pos: int, gold_spans: Sequence[Span]) -> bool:
    """Check if a character position falls within any gold span."""
    for s in gold_spans:
        if s.start <= pos < s.end:
            return True
    return False


def compute_faithfulness(
    originals: dict[str, str],
    deidentified: dict[str, str],
    gold_spans_by_doc: dict[str, list[Span]],
) -> FaithfulnessReport:
    """Compute faithfulness metrics for Mode B predictions.

    Args:
        originals: Mapping doc_id -> original text.
        deidentified: Mapping doc_id -> de-identified text.
        gold_spans_by_doc: Mapping doc_id -> gold PII spans.
    """
    report = FaithfulnessReport()
    report.total_docs = len(originals)
    faithfulness_scores = []

    for doc_id, original in originals.items():
        deid = deidentified.get(doc_id)
        if deid is None:
            continue

        gold_spans = gold_spans_by_doc.get(doc_id, [])
        alignment = align_texts(original, deid)

        # Count non-PII characters in original
        total_non_pii = 0
        for i in range(len(original)):
            if not _is_in_gold_span(i, gold_spans):
                total_non_pii += 1

        # Count corrupted non-PII characters
        corrupted_chars = 0
        corruptions = []
        for start, end, orig_text, replacement in alignment.corruptions:
            # Check how many chars in this region are non-PII
            non_pii_in_region = 0
            for i in range(start, end):
                if not _is_in_gold_span(i, gold_spans):
                    non_pii_in_region += 1
            if non_pii_in_region > 0:
                corrupted_chars += non_pii_in_region
                corruptions.append(
                    {
                        "start": start,
                        "end": end,
                        "original": orig_text[:100],
                        "replacement": replacement[:100],
                        "non_pii_chars_affected": non_pii_in_region,
                    }
                )

        # Also check predicted spans that don't overlap gold
        for pred_span in alignment.predicted_spans:
            non_pii_in_pred = 0
            for i in range(pred_span.start, pred_span.end):
                if not _is_in_gold_span(i, gold_spans):
                    non_pii_in_pred += 1
            if non_pii_in_pred > 0:
                corrupted_chars += non_pii_in_pred

        faithfulness = (
            (total_non_pii - corrupted_chars) / total_non_pii
            if total_non_pii > 0
            else 1.0
        )

        doc_result = FaithfulnessResult(
            doc_id=doc_id,
            faithfulness_score=faithfulness,
            total_non_pii_chars=total_non_pii,
            corrupted_chars=corrupted_chars,
            corruptions=corruptions,
        )
        report.per_doc.append(doc_result)
        faithfulness_scores.append(faithfulness)

        if corruptions:
            report.docs_with_corruptions += 1
            report.total_corruptions += len(corruptions)

        if faithfulness < report.min_faithfulness:
            report.min_faithfulness = faithfulness

    report.mean_faithfulness = (
        sum(faithfulness_scores) / len(faithfulness_scores)
        if faithfulness_scores
        else 1.0
    )

    return report
