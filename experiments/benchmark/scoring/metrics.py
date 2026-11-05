"""Detection metrics: Precision, Recall, F1 with multi-dimensional slicing.

All metrics are detection-only. The system predicts *which* spans contain
PII; gold labels are used to slice and analyse but not to evaluate
classification accuracy.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Sequence

from experiments.benchmark.scoring.matching import MatchResult, Span

# Label families for grouping — aligned with generator/labels.py Label enum
LABEL_FAMILIES: dict[str, str] = {
    # NAMES
    "PATIENT_NAME": "NAMES",
    "RELATIVE_NAME": "NAMES",
    "STAFF_NAME": "NAMES",
    "OTHER_PERSON_NAME": "NAMES",
    # DATES
    "DATE_BIRTH": "DATES",
    "DATE_ADMISSION": "DATES",
    "DATE_DISCHARGE": "DATES",
    "DATE_OTHER": "DATES",
    # IDS
    "ID_INSURANCE": "IDS",
    "ID_CASE": "IDS",
    "ID_PATIENT": "IDS",
    "ID_OTHER": "IDS",
    # CONTACT
    "PHONE": "CONTACT",
    "FAX": "CONTACT",
    "EMAIL": "CONTACT",
    "URL": "CONTACT",
    # LOCATION
    "ADDRESS": "LOCATION",
    "ZIP": "LOCATION",
    "CITY": "LOCATION",
    "INSTITUTION": "LOCATION",
    # QUASI
    "AGE": "QUASI",
    "PROFESSION": "QUASI",
    "EMPLOYER": "QUASI",
    "NATIONALITY": "QUASI",
    "INSURER": "QUASI",
}

# PII tiers — derived directly from generator.labels.TIER_OF so this can
# never drift out of sync with the schema.
from generator.labels import TIER_OF as _GENERATOR_TIER_OF

PII_TIERS: dict[str, str] = {
    lbl.value: tier.value for lbl, tier in _GENERATOR_TIER_OF.items()
}
# Everything else (NONPII negatives, unknown labels) defaults to QUASI


def get_tier(label: str) -> str:
    """Get the PII tier for a label."""
    return PII_TIERS.get(label, "QUASI")


def get_family(label: str) -> str:
    """Get the label family for a label."""
    return LABEL_FAMILIES.get(label, "OTHER")


@dataclass
class PRF:
    """Precision / Recall / F1 triple with support counts."""

    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
        }


@dataclass
class DocumentMetrics:
    """Per-document metrics summary."""

    doc_id: str
    f1: float
    precision: float
    recall: float
    tp: int
    fp: int
    fn: int


@dataclass
class MetricsReport:
    """Full metrics report across all documents."""

    # Overall micro-averaged
    micro: PRF = field(default_factory=PRF)

    # Sliced metrics
    by_label: dict[str, PRF] = field(default_factory=lambda: defaultdict(PRF))
    by_family: dict[str, PRF] = field(default_factory=lambda: defaultdict(PRF))
    by_tier: dict[str, PRF] = field(default_factory=lambda: defaultdict(PRF))
    by_difficulty: dict[str, PRF] = field(default_factory=lambda: defaultdict(PRF))
    by_style: dict[str, PRF] = field(default_factory=lambda: defaultdict(PRF))
    by_specialty: dict[str, PRF] = field(default_factory=lambda: defaultdict(PRF))
    by_split: dict[str, PRF] = field(default_factory=lambda: defaultdict(PRF))

    # Per-document distribution
    per_doc: list[DocumentMetrics] = field(default_factory=list)

    @property
    def perfect_doc_rate(self) -> float:
        """Fraction of documents with F1 = 1.0."""
        if not self.per_doc:
            return 0.0
        return sum(1 for d in self.per_doc if d.f1 == 1.0) / len(self.per_doc)

    @property
    def failure_doc_rate(self) -> float:
        """Fraction of documents with F1 = 0.0 (and at least one gold span)."""
        if not self.per_doc:
            return 0.0
        failures = sum(1 for d in self.per_doc if d.f1 == 0.0 and d.fn > 0)
        return failures / len(self.per_doc)

    def worst_n_docs(self, n: int = 10) -> list[DocumentMetrics]:
        """Get the N documents with lowest F1."""
        return sorted(self.per_doc, key=lambda d: d.f1)[:n]

    def to_dict(self) -> dict:
        return {
            "micro": self.micro.to_dict(),
            "by_label": {k: v.to_dict() for k, v in sorted(self.by_label.items())},
            "by_family": {k: v.to_dict() for k, v in sorted(self.by_family.items())},
            "by_tier": {k: v.to_dict() for k, v in sorted(self.by_tier.items())},
            "by_difficulty": {
                k: v.to_dict() for k, v in sorted(self.by_difficulty.items())
            },
            "by_style": {k: v.to_dict() for k, v in sorted(self.by_style.items())},
            "by_specialty": {
                k: v.to_dict() for k, v in sorted(self.by_specialty.items())
            },
            "by_split": {k: v.to_dict() for k, v in sorted(self.by_split.items())},
            "perfect_doc_rate": round(self.perfect_doc_rate, 4),
            "failure_doc_rate": round(self.failure_doc_rate, 4),
            "num_documents": len(self.per_doc),
        }


def compute_metrics(
    match_results: Sequence[MatchResult],
    doc_metadata: dict[str, dict] | None = None,
) -> MetricsReport:
    """Compute detection metrics from match results.

    Args:
        match_results: Per-document matching outcomes.
        doc_metadata: Optional mapping of doc_id -> metadata dict with keys
            like 'difficulty', 'style', 'specialty', 'split'.
    """
    report = MetricsReport()
    doc_metadata = doc_metadata or {}

    for mr in match_results:
        doc_tp = len(mr.true_positives)
        doc_fp = len(mr.false_positives)
        doc_fn = len(mr.false_negatives)

        # Micro aggregation
        report.micro.tp += doc_tp
        report.micro.fp += doc_fp
        report.micro.fn += doc_fn

        # Per-document
        doc_p = doc_tp / (doc_tp + doc_fp) if (doc_tp + doc_fp) > 0 else 0.0
        doc_r = doc_tp / (doc_tp + doc_fn) if (doc_tp + doc_fn) > 0 else 0.0
        doc_f1 = 2 * doc_p * doc_r / (doc_p + doc_r) if (doc_p + doc_r) > 0 else 0.0
        report.per_doc.append(
            DocumentMetrics(
                doc_id=mr.doc_id,
                f1=doc_f1,
                precision=doc_p,
                recall=doc_r,
                tp=doc_tp,
                fp=doc_fp,
                fn=doc_fn,
            )
        )

        # Get document metadata
        meta = doc_metadata.get(mr.doc_id, {})
        difficulty = meta.get("difficulty", "unknown")
        style = meta.get("style", "unknown")
        specialty = meta.get("fachabteilung", meta.get("specialty", "unknown"))
        splits = meta.get("splits", {})
        split_domain = (
            splits.get("by_domain", "unknown")
            if isinstance(splits, dict)
            else str(splits)
        )

        # Slice by document-level metadata
        _add_to_prf(report.by_difficulty[difficulty], doc_tp, doc_fp, doc_fn)
        _add_to_prf(report.by_style[style], doc_tp, doc_fp, doc_fn)
        _add_to_prf(report.by_specialty[specialty], doc_tp, doc_fp, doc_fn)
        _add_to_prf(report.by_split[split_domain], doc_tp, doc_fp, doc_fn)

        # Slice by gold label (TP and FN have gold labels)
        for gold, _pred in mr.true_positives:
            label = gold.label
            family = get_family(label)
            tier = get_tier(label)
            _add_to_prf(report.by_label[label], 1, 0, 0)
            _add_to_prf(report.by_family[family], 1, 0, 0)
            _add_to_prf(report.by_tier[tier], 1, 0, 0)

        for gold in mr.false_negatives:
            label = gold.label
            family = get_family(label)
            tier = get_tier(label)
            _add_to_prf(report.by_label[label], 0, 0, 1)
            _add_to_prf(report.by_family[family], 0, 0, 1)
            _add_to_prf(report.by_tier[tier], 0, 0, 1)

        # FP: no gold label, but we still count them in by_label as "SPURIOUS"
        # (they don't have a gold label since they are false positives)
        # We skip per-label FP attribution since system is detection-only

    return report


def _add_to_prf(prf: PRF, tp: int, fp: int, fn: int) -> None:
    prf.tp += tp
    prf.fp += fp
    prf.fn += fn
