"""Risk-weighted evaluation and breach analysis.

Assigns weights to PII categories by re-identification risk:
- DIRECT_HIGH (3.0): Patient name, DOB, full address — immediate re-identification
- DIRECT (2.0): Other direct identifiers (IDs, phone, email, doctor names)
- QUASI (1.0): Quasi-identifiers (age, occupation, nationality)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from experiments.benchmark.scoring.matching import MatchResult, Span

# Risk weights by label — aligned with generator/labels.py
DIRECT_HIGH_LABELS = {
    "PATIENT_NAME",
    "DATE_BIRTH",
    "ADDRESS",
}

DIRECT_LABELS = {
    "RELATIVE_NAME",
    "STAFF_NAME",
    "OTHER_PERSON_NAME",
    "DATE_ADMISSION",
    "DATE_DISCHARGE",
    "DATE_OTHER",
    "ZIP",
    "CITY",
    "ID_INSURANCE",
    "ID_CASE",
    "ID_PATIENT",
    "ID_OTHER",
    "PHONE",
    "FAX",
    "EMAIL",
    "URL",
    "INSTITUTION",
    "INSURER",
    "AGE",
}

RISK_WEIGHTS = {
    "DIRECT_HIGH": 3.0,
    "DIRECT": 2.0,
    "QUASI": 1.0,
}


def get_risk_tier(label: str) -> str:
    """Get risk tier for a label."""
    if label in DIRECT_HIGH_LABELS:
        return "DIRECT_HIGH"
    elif label in DIRECT_LABELS:
        return "DIRECT"
    return "QUASI"


def get_risk_weight(label: str) -> float:
    """Get numeric risk weight for a label."""
    return RISK_WEIGHTS[get_risk_tier(label)]


@dataclass
class DocumentRisk:
    """Risk classification for a single document."""

    doc_id: str
    classification: str = "SAFE"  # SAFE, AT_RISK, BREACHED
    missed_direct_high: list[Span] = field(default_factory=list)
    missed_direct: list[Span] = field(default_factory=list)
    missed_quasi: list[Span] = field(default_factory=list)
    risk_score: float = 0.0


@dataclass
class RiskReport:
    """Risk-weighted evaluation results."""

    # Risk-weighted recall
    weighted_recall: float = 0.0
    weighted_total: float = 0.0
    weighted_detected: float = 0.0

    # Breach rate: % docs with >= 1 missed DIRECT_HIGH
    breach_rate: float = 0.0
    breached_count: int = 0
    total_docs: int = 0

    # Per-document risk classification
    doc_risks: list[DocumentRisk] = field(default_factory=list)

    # Critical misses: all missed DIRECT_HIGH spans
    critical_misses: list[tuple[str, Span]] = field(
        default_factory=list
    )  # (doc_id, span)

    @property
    def safe_rate(self) -> float:
        if not self.doc_risks:
            return 0.0
        return sum(1 for d in self.doc_risks if d.classification == "SAFE") / len(
            self.doc_risks
        )

    def to_dict(self) -> dict:
        return {
            "weighted_recall": round(self.weighted_recall, 4),
            "breach_rate": round(self.breach_rate, 4),
            "safe_rate": round(self.safe_rate, 4),
            "breached_count": self.breached_count,
            "total_docs": self.total_docs,
            "critical_misses_count": len(self.critical_misses),
            "risk_distribution": {
                "SAFE": sum(1 for d in self.doc_risks if d.classification == "SAFE"),
                "AT_RISK": sum(
                    1 for d in self.doc_risks if d.classification == "AT_RISK"
                ),
                "BREACHED": sum(
                    1 for d in self.doc_risks if d.classification == "BREACHED"
                ),
            },
        }


def compute_risk(match_results: Sequence[MatchResult]) -> RiskReport:
    """Compute risk-weighted metrics from match results.

    Document classification:
    - BREACHED: >= 1 missed DIRECT_HIGH span
    - AT_RISK: >= 1 missed DIRECT span (but no DIRECT_HIGH misses)
    - SAFE: all misses are QUASI only (or no misses)
    """
    report = RiskReport()
    report.total_docs = len(match_results)

    total_weight = 0.0
    detected_weight = 0.0

    for mr in match_results:
        doc_risk = DocumentRisk(doc_id=mr.doc_id)
        has_direct_high_miss = False
        has_direct_miss = False

        # Count detected gold spans (TPs)
        for gold, _pred in mr.true_positives:
            w = get_risk_weight(gold.label)
            total_weight += w
            detected_weight += w

        # Count missed gold spans (FNs)
        for gold in mr.false_negatives:
            w = get_risk_weight(gold.label)
            total_weight += w
            tier = get_risk_tier(gold.label)

            if tier == "DIRECT_HIGH":
                has_direct_high_miss = True
                doc_risk.missed_direct_high.append(gold)
                report.critical_misses.append((mr.doc_id, gold))
            elif tier == "DIRECT":
                has_direct_miss = True
                doc_risk.missed_direct.append(gold)
            else:
                doc_risk.missed_quasi.append(gold)

        # Classify document
        if has_direct_high_miss:
            doc_risk.classification = "BREACHED"
            report.breached_count += 1
        elif has_direct_miss:
            doc_risk.classification = "AT_RISK"
        else:
            doc_risk.classification = "SAFE"

        # Per-doc risk score (sum of missed weights)
        doc_risk.risk_score = (
            sum(get_risk_weight(s.label) for s in doc_risk.missed_direct_high)
            + sum(get_risk_weight(s.label) for s in doc_risk.missed_direct)
            + sum(get_risk_weight(s.label) for s in doc_risk.missed_quasi)
        )

        report.doc_risks.append(doc_risk)

    report.weighted_total = total_weight
    report.weighted_detected = detected_weight
    report.weighted_recall = detected_weight / total_weight if total_weight > 0 else 1.0
    report.breach_rate = (
        report.breached_count / report.total_docs if report.total_docs > 0 else 0.0
    )

    return report
