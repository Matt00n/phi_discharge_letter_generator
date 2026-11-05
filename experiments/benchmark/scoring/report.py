"""Report formatting: JSON output and console summary table."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.benchmark.scoring.faithfulness import FaithfulnessReport
from experiments.benchmark.scoring.metrics import MetricsReport
from experiments.benchmark.scoring.negatives import NegativesReport
from experiments.benchmark.scoring.risk import RiskReport


def format_console_summary(
    metrics: MetricsReport,
    risk: RiskReport,
    negatives: NegativesReport | None = None,
    faithfulness: FaithfulnessReport | None = None,
    matching_mode: str = "lenient",
) -> str:
    """Format a human-readable console summary."""
    lines = []
    lines.append("=" * 70)
    lines.append("  ANONYMIZATION EVALUATION REPORT")
    lines.append("=" * 70)
    lines.append("")

    # Overall detection metrics
    lines.append(f"  Matching mode: {matching_mode}")
    lines.append(f"  Documents evaluated: {len(metrics.per_doc)}")
    lines.append("")
    lines.append("  DETECTION METRICS (micro-averaged)")
    lines.append("  " + "-" * 40)
    lines.append(f"  Precision:  {metrics.micro.precision:.4f}")
    lines.append(f"  Recall:     {metrics.micro.recall:.4f}")
    lines.append(f"  F1:         {metrics.micro.f1:.4f}")
    lines.append(
        f"  TP: {metrics.micro.tp}  FP: {metrics.micro.fp}  FN: {metrics.micro.fn}"
    )
    lines.append("")

    # Document-level stats
    lines.append("  DOCUMENT-LEVEL")
    lines.append("  " + "-" * 40)
    lines.append(f"  Perfect documents (F1=1.0): {metrics.perfect_doc_rate:.1%}")
    lines.append(f"  Failed documents (F1=0.0):  {metrics.failure_doc_rate:.1%}")
    lines.append("")

    # Risk analysis
    lines.append("  RISK ANALYSIS")
    lines.append("  " + "-" * 40)
    lines.append(f"  Risk-weighted recall: {risk.weighted_recall:.4f}")
    lines.append(f"  Breach rate:          {risk.breach_rate:.1%}")
    lines.append(f"  Safe rate:            {risk.safe_rate:.1%}")
    lines.append(f"  Critical misses:      {len(risk.critical_misses)}")
    lines.append("")

    # By PII tier
    lines.append("  BY PII TIER")
    lines.append("  " + "-" * 40)
    lines.append(f"  {'Tier':<15} {'P':>8} {'R':>8} {'F1':>8} {'TP':>6} {'FN':>6}")
    for tier in ["DIRECT", "QUASI"]:
        if tier in metrics.by_tier:
            prf = metrics.by_tier[tier]
            lines.append(
                f"  {tier:<15} {prf.precision:>8.4f} {prf.recall:>8.4f} "
                f"{prf.f1:>8.4f} {prf.tp:>6} {prf.fn:>6}"
            )
    lines.append("")

    # By label family
    lines.append("  BY LABEL FAMILY")
    lines.append("  " + "-" * 40)
    lines.append(f"  {'Family':<15} {'P':>8} {'R':>8} {'F1':>8} {'TP':>6} {'FN':>6}")
    for family in sorted(metrics.by_family.keys()):
        prf = metrics.by_family[family]
        lines.append(
            f"  {family:<15} {prf.precision:>8.4f} {prf.recall:>8.4f} "
            f"{prf.f1:>8.4f} {prf.tp:>6} {prf.fn:>6}"
        )
    lines.append("")

    # By difficulty
    if metrics.by_difficulty:
        lines.append("  BY DIFFICULTY")
        lines.append("  " + "-" * 40)
        lines.append(
            f"  {'Difficulty':<15} {'P':>8} {'R':>8} {'F1':>8} {'TP':>6} {'FN':>6}"
        )
        for diff in ["easy", "medium", "hard", "adversarial"]:
            if diff in metrics.by_difficulty:
                prf = metrics.by_difficulty[diff]
                lines.append(
                    f"  {diff:<15} {prf.precision:>8.4f} {prf.recall:>8.4f} "
                    f"{prf.f1:>8.4f} {prf.tp:>6} {prf.fn:>6}"
                )
        # Include any unexpected keys
        for diff in sorted(metrics.by_difficulty.keys()):
            if diff not in ("easy", "medium", "hard", "adversarial"):
                prf = metrics.by_difficulty[diff]
                lines.append(
                    f"  {diff:<15} {prf.precision:>8.4f} {prf.recall:>8.4f} "
                    f"{prf.f1:>8.4f} {prf.tp:>6} {prf.fn:>6}"
                )
        lines.append("")

    # Negatives
    if negatives and negatives.total_negatives > 0:
        lines.append("  OVER-ANONYMIZATION (negatives)")
        lines.append("  " + "-" * 40)
        lines.append(f"  Total negative spans:   {negatives.total_negatives}")
        lines.append(f"  Incorrectly redacted:   {negatives.redacted_negatives}")
        lines.append(
            f"  Over-anonymization rate: {negatives.over_anonymization_rate:.1%}"
        )
        lines.append(f"  Clinical corruptions:   {negatives.clinical_corruptions}")
        lines.append("")

    # Faithfulness
    if faithfulness:
        lines.append("  FAITHFULNESS (Mode B)")
        lines.append("  " + "-" * 40)
        lines.append(f"  Mean faithfulness:     {faithfulness.mean_faithfulness:.4f}")
        lines.append(f"  Min faithfulness:      {faithfulness.min_faithfulness:.4f}")
        lines.append(f"  Docs with corruptions: {faithfulness.docs_with_corruptions}")
        lines.append(f"  Total corruptions:     {faithfulness.total_corruptions}")
        lines.append("")

    # Worst documents
    worst = metrics.worst_n_docs(5)
    if worst and worst[0].f1 < 1.0:
        lines.append("  WORST 5 DOCUMENTS")
        lines.append("  " + "-" * 40)
        lines.append(f"  {'Doc ID':<30} {'F1':>8} {'TP':>5} {'FP':>5} {'FN':>5}")
        for d in worst:
            doc_label = d.doc_id[:28] if len(d.doc_id) > 28 else d.doc_id
            lines.append(
                f"  {doc_label:<30} {d.f1:>8.4f} {d.tp:>5} {d.fp:>5} {d.fn:>5}"
            )
        lines.append("")

    lines.append("=" * 70)
    return "\n".join(lines)


def build_full_report(
    metrics: MetricsReport,
    risk: RiskReport,
    negatives: NegativesReport | None = None,
    faithfulness: FaithfulnessReport | None = None,
    matching_mode: str = "lenient",
    boundary_metrics: dict | None = None,
) -> dict:
    """Build the full JSON-serializable report."""
    report = {
        "matching_mode": matching_mode,
        "detection": metrics.to_dict(),
        "risk": risk.to_dict(),
    }
    if negatives:
        report["negatives"] = negatives.to_dict()
    if faithfulness:
        report["faithfulness"] = faithfulness.to_dict()
    if boundary_metrics:
        report["boundary_quality"] = boundary_metrics
    return report


def write_report(report: dict, output_path: Path, verbose: bool = False) -> None:
    """Write JSON report to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
