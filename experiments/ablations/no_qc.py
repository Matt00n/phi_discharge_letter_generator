"""Ablation 1 — NO QC.

Analysis only (no API, no generation). Takes the full *pre-QC* corpus of the
main run — the kept dataset PLUS every doc the red-team/deterministic checks
excluded to the review queue — and reports what would have shipped if Stage 4
were switched off: the residual flag rate and its breakdown by mechanism, label,
and red-team category.

This is the counterfactual behind the paper's QC section: every review-queue doc
is a defect Stage 4 caught that a no-QC pipeline would have released.

Usage:
    python -m experiments.ablations.no_qc [--baseline-dir DIR]
"""

from __future__ import annotations

import argparse
import os

from experiments.ablations._common import (
    DEFAULT_BASELINE_DIR,
    load_pre_qc_corpus,
    utf8_stdout,
)
from generator.exporter import Exporter

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT_DIR = os.path.join(_HERE, "..", "results", "ablation_no_qc")


def _summary_md(prov: dict, run_dir: str) -> str:
    y = prov["yield"]
    generated = y["generated_ok"]
    flagged = y["flagged"]
    qc = prov["qc_exclusions"]
    mech = qc["by_mechanism"]
    ov = qc["deterministic_vs_redteam_overlap"]
    by_label = prov["by_label"]

    lines: list[str] = []
    lines.append("# Ablation 1 — No QC (counterfactual)")
    lines.append("")
    lines.append(f"Source run: `{run_dir}`")
    lines.append("")
    lines.append(
        "If Stage-4 QC were disabled, the dataset would be the full generated "
        "corpus below, defects included. Each flagged document is a defect QC "
        "actually removed."
    )
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append(f"- Generated documents (pre-QC): **{generated}**")
    lines.append(
        f"- Would ship with a defect (flagged): **{flagged}** "
        f"({flagged / generated:.1%} of the corpus)"
    )
    lines.append(
        f"- Clean documents QC keeps: **{generated - flagged}** "
        f"({(generated - flagged) / generated:.1%})"
    )
    lines.append("")
    lines.append("## Exclusions by mechanism")
    lines.append("")
    lines.append("| Mechanism | Docs |")
    lines.append("| --- | ---: |")
    for k, v in mech.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append("## Deterministic vs. red-team overlap")
    lines.append("")
    lines.append(f"- Deterministic only: {ov['deterministic_only']}")
    lines.append(f"- Red-team only: {ov['redteam_only']}")
    lines.append(f"- Both: {ov['both']}")
    lines.append(
        f"- (Red-team total: {ov['redteam_total']}, "
        f"deterministic total: {ov['deterministic_total']})"
    )
    lines.append("")
    lines.append("## Red-team findings by category")
    lines.append("")
    lines.append("| Category | Findings |")
    lines.append("| --- | ---: |")
    for k, v in by_label["redteam_findings_by_category"].items():
        lines.append(f"| {k} | {v} |")
    lines.append("")
    if by_label["deterministic_leaks"]:
        lines.append("## Deterministic PII leaks by label")
        lines.append("")
        lines.append("| Label | Leaks |")
        lines.append("| --- | ---: |")
        for k, v in by_label["deterministic_leaks"].items():
            lines.append(f"| {k} | {v} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    utf8_stdout()
    ap = argparse.ArgumentParser(description="Ablation 1: no-QC counterfactual.")
    ap.add_argument("--baseline-dir", default=DEFAULT_BASELINE_DIR)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    out_dir = args.out_dir
    run_dir = args.baseline_dir
    if not os.path.exists(os.path.join(run_dir, "dataset.jsonl")):
        raise SystemExit(
            f"No dataset.jsonl in {run_dir}. The main run must finish first "
            f"(it writes outputs only at the end)."
        )

    corpus = load_pre_qc_corpus(run_dir)
    exporter = Exporter(output_dir=out_dir)
    # provenance aggregator computes the exact yield/mechanism/label breakdown
    # the pipeline reports — but here over the pre-QC corpus with no auto-filter.
    _, prov = exporter.export_provenance(corpus, [], auto_mode=True)

    md = _summary_md(prov, run_dir)
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(md)

    print(md)
    print(f"\nWrote {out_dir}/provenance.json and {out_dir}/summary.md")


if __name__ == "__main__":
    main()
