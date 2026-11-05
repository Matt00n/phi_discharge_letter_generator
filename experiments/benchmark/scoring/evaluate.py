"""CLI entry point: python -m experiments.benchmark.scoring.evaluate

Evaluates anonymization system predictions against gold-standard annotations.

Modes:
  spans  — Predictions are character-offset spans (Mode A)
  text   — Predictions are de-identified text with [REDACTED] markers (Mode B)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from experiments.benchmark.scoring.alignment import align_texts
from experiments.benchmark.scoring.faithfulness import compute_faithfulness
from experiments.benchmark.scoring.matching import (
    MatchResult,
    Span,
    boundary_quality,
    match_lenient,
    match_strict,
    match_token_level,
)
from experiments.benchmark.scoring.metrics import compute_metrics
from experiments.benchmark.scoring.negatives import NEGATIVE_LABELS, compute_negatives
from experiments.benchmark.scoring.report import (
    build_full_report,
    format_console_summary,
    write_report,
)
from experiments.benchmark.scoring.risk import compute_risk


def load_gold(path: Path) -> list[dict]:
    """Load gold standard dataset (JSONL).

    Each line must have:
      - doc_id: str
      - text: str (original text)
      - annotations: list of {start, end, label, ...}
      - metadata: dict (optional, with difficulty/style/specialty/split)
    """
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))
    return docs


def load_predictions_spans(path: Path) -> dict[str, list[Span]]:
    """Load span-based predictions (Mode A).

    Each line: {"doc_id": "...", "spans": [{"start": N, "end": N}, ...]}
    """
    preds: dict[str, list[Span]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            doc_id = obj["doc_id"]
            spans = [
                Span(start=s["start"], end=s["end"], label=s.get("label", ""))
                for s in obj.get("spans", [])
            ]
            preds[doc_id] = spans
    return preds


def load_predictions_text(path: Path) -> dict[str, str]:
    """Load text-based predictions (Mode B).

    Each line: {"doc_id": "...", "text": "...with [REDACTED] markers..."}
    """
    preds: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            preds[obj["doc_id"]] = obj["text"]
    return preds


def gold_spans_from_doc(doc: dict, exclude_negatives: bool = True) -> list[Span]:
    """Extract gold Span objects from a document's spans + negatives."""
    result = []
    # Main PII spans
    for ann in doc.get("spans", []):
        label = ann["label"]
        if exclude_negatives and label in NEGATIVE_LABELS:
            continue
        result.append(
            Span(
                start=ann["start"],
                end=ann["end"],
                label=label,
                entity_id=ann.get("entity_id", ""),
                tier=ann.get("tier", ""),
                form=ann.get("form", ""),
            )
        )
    return result


def run_evaluation(
    gold_path: Path,
    pred_path: Path,
    mode: str,
    matching: str = "lenient",
    output_path: Path | None = None,
    verbose: bool = False,
    iou_threshold: float = 0.5,
) -> dict:
    """Run the full evaluation pipeline."""
    # Load data
    gold_docs = load_gold(gold_path)

    if not gold_docs:
        print("ERROR: No gold documents loaded.", file=sys.stderr)
        sys.exit(1)

    # Build metadata index
    doc_metadata: dict[str, dict] = {}
    for doc in gold_docs:
        meta = doc.get("metadata", {})
        doc_metadata[doc["doc_id"]] = meta

    # Get predictions and perform matching
    match_results: list[MatchResult] = []
    pred_spans_by_doc: dict[str, list[Span]] = {}
    originals: dict[str, str] = {}
    deidentified: dict[str, str] = {}
    gold_spans_by_doc: dict[str, list[Span]] = {}

    if mode == "spans":
        pred_spans_map = load_predictions_spans(pred_path)

        for doc in gold_docs:
            doc_id = doc["doc_id"]
            gold_spans = gold_spans_from_doc(doc)
            pred_spans = pred_spans_map.get(doc_id, [])
            pred_spans_by_doc[doc_id] = pred_spans
            gold_spans_by_doc[doc_id] = gold_spans

            if matching == "strict":
                mr = match_strict(gold_spans, pred_spans, doc_id=doc_id)
            elif matching == "token":
                mr = match_token_level(
                    doc["text"], gold_spans, pred_spans, doc_id=doc_id
                )
            else:  # lenient
                mr = match_lenient(
                    gold_spans, pred_spans, doc_id=doc_id, iou_threshold=iou_threshold
                )
            match_results.append(mr)

    elif mode == "text":
        pred_texts = load_predictions_text(pred_path)

        for doc in gold_docs:
            doc_id = doc["doc_id"]
            original = doc["text"]
            originals[doc_id] = original
            gold_spans = gold_spans_from_doc(doc)
            gold_spans_by_doc[doc_id] = gold_spans

            deid_text = pred_texts.get(doc_id)
            if deid_text is None:
                # Missing prediction — all gold are FN
                match_results.append(
                    MatchResult(
                        doc_id=doc_id,
                        true_positives=[],
                        false_negatives=gold_spans,
                        false_positives=[],
                    )
                )
                continue

            deidentified[doc_id] = deid_text

            # Align to get predicted spans
            alignment = align_texts(original, deid_text)
            pred_spans = alignment.predicted_spans
            pred_spans_by_doc[doc_id] = pred_spans

            if matching == "strict":
                mr = match_strict(gold_spans, pred_spans, doc_id=doc_id)
            elif matching == "token":
                mr = match_token_level(original, gold_spans, pred_spans, doc_id=doc_id)
            else:
                mr = match_lenient(
                    gold_spans, pred_spans, doc_id=doc_id, iou_threshold=iou_threshold
                )
            match_results.append(mr)
    else:
        print(f"ERROR: Unknown mode '{mode}'. Use 'spans' or 'text'.", file=sys.stderr)
        sys.exit(1)

    # Compute metrics
    metrics_report = compute_metrics(match_results, doc_metadata)
    risk_report = compute_risk(match_results)

    # Compute negatives (over-anonymization)
    negatives_report = compute_negatives(gold_docs, pred_spans_by_doc)

    # Compute faithfulness (Mode B only)
    faithfulness_report = None
    if mode == "text" and originals:
        faithfulness_report = compute_faithfulness(
            originals, deidentified, gold_spans_by_doc
        )

    # Compute boundary quality
    bq = None
    if matching == "lenient":
        # Aggregate boundary quality across all docs
        all_tps = []
        for mr in match_results:
            all_tps.extend(mr.true_positives)
        if all_tps:
            # Create a synthetic match result for boundary analysis
            combined = MatchResult(doc_id="__all__", true_positives=all_tps)
            bq = boundary_quality(combined)

    # Build report
    full_report = build_full_report(
        metrics=metrics_report,
        risk=risk_report,
        negatives=negatives_report,
        faithfulness=faithfulness_report,
        matching_mode=matching,
        boundary_metrics=bq,
    )

    # Console output
    console_summary = format_console_summary(
        metrics=metrics_report,
        risk=risk_report,
        negatives=negatives_report,
        faithfulness=faithfulness_report,
        matching_mode=matching,
    )
    print(console_summary)

    # Write JSON report
    if output_path:
        write_report(full_report, output_path)
        print(f"\nFull report written to: {output_path}")

    if verbose:
        # Print worst documents detail
        print("\n  DETAILED WORST DOCUMENTS:")
        for doc_m in metrics_report.worst_n_docs(10):
            if doc_m.f1 < 1.0:
                print(
                    f"    {doc_m.doc_id}: F1={doc_m.f1:.4f} TP={doc_m.tp} FP={doc_m.fp} FN={doc_m.fn}"
                )

        # Print critical misses
        if risk_report.critical_misses:
            print(f"\n  CRITICAL MISSES ({len(risk_report.critical_misses)}):")
            for doc_id, span in risk_report.critical_misses[:20]:
                print(f"    [{doc_id}] {span.label} @ {span.start}-{span.end}")

    return full_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate anonymization predictions against gold standard.",
        prog="python -m experiments.benchmark.scoring.evaluate",
    )
    parser.add_argument(
        "--gold",
        type=Path,
        required=True,
        help="Path to gold-standard JSONL file.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Path to predictions JSONL file.",
    )
    parser.add_argument(
        "--mode",
        choices=["spans", "text"],
        required=True,
        help="Prediction format: 'spans' (char offsets) or 'text' ([REDACTED] markers).",
    )
    parser.add_argument(
        "--matching",
        choices=["strict", "lenient", "token"],
        default="lenient",
        help="Matching strategy (default: lenient).",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for lenient matching (default: 0.5).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write JSON report.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed failure analysis.",
    )

    args = parser.parse_args()
    run_evaluation(
        gold_path=args.gold,
        pred_path=args.predictions,
        mode=args.mode,
        matching=args.matching,
        output_path=args.output,
        verbose=args.verbose,
        iou_threshold=args.iou_threshold,
    )


if __name__ == "__main__":
    main()
