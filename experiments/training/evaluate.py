"""Evaluate a fine-tuned PII detector on a target corpus.

Runs the model over every document with sliding-window chunking, reconstructs
character spans, and scores them against gold with the existing benchmark
scorer (lenient IoU matching) so numbers are directly comparable to the
de-identification benchmark. Writes report.json + a console summary.

Examples:
    # held-out synthetic
    python -m experiments.training.evaluate --model-dir <dir> \
        --eval-synthetic experiments/datasets/sparcling-de/dataset.jsonl

    # real target (cross-generator transfer)
    python -m experiments.training.evaluate --model-dir <dir> \
        --eval-grascco experiments/datasets/grascco_phi_lohr_2024
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from experiments.benchmark.scoring.matching import (
    Span,
    match_lenient,
    match_token_level,
)
from experiments.benchmark.scoring.metrics import compute_metrics
from experiments.training.data import (
    Doc,
    encode_doc,
    kfold_split,
    load_grascco,
    load_synthetic,
    reconstruct_spans,
    split_train_test,
)


@torch.no_grad()
def predict_doc(
    doc: Doc, model, tokenizer, max_length: int, stride: int, device, batch_size: int
) -> list[tuple[int, int]]:
    """Predict merged character PII spans for one document."""
    enc, _ = encode_doc(
        doc, tokenizer, max_length=max_length, stride=stride, with_labels=False
    )
    n_chunks = len(enc["input_ids"])
    offsets_per_chunk = enc["offset_mapping"]
    seqids_per_chunk = enc["sequence_ids"]
    preds_per_chunk: list[list[int]] = []

    for start in range(0, n_chunks, batch_size):
        ids = enc["input_ids"][start : start + batch_size]
        att = enc["attention_mask"][start : start + batch_size]
        maxlen = max(len(x) for x in ids)
        pad_id = tokenizer.pad_token_id or 0
        input_ids = torch.tensor([x + [pad_id] * (maxlen - len(x)) for x in ids])
        attention = torch.tensor([x + [0] * (maxlen - len(x)) for x in att])
        logits = model(
            input_ids=input_ids.to(device),
            attention_mask=attention.to(device),
        ).logits
        argmax = logits.argmax(-1).cpu().tolist()
        for bi in range(len(ids)):
            preds_per_chunk.append(argmax[bi][: len(ids[bi])])

    return reconstruct_spans(
        doc.text, offsets_per_chunk, preds_per_chunk, seqids_per_chunk
    )


def _prf(tp: int, fp: int, fn: int) -> dict:
    """Precision / recall / F1 from raw counts (0.0 when undefined)."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def _char_stats(
    doc: Doc, pred_char: list[tuple[int, int]]
) -> tuple[int, int, int, int]:
    """(gold_pii_chars, pred_pii_chars, overlap_chars, total_chars) for one doc."""
    n = len(doc.text)
    gmask = bytearray(n)
    pmask = bytearray(n)
    for s, e, _ in doc.spans:
        for k in range(max(0, s), min(n, e)):
            gmask[k] = 1
    for s, e in pred_char:
        for k in range(max(0, s), min(n, e)):
            pmask[k] = 1
    gold = sum(gmask)
    pred = sum(pmask)
    overlap = sum(a & b for a, b in zip(gmask, pmask, strict=True))
    return gold, pred, overlap, n


def _print_summary(name: str, report, char: dict, token: dict) -> None:
    m = report.micro
    print(f"\n=== {name} ===")
    print(
        f"span   P={m.precision:.4f}  R={m.recall:.4f}  F1={m.f1:.4f}  "
        f"(tp={m.tp} fp={m.fp} fn={m.fn})"
    )
    print(
        f"token  P={token['precision']:.4f}  R={token['recall']:.4f}  "
        f"F1={token['f1']:.4f}"
    )
    print(
        f"char   P={char['precision']:.4f}  R={char['recall']:.4f}  "
        f"F1={char['f1']:.4f}  over_redaction={char['over_redaction_rate']:.4f}"
    )
    if report.by_family:
        print("by_family (span-level):")
        for fam, prf in sorted(report.by_family.items()):
            print(
                f"  {fam:<10} P={prf.precision:.3f} R={prf.recall:.3f} "
                f"F1={prf.f1:.3f} (n={prf.tp + prf.fn})"
            )


def _evaluate(docs: list[Doc], model, tokenizer, args, device, name: str) -> dict:
    match_results = []
    meta = {}
    tok_tp = tok_fp = tok_fn = 0
    c_gold = c_pred = c_overlap = c_total = 0
    for i, doc in enumerate(docs, 1):
        pred_char = predict_doc(
            doc, model, tokenizer, args.max_length, args.stride, device, args.batch_size
        )
        gold = [Span(s, e, label=lab) for (s, e, lab) in doc.spans]
        pred = [Span(s, e) for (s, e) in pred_char]
        match_results.append(match_lenient(gold, pred, doc.doc_id, args.iou_threshold))
        # Merge-unbiased coverage metrics (token- and char-level).
        tmr = match_token_level(doc.text, gold, pred, doc.doc_id)
        tok_tp += len(tmr.true_positives)
        tok_fp += len(tmr.false_positives)
        tok_fn += len(tmr.false_negatives)
        g, p, ov, n = _char_stats(doc, pred_char)
        c_gold += g
        c_pred += p
        c_overlap += ov
        c_total += n
        meta[doc.doc_id] = doc.metadata
        if i % 50 == 0:
            print(f"  [{name}] {i}/{len(docs)}")
    report = compute_metrics(match_results, meta)

    char = {
        "recall": c_overlap / c_gold if c_gold else 0.0,
        "precision": c_overlap / c_pred if c_pred else 0.0,
        # fraction of non-PII characters wrongly redacted (over-redaction rate)
        "over_redaction_rate": (
            (c_pred - c_overlap) / (c_total - c_gold) if (c_total - c_gold) else 0.0
        ),
        "gold_pii_chars": c_gold,
        "pred_pii_chars": c_pred,
        "overlap_chars": c_overlap,
        "total_chars": c_total,
    }
    char["f1"] = (
        2 * char["precision"] * char["recall"] / (char["precision"] + char["recall"])
        if (char["precision"] + char["recall"])
        else 0.0
    )
    token = _prf(tok_tp, tok_fp, tok_fn)

    _print_summary(name, report, char, token)
    out = report.to_dict()
    out["char_level"] = char
    out["token_level"] = token
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True, help="fine-tuned model dir")
    ap.add_argument("--eval-synthetic", default=None, help="dataset.jsonl to eval on")
    ap.add_argument(
        "--synthetic-holdout-only",
        action="store_true",
        help="evaluate only the stable held-out synthetic test split (test_frac=0.1)",
    )
    ap.add_argument("--eval-grascco", default=None, help="GRASCCO dir to eval on")
    ap.add_argument(
        "--grascco-kfold",
        type=int,
        default=0,
        help="restrict GRASCCO eval to the held-out fold of a k-fold split (0=all)",
    )
    ap.add_argument("--grascco-fold", type=int, default=0, help="held-out fold index")
    ap.add_argument("--grascco-kfold-seed", type=int, default=42, help="fold seed")
    ap.add_argument("--out", default=None, help="report.json path (default: model-dir)")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--iou-threshold", type=float, default=0.5)
    args = ap.parse_args()

    if not args.eval_synthetic and not args.eval_grascco:
        raise SystemExit("Provide --eval-synthetic and/or --eval-grascco.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.to(device).eval()

    results: dict = {"model_dir": args.model_dir}
    if args.eval_synthetic:
        docs = load_synthetic(args.eval_synthetic)
        if args.synthetic_holdout_only:
            _train, docs = split_train_test(docs, test_frac=0.1, seed=42)
        results["synthetic"] = _evaluate(
            docs, model, tokenizer, args, device, "synthetic"
        )
    if args.eval_grascco:
        docs = load_grascco(args.eval_grascco)
        name = "grascco"
        if args.grascco_kfold:
            _train, docs = kfold_split(
                docs,
                args.grascco_kfold,
                args.grascco_fold,
                seed=args.grascco_kfold_seed,
            )
            name = f"grascco_fold{args.grascco_fold}"
        results[name] = _evaluate(docs, model, tokenizer, args, device, name)

    out = args.out or os.path.join(args.model_dir, "report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[eval] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
