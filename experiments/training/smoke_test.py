"""Fast, model-light smoke test for the training harness.

Validates the parts that are easy to get subtly wrong (char<->token offset
projection, sliding-window chunk reconstruction, scorer wiring) *without*
training anything. The one network dependency is the tokenizer (a few MB);
pass --tokenizer to override GerMedBERT/medbert-512.

Run:
    python -m experiments.training.smoke_test
"""

from __future__ import annotations

import argparse

from experiments.benchmark.scoring.matching import Span, match_lenient
from experiments.benchmark.scoring.metrics import compute_metrics
from experiments.training.data import (
    PII_LABEL,
    encode_doc,
    filter_adversarial,
    load_grascco,
    load_synthetic,
    reconstruct_spans,
    split_train_test,
    subsample,
)

SYN = "experiments/datasets/sparcling-de/dataset.jsonl"
GRASCCO = "experiments/datasets/grascco_phi_lohr_2024"


def _check(cond: bool, msg: str) -> None:
    print(f"  [{'ok' if cond else 'FAIL'}] {msg}")
    if not cond:
        raise SystemExit(f"smoke test failed: {msg}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokenizer", default="GerMedBERT/medbert-512")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--n", type=int, default=40, help="docs for the roundtrip check")
    args = ap.parse_args()

    print("== 1. corpus loading ==")
    syn = load_synthetic(SYN)
    _check(len(syn) > 900, f"loaded {len(syn)} synthetic docs")
    _check(all(d.text for d in syn), "every synthetic doc has text")
    _check(any(d.spans for d in syn), "synthetic docs carry gold spans")
    diffs = {d.metadata.get("difficulty") for d in syn}
    _check("adversarial" in diffs, f"difficulty tags present: {sorted(diffs)}")

    gra = load_grascco(GRASCCO)
    _check(len(gra) > 50, f"loaded {len(gra)} GRASCCO docs")
    _check(all(d.spans for d in gra), "every GRASCCO doc has gold spans")

    print("== 2. split / filter / subsample determinism ==")
    tr1, te1 = split_train_test(syn, test_frac=0.1, seed=42)
    tr2, te2 = split_train_test(syn, test_frac=0.1, seed=42)
    ids1 = {d.doc_id for d in te1}
    _check(ids1 == {d.doc_id for d in te2}, "held-out split is deterministic")
    _check(not (ids1 & {d.doc_id for d in tr1}), "no train/test leakage")
    _check(0.06 < len(te1) / len(syn) < 0.14, f"holdout frac ~10% ({len(te1)})")
    no_adv = filter_adversarial(syn, keep_adversarial=False)
    _check(
        all(d.metadata.get("difficulty") != "adversarial" for d in no_adv),
        f"--drop-adversarial removes {len(syn) - len(no_adv)} adversarial docs",
    )
    _check(len(subsample(syn, 100, seed=42)) == 100, "subsample returns N docs")

    print(f"== 3. chunk->reconstruct roundtrip (tokenizer={args.tokenizer}) ==")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    _check(tok.is_fast, "fast tokenizer (offset mapping available)")

    # Feed GOLD token labels back through reconstruct_spans as if a perfect model
    # emitted them; recovered char spans must match gold at lenient IoU>=0.5.
    sample = subsample(syn, args.n, seed=7)
    multichunk = 0
    match_results = []
    meta = {}
    for doc in sample:
        enc, labels = encode_doc(
            doc, tok, max_length=args.max_length, stride=args.stride, with_labels=True
        )
        n_chunks = len(enc["input_ids"])
        multichunk += n_chunks > 1
        offsets = enc["offset_mapping"]
        seqids = enc["sequence_ids"]
        # gold labels already use PII_LABEL/O_LABEL/IGNORE; treat IGNORE as O here
        preds = [[p if p == PII_LABEL else 0 for p in lab] for lab in labels]
        rec = reconstruct_spans(doc.text, offsets, preds, seqids)
        gold = [Span(s, e, label=lab) for (s, e, lab) in doc.spans]
        pred = [Span(s, e) for (s, e) in rec]
        match_results.append(match_lenient(gold, pred, doc.doc_id, 0.5))
        meta[doc.doc_id] = doc.metadata

    report = compute_metrics(match_results, meta)
    f1 = report.micro.f1
    _check(multichunk > 0, f"{multichunk}/{len(sample)} docs needed >1 chunk")
    # Ceiling < 1.0 is expected, not a bug: under binary IO tagging, distinct
    # entities separated only by a space (e.g. German 'ZIP CITY' -> '07743 Jena')
    # are indistinguishable from one entity and merge into a single span, costing
    # ~1 recall each. 0.96+ from perfect gold labels is the reconstruction ceiling.
    _check(
        f1 > 0.95,
        f"gold-relabel roundtrip micro-F1={f1:.4f} "
        f"(tp={report.micro.tp} fp={report.micro.fp} fn={report.micro.fn})",
    )

    print("\nALL SMOKE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
