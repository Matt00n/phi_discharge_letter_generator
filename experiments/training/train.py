"""Fine-tune a German encoder as a binary PII token classifier.

Trains on synthetic SPARCLING-DE with sliding-window chunking and saves the
model + tokenizer to an output dir for later evaluation (see evaluate.py).

Everything needed to report the run is persisted next to the model:
  - `train_meta.json`   : args, provenance (git SHA, versions), device, timings
  - `training_log.json` : per-step loss + per-epoch validation metrics (curves)
  - checkpoints (best kept) as crash insurance on long/unattended runs

Example:
    python -m experiments.training.train \
        --train-data experiments/datasets/sparcling-de/dataset.jsonl \
        --model GerMedBERT/medbert-512 \
        --out-dir experiments/results/training/full_synthetic
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from experiments.training.data import (
    ID2LABEL,
    IGNORE_INDEX,
    LABEL2ID,
    PII_LABEL,
    encode_doc,
    filter_adversarial,
    kfold_split,
    load_grascco,
    load_synthetic,
    split_train_test,
    subsample,
)

DEFAULT_MODEL = "GerMedBERT/medbert-512"


class ChunkDataset(Dataset):
    """Flattens all docs' sliding-window chunks into training examples."""

    def __init__(self, docs, tokenizer, max_length: int, stride: int):
        self.examples: list[dict] = []
        for doc in docs:
            enc, labels = encode_doc(
                doc, tokenizer, max_length=max_length, stride=stride, with_labels=True
            )
            for ci in range(len(enc["input_ids"])):
                self.examples.append(
                    {
                        "input_ids": enc["input_ids"][ci],
                        "attention_mask": enc["attention_mask"][ci],
                        "labels": labels[ci],
                    }
                )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> dict:
        return self.examples[i]


def _token_metrics(eval_pred):
    """Wordpiece-token-level binary P/R/F1 for the validation curve.

    This is a fast, in-loop convergence signal (not the final span metric —
    that comes from evaluate.py). Special/pad tokens (IGNORE_INDEX) are dropped.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    mask = labels != IGNORE_INDEX
    p = preds[mask]
    lbl = labels[mask]
    tp = int(((p == PII_LABEL) & (lbl == PII_LABEL)).sum())
    fp = int(((p == PII_LABEL) & (lbl != PII_LABEL)).sum())
    fn = int(((p != PII_LABEL) & (lbl == PII_LABEL)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-data", default=None, help="synthetic dataset.jsonl")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="HF encoder id or path")
    ap.add_argument("--out-dir", required=True)
    # Real-data / cross-validation / fine-tuning knobs (experiment 3)
    ap.add_argument(
        "--train-grascco",
        default=None,
        help="GRASCCO dir: train on real folds instead of synthetic",
    )
    ap.add_argument(
        "--kfold", type=int, default=0, help="k for GRASCCO cross-val (0=off)"
    )
    ap.add_argument("--fold", type=int, default=0, help="held-out GRASCCO fold index")
    ap.add_argument("--kfold-seed", type=int, default=42, help="GRASCCO fold seed")
    ap.add_argument(
        "--init-from",
        default=None,
        help="load weights+tokenizer from this dir (fine-tune) instead of --model",
    )
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    # Ablation knobs
    ap.add_argument(
        "--drop-adversarial",
        action="store_true",
        help="exclude the adversarial-difficulty slice from training",
    )
    ap.add_argument(
        "--subsample",
        type=int,
        default=0,
        help="train on at most N docs (0 = all) for size-scaling",
    )
    ap.add_argument(
        "--test-frac",
        type=float,
        default=0.1,
        help="held-out synthetic fraction (kept identical across runs)",
    )
    ap.add_argument(
        "--val-frac",
        type=float,
        default=0.1,
        help="validation fraction carved from train for per-epoch curves",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.train_grascco:
        # Real-data training (exp. 3: real-only / synthetic+real fine-tune).
        docs = load_grascco(args.train_grascco)
        if args.kfold:
            docs, _test = kfold_split(docs, args.kfold, args.fold, seed=args.kfold_seed)
        else:
            docs, _test = split_train_test(docs, test_frac=args.test_frac, seed=42)
    else:
        if not args.train_data:
            raise SystemExit("Provide --train-data (synthetic) or --train-grascco.")
        docs = load_synthetic(args.train_data)
        # Reserve a stable held-out synthetic test set; train on the remainder.
        docs, _test = split_train_test(docs, test_frac=args.test_frac, seed=42)
        docs = filter_adversarial(docs, keep_adversarial=not args.drop_adversarial)
        if args.subsample:
            docs = subsample(docs, args.subsample, seed=args.seed)
    # Carve a validation slice from train (disjoint from the test hold-out) so
    # we can plot a real per-epoch validation curve without touching the test set.
    train_docs, val_docs = split_train_test(docs, test_frac=args.val_frac, seed=43)
    model_src = args.init_from or args.model
    print(
        f"[train] {len(train_docs)} train / {len(val_docs)} val docs "
        f"({len(_test)} held out) (model={model_src}"
        + (f", init_from={args.init_from}" if args.init_from else "")
        + (f", grascco fold {args.fold}/{args.kfold}" if args.kfold else "")
        + ")"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_src, use_fast=True)
    if not tokenizer.is_fast:
        raise SystemExit("A fast tokenizer is required for offset mapping.")
    model = AutoModelForTokenClassification.from_pretrained(
        model_src,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_ds = ChunkDataset(train_docs, tokenizer, args.max_length, args.stride)
    val_ds = ChunkDataset(val_docs, tokenizer, args.max_length, args.stride)
    has_val = len(val_ds) > 0
    print(f"[train] {len(train_ds)} train chunks / {len(val_ds)} val chunks")
    if not has_val:
        print("[train] empty validation set -> no per-epoch eval curve / best-model")

    use_cuda = torch.cuda.is_available()
    collator = DataCollatorForTokenClassification(tokenizer)
    # With a validation set we track a per-epoch curve and keep the best model;
    # without one (degenerate tiny subsample) we just checkpoint each epoch.
    targs = TrainingArguments(
        output_dir=os.path.join(args.out_dir, "_hf"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(args.batch_size, 16),
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=10,
        eval_strategy="epoch" if has_val else "no",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=has_val,
        metric_for_best_model="f1" if has_val else None,
        greater_is_better=True,
        seed=args.seed,
        fp16=use_cuda,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds if has_val else None,
        data_collator=collator,
        compute_metrics=_token_metrics if has_val else None,
    )

    t0 = time.time()
    trainer.train()
    train_seconds = time.time() - t0

    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)

    # Persist the loss/metric curve (per-step loss + per-epoch val metrics).
    with open(
        os.path.join(args.out_dir, "training_log.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(trainer.state.log_history, f, ensure_ascii=False, indent=2)

    # Persist provenance + timing so the run is fully reproducible / reportable.
    import transformers

    meta = {
        "args": vars(args),
        "train_source": "grascco" if args.train_grascco else "synthetic",
        "model_src": model_src,
        "git_sha": _git_sha(),
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "python": platform.python_version(),
        },
        "device": {
            "cuda": use_cuda,
            "gpu_name": torch.cuda.get_device_name(0) if use_cuda else None,
            "n_gpu": torch.cuda.device_count() if use_cuda else 0,
            "cpu_threads": torch.get_num_threads(),
        },
        "data": {
            "n_train_docs": len(train_docs),
            "n_val_docs": len(val_docs),
            "n_test_docs": len(_test),
            "n_train_chunks": len(train_ds),
            "n_val_chunks": len(val_ds),
        },
        "timing": {
            "train_seconds": train_seconds,
            "train_chunks_per_second": (
                len(train_ds) * args.epochs / train_seconds if train_seconds else None
            ),
        },
        "best_metric": trainer.state.best_metric,
    }
    with open(
        os.path.join(args.out_dir, "train_meta.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(
        f"[train] done in {train_seconds:.0f}s "
        f"(best val f1={trainer.state.best_metric}); saved to {args.out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
