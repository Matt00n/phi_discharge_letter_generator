"""Benchmark orchestration: run an anonymizer, then score it.

Two stages, both resumable via the anonymizer's response cache:

1. **Predict** — run the anonymizer over every document and write full
   PHI-replaced text to ``deid_texts.jsonl``.
2. **Evaluate** — hand that file to the :mod:`experiments.benchmark.scoring` package in
   text mode, which aligns the redacted text back to character offsets and
   scores it against the gold spans. The JSON report is written to
   ``report.json``.
"""

from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from experiments.benchmark.scoring.evaluate import run_evaluation

from experiments.benchmark.anonymizers import Anonymizer, build_anonymizer
from experiments.benchmark.config import BenchmarkConfig

logger = logging.getLogger(__name__)


def load_dataset(path: Path) -> list[dict]:
    """Load a gold-standard JSONL dataset (doc_id/text/spans per line)."""
    docs: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def predict_dataset(
    anonymizer: Anonymizer,
    docs: list[dict],
    output_path: Path,
    *,
    workers: int = 1,
) -> Path:
    """Run ``anonymizer`` over ``docs`` and write de-identified text as JSONL.

    Each output line is ``{"doc_id": ..., "text": <PHI-replaced text>}``. On a
    per-document failure the original text is written unchanged so evaluation
    still scores the document (as a full set of false negatives) rather than
    silently dropping it.
    """
    n = len(docs)
    completed = 0
    print_lock = threading.Lock()
    predictions: list[tuple[int, dict]] = []  # (original_index, record)

    def _process_one(idx: int, doc: dict) -> tuple[int, dict]:
        deid = anonymizer.anonymize_text(doc["text"])
        return idx, {"doc_id": doc["doc_id"], "text": deid}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_process_one, i, doc): i for i, doc in enumerate(docs)
        }
        for future in as_completed(futures):
            completed += 1
            i = futures[future]
            doc_id = docs[i]["doc_id"]
            try:
                idx, pred = future.result()
                predictions.append((idx, pred))
                n_markers = pred["text"].count("[REDACTED]")
                with print_lock:
                    print(
                        f"  [{completed}/{n}] {doc_id} "
                        f"({len(docs[i]['text'])} chars)... {n_markers} redactions"
                    )
            except Exception as exc:  # noqa: BLE001 — keep the batch going
                with print_lock:
                    print(f"  [{completed}/{n}] {doc_id} ERROR: {exc}")
                predictions.append((i, {"doc_id": doc_id, "text": docs[i]["text"]}))

    predictions.sort(key=lambda item: item[0])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out:
        for _, pred in predictions:
            out.write(json.dumps(pred, ensure_ascii=False) + "\n")
    return output_path


def run_benchmark(
    config: BenchmarkConfig,
    *,
    output_dir: Path | None = None,
    limit: int | None = None,
) -> dict:
    """Run the full predict + evaluate benchmark for one anonymizer.

    Returns the evaluation report dict. Artifacts are written under
    ``<output_dir>/<anonymizer.name>/``.
    """
    docs = load_dataset(config.dataset_path)
    if not docs:
        raise ValueError(f"No documents loaded from {config.dataset_path}")

    anonymizer = build_anonymizer(config.anonymizer)

    base_dir = (output_dir or config.output_dir).resolve()
    run_dir = base_dir / anonymizer.name
    run_dir.mkdir(parents=True, exist_ok=True)
    pred_path = run_dir / "deid_texts.jsonl"
    report_path = run_dir / "report.json"

    # When limiting to a subset, score against exactly the documents predicted —
    # otherwise the full gold file's un-predicted docs would count as all-misses.
    effective_limit = limit if limit is not None else config.limit
    gold_eval_path = config.dataset_path
    if effective_limit is not None:
        docs = docs[:effective_limit]
        gold_eval_path = run_dir / "gold_subset.jsonl"
        with open(gold_eval_path, "w", encoding="utf-8") as subset:
            for doc in docs:
                subset.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(
        f"Benchmarking '{anonymizer.name}' on {len(docs)} documents "
        f"(workers={config.anonymizer.workers})"
    )
    logger.info("Predicting de-identified text -> %s", pred_path)
    predict_dataset(anonymizer, docs, pred_path, workers=config.anonymizer.workers)

    logger.info("Scoring predictions against gold spans (text mode)")
    report = run_evaluation(
        gold_path=gold_eval_path,
        pred_path=pred_path,
        mode="text",
        matching=config.evaluation.matching,
        output_path=report_path,
        verbose=config.evaluation.verbose,
        iou_threshold=config.evaluation.iou_threshold,
    )
    return report
