"""Run the blinded Gemini clinical-realism judge on the German discharge-letter
corpora and write per-corpus + per-difficulty-tier reports.

Judged corpora (all genuine discharge letters/summaries). gernermed (~85-char
NER snippets) and i2b2 (~788-char nursing/progress-note fragments) are excluded
because they are not discharge letters, so the discharge-letter rubric does not
apply:

  * sparcling_de      — our full synthetic corpus (per-difficulty-tier breakdown)
  * sparcling_de_63    — the size-matched n=63 subsample (comparable to grascco)
  * grascco            — the real German reference corpus (n=63)
  * mimic              — real English discharge summaries (n=1000); cross-language
                         reference — compare with care vs the German corpora

Judgments are cached in results/gemini_realism_judge.sqlite3 keyed on document
text, so the 63 subsample docs reuse the full-corpus calls and re-runs are free.

Outputs (all under experiments/results/):
  * judge_<corpus>.json  — the full llm_judge_metrics report per corpus
  * judge_summary.json   — headline dimension means per corpus + the
                            per-difficulty-tier breakdown for sparcling_de(_63)

Usage:
    python experiments/run_judge.py
    python experiments/run_judge.py --sample-size 200   # override (default: full)
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from experiments.evaluation.config import DatasetConfig, LLMJudgeConfig
from experiments.evaluation.corpus import Corpus, load_corpus
from experiments.evaluation.llm_judge import (
    PROMPT_VERSION,
    SCORE_FIELDS,
    SYSTEM_INSTRUCTION,
    GeminiClinicalJudge,
    JudgmentRecord,
    _sample_document_indices,
    aggregate_judgments,
)
from experiments.evaluation.statistics import numeric_summary

LOGGER = logging.getLogger("run_judge")

# Force UTF-8 on the Windows console so German output does not crash on print.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
CACHE_PATH = RESULTS / "gemini_realism_judge.sqlite3"

SEED = 1729
BOOTSTRAP = 2000
CONFIDENCE = 0.95
PASSING = 4
TIER_ORDER = ["easy", "medium", "hard", "adversarial"]

# Match the generation pipeline's Vertex AI / Google AI setup:
DEFAULT_PROJECT = os.environ.get("GCP_PROJECT_ID")
DEFAULT_LOCATION = os.environ.get("GCP_LOCATION", "eu")

# (name, path relative to experiments/, has difficulty metadata, language)
# gernermed and i2b2 are deliberately excluded: both are short note/snippet
# corpora (medians ~85 and ~788 chars), not discharge letters, so the
# discharge-letter rubric does not apply. mimic (real English discharge
# summaries) is included as a real-corpus reference — note the cross-language
# caveat vs the German corpora.
CORPORA = [
    ("sparcling_de", "datasets/sparcling-de/dataset.jsonl", True, "de"),
    ("sparcling_de_63", "datasets/sparcling-de/subsample63.jsonl", True, "de"),
    ("grascco", "datasets/_adapted/grascco.jsonl", False, "de"),
    ("mimic", "datasets/_adapted/mimic.jsonl", False, "en"),
]


def _dataset_config(name: str, rel_path: str, language: str = "de") -> DatasetConfig:
    # The judge only needs text; disable span/ICD/spaCy work for a fast, robust
    # load across corpora with heterogeneous schemas.
    return DatasetConfig(
        name=name,
        path=ROOT / rel_path,
        format="jsonl",
        text_field="text",
        id_field="doc_id",
        phi_spans_field=None,
        validate_phi_offsets=False,
        extract_icd_from_text=False,
        extract_ops_from_text=False,
        language=language,
        spacy_model=None,
    )


def _difficulty_by_id(rel_path: str) -> dict[str, str]:
    tiers: dict[str, str] = {}
    for line in (ROOT / rel_path).open(encoding="utf-8"):
        record = json.loads(line)
        tiers[record["doc_id"]] = record.get("metadata", {}).get("difficulty", "?")
    return tiers


def _document_level(result: dict) -> list[dict]:
    """Collapse per-repetition judgments to one row per document.

    Score per dimension is the mean over repetitions; the binary flags are ORed
    (matches the aggregate's "in any repetition" semantics).
    """
    rows: list[dict] = []
    for doc in result["document_results"]:
        reps = doc["repetitions"]
        scores = {
            field: sum(r["scores"][field] for r in reps) / len(reps)
            for field in SCORE_FIELDS
        }
        rows.append(
            {
                "document_index": doc["document_index"],
                "scores": scores,
                "appears_synthetic": any(r["appears_synthetic"] for r in reps),
                "has_major_error": any(r["has_major_error"] for r in reps),
                "failure_tags": sorted(
                    {tag for r in reps for tag in r["failure_tags"]}
                ),
            }
        )
    return rows


def _per_tier_report(
    result: dict, corpus: Corpus, tier_by_id: dict[str, str]
) -> dict[str, object]:
    """Aggregate document-level judgments by difficulty tier."""
    rows = _document_level(result)
    by_tier: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        doc_id = corpus.documents[row["document_index"]].doc_id
        by_tier[tier_by_id.get(doc_id, "?")].append(row)

    ordered = [t for t in TIER_ORDER if t in by_tier] + [
        t for t in by_tier if t not in TIER_ORDER
    ]
    report: dict[str, object] = {}
    for tier_index, tier in enumerate(ordered):
        tier_rows = by_tier[tier]
        n = len(tier_rows)
        dimensions: dict[str, object] = {}
        for dim_index, field in enumerate(SCORE_FIELDS):
            values = [row["scores"][field] for row in tier_rows]
            summary = numeric_summary(
                values,
                bootstrap_samples=BOOTSTRAP,
                confidence_level=CONFIDENCE,
                seed=SEED + tier_index * 100 + dim_index,
            )
            dimensions[field] = {
                "mean": summary["mean"],
                "mean_ci": summary.get("mean_confidence_interval"),
                "pass_rate": sum(v >= PASSING for v in values) / n,
            }
        report[tier] = {
            "n": n,
            "dimensions": dimensions,
            "documents_appearing_synthetic": sum(
                row["appears_synthetic"] for row in tier_rows
            ),
            "documents_with_major_error": sum(
                row["has_major_error"] for row in tier_rows
            ),
            "failure_tag_document_counts": dict(
                sorted(
                    Counter(
                        tag for row in tier_rows for tag in row["failure_tags"]
                    ).items()
                )
            ),
        }
    return report


def _headline(result: dict) -> dict[str, object]:
    dims = result["dimensions"]
    return {
        "corpus_documents": result["method"]["corpus_documents"],
        "sample_size": result["method"]["sample_size"],
        "documents_judged": result["method"].get("documents_judged"),
        "documents_failed": result["method"].get("documents_failed"),
        "dimensions": {
            field: {
                "mean": dims[field]["document_mean_score"]["mean"],
                "mean_ci": dims[field]["document_mean_score"].get(
                    "mean_confidence_interval"
                ),
                "pass_rate": dims[field]["document_pass_rate"],
            }
            for field in SCORE_FIELDS
        },
        "binary_flags": result["binary_flags"],
        "failure_tags": result["failure_tags"]["document_any_repetition_counts"],
        "requests": result["requests"],
    }


def _judge_corpus_resilient(
    corpus: Corpus,
    cfg: LLMJudgeConfig,
    *,
    bootstrap_samples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, object]:
    """Like ``llm_judge_metrics`` but tolerant of individual document failures.

    The shipped ``llm_judge_metrics`` is fail-fast: one document that returns no
    parseable response (e.g. the judge's thinking budget consumes the whole
    output allowance on a hard case) aborts the entire batch. Over ~1000 docs
    that is too brittle, so here a failed document is logged, recorded, and
    skipped; every successful (and cached) judgment is retained. Config is left
    byte-identical to the standard judge so cached judgments are reused.
    """
    indices = _sample_document_indices(len(corpus.documents), cfg.sample_size, seed)
    judge = GeminiClinicalJudge(cfg)
    tasks = [
        (document_index, repetition)
        for document_index in indices
        for repetition in range(cfg.repetitions)
    ]
    records: list[JudgmentRecord] = []
    failed: list[dict[str, object]] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=cfg.max_concurrent_requests) as executor:
        future_to_task = {
            executor.submit(
                judge.judge, corpus.documents[document_index].text, repetition
            ): (document_index, repetition)
            for document_index, repetition in tasks
        }
        for future in as_completed(future_to_task):
            document_index, repetition = future_to_task[future]
            completed += 1
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 — skip and record, never abort
                failed.append(
                    {
                        "doc_id": corpus.documents[document_index].doc_id,
                        "document_index": document_index,
                        "repetition": repetition,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                LOGGER.warning(
                    "judge FAILED corpus=%s doc=%s (%d/%d): %s",
                    corpus.name,
                    corpus.documents[document_index].doc_id,
                    completed,
                    len(tasks),
                    exc,
                )
                continue
            records.append(
                JudgmentRecord(
                    document_index=document_index,
                    repetition=repetition,
                    result=result,
                )
            )
            if completed % 50 == 0 or completed == len(tasks):
                LOGGER.info(
                    "judge corpus=%s completed=%d/%d failed=%d",
                    corpus.name,
                    completed,
                    len(tasks),
                    len(failed),
                )
    if not records:
        raise RuntimeError(f"every judge call failed for corpus {corpus.name}")
    records.sort(key=lambda record: (record.document_index, record.repetition))
    aggregate = aggregate_judgments(
        records,
        cfg,
        bootstrap_samples=bootstrap_samples,
        confidence_level=confidence_level,
        seed=seed,
    )
    return {
        "status": "computed",
        "method": {
            "provider": "Google",
            "backend": cfg.backend,
            "requested_model": cfg.model,
            "prompt_version": PROMPT_VERSION,
            "prompt_sha256": hashlib.sha256(SYSTEM_INSTRUCTION.encode()).hexdigest(),
            "thinking_level": cfg.thinking_level,
            "rubric_scale": "integer 1--5 with fixed common anchors",
            "sample_size": len(indices),
            "corpus_documents": len(corpus.documents),
            "documents_judged": len({r.document_index for r in records}),
            "documents_failed": len({f["document_index"] for f in failed}),
            "repetitions_per_document": cfg.repetitions,
            "passing_score": cfg.passing_score,
            "blinded_to_dataset_identity": True,
        },
        **aggregate,
        "failed_documents": failed,
        "interpretation": (
            "LLM ratings are model-based measurements, not ground truth. Documents "
            "that returned no parseable judgment after retries were skipped and are "
            "listed under 'failed_documents'."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Gemini clinical-realism judge runner.")
    ap.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Docs to judge per corpus (default: full corpus).",
    )
    ap.add_argument("--model", default=None, help="Override judge model.")
    ap.add_argument("--project", default=DEFAULT_PROJECT, help="Vertex AI project.")
    ap.add_argument("--location", default=DEFAULT_LOCATION, help="Vertex AI location.")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    RESULTS.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {"corpora": {}, "per_difficulty": {}}

    for name, rel_path, has_difficulty, language in CORPORA:
        corpus = load_corpus(_dataset_config(name, rel_path, language))
        n_docs = len(corpus.documents)
        sample = min(args.sample_size, n_docs) if args.sample_size else n_docs
        print("=" * 70)
        print(f"  JUDGE  {name}   documents={n_docs}  sample={sample}")
        print("=" * 70)

        judge_kwargs = {
            "enabled": True,
            "allow_external_api_calls": True,
            "backend": "vertex_ai",
            "project": args.project,
            "location": args.location,
            "thinking_level": "medium",
            "sample_size": sample,
            "repetitions": 1,
            "max_concurrent_requests": 12,
            "include_rationales_in_report": True,
            "cache_path": CACHE_PATH,
        }
        if args.model:
            judge_kwargs["model"] = args.model
        cfg = LLMJudgeConfig(**judge_kwargs)

        result = _judge_corpus_resilient(
            corpus,
            cfg,
            bootstrap_samples=BOOTSTRAP,
            confidence_level=CONFIDENCE,
            seed=SEED,
        )

        out = RESULTS / f"judge_{name}.json"
        out.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary["corpora"][name] = _headline(result)

        req = result["requests"]
        n_failed = result["method"]["documents_failed"]
        print(
            f"  api_calls={req['api_calls_this_run']} "
            f"cache_hits={req['cache_hits_this_run']} failed={n_failed}  ->  {out.name}"
        )
        dims = result["dimensions"]
        for field in SCORE_FIELDS:
            m = dims[field]["document_mean_score"]["mean"]
            pr = dims[field]["document_pass_rate"]
            print(f"    {field:<34} mean={m:.2f}  pass(>= {PASSING})={pr:.0%}")

        if has_difficulty:
            tiers = _per_tier_report(result, corpus, _difficulty_by_id(rel_path))
            summary["per_difficulty"][name] = tiers
            print("  --- per difficulty tier (overall_realism / doc_quality) ---")
            for tier, rep in tiers.items():
                ov = rep["dimensions"]["overall_realism"]["mean"]
                dq = rep["dimensions"]["documentation_quality"]["mean"]
                print(
                    f"    {tier:<12} n={rep['n']:<4} overall={ov:.2f} "
                    f"doc_qual={dq:.2f} "
                    f"synthetic={rep['documents_appearing_synthetic']}/{rep['n']} "
                    f"major_err={rep['documents_with_major_error']}/{rep['n']}"
                )

    summary_path = RESULTS / "judge_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("=" * 70)
    print(f"Wrote {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
