"""SPARCLING CLI: typed-placeholder synthesis pipeline.

Generates the benchmark, runs QC, assigns splits, and writes:
  - dataset.jsonl       gold records (text + spans + negatives + metadata + qc)
  - review_queue.jsonl  flagged documents for human review
  - diversity.json      diversity audit report

Use ``--dry-run`` to run the full pipeline offline (no API) with a deterministic
skeleton author - useful for testing and inspection.
"""

from __future__ import annotations

import argparse
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict

import yaml

from generator.clinical_seeds import ClinicalSeedsManager
from generator.config_schema import PipelineConfig
from generator.diversity import audit_by_splits
from generator.dry_run import template_skeleton
from generator.exporter import Exporter
from generator.pipeline import Pipeline, SkeletonDegenerationError
from generator.render import RenderError, SpanIntegrityError
from generator.splits import assign_splits


def _classify_drop_stage(exc: Exception) -> str:
    """Map a generation exception to the pipeline stage that dropped the doc.

    Distinguishes render/skeleton/span failures from LLM-authoring API failures
    so the provenance report can attribute the yield gap. Red-team call failures
    do NOT reach here: they are captured as a recorded QC exclusion instead of a
    silent drop (see Pipeline._red_team).
    """
    if isinstance(exc, SkeletonDegenerationError):
        return "skeleton_quality"
    if isinstance(exc, RenderError):
        return "render"
    if isinstance(exc, SpanIntegrityError):
        return "span_validation"
    if "structured generation" in str(exc) or type(exc).__name__ in (
        "ClientError",
        "ServerError",
    ):
        return "llm_authoring"
    return "unknown"


DEFAULT_CONFIG = {
    "num_letters": 10,
    "output_dir": "./outputs/default",
    "cache_db_path": "./gemini_cache.db",
    "gemini_model": "gemini-3.7-flash",
    "api_retry_attempts": 5,
    "api_retry_delay_seconds": 2,
    "red_team": True,
    "auto": True,
    "base_seed": 1000,
    "parallel_workers": 8,
}


def load_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        print(f"Config not found at {path}; using defaults.")
        return dict(DEFAULT_CONFIG)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    merged = dict(DEFAULT_CONFIG)
    merged.update(cfg)
    return merged


def build_pipeline(config: Dict[str, Any], dry_run: bool) -> Pipeline:
    seeds = ClinicalSeedsManager(data_dir="data")
    # Tunable hyperparameters (sampling / generation / red_team) live under the
    # optional ``pipeline_config`` key; absent keys fall back to the historical
    # defaults baked into PipelineConfig.
    pipeline_config = PipelineConfig.from_dict(config.get("pipeline_config"))
    if dry_run:
        return Pipeline(
            seeds, skeleton_provider=template_skeleton, config=pipeline_config
        )
    from generator.llm_client import GeminiCachedClient

    llm = GeminiCachedClient(
        db_path=config["cache_db_path"],
        model_name=config["gemini_model"],
        retry_attempts=config["api_retry_attempts"],
        retry_delay_seconds=config["api_retry_delay_seconds"],
    )
    return Pipeline(
        seeds,
        llm_client=llm,
        red_team=config.get("red_team", False),
        config=pipeline_config,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="SPARCLING generator")
    ap.add_argument("--num-letters", type=int)
    ap.add_argument("--output-dir", type=str)
    ap.add_argument("--config", type=str, default="generator/config.yaml")
    ap.add_argument("--dry-run", action="store_true", help="Run offline without API.")
    ap.add_argument(
        "--style",
        type=str,
        help="Force style (Standard-Klinik, Akademisch-Ausführlich, Telegrammstil-Kompakt).",
    )
    ap.add_argument(
        "--difficulty",
        type=str,
        help="Force difficulty (easy, medium, hard, adversarial).",
    )
    ap.add_argument(
        "--length",
        type=str,
        help="Force length (kurz, mittel, ausführlich, sehr_lang).",
    )
    args = ap.parse_args()

    config = load_config(args.config)
    n = args.num_letters or config["num_letters"]
    output_dir = args.output_dir or config["output_dir"]
    base_seed = config.get("base_seed", 1000)
    workers = config.get("parallel_workers", 1)

    pipeline = build_pipeline(config, args.dry_run)
    exporter = Exporter(output_dir=output_dir)

    mode_label = "DRY-RUN" if args.dry_run else "LLM"
    extras = []
    if args.style:
        extras.append(f"style={args.style}")
    if args.difficulty:
        extras.append(f"difficulty={args.difficulty}")
    if args.length:
        extras.append(f"length={args.length}")
    extra_str = f"  {' '.join(extras)}" if extras else ""

    print("=" * 64)
    print("  SPARCLING  (typed-placeholder synthesis)")
    print(
        f"  documents={n}  mode={mode_label}  workers={workers}  out={output_dir}{extra_str}"
    )
    print("=" * 64)

    records: list[dict] = []
    failures: list[dict] = []
    flagged = 0
    completed = 0
    _print_lock = threading.Lock()

    def _generate_one(i: int) -> tuple[int, str, Any]:
        doc_id = f"doc_{i:05d}"
        seed = base_seed + i
        rec = pipeline.generate_record(
            doc_id,
            seed,
            style=args.style,
            difficulty=args.difficulty,
            length=args.length,
        )
        return i, doc_id, rec

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_generate_one, i): i for i in range(n)}
        for future in as_completed(futures):
            completed += 1
            try:
                i, doc_id, rec = future.result()
            except Exception as e:  # noqa: BLE001
                i = futures[future]
                doc_id = f"doc_{i:05d}"
                stage = _classify_drop_stage(e)
                failures.append(
                    {
                        "doc_id": doc_id,
                        "seed": base_seed + i,
                        "stage": stage,
                        "error_type": type(e).__name__,
                        "error": str(e),
                    }
                )
                with _print_lock:
                    print(f"[{completed}/{n}] {doc_id}  ERROR [{stage}]: {e}")
                continue
            records.append(rec.as_dict())
            m = rec.metadata
            flag = "FLAG" if rec.qc["flagged"] else "ok"
            if rec.qc["flagged"]:
                flagged += 1
            with _print_lock:
                print(
                    f"[{completed}/{n}] {doc_id}  {m['difficulty']:<6} {m.get('fachabteilung', ''):<16} "
                    f"spans={m['n_spans']:<3} neg={m['n_negatives']:<2} {flag}"
                )

    # Sort by doc_id so output is deterministic regardless of completion order.
    records.sort(key=lambda r: r["doc_id"])
    failures.sort(key=lambda f: f["doc_id"])

    # Persist the drop list (Stages 1-3 hard failures) — always, even if empty,
    # so the yield chain is complete and reproducible.
    failures_path = exporter.export_failures(failures)

    if not records:
        print(f"No records generated. ({len(failures)} dropped; see {failures_path})")
        return

    # Persist every exclusion decision (deterministic + red-team) BEFORE the
    # auto-filter drops flagged docs, so the review queue is always complete.
    review_path = exporter.export_review_queue(records)

    # Full provenance / yield report, computed on the COMPLETE generated set
    # (before auto-filtering): yield chain, generation drops by stage, QC
    # exclusions by mechanism + deterministic/red-team overlap, findings by label.
    auto_mode = config.get("auto", False)
    provenance_path, provenance = exporter.export_provenance(
        records, failures, auto_mode=auto_mode
    )

    # Auto mode: exclude flagged documents from the dataset
    if auto_mode:
        excluded = [r for r in records if r["qc"]["flagged"]]
        records = [r for r in records if not r["qc"]["flagged"]]
        if excluded:
            print(
                f"  [auto] Excluded {len(excluded)} flagged doc(s): "
                + ", ".join(r["doc_id"] for r in excluded)
            )

    assign_splits(records)

    run_provenance = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_documents": len(records),
        "base_seed": base_seed,
        "mode": "dry-run" if args.dry_run else "llm",
        "model": config.get("gemini_model") if not args.dry_run else None,
        "red_team": config.get("red_team", False),
        "auto": auto_mode,
    }

    dataset_path = exporter.write_jsonl(records, "dataset.jsonl")
    report = audit_by_splits(records)
    report["provenance"] = run_provenance
    report_path = exporter.write_json(report, "diversity.json")

    y = provenance["yield"]
    print("-" * 64)
    print(f"dataset:       {dataset_path}  ({len(records)} docs)")
    print(f"review queue:  {review_path}  ({flagged} flagged)")
    print(f"failures:      {failures_path}  ({len(failures)} dropped)")
    print(f"provenance:    {provenance_path}")
    print(
        f"yield: attempted={y['attempted']} dropped={y['dropped_generation_errors']} "
        f"excluded={y['excluded']} kept={y['kept']} "
        f"(overall {y['rates']['overall_yield']:.1%})"
    )
    ov = provenance["qc_exclusions"]["deterministic_vs_redteam_overlap"]
    print(
        f"QC overlap: det-only={ov['deterministic_only']} "
        f"redteam-only={ov['redteam_only']} both={ov['both']}"
    )
    print(f"diversity:     {report_path}")
    overall = report["overall"]
    print(
        f"unique surnames: {overall['unique_surnames']}  "
        f"surname entropy: {overall['surname_entropy_bits']} bits"
    )
    print(f"tiers: {overall['tier_distribution']}")
    print(
        f"aggregate diversity score: {overall['aggregate']['aggregate_diversity_score']}"
    )
    print("=" * 64)


if __name__ == "__main__":
    main()
