"""Interactive review of flagged documents.

Presents each flagged document to the human reviewer with context about
what was flagged, and allows accept/drop decisions. Supports resume.

Usage:
    python -m scripts.review_flagged outputs/easy30/
    python -m scripts.review_flagged outputs/easy30/ --resume
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def load_dataset(dataset_path: Path) -> list[dict]:
    records = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_review_log(log_path: Path) -> dict[str, dict]:
    """Load existing review decisions keyed by doc_id."""
    if not log_path.exists():
        return {}
    decisions = {}
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                decisions[entry["doc_id"]] = entry
    return decisions


def save_decision(log_path: Path, decision: dict) -> None:
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(decision, ensure_ascii=False) + "\n")


def display_flagged_doc(rec: dict) -> None:
    """Display a flagged document summary for review."""
    meta = rec["metadata"]
    qc = rec["qc"]
    text = rec["text"]

    print("\n" + "=" * 70)
    print(f"  DOC: {rec['doc_id']}")
    print(
        f"  Difficulty: {meta['difficulty']}  Style: {meta['style']}  "
        f"Specialty: {meta.get('fachabteilung', '?')}"
    )
    print(
        f"  Spans: {meta.get('n_spans', '?')}  "
        f"Negatives: {meta.get('n_negatives', '?')}  "
        f"Length: {len(text)} chars"
    )
    print("=" * 70)

    # --- Flag reasons ---
    print("\n  FLAG REASONS:")
    for reason in qc["reasons"]:
        print(f"    - {reason}")

    # --- PII leaks ---
    if qc.get("leaks"):
        print("\n  PII LEAKS:")
        for leak in qc["leaks"]:
            surface = leak.get("surface", "?")
            entity_id = leak.get("entity_id", "?")
            start = leak.get("start", 0)
            end = leak.get("end", 0)
            ctx_start = max(0, start - 50)
            ctx_end = min(len(text), end + 50)
            context = text[ctx_start:ctx_end]
            # Highlight the leak in context
            rel_start = start - ctx_start
            rel_end = end - ctx_start
            highlighted = (
                context[:rel_start]
                + ">>>"
                + context[rel_start:rel_end]
                + "<<<"
                + context[rel_end:]
            )
            print(f"    [{entity_id}] '{surface}'")
            print(f"      ...{highlighted}...")

    # --- Missing persons ---
    if qc.get("missing_planned_persons"):
        print(
            f"\n  MISSING PLANNED PERSONS: {', '.join(qc['missing_planned_persons'])}"
        )

    # --- Negative integrity issues ---
    if qc.get("negative_findings"):
        print("\n  NEGATIVE INTEGRITY ISSUES:")
        for nf in qc["negative_findings"]:
            print(
                f"    [{nf.get('entity_id')}] '{nf.get('surface')}' — {nf.get('reason')}"
            )

    # --- Red-team findings ---
    red_team = qc.get("red_team")
    if red_team and red_team.get("review_items"):
        print("\n  RED-TEAM FINDINGS:")
        for item in red_team["review_items"]:
            if isinstance(item, dict):
                print(f"    - {item.get('issue', item)}")
                if item.get("context"):
                    print(f"      context: {item['context'][:120]}")
            else:
                print(f"    - {item}")

    # --- Text preview with inline annotations ---
    print("\n  TEXT PREVIEW (first 800 chars, spans marked with [LABEL]):")
    preview_end = min(len(text), 800)
    # Build annotated preview
    spans_in_range = sorted(
        [s for s in rec["spans"] if s["start"] < preview_end],
        key=lambda s: s["start"],
    )
    annotated = []
    pos = 0
    for span in spans_in_range:
        s, e = span["start"], min(span["end"], preview_end)
        if s > pos:
            annotated.append(text[pos:s])
        annotated.append(f"[{span['label']}:{text[s:e]}]")
        pos = e
    if pos < preview_end:
        annotated.append(text[pos:preview_end])
    preview = "".join(annotated)
    # Indent
    for line in preview.split("\n"):
        print(f"    {line}")
    if len(text) > 800:
        print(f"    ... ({len(text) - 800} more chars)")

    print()


def prompt_decision() -> str:
    """Prompt the user for a decision. Returns 'accept', 'drop', 'skip', or 'quit'."""
    while True:
        choice = (
            input("  Decision: [a]ccept  [d]rop  [s]kip  [q]uit → ").strip().lower()
        )
        if choice in ("a", "accept"):
            return "accept"
        elif choice in ("d", "drop"):
            return "drop"
        elif choice in ("s", "skip"):
            return "skip"
        elif choice in ("q", "quit"):
            return "quit"
        else:
            print("    Invalid choice. Use a/d/s/q.")


def apply_decisions(
    dataset_path: Path, records: list[dict], log: dict[str, dict]
) -> None:
    """Rewrite dataset.jsonl excluding dropped documents."""
    dropped = {doc_id for doc_id, entry in log.items() if entry["decision"] == "drop"}
    if not dropped:
        print("\n  No documents to drop.")
        return

    kept = [r for r in records if r["doc_id"] not in dropped]
    # Write updated dataset
    with open(dataset_path, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n  Dataset updated: {len(kept)} kept, {len(dropped)} dropped.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Review flagged documents interactively")
    ap.add_argument("output_dir", type=str, help="Path to the output directory")
    ap.add_argument("--resume", action="store_true", help="Skip already-reviewed docs")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    dataset_path = output_dir / "dataset.jsonl"
    log_path = output_dir / "review_log.jsonl"

    if not dataset_path.exists():
        print(f"Error: {dataset_path} not found.")
        sys.exit(1)

    records = load_dataset(dataset_path)
    existing_log = load_review_log(log_path) if args.resume else {}

    # Filter to flagged documents
    flagged = [r for r in records if r["qc"]["flagged"]]

    if not flagged:
        print("No flagged documents to review.")
        return

    # Skip already-reviewed if resuming
    to_review = [r for r in flagged if r["doc_id"] not in existing_log]

    print(f"\n  Total docs: {len(records)}")
    print(f"  Flagged: {len(flagged)}")
    print(f"  Already reviewed: {len(flagged) - len(to_review)}")
    print(f"  Remaining to review: {len(to_review)}")

    if not to_review:
        print("  All flagged documents already reviewed.")
        # Still apply pending decisions
        full_log = load_review_log(log_path)
        apply_decisions(dataset_path, records, full_log)
        return

    reviewed = 0
    for i, rec in enumerate(to_review):
        display_flagged_doc(rec)
        print(f"  [{i + 1}/{len(to_review)}]")
        decision = prompt_decision()

        if decision == "quit":
            print("\n  Saving progress and exiting.")
            break

        if decision == "skip":
            continue

        entry = {
            "doc_id": rec["doc_id"],
            "decision": decision,
            "reasons": rec["qc"]["reasons"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        save_decision(log_path, entry)
        existing_log[rec["doc_id"]] = entry
        reviewed += 1
        print(f"  → {decision.upper()}")

    # Apply all decisions to the dataset
    full_log = load_review_log(log_path)
    apply_decisions(dataset_path, records, full_log)

    # Summary
    accepted = sum(1 for e in full_log.values() if e["decision"] == "accept")
    dropped = sum(1 for e in full_log.values() if e["decision"] == "drop")
    print(
        f"\n  Review summary: {accepted} accepted, {dropped} dropped, "
        f"{len(to_review) - reviewed} pending"
    )


if __name__ == "__main__":
    main()
