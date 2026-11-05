"""Run red-team verification on already-generated docs.

Usage:
    python -m scripts.run_red_team outputs/test_p5d_adversarial --first 10
    python -m scripts.run_red_team outputs/test_p5_easy outputs/test_p5_medium
"""

import argparse
import json
import pathlib
import sys

from generator.llm_client import GeminiCachedClient
from generator.red_team import (
    RED_TEAM_SYSTEM,
    RedTeamResponse,
    build_red_team_user,
    cache_key_for,
    compare,
)
from generator.render import RenderResult, Span


def _reconstruct_render_result(doc: dict) -> RenderResult:
    """Rebuild a RenderResult from a serialized dataset record."""
    spans = [
        Span(
            start=s["start"],
            end=s["end"],
            label=s["label"],
            tier=s["tier"],
            entity_id=s["entity_id"],
            form=s.get("form"),
            case=s.get("case"),
        )
        for s in doc["spans"]
    ]
    negatives = [
        Span(
            start=s["start"],
            end=s["end"],
            label=s["label"],
            tier=s["tier"],
            entity_id=s["entity_id"],
            form=s.get("form"),
            case=s.get("case"),
        )
        for s in doc.get("negatives", [])
    ]
    return RenderResult(text=doc["text"], spans=spans, negative_spans=negatives)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "output_dirs", nargs="+", type=str, help="One or more output directories"
    )
    ap.add_argument("--first", type=int, default=None, help="Limit docs per directory")
    args = ap.parse_args()

    # Collect docs from all dirs
    all_docs = []
    for d in args.output_dirs:
        ds_path = pathlib.Path(d) / "dataset.jsonl"
        if not ds_path.exists():
            print(f"WARNING: {ds_path} not found, skipping.")
            continue
        docs = [
            json.loads(line)
            for line in ds_path.read_text(encoding="utf-8").splitlines()
        ]
        if args.first:
            docs = docs[: args.first]
        for doc in docs:
            doc["_source_dir"] = d
        all_docs.extend(docs)

    if not all_docs:
        print("No docs found.")
        sys.exit(1)

    llm = GeminiCachedClient(db_path="./gemini_cache.db", model_name="gemini-3.7-flash")

    print(
        f"Running red-team on {len(all_docs)} docs from {len(args.output_dirs)} dir(s)"
    )
    print("=" * 70)

    results = []
    for i, doc in enumerate(all_docs):
        meta = doc["metadata"]
        mode = meta.get("adversarial_mode", "-")
        seed = meta.get("seed", "?")
        difficulty = meta.get("difficulty", "?")
        source = doc.get("_source_dir", "?")

        result = _reconstruct_render_result(doc)
        user_content = build_red_team_user(result)

        resp = llm.generate_structured(
            system_instruction=RED_TEAM_SYSTEM,
            user_content=user_content,
            response_schema=RedTeamResponse,
            cache_key=cache_key_for(result.text),
            temperature=0.0,
        )

        agreement = compare(result, resp.get("findings", []))
        n_findings = agreement.n_findings
        n_review = len(agreement.review_items)

        status = "CLEAN" if n_review == 0 else f"FLAGGED ({n_review})"
        print(
            f"  [{i + 1}/{len(all_docs)}] {doc['doc_id']}  {difficulty:<12}  "
            f"mode={mode:20s}  {status}"
        )

        if agreement.review_items:
            for ri in agreement.review_items:
                print(f'    -> [{ri.category}] "{ri.text}"')
                print(f"       {ri.reasoning}")

        results.append(
            {
                "doc_id": doc["doc_id"],
                "source": source,
                "seed": seed,
                "difficulty": difficulty,
                "adversarial_mode": mode,
                "n_findings": n_findings,
                "n_review_items": n_review,
                "review_items": [vars(ri) for ri in agreement.review_items],
            }
        )

    # Summary
    print("\n" + "=" * 70)
    n_flagged = sum(1 for r in results if r["n_review_items"] > 0)
    total_findings = sum(r["n_review_items"] for r in results)
    print(
        f"SUMMARY: {n_flagged}/{len(results)} docs flagged by red-team ({total_findings} total findings)"
    )

    # Per-difficulty breakdown
    by_diff = {}
    for r in results:
        d = r["difficulty"]
        by_diff.setdefault(d, {"total": 0, "flagged": 0, "findings": 0})
        by_diff[d]["total"] += 1
        if r["n_review_items"] > 0:
            by_diff[d]["flagged"] += 1
        by_diff[d]["findings"] += r["n_review_items"]
    print("\nPer-difficulty:")
    for d, v in sorted(by_diff.items()):
        print(
            f"  {d:12s}: {v['flagged']}/{v['total']} flagged, {v['findings']} findings"
        )

    # Per-category breakdown
    cats = {}
    for r in results:
        for ri in r["review_items"]:
            cat = ri["category"]
            cats[cat] = cats.get(cat, 0) + 1
    if cats:
        print("\nPer-category:")
        for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
            print(f"  {cat:25s}: {count}")

    # Save results
    out_path = pathlib.Path("outputs") / "red_team_full_results.json"
    out_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
