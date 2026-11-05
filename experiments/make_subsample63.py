"""Draw a stable, representative 63-document subsample of a German synth corpus.

Corpus is selected with ``--corpus`` (default: sparcling-de).

Size 63 matches the GRASSCO corpus for a size-controlled comparison (many
lexical/similarity metrics are sample-size sensitive). The subsample is written
once and reused; this script is deterministic given the accepted seed.

Representativeness is verified against the full corpus on the controlled design
axes (difficulty, style, department) via total-variation distance, and on the
continuous surface distributions (document length, PHI-span count, patient age)
via two-sample Kolmogorov-Smirnov tests. Candidate seeds are searched in order;
the first seed clearing every threshold is accepted. If none clears, the least
divergent candidate is reported and the run exits non-zero so the caller notices.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from scipy.stats import ks_2samp

HERE = Path(__file__).resolve().parent
# Which corpus to subsample. Overridable so the same representativeness-checked
# procedure can be applied to any German synthetic run (e.g. sparcling-de).
DEFAULT_CORPUS = "sparcling-de"

SUBSAMPLE_SIZE = 63
MAX_SEEDS = 2000

# Acceptance thresholds. Categorical: max total-variation distance. Continuous:
# minimum KS p-value (we must FAIL to reject "same distribution").
TVD_MAX = {"difficulty": 0.08, "style": 0.08, "fachabteilung": 0.18}
KS_P_MIN = {"char_len": 0.10, "n_spans": 0.10, "patient_age": 0.10}


def load_records(dataset: Path) -> list[dict]:
    with open(dataset, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def categorical_props(records: list[dict], field: str) -> Counter:
    counts = Counter(r["metadata"].get(field) for r in records)
    total = sum(counts.values())
    return Counter({k: v / total for k, v in counts.items()})


def tvd(full: Counter, sub: Counter) -> float:
    """Total-variation distance between two categorical distributions."""
    keys = set(full) | set(sub)
    return 0.5 * sum(abs(full.get(k, 0.0) - sub.get(k, 0.0)) for k in keys)


def continuous(records: list[dict], field: str) -> list[float]:
    if field == "char_len":
        return [float(len(r["text"])) for r in records]
    return [
        float(r["metadata"][field])
        for r in records
        if r["metadata"].get(field) is not None
    ]


def evaluate(full: list[dict], sub: list[dict]) -> dict:
    tvds = {
        f: tvd(categorical_props(full, f), categorical_props(sub, f))
        for f in TVD_MAX
    }
    ks = {}
    for f in KS_P_MIN:
        stat, p = ks_2samp(continuous(full, f), continuous(sub, f))
        ks[f] = {"statistic": float(stat), "p_value": float(p)}
    passed = all(tvds[f] <= TVD_MAX[f] for f in TVD_MAX) and all(
        ks[f]["p_value"] >= KS_P_MIN[f] for f in KS_P_MIN
    )
    # Lower is better: categorical divergence plus a small KS-shortfall penalty.
    score = sum(tvds.values()) + sum(
        max(0.0, KS_P_MIN[f] - ks[f]["p_value"]) for f in KS_P_MIN
    )
    return {"tvd": tvds, "ks": ks, "passed": passed, "score": score}


def main() -> int:
    ap = argparse.ArgumentParser(description="Draw a representative n=63 subsample.")
    ap.add_argument(
        "--corpus",
        default=DEFAULT_CORPUS,
        help="Dataset dir name under experiments/datasets/ (default: sparcling-de).",
    )
    args = ap.parse_args()
    corpus_dir = HERE / "datasets" / args.corpus
    dataset = corpus_dir / "dataset.jsonl"
    out_jsonl = corpus_dir / "subsample63.jsonl"
    out_meta = corpus_dir / "subsample63_meta.json"

    full = load_records(dataset)
    print(
        f"Full corpus ({args.corpus}): {len(full)} documents; "
        f"drawing {SUBSAMPLE_SIZE}."
    )

    best_seed = None
    best_eval = None
    accepted_seed = None
    for seed in range(MAX_SEEDS):
        sub = random.Random(seed).sample(full, SUBSAMPLE_SIZE)
        ev = evaluate(full, sub)
        if best_eval is None or ev["score"] < best_eval["score"]:
            best_seed, best_eval = seed, ev
        if ev["passed"]:
            accepted_seed, best_seed, best_eval = seed, seed, ev
            print(f"Accepted seed {seed} (cleared all thresholds after retries).")
            break

    if accepted_seed is None:
        print(
            f"WARNING: no seed in [0,{MAX_SEEDS}) cleared every threshold; "
            f"reporting least-divergent seed {best_seed}."
        )

    seed = best_seed
    sub = random.Random(seed).sample(full, SUBSAMPLE_SIZE)
    sub.sort(key=lambda r: r["doc_id"])
    ev = best_eval

    with open(out_jsonl, "w", encoding="utf-8") as fh:
        for r in sub:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    meta = {
        "source": str(dataset.relative_to(HERE)),
        "subsample_size": SUBSAMPLE_SIZE,
        "accepted_seed": seed,
        "passed_all_thresholds": ev["passed"],
        "thresholds": {"tvd_max": TVD_MAX, "ks_p_min": KS_P_MIN},
        "representativeness": ev,
        "doc_ids": [r["doc_id"] for r in sub],
    }
    with open(out_meta, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    print("\nRepresentativeness (subsample vs full):")
    for f, v in ev["tvd"].items():
        mark = "ok" if v <= TVD_MAX[f] else "FAIL"
        print(f"  TVD {f:<14} = {v:.4f}  (<= {TVD_MAX[f]}) [{mark}]")
    for f, v in ev["ks"].items():
        mark = "ok" if v["p_value"] >= KS_P_MIN[f] else "FAIL"
        print(
            f"  KS  {f:<14} D={v['statistic']:.4f} p={v['p_value']:.4f} "
            f"(p >= {KS_P_MIN[f]}) [{mark}]"
        )
    print(f"\nWrote {out_jsonl.name} ({len(sub)} docs) and {out_meta.name}.")
    return 0 if ev["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
