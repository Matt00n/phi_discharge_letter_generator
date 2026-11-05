"""Aggregate all training-experiment reports into one summary.

Walks the results/training tree written by run_experiments.sh, pulls span /
token / char metrics out of every report.json, and emits:
  - summary.json  : machine-readable roll-up (incl. per-fold CV detail)
  - printed tables: synthetic-trained transfer + GRASCCO 5-fold three-way

Char-level recall is the de-id *safety* metric (share of sensitive characters
caught); span-F1 is reported too but is capped below 1.0 by the binary-IO merge
of space-adjacent entities, so read char/token first.
"""

from __future__ import annotations

import glob
import json
import os
import statistics as st

ROOT = os.environ.get("OUT") or os.path.join("experiments", "results", "training")


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _metrics(block: dict) -> dict:
    m = block["micro"]
    c = block["char_level"]
    t = block["token_level"]
    return {
        "span_f1": m["f1"],
        "span_p": m["precision"],
        "span_r": m["recall"],
        "token_f1": t["f1"],
        "token_r": t["recall"],
        "token_p": t["precision"],
        "char_f1": c["f1"],
        "char_r": c["recall"],
        "char_p": c["precision"],
        "over_redaction": c["over_redaction_rate"],
        "n_docs": block.get("num_documents"),
    }


def _grascco_block(report: dict) -> dict | None:
    """Return the GRASCCO metrics block whether full-corpus or a CV fold."""
    if "grascco" in report:
        return report["grascco"]
    key = next((k for k in report if k.startswith("grascco_fold")), None)
    return report[key] if key else None


def _environment() -> dict:
    """Roll up hardware / software / hyperparameters / wall-clock for the paper.

    Every trained run persists the same environment in its train_meta.json; we
    read them all, assert the hardware+versions are consistent, and total the
    training wall-clock. The 5 synth_only CV folds are eval-only (no meta).
    """
    metas = sorted(glob.glob(os.path.join(ROOT, "**", "train_meta.json"),
                             recursive=True))
    gpus, versions = set(), set()
    total_seconds = 0.0
    per_run = {}
    args_ref = None
    for m in metas:
        mm = _load(m)
        dev = mm["device"]
        gpus.add(f"{dev['gpu_name']} x{dev['n_gpu']}" if dev["cuda"] else "cpu")
        v = mm["versions"]
        versions.add((v["torch"], v["transformers"], v["python"]))
        total_seconds += mm["timing"]["train_seconds"]
        name = os.path.relpath(os.path.dirname(m), ROOT).replace(os.sep, "/")
        per_run[name] = round(mm["timing"]["train_seconds"], 1)
        args_ref = args_ref or mm["args"]
    v0 = next(iter(versions), (None, None, None))
    a = args_ref or {}
    return {
        "gpu": " | ".join(sorted(gpus)) or None,
        "consistent_hardware": len(gpus) <= 1,
        "torch": v0[0],
        "transformers": v0[1],
        "python": v0[2],
        "consistent_versions": len(versions) <= 1,
        "hyperparameters": {
            "base_encoder": a.get("model"),
            "lr": a.get("lr"),
            "per_device_batch_size": a.get("batch_size"),
            "grad_accum": a.get("grad_accum"),
            "effective_batch_size": (
                a.get("batch_size", 0) * a.get("grad_accum", 0) or None
            ),
            "max_length": a.get("max_length"),
            "stride": a.get("stride"),
            "weight_decay": a.get("weight_decay"),
            "warmup_ratio": a.get("warmup_ratio"),
            "epochs_synthetic": 3.0,
            "epochs_grascco_fold": 10.0,
            "precision": "fp16",
        },
        "n_trained_runs": len(metas),
        "train_seconds_per_run": per_run,
        "total_train_seconds": round(total_seconds, 1),
        "total_train_minutes": round(total_seconds / 60, 1),
    }


def main() -> int:
    summary: dict = {"environment": _environment(), "synthetic_trained": {},
                     "kfold": {}}

    # -- synthetic-trained models (full/seeds/scaling/drop-adversarial) --------
    for d in sorted(glob.glob(os.path.join(ROOT, "*"))):
        rp = os.path.join(d, "report.json")
        if not os.path.isfile(rp):
            continue
        name = os.path.basename(d)
        r = _load(rp)
        entry: dict = {}
        if "synthetic" in r:
            entry["synthetic_holdout"] = _metrics(r["synthetic"])
        if "grascco" in r:
            entry["grascco"] = _metrics(r["grascco"])
        mp = os.path.join(d, "train_meta.json")
        if os.path.isfile(mp):
            mm = _load(mp)
            entry["n_train_docs"] = mm["data"]["n_train_docs"]
            entry["train_seconds"] = round(mm["timing"]["train_seconds"], 1)
            entry["best_val_f1"] = mm.get("best_metric")
        summary["synthetic_trained"][name] = entry

    # -- GRASCCO 5-fold, three conditions -------------------------------------
    kdir = os.path.join(ROOT, "kfold")
    conds = {
        "real_only": "real_only_fold",
        "synth_only": "synth_only_fold",
        "synth_real": "synth_real_fold",
    }
    for cond, prefix in conds.items():
        folds = []
        for d in sorted(glob.glob(os.path.join(kdir, prefix + "*"))):
            rp = os.path.join(d, "report.json")
            if not os.path.isfile(rp):
                continue
            block = _grascco_block(_load(rp))
            if block:
                folds.append(_metrics(block))
        if not folds:
            continue
        agg: dict = {"n_folds": len(folds), "per_fold": folds}
        for k in ("span_f1", "token_f1", "char_f1", "char_r", "over_redaction"):
            vals = [f[k] for f in folds]
            agg[f"{k}_mean"] = st.mean(vals)
            agg[f"{k}_std"] = st.pstdev(vals) if len(vals) > 1 else 0.0
        summary["kfold"][cond] = agg

    with open(os.path.join(ROOT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # -- printed tables --------------------------------------------------------
    nan = float("nan")
    env = summary["environment"]
    hp = env["hyperparameters"]
    print("\n=== ENVIRONMENT (report in paper) ===")
    print(f"GPU:      {env['gpu']}")
    print(
        f"software: torch {env['torch']} / transformers {env['transformers']} "
        f"/ python {env['python']}"
    )
    print(
        f"encoder:  {hp['base_encoder']}  |  lr {hp['lr']}  "
        f"eff.batch {hp['effective_batch_size']} ({hp['per_device_batch_size']}"
        f"x{hp['grad_accum']})  max_len {hp['max_length']}/stride {hp['stride']}  "
        f"{hp['precision']}"
    )
    print(
        f"epochs:   {hp['epochs_synthetic']} (synthetic) / "
        f"{hp['epochs_grascco_fold']} (GRASCCO fold)  |  "
        f"warmup {hp['warmup_ratio']}  wd {hp['weight_decay']}"
    )
    print(
        f"wall-clock: {env['total_train_minutes']} min total over "
        f"{env['n_trained_runs']} trained runs (single GPU)"
    )

    print("\n=== SYNTHETIC-TRAINED -> real GRASCCO (char / token / span) ===")
    print(
        f"{'run':22} {'#tr':>5} {'sec':>5} | {'graChR':>7} {'graTokF1':>8} "
        f"{'graSpF1':>7} | {'holdChR':>7} {'holdSpF1':>8}"
    )
    for name, e in summary["synthetic_trained"].items():
        g = e.get("grascco", {})
        h = e.get("synthetic_holdout", {})
        print(
            f"{name:22} {e.get('n_train_docs',''):>5} "
            f"{e.get('train_seconds',nan):>5.0f} | "
            f"{g.get('char_r',nan):>7.4f} {g.get('token_f1',nan):>8.4f} "
            f"{g.get('span_f1',nan):>7.4f} | "
            f"{h.get('char_r',nan):>7.4f} {h.get('span_f1',nan):>8.4f}"
        )

    print("\n=== GRASCCO 5-fold CV — three-way (mean ± std over folds) ===")
    print(f"{'condition':14} | {'char recall':>16} {'token F1':>16} {'span F1':>16}")
    labels = {
        "real_only": "real-only",
        "synth_only": "synthetic-only",
        "synth_real": "synth+real",
    }
    for cond in ("real_only", "synth_only", "synth_real"):
        a = summary["kfold"].get(cond)
        if not a:
            continue
        print(
            f"{labels[cond]:14} | "
            f"{a['char_r_mean']:>7.4f} ± {a['char_r_std']:<6.4f} "
            f"{a['token_f1_mean']:>7.4f} ± {a['token_f1_std']:<6.4f} "
            f"{a['span_f1_mean']:>7.4f} ± {a['span_f1_std']:<6.4f}"
        )
    print(f"\n[aggregate] wrote {os.path.join(ROOT, 'summary.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
