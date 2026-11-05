"""Run the full corpus-evaluation suite for the paper (cross-shell, no bash).

Two sequential passes:
  1. German corpora  - sparcling-de + its n=63 subsample + grascco +
     gernermed. Neural metrics ON. Per-corpus metrics AND the meaningful
     (same-language) cross-corpus distances.
  2. English baselines - i2b2 + mimic. Neural metrics ON (intra-corpus diversity
     reported); no German-vs-English cross pair is computed.

Passes run sequentially on purpose: no GPU here, and xlm-roberta-large already
multithreads across CPU cores, so parallel processes would only contend. The
LLM judge is OFF in both configs.

Usage (from an activated environment, any shell):
    python experiments/run_all_evals.py                 # first run
    python experiments/run_all_evals.py --overwrite     # replace existing reports
Extra arguments are forwarded verbatim to both evaluation invocations.

Output is streamed live to the console (progress bars preserved) AND teed to a
per-pass .log file under experiments/results/.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "experiments" / "results"
SUBSAMPLE = (
    REPO_ROOT / "experiments" / "datasets" / "sparcling-de" / "subsample63.jsonl"
)

PASSES = [
    ("1/2  German corpora (synth + synth63 + grascco + gernermed)",
     "experiments/config_eval_sparcling_de.yaml", "eval_sparcling_de.log"),
    ("2/2  English baselines (i2b2 + mimic)",
     "experiments/config_eval_en_baselines.yaml", "eval_en_baselines.log"),
]


def _tee(cmd: list[str], log_path: Path, env: dict) -> int:
    """Run cmd, streaming merged stdout/stderr to console and a log file.

    Reads raw bytes so tqdm/bert_score carriage-return progress bars render live
    instead of being buffered until the next newline.
    """
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        assert proc.stdout is not None
        fd = proc.stdout.fileno()
        out = sys.stdout.buffer
        while True:
            chunk = os.read(fd, 4096)
            if not chunk:
                break
            out.write(chunk)
            out.flush()
            log.write(chunk)
            log.flush()
        return proc.wait()


def main() -> int:
    forwarded = sys.argv[1:]
    RESULTS.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    if not SUBSAMPLE.exists():
        print(f"=== Subsample missing; generating {SUBSAMPLE} ===", flush=True)
        rc = subprocess.call(
            [sys.executable, "experiments/make_subsample63.py"], cwd=REPO_ROOT, env=env
        )
        if rc != 0:
            print(f"Subsample generation failed (exit {rc}); aborting.", flush=True)
            return rc

    for title, config, log_name in PASSES:
        print("=" * 64, flush=True)
        print(f"  [{title}]", flush=True)
        print("=" * 64, flush=True)
        cmd = [sys.executable, "-m", "experiments.evaluation", "--config", config, *forwarded]
        rc = _tee(cmd, RESULTS / log_name, env)
        if rc != 0:
            print(f"\nPass failed (exit {rc}); aborting before remaining passes.", flush=True)
            return rc

    print("=" * 64, flush=True)
    print("  DONE. Reports:", flush=True)
    print(f"    {RESULTS / 'eval_sparcling_de.json'}", flush=True)
    print(f"    {RESULTS / 'eval_en_baselines.json'}", flush=True)
    print("=" * 64, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
