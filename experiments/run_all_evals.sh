#!/usr/bin/env bash
# Run the full corpus-evaluation suite for the paper, in two passes:
#   1. German corpora  — sparcling-de + its n=63 subsample + grascco +
#      gernermed. Neural metrics ON. Produces per-corpus metrics AND the
#      meaningful (same-language) cross-corpus distances.
#   2. English baselines — i2b2 + mimic. Neural metrics ON (intra-corpus
#      diversity reported); no German-vs-English cross pair is computed.
#
# Passes run SEQUENTIALLY on purpose: no GPU here, and xlm-roberta-large already
# multithreads across CPU cores, so parallel processes would only contend.
#
# The LLM judge is OFF in both configs (as requested).
#
# Usage:
#   bash experiments/run_all_evals.sh            # first run
#   bash experiments/run_all_evals.sh --overwrite  # replace existing reports
#
# Any extra arguments (e.g. --overwrite, --log-level DEBUG) are forwarded to
# both evaluation invocations.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source activate dataset_generator
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

mkdir -p experiments/results

# Regenerate the n=63 subsample only if it is missing; it is a stable, reusable
# artifact and must not silently change between runs.
if [[ ! -f experiments/datasets/sparcling-de/subsample63.jsonl ]]; then
  echo "=== Subsample missing; generating experiments/datasets/sparcling-de/subsample63.jsonl ==="
  python experiments/make_subsample63.py
fi

echo "================================================================"
echo "  [1/2] German corpora (synth + synth63 + grascco + gernermed)"
echo "================================================================"
python -m experiments.evaluation \
  --config experiments/config_eval_sparcling_de.yaml "$@" \
  2>&1 | tee experiments/results/eval_sparcling_de.log

echo "================================================================"
echo "  [2/2] English baselines (i2b2 + mimic)"
echo "================================================================"
python -m experiments.evaluation \
  --config experiments/config_eval_en_baselines.yaml "$@" \
  2>&1 | tee experiments/results/eval_en_baselines.log

echo "================================================================"
echo "  DONE. Reports:"
echo "    experiments/results/eval_sparcling_de.json"
echo "    experiments/results/eval_en_baselines.json"
echo "================================================================"
