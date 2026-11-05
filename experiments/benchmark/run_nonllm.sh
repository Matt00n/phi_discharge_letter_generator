#!/usr/bin/env bash
# Run the non-LLM de-identification benchmarks sequentially against SPARCLING-DE.
# Each method writes to experiments/results/benchmark/<name>/{deid_texts.jsonl,report.json}.
# Sequential on purpose: scoring is CPU-bound and the neural methods multithread,
# so parallel runs would only contend.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source activate dataset_generator
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

OUT="experiments/results/benchmark"
mkdir -p "$OUT"

run() {
  local name="$1"
  echo "================================================================"
  echo "  de-id benchmark: ${name}"
  echo "================================================================"
  python -m experiments.benchmark \
    --config "experiments/benchmark/${name}_config.yaml" \
    2>&1 | tee "${OUT}/${name}.log"
  echo "  [done] ${name}"
}

for m in "$@"; do
  run "$m"
done

echo "================================================================"
echo "  ALL DONE. Reports under ${OUT}/<method>/report.json"
echo "================================================================"
