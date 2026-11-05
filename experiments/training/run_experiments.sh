#!/usr/bin/env bash
# Full training-experiment matrix for the SPARCLING-DE de-identification utility
# study. Trains medBERT.de as a binary PII detector and evaluates cross-generator
# transfer to real GRASCCO. Runs are fast on a GPU (minutes each); slow on CPU —
# prefer a CUDA VM. Every run writes a model dir + report.json under $OUT/.
#
# Env overrides:
#   PY=python3          interpreter (VM deep-learning image has no `python` alias)
#   OUT=~/results/training   output root
#   SKIP_ACTIVATE=1     skip `conda activate` (set on the VM)
#   MODEL=...           base encoder
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="${PYTHONPATH:-$REPO_ROOT}"
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
if command -v conda >/dev/null 2>&1 && [ -z "${SKIP_ACTIVATE:-}" ]; then
  # shellcheck disable=SC1091
  source activate dataset_generator
fi

PY="${PY:-python}"
MODEL="${MODEL:-GerMedBERT/medbert-512}"
SYN="experiments/datasets/sparcling-de/dataset.jsonl"
GRASCCO="experiments/datasets/grascco_phi_lohr_2024"
OUT="${OUT:-experiments/results/training}"
K=5
mkdir -p "$OUT/kfold"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# Train on synthetic + eval (held-out synthetic + full GRASCCO).
syn_run() {  # name  <extra train args...>
  local name="$1"; shift
  local dir="$OUT/$name"
  log "======== $name ========"
  $PY -m experiments.training.train \
    --train-data "$SYN" --model "$MODEL" --out-dir "$dir" "$@" \
    2>&1 | tee "$OUT/$name.train.log"
  $PY -m experiments.training.evaluate \
    --model-dir "$dir" \
    --eval-synthetic "$SYN" --synthetic-holdout-only \
    --eval-grascco "$GRASCCO" \
    2>&1 | tee "$OUT/$name.eval.log"
}

# --- Experiment 1 + 5: full synthetic -> GRASCCO, 3 seeds (headline TSTR + error bars)
for s in 42 43 44; do
  if [ "$s" = 42 ]; then syn_run full_synthetic --seed "$s"
  else syn_run "full_synthetic_seed$s" --seed "$s"; fi
done

# --- Experiment 2: synthetic size scaling (data-value curve); "all" == full_synthetic
for n in 100 300 600; do
  syn_run "scaling_$n" --subsample "$n" --seed 42
done

# --- Experiment 4: with vs without adversarial cases (generator differentiator)
syn_run drop_adversarial --drop-adversarial --seed 42

# --- Experiment 3: GRASCCO 5-fold CV, three conditions (data-scarcity argument)
SYN_MODEL="$OUT/full_synthetic"   # synthetic pretrain: reused for (b) init + (c) eval
for f in $(seq 0 $((K - 1))); do
  # (a) real-only: base encoder trained on GRASCCO train folds
  dir="$OUT/kfold/real_only_fold$f"
  log "======== kfold real_only fold=$f ========"
  $PY -m experiments.training.train \
    --train-grascco "$GRASCCO" --kfold $K --fold "$f" \
    --model "$MODEL" --out-dir "$dir" --epochs 10 --seed 42 \
    2>&1 | tee "$OUT/kfold/real_only_fold$f.train.log"
  $PY -m experiments.training.evaluate --model-dir "$dir" \
    --eval-grascco "$GRASCCO" --grascco-kfold $K --grascco-fold "$f" \
    2>&1 | tee "$OUT/kfold/real_only_fold$f.eval.log"

  # (b) synthetic+real: fine-tune the synthetic-pretrained model on GRASCCO folds
  dir="$OUT/kfold/synth_real_fold$f"
  log "======== kfold synth_real fold=$f ========"
  $PY -m experiments.training.train \
    --train-grascco "$GRASCCO" --kfold $K --fold "$f" \
    --init-from "$SYN_MODEL" --out-dir "$dir" --epochs 10 --seed 42 \
    2>&1 | tee "$OUT/kfold/synth_real_fold$f.train.log"
  $PY -m experiments.training.evaluate --model-dir "$dir" \
    --eval-grascco "$GRASCCO" --grascco-kfold $K --grascco-fold "$f" \
    2>&1 | tee "$OUT/kfold/synth_real_fold$f.eval.log"

  # (c) synthetic-only on this held-out fold (no training; reuse synthetic model)
  dir="$OUT/kfold/synth_only_fold$f"; mkdir -p "$dir"
  log "======== kfold synth_only fold=$f ========"
  $PY -m experiments.training.evaluate --model-dir "$SYN_MODEL" \
    --eval-grascco "$GRASCCO" --grascco-kfold $K --grascco-fold "$f" \
    --out "$dir/report.json" \
    2>&1 | tee "$OUT/kfold/synth_only_fold$f.eval.log"
done

# --- Aggregate everything into summary.json + printed tables
$PY -m experiments.training.aggregate

log "======== ALL DONE — reports under $OUT/ (summary.json) ========"
touch "$OUT/DONE"
