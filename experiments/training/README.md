# Training harness — synthetic-data utility for German clinical de-identification

Fine-tunes a small German encoder as a **binary PII / not-PII** token detector on
the synthetic **SPARCLING-DE** corpus and measures whether training on our
synthetic data produces a usable de-identifier — in particular one that
**transfers to real clinical text it was not generated from**.

This is the *utility* argument a synthetic-dataset paper needs: the
de-identification benchmark (`experiments/benchmark/`) shows our test set is
*discriminative*; this harness shows the dataset is *useful for training*.

## Strategy

- **Model.** `medBERT.de` (`GerMedBERT/medbert-512`), a German clinical BERT —
  domain-matched. Any HF encoder works via `--model` (e.g. `deepset/gelectra-large`,
  `deepset/gbert-base` as fallbacks).
- **Task.** Binary token classification (IO tagging: `O` / `PII`). We predict
  *where* PII is, not its type — matching the detection-only benchmark scorer.
- **Long documents.** medBERT.de caps at **512 tokens** but 99.5% of our letters
  are longer (median ≈1.7k, max ≈8k tokens). We use **sliding-window chunking**
  (`max_length=512`, `stride=128`): overlapping windows cover the whole letter,
  per-token predictions are mapped back to **character offsets** via the fast
  tokenizer's `offset_mapping`, unioned across overlaps, and merged into spans.
- **Scoring.** Reconstructed character spans are scored with the **existing
  benchmark scorer** (lenient IoU ≥ 0.5), so fine-tuned numbers sit in the same
  table as Gemini / rule_based / flair / etc. Alongside span-F1 we report
  **merge-unbiased coverage metrics** — character- and token-level
  precision/recall — because binary IO tagging cannot separate space-adjacent
  distinct entities (e.g. `ZIP CITY` → `07743 Jena`), capping span-F1 below 1.0
  by construction. **Character recall is the de-id safety metric** ("% of
  sensitive material caught"); the char over-redaction rate is the false-positive
  cost. These are cheap (pure-python over already-computed predictions).
- **Real target.** **GRASCCO** (63 real annotated German letters, WebAnno UIMA
  CAS JSON) is the cross-generator test set — never trained on. A stable 10%
  synthetic hold-out (keyed on `doc_id`) provides the in-distribution reference.
- **Negatives.** NONPII decoys (eponyms/lookalikes) are excluded from gold, so
  the model is not taught to redact them.

## Planned experiments

A T4 makes a full train+eval run ~8 min / ~$0.10, so the plan is sized for a
coherent *utility* story with error bars rather than a single headline number.

**Core — utility story**

1. **Full synthetic → GRASCCO + held-out synthetic** — headline train-on-
   synthetic-test-on-real (TSTR) transfer result. *Must-have.* Supported today.
2. **Synthetic size scaling** (`--subsample 100/300/600/all`) — data-value curve
   (GRASCCO char-recall + span-F1 vs corpus size); shows whether utility has
   saturated. Supported today.
3. **Three-way real-data comparison — the data-scarcity argument.** Only ~63 real
   annotated German letters exist (GRASCCO), so we use **5-fold cross-validation**
   over GRASCCO and compare, on each held-out fold:
   - **real-only** — train on the GRASCCO train folds (the "all the real data you
     actually have" baseline);
   - **synthetic-only** — full synthetic, zero-shot onto the fold (= exp. 1,
     re-scored per fold);
   - **synthetic + real** — synthetic pretrain, then fine-tune on the GRASCCO
     train folds.

   This is the paper's practical headline: *does our synthetic corpus match
   real-only training, and does it augment the scarce real data?* If
   synthetic-only ≈ real-only it stands alone; if synthetic+real > real-only,
   that's the strongest claim. *Needs a small harness addition:* a deterministic
   GRASCCO k-fold split (`data.py`) and a `--init-from` / fine-tune path in
   `train.py`.

**Ablation**

4. **With vs without adversarial cases** (`--drop-adversarial`) — isolates the
   value of the generator's adversarial/red-team design. *Highest-ROI ablation,
   unique to this dataset.* Supported today.

**Rigor**

5. **Error bars** — repeat the headline configs (1 and 3) over **3 seeds**
   (`--seed`) and report mean ± std. Cheap now; small-corpus transfer numbers
   need variance to be credible. Supported today (loop over `--seed`).

**Reference (produced via `experiments/benchmark/`, not here)**

6. **Zero-shot off-the-shelf** flair/openmed on GRASCCO with *no* fine-tuning —
   the "no task-specific training" floor.

## Layout

| File | Role |
|------|------|
| `data.py` | corpus loaders (synthetic + GRASCCO), **manual** sliding-window chunking (version-robust; see note), span↔token-label projection, span reconstruction, deterministic train/test split, ablation filters. *Exp. 3 adds a GRASCCO k-fold split here.* |
| `train.py` | fine-tune + save model, writes `train_meta.json` (provenance/timing/device) + `training_log.json` (loss + per-epoch val curve). Flags: `--drop-adversarial`, `--subsample`, `--test-frac`, `--val-frac`, `--seed`. *Exp. 3 adds an `--init-from` fine-tune path.* |
| `evaluate.py` | predict + score, write `report.json`: span metrics (benchmark scorer) + `char_level`/`token_level` coverage + per-family/tier/difficulty slices (`--eval-synthetic --synthetic-holdout-only`, `--eval-grascco`) |
| `smoke_test.py` | fast, model-light harness check (loaders, split determinism, no leakage, chunk→reconstruct roundtrip); tokenizer-only, no training |
| `run_experiments.sh` | driver for the **full matrix** — exp. 1/5 (3 seeds), exp. 2 (scaling), exp. 4 (drop-adversarial), exp. 3 (GRASCCO 5-fold × {real-only, synthetic-only, synth+real}) — then runs `aggregate` and touches `DONE` |
| `aggregate.py` | walks `results/training/`, rolls up span/token/char metrics + hardware/timing/versions into `summary.json`, prints the environment + result tables |

## Usage

```bash
# one run
python -m experiments.training.train \
  --train-data experiments/datasets/sparcling-de/dataset.jsonl \
  --model GerMedBERT/medbert-512 \
  --out-dir experiments/results/training/full_synthetic

python -m experiments.training.evaluate \
  --model-dir experiments/results/training/full_synthetic \
  --eval-synthetic experiments/datasets/sparcling-de/dataset.jsonl --synthetic-holdout-only \
  --eval-grascco experiments/datasets/grascco_phi_lohr_2024

# full matrix (all experiments) -> results/training/summary.json + DONE
bash experiments/training/run_experiments.sh
# on a GPU VM with no conda + a `python3` alias:
#   OUT=$HOME/results/training PY=python3 SKIP_ACTIVATE=1 \
#     bash experiments/training/run_experiments.sh
```

## Results & what to report in the paper

The full matrix was run on a **single NVIDIA Tesla T4 (16 GB, GCP)**; the whole
sweep (17 trained runs + 5 eval-only CV folds) took **~35 min of GPU wall-clock**.
Everything below is machine-readable in
[`../results/training/summary.json`](../results/training/) (`environment` block)
and per-run in each `train_meta.json`.

**Setup to report** (identical across all runs unless noted):

| | |
|---|---|
| Base encoder | `medBERT.de` (`GerMedBERT/medbert-512`) |
| Hardware | 1× NVIDIA Tesla T4 (16 GB), fp16 |
| Software | torch 2.9.1+cu129, transformers 4.57.6, Python 3.10.12 |
| Optimizer | AdamW, lr 3e-5, weight decay 0.01, warmup ratio 0.1 |
| Batch | 8 × grad-accum 2 = **effective 16** |
| Chunking | sliding window, `max_length=512`, `stride=128` |
| Epochs | 3 (synthetic runs) / 10 (GRASCCO CV folds, 42–47 train docs each) |
| Seeds | 42 / 43 / 44 for the headline configs (error bars) |
| Train time | full synthetic run ≈ 5.1 min; each GRASCCO fold ≈ 1 min |

**Headline numbers** (from `summary.json`; char recall = de-id safety metric):

- **GRASCCO 5-fold CV, three-way** — real-only char-R **0.955 ± 0.022**,
  synthetic-only **0.834 ± 0.032**, **synth+real 0.975 ± 0.013** (also best
  token-F1 0.968 and span-F1 0.873, with the lowest variance). Synthetic
  pretraining *augments* the scarce real data and stabilises it.
- **TSTR transfer** (train synthetic, test real GRASCCO): char-R 0.836 / token-F1
  0.869 / span-F1 0.762, tight across seeds; held-out synthetic char-R 0.998.
- **Scaling:** span-F1 climbs 0.70 → 0.76 from 100 → 821 synthetic docs while
  char recall saturates early (~0.88 by 100 docs).

Regenerate the tables/rollup any time with:

```bash
OUT=experiments/results/training python -m experiments.training.aggregate
```

> Provenance note: `git_sha` is `null` in the saved `train_meta.json` because the
> VM ran an uncommitted working copy of this harness. Cite the commit that lands
> these files as the code provenance.

## Notes / requirements

- Deps: `torch`, `transformers`, `accelerate` (present). Intentionally **no**
  `datasets`/`seqeval` dependency. **Pin `transformers==4.57.6`** (matches local):
  5.x drops `TrainingArguments(warmup_ratio=...)` and eagerly imports a broken
  `torchaudio` on the GCP DL image.
- **Chunking is done manually** (`data.py:_sliding_windows`), not via the
  tokenizer's `return_overflowing_tokens`: the latter silently emits only one
  overflow window under transformers 5.x, truncating ~90% of long letters. The
  manual window is correct and version-independent.
- **Compute:** training is trivial on a GPU (measured **~35 train chunks/s on a
  T4**, fp16 — a full synthetic run ≈ 5 min, the whole matrix ≈ 35 min / ~$0.30)
  but slow on CPU (~1 chunk/s). The
  local box is CPU-only (`torch==*+cpu`); run the real experiments on a **CUDA
  GCP VM** (T4/L4; watch for zonal stockouts). `evaluate.py`/`train.py`
  auto-detect CUDA and enable fp16. See the `vm-neural-training-run` memo for the
  VM recipe.
- Labels are binary by design; the scorer still slices gold by family/tier for
  reporting because matching is label-agnostic.
