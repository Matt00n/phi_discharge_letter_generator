# Ablations

Three ablations, each removing one load-bearing stage of the SPARCLING-DE
pipeline to quantify what it contributes. Every arm reuses the **identical**
Stage-4 QC (`generator.red_team` + `generator.render.scan_leaks`) and Stage-5
diversity (`generator.diversity`) code as the real pipeline, so comparisons are
apples-to-apples. Ablation LLM calls use a separate cache DB
(`ablations_cache.db`) so they never collide with the production run's cache.

Baselines are drawn from the main run at `experiments/datasets/sparcling-de/`
(override with `--baseline-dir`). Because entropy/diversity metrics are
N-sensitive, every comparison is made against an **N-matched slice** of the main
run.

## 1. No QC — `no_qc.py`

Analysis only (no API). Reconstructs the full pre-QC corpus (kept dataset ∪
review-queue exclusions) and reports what would ship if Stage 4 were off: the
residual flag rate and its breakdown by mechanism, red-team category, and leaked
label. Each review-queue doc is a defect QC actually removed.

```
python -m experiments.ablations.no_qc [--baseline-dir DIR] [--out-dir DIR]
```

Outputs (default `experiments/results/ablation_no_qc/`): `provenance.json`,
`summary.md`.

## 2. No templating — `no_templating.py`

For each of the same first-N sampled profiles (base_seed 1,000,000), the LLM is
handed the **concrete** sampled PII and writes the letter as free text, then
self-annotates the PII it used — one structured call. We then run the identical
Stage-4 QC (deterministic `scan_leaks` against the known values + the red-team
auditor on the self-annotated text). Shows QC fires heavily: free-text authoring
inflects, paraphrases, drops/invents identifiers, and self-annotation misses
spans.

```
python -m experiments.ablations.no_templating [--num 100] [--baseline-dir DIR] [--out-dir DIR]
```

Outputs (default `experiments/results/ablation_no_templating/`):
`{dataset,review_queue}.jsonl`, `provenance.json`. Reports the no-templating flag
rate vs. the templated baseline on the same docs.

## 3. No sampling — `no_sampling.py`

The Stage-1 sampler is removed. Two LLM passes per doc: (1) author a template
with `[[ref]]` placeholders from a fixed field schema, inventing the clinical
scenario; (2) fill every placeholder with an invented value. The deterministic
renderer substitutes them, so spans stay correct by construction (QC passes) —
but the PII distribution is now the LLM's. We run the identical diversity audit
and compare to an N-matched baseline slice. Shows diversity collapses
(surname/age/city entropy, PII-surface variety drop; LLMs fall back on
"Max Mustermann", "Musterstadt", round numbers).

```
python -m experiments.ablations.no_sampling [--num 100] [--baseline-dir DIR] [--out-dir DIR]
```

Outputs (default `experiments/results/ablation_no_sampling/`):
`{dataset.jsonl,diversity.json,summary.md}`.

> Caveat: `patient_origin`, `style`, `difficulty`, section structure, and the
> free-form `fachabteilung` string are sampler concepts not normalized in the
> no-sampling arm, so the single aggregate diversity score is not directly
> comparable. The per-dimension surname/age/PII-surface/lexical metrics in
> `summary.md` are the fair comparison.

## Build/validation status

All three ablations run to completion against `sparcling-de`; outputs live
under `experiments/results/ablation_*/`. Headlines:
- **No QC:** 91/1110 (8.2%) docs would ship with a potential PHI leak (68 red-team,
  16 deterministic-leak, 7 both).
- **No templating (n=100):** flag rate 21% vs. 9% for the templated baseline
  on the same profiles.
- **No sampling (n=100):** PII diversity collapses — surname entropy
  3.14 vs. 6.52 bits (21 vs. 94 unique), unique ages 14 vs. 61, unique PII
  surfaces 560 vs. 2339.
