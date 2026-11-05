# Experiments

This directory contains the paper-facing experiments for **SPARCLING** (*Generating Diverse and Verified Synthetic Clinical Notes with Labels by Construction*).

- [`evaluation/`](evaluation/) — Corpus diversity, controllability, sampled-profile separation, duplication, clinical/textual realism (readability + blinded Gemini LLM judge), and cross-corpus distribution shift metrics (Sections 4.2 & 4.3).
- [`ablations/`](ablations/) — Ablation studies isolating the contributions of dual quality control (`no_qc`), placeholder templating (`no_templating`), and structured attribute sampling (`no_sampling`) (Section 4.4).
- [`benchmark/`](benchmark/) — De-identification benchmark harness evaluating 6 classes of methods (`gemini`, `lmstudio` / Gemma 4 E2B, `flair`, `presidio` + spaCy, `openmed`, `rule_based`) against character-exact gold spans (Section 4.5.1). Includes the [`scoring/`](benchmark/scoring/) package.
- [`training/`](training/) — Downstream utility experiments fine-tuning `medBERT.de` (`GerMedBERT/medbert-512`) for synthetic-to-real transfer, 5-fold cross-validation data augmentation on GraSCCo, data scaling, and adversarial case ablations (Section 4.5.2).

## Setup & Quick Commands

Install the experiment dependency group:

```bash
poetry install --with experiments
```

1. **Generate `SPARCLING-DE` (or download from Hugging Face into `experiments/datasets/sparcling-de/`):**
   ```bash
   poetry run python -m generator.main --config experiments/config_sparcling_de.yaml
   poetry run python experiments/make_subsample63.py
   ```
2. **Corpus Diversity & Realism Evaluation (Sections 4.2 & 4.3):**
   ```bash
   poetry run python experiments/run_all_evals.py
   poetry run python experiments/run_judge.py
   ```
3. **Ablations (Section 4.4):**
   ```bash
   poetry run python -m experiments.ablations.no_qc
   poetry run python -m experiments.ablations.no_sampling --num 100
   poetry run python -m experiments.ablations.no_templating --num 100
   ```
4. **De-identification Benchmark (Section 4.5.1):**
   ```bash
   poetry run python -m experiments.benchmark --config experiments/benchmark/rule_based_config.yaml
   ```
5. **Downstream Model Training (Section 4.5.2):**
   ```bash
   bash experiments/training/run_experiments.sh
   ```
