# SPARCLING: Synthetic, Provenance-Annotated, Red-teamed, CLINical data Generation

Official repository for the paper:

> **Generating Diverse and Verified Synthetic Clinical Notes with Labels by Construction**  
> Matthias Lehmann, Madeleine Werthebach  
> *GenAI for Health Workshop @ NeurIPS 2026*

---

## Overview

Evaluating and training clinical de-identification models requires benchmarks with three foundational properties:
1. **Exact and complete character-level PHI span annotations**
2. **Surface plausibility** sufficient to embed Protected Health Information (PHI) in realistic clinical context
3. **Pervasive yet controlled diversity** covering varied document styles, clinical specialties, demographic distributions, and adversarial edge cases.

**SPARCLING** factors synthetic clinical note generation into four complementary stages so that PHI labels are **exact by construction** rather than noisily annotated post-hoc:

```text
Stage 1  Structured Attribute Sampling   Sample patient, clinical, document, and meta attributes z_i ~ P(z)
   │                                     independently of the LLM (hierarchical & difficulty-conditioned)
   ▼
Stage 2  Controlled Template Authoring   Mask PHI values (z̃_i) and prompt the generation LLM (M_gen) to author
   │                                     a clinical narrative using typed, inflection-tagged placeholders [[ref ...]]
   ▼
Stage 3  Deterministic Rendering         Substitute morphologically inflected surface forms and record character
   │                                     offsets (x_i, Y_i) = g_fill(t_i, z_i) -> exact spans by construction
   ▼
Stage 4  Dual Quality Control            (1) Deterministic known-PHI leak & negative-trap consistency checks
                                         (2) Adversarial red-teaming LLM (M_qc) on masked text to catch residual PHI
```

---

## Installation

Requires **Python 3.13+** and [Poetry](https://python-poetry.org/) (or standard `pip`).

```bash
# Core generation & red-teaming pipeline
poetry install

# Full suite including evaluation, NER baselines, and model training
poetry install --with dev,experiments
```

### API Configuration (Gemini / Vertex AI)

The generation pipeline, red-team auditor, and LLM realism judge use `google-genai` (`gemini-3.7-flash` by default). Configure either **Google Cloud Vertex AI** (recommended) or **Google AI Studio**:

```bash
# Option A: Vertex AI (Application Default Credentials)
export GCP_PROJECT_ID="your-gcp-project-id"
export GCP_LOCATION="eu"   # optional, defaults to "eu"
# If using a service account key file:
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"

# Option B: Google AI Studio API Key
export GEMINI_API_KEY="your-gemini-api-key"
```

> **Offline mode:** You can test the entire pipeline deterministically without any API credentials using `--dry-run`.

---

## Quick Start: Generating Synthetic Clinical Notes

### 1. Offline Smoke Test (No API Key Needed)

```bash
poetry run python -m generator.main --num-letters 20 --dry-run --output-dir outputs/smoke
```

### 2. Generate a Custom Batch with LLM & Red-Team QC

```bash
poetry run python -m generator.main \
  --num-letters 50 \
  --difficulty hard \
  --output-dir outputs/hard50
```

### 3. Reproduce the Full `SPARCLING-DE` Dataset (Paper Configuration)

To generate the full German discharge-letter benchmark (`1,150` attempted $\rightarrow$ `1,019` verified documents after placeholder validation and dual QC filtering):

```bash
poetry run python -m generator.main --config experiments/config_sparcling_de.yaml
```

*(Alternatively, download the pre-generated **SPARCLING-DE** dataset from Hugging Face and place `dataset.jsonl`, `subsample63.jsonl`, and `review_queue.jsonl` in `experiments/datasets/sparcling-de/`.)*

### CLI Options (`python -m generator.main`)

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--config PATH` | `generator/config.yaml` | Path to YAML configuration file |
| `--num-letters N` | `10` | Number of documents to generate |
| `--output-dir DIR` | `./outputs/default` | Output directory |
| `--dry-run` | `off` | Deterministic offline skeleton author (no API calls) |
| `--style STYLE` | sampled | Force style: `Standard-Klinik`, `Akademisch-Ausführlich`, `Telegrammstil-Kompakt` |
| `--difficulty DIFF` | sampled | Force difficulty: `easy`, `medium`, `hard`, `adversarial` |
| `--length LEN` | sampled | Force length: `kurz`, `mittel`, `ausführlich`, `sehr_lang` |

### Output Artifacts

Each run writes the following files to `--output-dir`:
- `dataset.jsonl` — Verified documents (`doc_id`, `text`, character-exact `spans`, `negatives`, `metadata`, `qc`, and `splits`).
- `review_queue.jsonl` — Documents flagged by deterministic leak checks or the red-team auditor (excluded when `auto: true`).
- `failures.jsonl` — Generation/rendering drops by stage.
- `provenance.json` — Full yield chain, QC exclusion counts by mechanism, deterministic vs. red-team overlap, and per-label breakdowns.
- `diversity.json` — Built-in 6-dimension diversity audit report.

---

## Reproducing Paper Experiments

All paper experiments live under [`experiments/`](experiments/). Install experiment dependencies first (`poetry install --with experiments`).

### 1. Quality Control Analysis (Section 4.1)

The generation run automatically produces `provenance.json` with exact exclusion counts and overlap statistics. To render a human-readable Markdown report of all excluded documents and their offending sentences:

```bash
poetry run python experiments/make_exclusions_md.py --corpus sparcling-de
```

### 2. Corpus Diversity, Controllability & Plausibility (Sections 4.2 & 4.3)

 See [`experiments/evaluation/README.md`](experiments/evaluation/README.md) for full metric specifications.

```bash
# 1. Create the size-matched n=63 subsample (for fair comparison against GraSCCo)
poetry run python experiments/make_subsample63.py --corpus sparcling-de

# 2. Run lexical, structural, profile-separation (Gower), readability (SMOG/FKGL),
#    and neural similarity/distance metrics (Self-BLEU, MTLD, BERTScore, Fréchet, MMD)
poetry run python experiments/run_all_evals.py

# 3. Run the blinded Gemini 3.7 Flash clinical-realism judge (9 dimensions, 1-5 scale)
poetry run python experiments/run_judge.py
```

### 3. Ablation Studies (Section 4.4)

See [`experiments/ablations/README.md`](experiments/ablations/README.md) for details on the three ablations:

```bash
# Ablation 1: No Dual Quality Control (counterfactual analysis on pre-QC corpus)
poetry run python -m experiments.ablations.no_qc

# Ablation 2: No Structured Sampling (LLM invents case + fills placeholders)
poetry run python -m experiments.ablations.no_sampling --num 100

# Ablation 3: No Placeholder Generation (LLM writes free text from unmasked profile + self-annotates)
poetry run python -m experiments.ablations.no_templating --num 100
```

### 4. De-identification Benchmark (Section 4.5.1)

See [`experiments/benchmark/README.md`](experiments/benchmark/README.md). Evaluates six de-identification methods on `SPARCLING-DE` at lenient span matching ($\text{IoU} \ge 0.5$), reporting micro Precision/Recall/F1, per-label recall, difficulty stratification, and clinical over-redaction on planted negative traps (medical eponyms, look-alikes, non-PII IDs):

```bash
# Non-LLM methods (rule-based, Presidio+spaCy, Presidio+Flair, OpenMed PII)
bash experiments/benchmark/run_nonllm.sh rule_based presidio flair openmed

# Frontier LLM (Gemini 3.7 Flash)
poetry run python -m experiments.benchmark --config experiments/benchmark/gemini_config.yaml

# Open-weight local LLM via LM Studio (Gemma 4 E2B)
poetry run python -m experiments.benchmark --config experiments/benchmark/lmstudio_gemma_config.yaml
```

You can also score any external de-identification system's predictions directly using the standalone scorer:

```bash
# Mode A: character-offset span predictions
poetry run python -m experiments.benchmark.scoring.evaluate \
  --gold experiments/datasets/sparcling-de/dataset.jsonl \
  --predictions pred_spans.jsonl \
  --mode spans --matching lenient --output report.json

# Mode B: de-identified text with [REDACTED] markers
poetry run python -m experiments.benchmark.scoring.evaluate \
  --gold experiments/datasets/sparcling-de/dataset.jsonl \
  --predictions deid_texts.jsonl \
  --mode text --matching lenient --output report.json
```

### 5. Downstream Model Training Utility (Section 4.5.2)

See [`experiments/training/README.md`](experiments/training/README.md). Fine-tunes `medBERT.de` (`GerMedBERT/medbert-512`) with sliding-window chunking (`max_length=512`, `stride=128`) to evaluate synthetic-to-real transfer (TSTR), 5-fold cross-validation pretraining + fine-tuning on GraSCCo, synthetic data scaling, and the drop-adversarial ablation:

```bash
# Run the full training matrix (requires CUDA GPU, ~35 mins on a single Tesla T4)
bash experiments/training/run_experiments.sh

# Aggregate results into summary tables
OUT=experiments/results/training poetry run python -m experiments.training.aggregate
```

---

## Label Ontology & Placeholder System

### PII Ontology (25 PHI Labels + 3 Non-PII Negative Traps)

| Tier | Family | Labels |
| :--- | :--- | :--- |
| **Direct Identifiers** (17) | Names | `PATIENT_NAME`, `RELATIVE_NAME`, `STAFF_NAME`, `OTHER_PERSON_NAME` |
| | Dates | `DATE_BIRTH`, `DATE_ADMISSION`, `DATE_DISCHARGE`, `DATE_OTHER` |
| | Location | `ADDRESS` |
| | Contact | `PHONE`, `FAX`, `EMAIL`, `URL` |
| | Identifiers | `ID_INSURANCE` (KVNR), `ID_CASE` (Fallnr.), `ID_PATIENT`, `ID_OTHER` (LANR/BSNR/IK) |
| **Quasi-Identifiers** (8) | Location | `ZIP`, `CITY` |
| | Organizations | `INSTITUTION`, `INSURER` |
| | Demographics | `AGE`, `PROFESSION`, `EMPLOYER`, `NATIONALITY` |
| **Non-PII Negatives** (3) | Decoy Traps | `EPONYM` (e.g., *Morbus Crohn*), `LOOKALIKE` (e.g., *Swan-Ganz-Katheter*), `NONPII_ID` (e.g., *PZN*) |

> **Note:** Non-PII negative traps are never included in the ground-truth PHI span set $Y_i$. They are tracked separately to whitelist benign look-alikes during deterministic leak scanning and to measure clinical over-redaction in downstream de-identifiers.

### Placeholder Grammar

During Stage 2, the LLM authors templates using typed placeholders:
- **Person references:** `[[<ref> form=<form> case=<case>]]`  
  - `form` $\in$ `{first_last, hon_first_last, hon_last, last, first, initial_last, hon_initial, salutation}`
  - `case` $\in$ `{nom, gen, dat, akk}` (German grammatical case)
- **Atomic references:** `[[<ref>]]` or `[[<ref> variant=<variant>]]` (e.g., `[[geburtsdatum variant=textual]]`, `[[alter variant=inline]]`)

The deterministic renderer (`generator/render.py` + `generator/morphology.py`) inflects names (e.g., genitive *-s*) and splits honorific prefixes (*"Herrn "*, *"Dr. med. "*) from the identifying name core so that only the PHI core is emitted as a gold span.

---

## Repository Structure

```text
├── generator/                  # Core SPARCLING generation pipeline
│   ├── main.py                 # CLI entry point (python -m generator.main)
│   ├── pipeline.py             # Per-document orchestration (Stages 1–4)
│   ├── config.yaml             # Default pipeline configuration
│   ├── config_schema.py        # Typed hyperparameter schema (Table 11–14)
│   ├── profile_sampler.py      # Stage 1: Hierarchical attribute & case sampling
│   ├── profile_data.py         # Document controls, section & style distributions
│   ├── name_data.py            # Origin-weighted demographic name pools
│   ├── specialty_data.py       # Specialty-conditioned clinical pools
│   ├── clinical_seeds.py       # ICD-10-GM & OPS catalog loaders
│   ├── eponym_data.py          # Curated medical eponyms & look-alike traps
│   ├── atom_samplers.py        # Format-varied atomic PII samplers
│   ├── skeleton_prompt.py      # Stage 2: German system & user prompt templates
│   ├── llm_client.py           # Cached Gemini client (Vertex AI / Google AI Studio)
│   ├── render.py               # Stage 3: Deterministic placeholder rendering & leak scan
│   ├── morphology.py           # German grammatical case & mention-form inflection
│   ├── typo_injection.py       # Post-render adversarial typo perturbations
│   ├── red_team.py             # Stage 4: Adversarial LLM residual-PHI auditor
│   ├── splits.py               # Domain, difficulty, and surface split assignment
│   ├── diversity.py            # Built-in 6-dimension diversity audit
│   ├── exporter.py             # JSONL & provenance report serialization
│   └── dry_run.py              # Offline deterministic template author
├── data/
│   ├── icd10_codes.txt         # ICD-10-GM diagnosis catalog
│   └── ops_codes.txt           # OPS procedure catalog
├── experiments/                # Paper evaluation, ablations, benchmark & training
│   ├── config_sparcling_de.yaml
│   ├── config_eval_sparcling_de.yaml
│   ├── config_eval_en_baselines.yaml
│   ├── evaluation/             # Corpus diversity, realism & LLM judge suite
│   ├── ablations/              # No-QC, No-Sampling, and No-Templating ablations
│   ├── benchmark/              # De-identification benchmark & scoring package
│   └── training/               # medBERT.de fine-tuning & transfer harness
├── scripts/
│   ├── inspect_output.py       # Format dataset.jsonl into readable markdown
│   ├── review_flagged.py       # Interactive CLI for reviewing QC-flagged docs
│   └── run_red_team.py         # Standalone red-team runner on existing outputs
├── tests/                      # Unit & integration test suite (pytest)
└── pyproject.toml              # Dependencies & project metadata
```

---

## Running Tests

```bash
poetry run pytest -q
```

---

## Citation

If you use **SPARCLING** or **SPARCLING-DE** in your research, please cite our paper:

```bibtex
@inproceedings{lehmann2026sparcling,
  title     = {Generating Diverse and Verified Synthetic Clinical Notes with Labels by Construction},
  author    = {Lehmann, Matthias and Werthebach, Madeleine},
  booktitle = {GenAI for Health Workshop at NeurIPS 2026},
  year      = {2026}
}
```
