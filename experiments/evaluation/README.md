# Dataset evaluation

This package produces a single, provenance-rich JSON report for one or more corpora.
It covers diversity/controllability and clinical/textual realism. It does not import
from the existing top-level `evaluation/` package. Dataset schemas are mapped in YAML,
so the same implementation evaluates the synthetic corpus and external comparison
corpora.

## Reproducible run

```bash
poetry install --with experiments
poetry run python -m experiments.evaluation \
  --config experiments/evaluation/example_config.yaml
```

The German spaCy 3.8.0 pipeline is pinned in the experiment dependency group. For a
different language, install a compatible pipeline and set each dataset's
`spacy_model`. The runner refuses to replace an existing report unless
`--overwrite` is passed or `overwrite: true` is configured. Writes are atomic.

Before the paper's frozen run, replace both null model revisions in the YAML with
immutable Hugging Face commit hashes. The report independently records resolved model
commits, package versions, the Git commit/dirty state, all random seeds, dataset
SHA-256 hashes, and the complete configuration with machine-local paths made portable.
The report schema is currently version 2.

Paths in YAML are resolved relative to the YAML file. Supported inputs are JSONL,
JSON arrays (optionally below `records_field`), CSV, line-delimited text, and gzip
variants. Dotted field paths and `*` list traversal allow mappings such as
`diagnoses.*.icd_code`. Set `phi_spans_field: null` only when a corpus genuinely lacks
span annotations. Otherwise, missing fields, non-integer offsets, out-of-range spans,
overlaps, duplicate document IDs, and empty datasets fail fast.

The Gemini judge is disabled by default. Enabling it also requires
`allow_external_api_calls: true`; this is a deliberate data-governance acknowledgement.
With `backend: vertex_ai`, configure `project` or `GCP_PROJECT_ID` and configure
`location` or `GCP_LOCATION`. With `backend: google_ai`, set `GEMINI_API_KEY`. Never
enable it for clinical text that has not been approved for processing by the selected
service. Responses are cached in SQLite so interrupted and frozen runs resume without
silently replacing prior ratings.

## Metric contract

All floating-point scores are stored without display-time rounding. Summaries contain
the arithmetic mean, median, population and sample standard deviations,
5/25/75/95th percentiles, range, and sample size. Population standard deviation is the
main descriptive quantity when the loaded corpus is the complete dataset; the sample
version is retained for downstream sampling analyses. Document-level means use
deterministic percentile-bootstrap confidence intervals. Uniform pair-sample means use
a design-based normal interval with a finite-population correction (and zero sampling
uncertainty when all pairs are enumerated). Sentence-level distributions do not receive
naive sentence bootstrap intervals because sentences within a document are dependent.
“Average” and “mean” are synonyms, so the requested second location summary is the
median. The observation unit is explicit in each output path.

### Per-corpus metrics

- **Composition:** number of documents, total Unicode code points (reported as
  characters), total Unicode-regex word tokens, total PHI spans, and document-level
  distributions of character length, word-token length, and PHI-span count. The PHI
  total is `null`, rather than a misleading partial sum, if any document lacks span
  annotations. Character length and PHI spans per document additionally expose their
  population standard deviations in one compact result object.
- **De-duplication:** raw SHA-256 equality; NFKC/case/whitespace-normalized equality;
  and exact Jaccard verification on sets of within-document word 5-grams. Corpora up
  to the configured limit are checked exhaustively. Larger corpora use deterministic
  256-permutation MinHash LSH only to generate candidates, followed by exact Jaccard
  scoring. The report labels this approximate-recall regime and provides its candidate
  denominator, clusters, pair prevalence, and bounded examples. Examples use zero-based
  input row indices rather than potentially sensitive source document identifiers.
- **ICD controllability:** codes observed in text and configured structured fields are
  reported separately and jointly. Unique per-document code combinations include an
  explicit empty combination. When a catalogue is supplied, exact-entry coverage and
  three-character-category coverage use separate denominators; out-of-catalogue codes
  are retained for audit. This is observed-output coverage, not planned-generation
  coverage unless planned ICD fields are persisted and mapped with `icd_fields`.
- **OPS controllability:** when `extract_ops_from_text` is enabled, German OPS
  procedure codes are regex-extracted from document text. The report gives the number of
  unique observed codes, documents carrying at least one code, and the most frequent
  codes and one-digit chapters. Like text-side ICD extraction, this is a descriptive
  surface measure and shares its regex false-positive caveat.
- **PHI surface diversity:** using each annotation's offset-sliced surface string, label,
  and mention-form field, the report gives the number of unique surface strings and the
  surface-diversity ratio, the entropy of the label distribution, and the entropy of the
  mention-form distribution, with a per-label breakdown of unique surfaces and
  normalized mention-form entropy. This measures how varied the rendered identifiers and
  their chosen forms are, not merely how many spans exist. Reported only when the corpus
  carries PHI span annotations with validated offsets.
- **Negative-trap diversity:** planted non-PHI traps (eponyms, look-alikes, non-PHI
  identifiers, mapped through `negatives_field`) receive the same surface treatment:
  total traps, unique surface strings overall and per trap label, and the per-document
  trap density. Enabled only when `negatives_field` is configured.
- **Structural diversity:** treating each document's ordered section headings
  (`section_sequence_field`) as a pattern, the report gives the number of unique ordered
  section patterns, the section-pattern entropy and its normalization, the unique section
  labels and their entropy, and the per-document section-count distribution. Order is
  preserved and repeated headings are retained, so this captures layout variety that a
  set-valued profile field discards. Enabled only when `section_sequence_field` is set.
- **Aggregate diversity index:** a single caveated harmonic mean of the normalized
  entropies of the content dimensions already computed — every configured profile field
  plus the PHI-label, mention-form, and section-pattern distributions. The harmonic mean
  is dominated by the least diverse dimension, so a collapsed dimension cannot be masked;
  a zero-entropy dimension collapses the index to zero and is listed explicitly. Identity
  fields (surnames, entity identifiers, seeds) are excluded by construction because they
  are never configured as profile fields; their entropy is trivially high and would not
  reflect controllable content diversity. The report records exactly which dimensions
  contributed and which were omitted for the corpus.
- **Lexical statistics:** corpus vocabulary entropy in bits and normalized entropy;
  corpus type-token ratio and its per-document distribution; corpus hapax ratio (unigram
  types occurring exactly once over the vocabulary); per-document trigram novelty (the
  share of a document's distinct trigrams occurring in no other document); empirical
  conditional entropy `H(next token | current token)` from within-document bigrams; top
  unigram through 4-gram frequencies, type counts, and hapax counts; pooled Distinct-1/2;
  bidirectional MTLD with threshold 0.72; Yule's K; and explicit within-document 4-gram
  self-repetition. Self-repetition is the fraction of 4-gram occurrences beyond their
  first occurrence, aggregated with a micro denominator. No n-gram crosses a document
  boundary.
- **Sampled-profile diversity:** each dataset maps persisted control/profile fields as
  categorical, numeric, or set-valued. Marginal reports include missingness, unique
  values, entropy, effective category count, concentration, numeric distributions, and
  set/item coverage. Joint reports count complete and unique profile combinations.
  Every document pair is traversed exactly using a weighted Gower distance:
  categorical mismatch, range-scaled numeric distance, and set Jaccard distance.
  Pairwise missing fields are excluded with weight renormalization and audited. The
  “no close profiles” check passes only if every configured profile is complete, every
  pair is comparable, and no pair is strictly below the preregistered threshold. Exact
  close-pair counts and all per-document nearest-neighbor distances are retained; only
  the descriptive all-pair distance distribution may use a uniform reservoir sample.
  The run fails above the exhaustive limit rather than claiming separation from an
  approximate neighbor search.
- **Style and POS:** spaCy sentence length in non-space, non-punctuation tokens, both
  per sentence and as per-document means. Universal POS frequencies are reported for
  the full tag set and separately for `NOUN`, `PROPN`, `VERB`, `AUX`, and `NUM`, with
  combined noun/proper-noun and verb/auxiliary rates. This avoids silently conflating
  proper nouns or auxiliaries with their requested coarse groups.
- **Pairwise surface similarity:** unordered document pairs are sampled uniformly
  without replacement, or enumerated when feasible. ROUGE-1/2/L F1 uses one shared
  Unicode casefolded tokenizer and no English-only stemmer. BLEU is the mean of both
  sentence-BLEU directions, using SacreBLEU exponential smoothing and effective order
  on the same pretokenized text. This symmetry is necessary because a corpus-diversity
  pair has no natural hypothesis/reference direction.
- **Pairwise semantic similarity:** BERTScore is the mean F1 over both directions on a
  separately reported uniform pair subsample. Documents beyond the configured budget
  use equal head/tail retention and the truncation count is recorded. Cosine similarity
  uses L2-normalized document embeddings. Long documents are divided into fixed
  subword chunks; chunk embeddings are token-count-weighted, mean-pooled, then
  normalized. The mean fields of these score summaries are the average pairwise
  similarities.

Identity fields, row/document IDs, random seeds, names, and output-derived counts must
not be included in profile separation: they can make every profile trivially unique
without demonstrating clinical or controllability diversity. Prefer design-defined
numeric bounds (for example, the generator's allowed age range) to observed ranges; the
loader rejects configured-range violations. Fix profile fields, weights, bounds, and
the close threshold before examining final results.

ROUGE follows the [Google Research reference implementation](https://github.com/google-research/google-research/blob/master/rouge/rouge_scorer.py),
BLEU uses [SacreBLEU](https://github.com/mjpost/sacrebleu), semantic overlap uses
[BERTScore](https://github.com/Tiiiger/bert_score), and embedding cosine follows the
[Sentence Transformers semantic-similarity interface](https://sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html).
MTLD uses the bidirectional form described by
[McCarthy and Jarvis](https://pubmed.ncbi.nlm.nih.gov/20479170/).

### Clinical and textual realism

- **Flesch--Kincaid Grade Level:** evaluated per document and once on the pooled corpus
  using `0.39 W/S + 11.8 Y/W - 15.59`. Word, sentence, syllable, and
  polysyllabic-word counts come from `textstat`, with the configured language used for
  syllabification.
- **SMOG:** evaluated using `1.043 sqrt(P * 30/S) + 3.1291`. Scores available at the
  implementation minimum of three sentences and the validly normed subset of at least
  30 sentences are reported separately. Eligibility counts prevent a corpus dominated
  by short documents from yielding an apparently authoritative aggregate.
- **Gemini clinical-realism judge:** the default model is `gemini-3.7-flash`, with the
  supported thinking level explicitly fixed to `medium`. Each
  uniformly sampled document is evaluated without its dataset name using fixed 1--5
  anchors for structure/genre, clinical plausibility, internal consistency, temporal
  consistency, care-trajectory coherence, discharge-plan completeness, clinical
  language/register, and holistic realism. It also returns major-error and
  synthetic-artifact flags plus controlled failure tags. Documents are never silently
  truncated. Structured output is schema-validated; prompt version/hash, requested and
  resolved model versions, sample design, cache hits, and token usage are recorded.
  Rationales remain in the local cache but are excluded from the report by default to
  reduce copied-text and sensitive-content risk.

Flesch--Kincaid and SMOG are English educational-grade formulas. The
[`textstat` documentation](https://github.com/textstat/textstat#language-support)
states that changing language affects syllabification while these formulas have no
German variant, and that SMOG was normed on 30-sentence samples. Consequently, German
results are complexity proxies, not German school-grade estimates. State this
limitation in the paper; add a validated German measure as a sensitivity analysis rather
than relabelling these values. The model name and supported thinking levels follow the
[official Gemini 3.7 documentation](https://ai.google.dev/gemini-api/docs/models/gemini-3.7-flash).

LLM ratings are measurements from a mutable model service, not ground truth. The final
protocol should freeze the prompt/hash and cached responses, select the human-validation
subset before inspecting judge scores, have at least two clinically qualified raters
score it independently, report inter-human and human--LLM agreement, adjudicate major
disagreements, and repeat the principal analysis with an independently developed prompt
or judge model. Multiple calls to the same model estimate repeat stability only; they
are not independent raters. Statistical summaries first average repetitions within each
document and then treat documents as the analysis unit.

### Cross-corpus metrics

Every unordered corpus pair receives the following comparisons:

- **Lexical coverage gaps:** directional missing-type fraction and missing token-mass
  fraction. Both directions are required because coverage is asymmetric.
- **Lexical shift:** base-2 Jensen-Shannon divergence on the union support of empirical
  unigram and within-document bigram distributions. Values lie in `[0, 1]` bits.
- **Type-level coverage:** Jaccard similarity of unigram vocabularies, plus union and
  intersection sizes.
- **Subword entropy shift:** entropy in bits under one fixed tokenizer, signed
  right-minus-left entropy change, subword-support overlap, and subword JSD. A shared
  tokenizer is essential; entropy values from different tokenizers are not comparable.
- **Maximum mean discrepancy:** biased and unbiased empirical MMD² with an RBF kernel.
  The bandwidth `sigma` is the median nonzero Euclidean distance in a deterministic
  pooled sample, with kernel `exp(-||x-y||² / (2 sigma²))`. The unbiased finite-sample
  estimate can be negative and is therefore preserved; only its separately labelled
  square root is clipped. This follows the kernel two-sample formulation of
  [Gretton et al.](https://www.jmlr.org/papers/v13/gretton12a.html).
- **Gaussian Fréchet distance:** the squared 2-Wasserstein distance and its square root
  between Gaussian summaries of the same fixed document embeddings. Covariances use
  `ddof=1`, symmetric eigendecompositions, PSD clipping, and a reported scale-relative
  diagonal regularizer. This is an embedding-space descriptive distance, not image FID.

MMD and Fréchet use identically sampled documents per corpus, with the selected row-index
list hash in the report. They are descriptive effect sizes, not p-values. If inferential
claims are later made from them, the paper should preregister a permutation or bootstrap
procedure and correct across all tested corpus/metric comparisons.

## Interpretation and paper reporting

Similarity is not synonymous with quality: low pairwise overlap may reflect diversity,
but also incoherence, encoding corruption, or domain mismatch. Report surface and
semantic metrics together, alongside de-duplication examples and corpus-size controls.
For fair comparisons, use the same preprocessing, model revisions, sampling caps, and
random seed for every corpus. Where document lengths differ substantially, supplement
the main result with a length-matched sensitivity analysis; several lexical and neural
metrics remain length-sensitive despite MTLD and explicit truncation accounting.

The JSON hierarchy is:

```text
corpora.<name>.aggregate
corpora.<name>.diversity_and_controllability.{deduplication,icd,ops,lexical,sampled_profile_diversity}
corpora.<name>.diversity_and_controllability.{phi_surface_diversity,negative_trap_diversity,structural_diversity}
corpora.<name>.diversity_and_controllability.aggregate_diversity_index
corpora.<name>.diversity_and_controllability.pairwise_similarity
corpora.<name>.diversity_and_controllability.{style_and_pos,embedding,subword_distribution}
corpora.<name>.clinical_and_textual_realism.{readability,llm_judge}
comparisons.<left>__vs__<right>.{lexical,subword_entropy_shift,embedding_distributions}
```
