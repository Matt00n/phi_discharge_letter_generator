"""End-to-end orchestration and provenance-rich report serialization."""

from __future__ import annotations

import hashlib
import importlib.metadata
import itertools
import json
import logging
import os
import platform
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from experiments.evaluation.aggregate import aggregate_metrics, icd_metrics, ops_metrics
from experiments.evaluation.comparison import (
    embedding_distribution_comparison,
    lexical_comparison,
    subword_comparison,
)
from experiments.evaluation.composite import aggregate_diversity_index
from experiments.evaluation.config import EvaluationConfig
from experiments.evaluation.corpus import load_corpus
from experiments.evaluation.deduplication import deduplication_metrics
from experiments.evaluation.lexical import lexical_metrics
from experiments.evaluation.llm_judge import llm_judge_metrics
from experiments.evaluation.model_metrics import (
    DocumentEmbedder,
    EmbeddingResult,
    embedding_pair_similarity,
    linguistic_metrics,
    subword_counts,
)
from experiments.evaluation.phi_surface import (
    negative_trap_metrics,
    phi_surface_metrics,
)
from experiments.evaluation.profiles import profile_diversity_metrics
from experiments.evaluation.realism import readability_metrics
from experiments.evaluation.similarity import (
    bertscore_metrics,
    overlap_metrics,
    sample_document_pairs,
)
from experiments.evaluation.structural import structural_metrics

LOGGER = logging.getLogger(__name__)


def _stable_seed(base_seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _as_float(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) else None


def _diversity_index_components(
    profile_report: dict[str, Any],
    phi_surface_report: dict[str, Any],
    structural_report: dict[str, Any],
) -> dict[str, float | None]:
    """Collect normalized-entropy dimensions for the composite diversity index.

    Only content dimensions are included: every configured profile field plus
    the PHI-label, mention-form, and section-pattern distributions. Identity
    fields are excluded because the configuration deliberately omits them from
    ``profile_fields``.
    """

    components: dict[str, float | None] = {}
    profile_fields = profile_report.get("profile_fields")
    if isinstance(profile_fields, dict):
        for name, field_report in profile_fields.items():
            if isinstance(field_report, dict):
                components[f"profile:{name}"] = _as_float(
                    field_report.get("normalized_entropy")
                )
    if phi_surface_report.get("status") == "computed":
        labels = phi_surface_report.get("labels", {})
        forms = phi_surface_report.get("mention_forms", {})
        if isinstance(labels, dict):
            components["phi_labels"] = _as_float(labels.get("normalized_entropy"))
        if isinstance(forms, dict):
            components["mention_forms"] = _as_float(forms.get("normalized_entropy"))
    if structural_report.get("status") == "computed":
        components["section_patterns"] = _as_float(
            structural_report.get("section_pattern_normalized_entropy")
        )
    return components


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _portable_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return f"<external>/{resolved.name}"


def _report_configuration(config: EvaluationConfig) -> dict[str, object]:
    serialized = config.model_dump(mode="json")
    serialized["output_path"] = _portable_path(config.output_path)
    serialized["llm_judge"]["cache_path"] = _portable_path(config.llm_judge.cache_path)
    if serialized["llm_judge"]["project"]:
        serialized["llm_judge"]["project"] = "<configured>"
    for serialized_dataset, dataset in zip(
        serialized["datasets"], config.datasets, strict=True
    ):
        serialized_dataset["path"] = _portable_path(dataset.path)
        if dataset.icd_catalog_path:
            serialized_dataset["icd_catalog_path"] = _portable_path(
                dataset.icd_catalog_path
            )
    return serialized


def _git_provenance() -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "working_tree_dirty": dirty}
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"commit": None, "working_tree_dirty": None}


def _package_versions() -> dict[str, str | None]:
    packages = [
        "bert-score",
        "datasketch",
        "google-genai",
        "numpy",
        "pydantic",
        "rouge-score",
        "sacrebleu",
        "scipy",
        "sentence-transformers",
        "spacy",
        "textstat",
        "transformers",
    ]
    result: dict[str, str | None] = {}
    for package in packages:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def _optional(
    operation: str,
    function: Any,
    *args: Any,
    strict: bool,
    **kwargs: Any,
) -> Any:
    try:
        return function(*args, **kwargs)
    except (ImportError, ModuleNotFoundError, OSError, RuntimeError) as exc:
        if strict:
            raise RuntimeError(f"{operation} failed: {exc}") from exc
        return {"status": "unavailable", "reason": str(exc)}


def _write_report(path: Path, report: dict[str, object], *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing report: {path}; pass --overwrite or "
            "set overwrite: true"
        )
    serialized = json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.write("\n")
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def run_evaluation(
    config: EvaluationConfig,
    *,
    output_path: Path | None = None,
    overwrite: bool | None = None,
) -> dict[str, object]:
    """Execute all configured metrics and atomically write one JSON report."""

    LOGGER.info("Loading %d configured corpus/corpora", len(config.datasets))
    corpora = [load_corpus(dataset) for dataset in config.datasets]
    report: dict[str, object] = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "configuration": _report_configuration(config),
        "provenance": {
            "python": sys.version,
            "platform": platform.platform(),
            "git": _git_provenance(),
            "package_versions": _package_versions(),
            "inputs": {
                corpus.name: {
                    "path": _portable_path(corpus.source_path),
                    "sha256": _file_sha256(corpus.source_path),
                }
                for corpus in corpora
            },
        },
        "interpretation_notes": [
            (
                "Pairwise similarities are descriptive diversity proxies: lower "
                "overlap usually indicates greater diversity, but can also reflect "
                "noise or domain shift."
            ),
            (
                "Cross-corpus distances are effect sizes in the configured "
                "representation, not distribution-free significance tests."
            ),
            (
                "All n-grams are computed within documents; no n-gram crosses a "
                "document boundary."
            ),
            (
                "Document-level mean confidence intervals use a percentile bootstrap; "
                "pair-sample means use design-based finite-population intervals."
            ),
            (
                "Flesch--Kincaid and SMOG are English-calibrated formula proxies; "
                "non-English syllabification does not validate grade-level semantics."
            ),
            (
                "LLM-judge ratings are model-based measurements requiring human "
                "calibration and a judge-sensitivity analysis before inferential "
                "claims."
            ),
        ],
        "reproducibility_warnings": [
            message
            for message, revision in (
                (
                    "embedding_revision is unpinned; use the resolved commit recorded "
                    "in this report for the frozen paper configuration",
                    config.models.embedding_revision,
                ),
                (
                    "bertscore_revision is unpinned; use the resolved commit recorded "
                    "in this report for the frozen paper configuration",
                    config.models.bertscore_revision,
                ),
            )
            if config.models.enabled and revision is None
        ],
        "corpora": {},
        "comparisons": {},
    }

    pairs_by_corpus: dict[str, list[tuple[int, int]]] = {}
    corpus_reports: dict[str, dict[str, Any]] = {}
    for corpus in corpora:
        LOGGER.info(
            "Computing deterministic metrics for %s (%d documents)",
            corpus.name,
            len(corpus.documents),
        )
        corpus_seed = _stable_seed(config.random_seed, corpus.name)
        pairs, pair_metadata = sample_document_pairs(
            len(corpus.documents), config.pair_sample_size, seed=corpus_seed
        )
        pairs_by_corpus[corpus.name] = pairs
        profile_report = profile_diversity_metrics(
            corpus, config.profile_diversity, seed=corpus_seed + 3
        )
        phi_surface_report = phi_surface_metrics(
            corpus, max_reported_labels=config.top_k_ngrams
        )
        negative_trap_report = negative_trap_metrics(
            corpus, max_reported_labels=config.top_k_ngrams
        )
        structural_report = structural_metrics(
            corpus, max_reported_patterns=config.top_k_ngrams
        )
        composite_report = aggregate_diversity_index(
            _diversity_index_components(
                profile_report, phi_surface_report, structural_report
            )
        )
        corpus_reports[corpus.name] = {
            "aggregate": aggregate_metrics(
                corpus,
                bootstrap_samples=config.bootstrap_samples,
                confidence_level=config.confidence_level,
                seed=corpus_seed,
                casefold=config.lexical_casefold,
            ),
            "diversity_and_controllability": {
                "deduplication": deduplication_metrics(
                    corpus, config.deduplication, seed=corpus_seed
                ),
                "icd": icd_metrics(corpus, top_k=config.top_k_ngrams),
                "ops": ops_metrics(corpus, top_k=config.top_k_ngrams),
                "lexical": lexical_metrics(
                    corpus,
                    top_k=config.top_k_ngrams,
                    casefold=config.lexical_casefold,
                    bootstrap_samples=config.bootstrap_samples,
                    confidence_level=config.confidence_level,
                    seed=corpus_seed,
                ),
                "sampled_profile_diversity": profile_report,
                "phi_surface_diversity": phi_surface_report,
                "negative_trap_diversity": negative_trap_report,
                "structural_diversity": structural_report,
                "aggregate_diversity_index": composite_report,
                "pairwise_similarity": {
                    "pair_sample": pair_metadata,
                    "rouge_and_bleu": _optional(
                        "ROUGE/BLEU evaluation",
                        overlap_metrics,
                        corpus,
                        pairs,
                        confidence_level=config.confidence_level,
                        strict=config.strict_optional_dependencies,
                    ),
                },
            },
            "clinical_and_textual_realism": {
                "readability": (
                    _optional(
                        "readability evaluation",
                        readability_metrics,
                        corpus,
                        bootstrap_samples=config.bootstrap_samples,
                        confidence_level=config.confidence_level,
                        seed=corpus_seed + 4,
                        strict=config.strict_optional_dependencies,
                    )
                    if config.readability.enabled
                    else {"status": "disabled"}
                ),
                "llm_judge": llm_judge_metrics(
                    corpus,
                    config.llm_judge,
                    bootstrap_samples=config.bootstrap_samples,
                    confidence_level=config.confidence_level,
                    seed=corpus_seed + 5,
                ),
            },
        }
    report["corpora"] = corpus_reports

    embeddings: dict[str, object] = {}
    subwords: dict[str, Counter[int]] = {}
    if config.models.enabled:
        LOGGER.info("Initializing configured neural models")
        embedder = _optional(
            "embedding model initialization",
            DocumentEmbedder,
            config.models,
            strict=config.strict_optional_dependencies,
        )
        embedder_available = isinstance(embedder, DocumentEmbedder)
        for corpus in corpora:
            corpus_seed = _stable_seed(config.random_seed, corpus.name)
            diversity = corpus_reports[corpus.name]["diversity_and_controllability"]
            pairwise = diversity["pairwise_similarity"]
            pairwise["bertscore"] = _optional(
                "BERTScore evaluation",
                bertscore_metrics,
                corpus,
                pairs_by_corpus[corpus.name],
                config.models,
                confidence_level=config.confidence_level,
                seed=corpus_seed + 1,
                strict=config.strict_optional_dependencies,
            )
            diversity["style_and_pos"] = _optional(
                "spaCy linguistic evaluation",
                linguistic_metrics,
                corpus,
                bootstrap_samples=config.bootstrap_samples,
                confidence_level=config.confidence_level,
                seed=corpus_seed + 2,
                strict=config.strict_optional_dependencies,
            )
            subword_result = _optional(
                "subword distribution evaluation",
                subword_counts,
                corpus,
                config.models,
                strict=config.strict_optional_dependencies,
            )
            if isinstance(subword_result, tuple):
                subwords[corpus.name], subword_metadata = subword_result
                diversity["subword_distribution"] = subword_metadata
            else:
                diversity["subword_distribution"] = subword_result
            if embedder_available:
                embedding_result = _optional(
                    "document embedding evaluation",
                    embedder.encode,
                    corpus,
                    strict=config.strict_optional_dependencies,
                )
                if isinstance(embedding_result, EmbeddingResult):
                    embeddings[corpus.name] = embedding_result.values
                    diversity["embedding"] = embedding_result.metadata
                    pairwise["embedding_cosine"] = embedding_pair_similarity(
                        embedding_result.values,
                        pairs_by_corpus[corpus.name],
                        confidence_level=config.confidence_level,
                        total_possible_pairs=(
                            len(corpus.documents) * (len(corpus.documents) - 1) // 2
                        ),
                    )
                else:
                    diversity["embedding"] = embedding_result
                    pairwise["embedding_cosine"] = embedding_result
            else:
                diversity["embedding"] = embedder
                pairwise["embedding_cosine"] = embedder
    else:
        for corpus in corpora:
            diversity = corpus_reports[corpus.name]["diversity_and_controllability"]
            diversity["style_and_pos"] = {"status": "disabled"}
            diversity["subword_distribution"] = {"status": "disabled"}
            diversity["embedding"] = {"status": "disabled"}
            pairwise = diversity["pairwise_similarity"]
            pairwise["bertscore"] = {"status": "disabled"}
            pairwise["embedding_cosine"] = {"status": "disabled"}

    comparison_reports: dict[str, dict[str, object]] = {}
    for left, right in itertools.combinations(corpora, 2):
        LOGGER.info("Comparing corpora %s and %s", left.name, right.name)
        comparison_name = f"{left.name}__vs__{right.name}"
        comparison: dict[str, object] = {
            "left": left.name,
            "right": right.name,
            "lexical": lexical_comparison(
                left, right, casefold=config.lexical_casefold
            ),
        }
        if left.name in subwords and right.name in subwords:
            comparison["subword_entropy_shift"] = subword_comparison(
                subwords[left.name], subwords[right.name]
            )
        else:
            comparison["subword_entropy_shift"] = {"status": "unavailable"}
        if left.name in embeddings and right.name in embeddings:
            comparison["embedding_distributions"] = embedding_distribution_comparison(
                left,
                right,
                embeddings,
                maximum_documents=config.max_distribution_documents,
                bandwidth_sample_size=config.bandwidth_sample_size,
                seed=config.random_seed,
            )
        else:
            comparison["embedding_distributions"] = {"status": "unavailable"}
        comparison_reports[comparison_name] = comparison
    report["comparisons"] = comparison_reports

    destination = output_path or config.output_path
    report["written_to"] = _portable_path(destination)
    _write_report(
        destination,
        report,
        overwrite=config.overwrite if overwrite is None else overwrite,
    )
    LOGGER.info("Wrote evaluation report to %s", destination)
    return report
