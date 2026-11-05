"""Aggregate corpus and ICD controllability statistics."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from experiments.evaluation.corpus import ICD_PATTERN, Corpus
from experiments.evaluation.statistics import numeric_summary
from experiments.evaluation.text import word_tokens


def aggregate_metrics(
    corpus: Corpus,
    *,
    bootstrap_samples: int,
    confidence_level: float,
    seed: int,
    casefold: bool,
) -> dict[str, object]:
    """Compute counts and distributions without model dependencies."""

    character_counts = [len(document.text) for document in corpus.documents]
    token_counts = [
        len(word_tokens(document.text, casefold=casefold))
        for document in corpus.documents
    ]
    available_phi_counts = [
        document.phi_span_count
        for document in corpus.documents
        if document.phi_span_count is not None
    ]
    all_phi_available = len(available_phi_counts) == len(corpus.documents)
    summary_kwargs = {
        "bootstrap_samples": bootstrap_samples,
        "confidence_level": confidence_level,
    }
    document_length_summary = numeric_summary(
        character_counts, seed=seed + 1, **summary_kwargs
    )
    phi_count_summary = numeric_summary(
        available_phi_counts, seed=seed + 3, **summary_kwargs
    )
    return {
        "number_of_documents": len(corpus.documents),
        "total_characters": sum(character_counts),
        "total_word_tokens": sum(token_counts),
        "total_phi_spans": sum(available_phi_counts) if all_phi_available else None,
        "phi_span_annotations": {
            "available_for_all_documents": all_phi_available,
            "documents_with_span_annotations": len(available_phi_counts),
        },
        "document_length_characters": document_length_summary,
        "document_length_word_tokens": numeric_summary(
            token_counts, seed=seed + 2, **summary_kwargs
        ),
        "phi_spans_per_document": phi_count_summary,
        "standard_deviations_across_documents": {
            "definition": "population standard deviation across documents",
            "document_length_characters": document_length_summary["std_population"],
            "phi_spans_per_document": phi_count_summary["std_population"],
        },
    }


def _load_icd_catalog(path: str) -> set[str]:
    codes: set[str] = set()
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            match = ICD_PATTERN.search(line.upper())
            if match:
                codes.add(match.group(1).upper())
    if not codes:
        raise ValueError(f"ICD catalogue contains no recognizable codes: {path}")
    return codes


def _code_category(code: str) -> str:
    return code[:3]


def ops_metrics(corpus: Corpus, *, top_k: int) -> dict[str, object]:
    """Report OPS procedure codes regex-extracted from generated text."""

    if not any(document.ops_text_codes for document in corpus.documents):
        return {
            "status": "not_configured",
            "reason": (
                "no OPS codes were extracted; set extract_ops_from_text to enable "
                "this metric"
            ),
        }
    observed_codes = {
        code for document in corpus.documents for code in document.ops_text_codes
    }
    chapter_counts = Counter(
        code.split("-", 1)[0]
        for document in corpus.documents
        for code in document.ops_text_codes
    )
    code_counts = Counter(
        code for document in corpus.documents for code in document.ops_text_codes
    )
    return {
        "status": "computed",
        "source": "regex extraction from document text",
        "unique_codes_observed_in_text": len(observed_codes),
        "documents_with_at_least_one_code": sum(
            bool(document.ops_text_codes) for document in corpus.documents
        ),
        "top_codes": [
            {"code": code, "document_count": count}
            for code, count in code_counts.most_common(top_k)
        ],
        "top_chapters": [
            {"chapter": chapter, "occurrences": count}
            for chapter, count in chapter_counts.most_common(top_k)
        ],
    }


def icd_metrics(corpus: Corpus, *, top_k: int) -> dict[str, object]:
    """Report observed codes, combinations, and optional catalogue coverage."""

    text_codes = {code for doc in corpus.documents for code in doc.icd_text_codes}
    field_codes = {code for doc in corpus.documents for code in doc.icd_field_codes}
    observed_codes = text_codes | field_codes
    combinations = Counter(tuple(sorted(doc.icd_codes)) for doc in corpus.documents)
    nonempty_combinations = Counter(
        {
            combination: count
            for combination, count in combinations.items()
            if combination
        }
    )
    category_counts = Counter(
        _code_category(code)
        for document in corpus.documents
        for code in document.icd_codes
    )

    catalogue_result: dict[str, object] | None = None
    if corpus.icd_catalog_path:
        catalog_codes = _load_icd_catalog(corpus.icd_catalog_path)
        catalog_categories = {_code_category(code) for code in catalog_codes}
        observed_categories = {_code_category(code) for code in observed_codes}
        catalogue_result = {
            "path": corpus.icd_catalog_path,
            "catalogue_unique_codes": len(catalog_codes),
            "catalogue_unique_three_character_categories": len(catalog_categories),
            "exact_code_coverage": len(observed_codes & catalog_codes)
            / len(catalog_codes),
            "three_character_category_coverage": len(
                observed_categories & catalog_categories
            )
            / len(catalog_categories),
            "observed_codes_absent_from_catalogue": sorted(
                observed_codes - catalog_codes
            ),
        }

    return {
        "unique_codes_observed": len(observed_codes),
        "unique_codes_observed_in_text": len(text_codes),
        "unique_codes_observed_in_structured_fields": len(field_codes),
        "documents_with_at_least_one_code": sum(
            bool(document.icd_codes) for document in corpus.documents
        ),
        "unique_document_code_combinations_including_empty": len(combinations),
        "unique_nonempty_document_code_combinations": len(nonempty_combinations),
        "top_code_combinations": [
            {"codes": list(combination), "document_count": count}
            for combination, count in combinations.most_common(top_k)
        ],
        "top_three_character_categories": [
            {"category": category, "occurrences": count}
            for category, count in category_counts.most_common(top_k)
        ],
        "catalogue_coverage": catalogue_result,
    }
