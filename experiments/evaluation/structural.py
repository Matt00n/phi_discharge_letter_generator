"""Document-structure diversity metrics over ordered section sequences."""

from __future__ import annotations

import math
from collections import Counter

from experiments.evaluation.corpus import Corpus
from experiments.evaluation.lexical import entropy_bits
from experiments.evaluation.statistics import numeric_summary


def _normalized_entropy(entropy: float, support: int) -> float:
    return entropy / math.log2(support) if support > 1 else 0.0


def structural_metrics(
    corpus: Corpus, *, max_reported_patterns: int
) -> dict[str, object]:
    """Report ordered section-pattern variety, entropy, and section counts."""

    sequences = [
        document.section_sequence
        for document in corpus.documents
        if document.section_sequence is not None
    ]
    if not sequences:
        return {
            "status": "not_configured",
            "reason": (
                "no section sequences were loaded; set section_sequence_field to "
                "enable this metric"
            ),
        }

    pattern_counts = Counter(sequences)
    section_labels = Counter(section for sequence in sequences for section in sequence)
    pattern_entropy = entropy_bits(pattern_counts)
    label_entropy = entropy_bits(section_labels)
    top_patterns = [
        {"pattern": list(pattern), "document_count": count}
        for pattern, count in pattern_counts.most_common(max_reported_patterns)
    ]
    return {
        "status": "computed",
        "definition": (
            "a section pattern is the ordered tuple of section headings for one "
            "document; order is preserved and repeated headings are retained"
        ),
        "documents_with_sections": len(sequences),
        "unique_ordered_section_patterns": len(pattern_counts),
        "section_pattern_entropy_bits": pattern_entropy,
        "section_pattern_normalized_entropy": _normalized_entropy(
            pattern_entropy, len(pattern_counts)
        ),
        "unique_section_labels": len(section_labels),
        "section_label_entropy_bits": label_entropy,
        "section_label_normalized_entropy": _normalized_entropy(
            label_entropy, len(section_labels)
        ),
        "sections_per_document": numeric_summary(
            [len(sequence) for sequence in sequences]
        ),
        "top_section_patterns": top_patterns,
        "top_section_patterns_truncated": len(pattern_counts) > max_reported_patterns,
    }
