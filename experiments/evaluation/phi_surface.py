"""PHI surface-form diversity and negative-trap diversity metrics.

These statistics describe how varied the *rendered* mentions are, independent of
any schema: the distinct surface strings each label takes, the entropy of the
mention forms chosen per label, and the analogous variety of the negative traps
(eponyms, look-alikes, non-PHI identifiers) planted to probe over-redaction.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from experiments.evaluation.corpus import Corpus, SpanSurface
from experiments.evaluation.lexical import entropy_bits
from experiments.evaluation.statistics import numeric_summary


def _normalized_entropy(entropy: float, support: int) -> float:
    """Entropy divided by its maximum for the observed support."""

    return entropy / math.log2(support) if support > 1 else 0.0


def _label_surface_row(label: str, spans: list[SpanSurface]) -> dict[str, object]:
    surfaces = [span.surface for span in spans if span.surface]
    unique_surfaces = len(set(surfaces))
    forms = Counter(span.form for span in spans if span.form)
    form_support = len(forms)
    form_entropy = entropy_bits(forms)
    return {
        "label": label,
        "annotations": len(spans),
        "annotations_with_surface_string": len(surfaces),
        "unique_surface_strings": unique_surfaces,
        "surface_diversity_ratio": (
            unique_surfaces / len(surfaces) if surfaces else None
        ),
        "annotations_with_mention_form": sum(forms.values()),
        "unique_mention_forms": form_support,
        "mention_form_entropy_bits": form_entropy,
        "mention_form_normalized_entropy": _normalized_entropy(
            form_entropy, form_support
        ),
    }


def phi_surface_metrics(
    corpus: Corpus, *, max_reported_labels: int
) -> dict[str, object]:
    """Measure surface-string variety and mention-form entropy by PHI label."""

    if not any(document.phi_span_count is not None for document in corpus.documents):
        return {
            "status": "not_configured",
            "reason": "the dataset has no configured PHI span annotations",
        }

    spans_by_label: defaultdict[str, list[SpanSurface]] = defaultdict(list)
    all_spans: list[SpanSurface] = []
    for document in corpus.documents:
        for span in document.phi_spans:
            all_spans.append(span)
            spans_by_label[span.label or "<unlabeled>"].append(span)

    surfaces = [span.surface for span in all_spans if span.surface]
    unique_surfaces = len(set(surfaces))
    label_counts = Counter(span.label for span in all_spans if span.label is not None)
    label_entropy = entropy_bits(label_counts)
    form_counts = Counter(span.form for span in all_spans if span.form)
    form_entropy = entropy_bits(form_counts)

    per_label = sorted(
        (_label_surface_row(label, spans) for label, spans in spans_by_label.items()),
        key=lambda row: (-int(row["annotations"]), str(row["label"])),
    )
    return {
        "status": "computed",
        "definition": (
            "surface string = document text sliced by the annotated offsets; "
            "mention form = the configured rendering-strategy field per span"
        ),
        "total_annotations": len(all_spans),
        "annotations_with_surface_string": len(surfaces),
        "unique_surface_strings": unique_surfaces,
        "surface_diversity_ratio": (
            unique_surfaces / len(surfaces) if surfaces else None
        ),
        "labels": {
            "unique_labels": len(label_counts),
            "label_distribution_entropy_bits": label_entropy,
            "normalized_entropy": _normalized_entropy(label_entropy, len(label_counts)),
        },
        "mention_forms": {
            "annotations_with_mention_form": sum(form_counts.values()),
            "unique_mention_forms": len(form_counts),
            "entropy_bits": form_entropy,
            "normalized_entropy": _normalized_entropy(form_entropy, len(form_counts)),
        },
        "spans_per_document": numeric_summary(
            [len(document.phi_spans) for document in corpus.documents]
        ),
        "per_label": per_label[:max_reported_labels],
        "per_label_truncated": len(per_label) > max_reported_labels,
    }


def negative_trap_metrics(
    corpus: Corpus, *, max_reported_labels: int
) -> dict[str, object]:
    """Measure the variety and density of planted negative (non-PHI) traps."""

    if not any(document.negative_spans for document in corpus.documents):
        return {
            "status": "not_configured",
            "reason": (
                "no negative-trap spans were loaded; set negatives_field to enable "
                "this metric"
            ),
        }

    spans_by_label: defaultdict[str, list[SpanSurface]] = defaultdict(list)
    all_spans: list[SpanSurface] = []
    for document in corpus.documents:
        for span in document.negative_spans:
            all_spans.append(span)
            spans_by_label[span.label or "<unlabeled>"].append(span)

    surfaces = [span.surface for span in all_spans if span.surface]
    per_label = sorted(
        (
            {
                "label": label,
                "traps": len(spans),
                "unique_surface_strings": len(
                    {span.surface for span in spans if span.surface}
                ),
            }
            for label, spans in spans_by_label.items()
        ),
        key=lambda row: (-int(row["traps"]), str(row["label"])),
    )
    return {
        "status": "computed",
        "total_traps": len(all_spans),
        "unique_surface_strings": len(set(surfaces)),
        "unique_trap_labels": len(spans_by_label),
        "traps_per_document": numeric_summary(
            [len(document.negative_spans) for document in corpus.documents]
        ),
        "per_label": per_label[:max_reported_labels],
        "per_label_truncated": len(per_label) > max_reported_labels,
    }
