"""Diversity and exact separation checks for persisted sampling profiles."""

from __future__ import annotations

import heapq
import itertools
import math
import random
from collections import Counter

from experiments.evaluation.config import ProfileDiversityConfig, ProfileFieldConfig
from experiments.evaluation.corpus import Corpus
from experiments.evaluation.statistics import numeric_summary

type ProfileValue = str | float | tuple[str, ...] | None


def _entropy(counter: Counter[object]) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return -sum(
        (count / total) * math.log2(count / total) for count in counter.values()
    )


def _frequency_table(
    counter: Counter[object], *, include_labels: bool, maximum: int
) -> list[dict[str, object]]:
    ordered = sorted(counter.items(), key=lambda item: (-item[1], repr(item[0])))
    rows: list[dict[str, object]] = []
    for rank, (value, count) in enumerate(ordered[:maximum], start=1):
        row: dict[str, object] = {"rank": rank, "count": count}
        if include_labels:
            row["value"] = list(value) if isinstance(value, tuple) else value
        rows.append(row)
    return rows


def _categorical_summary(
    values: list[ProfileValue], field: ProfileFieldConfig, *, maximum: int
) -> dict[str, object]:
    observed = [value for value in values if value is not None]
    counts = Counter(observed)
    entropy = _entropy(counts)
    unique = len(counts)
    return {
        "kind": field.kind,
        "weight": field.weight,
        "observed_documents": len(observed),
        "missing_documents": len(values) - len(observed),
        "missing_rate": (len(values) - len(observed)) / len(values),
        "unique_values": unique,
        "entropy_bits": entropy,
        "normalized_entropy": entropy / math.log2(unique) if unique > 1 else 0.0,
        "effective_number_of_values": 2.0**entropy,
        "largest_value_prevalence": max(counts.values()) / len(observed)
        if observed
        else None,
        "value_labels_included": field.include_value_labels_in_report,
        "frequency_table": _frequency_table(
            counts,
            include_labels=field.include_value_labels_in_report,
            maximum=maximum,
        ),
        "frequency_table_truncated": unique > maximum,
    }


def _numeric_summary(
    values: list[ProfileValue], field: ProfileFieldConfig, *, seed: int
) -> dict[str, object]:
    observed = [float(value) for value in values if value is not None]
    configured = field.numeric_min is not None
    if configured:
        scale_min = float(field.numeric_min)
        scale_max = float(field.numeric_max)
    elif observed:
        scale_min = min(observed)
        scale_max = max(observed)
    else:
        scale_min = scale_max = None
    return {
        "kind": field.kind,
        "weight": field.weight,
        "observed_documents": len(observed),
        "missing_documents": len(values) - len(observed),
        "missing_rate": (len(values) - len(observed)) / len(values),
        "unique_values": len(set(observed)),
        "distribution": numeric_summary(observed, seed=seed),
        "gower_scaling": {
            "minimum": scale_min,
            "maximum": scale_max,
            "source": "configured" if configured else "observed_in_this_corpus",
        },
    }


def _set_summary(
    values: list[ProfileValue], field: ProfileFieldConfig, *, maximum: int
) -> dict[str, object]:
    result = _categorical_summary(values, field, maximum=maximum)
    observed = [value for value in values if isinstance(value, tuple)]
    items = Counter(item for value in observed for item in value)
    result.update(
        {
            "unique_set_combinations": result.pop("unique_values"),
            "mean_set_size": (
                sum(len(value) for value in observed) / len(observed)
                if observed
                else None
            ),
            "unique_items": len(items),
            "item_entropy_bits": _entropy(items),
            "item_frequency_table": _frequency_table(
                items,
                include_labels=field.include_value_labels_in_report,
                maximum=maximum,
            ),
            "item_frequency_table_truncated": len(items) > maximum,
        }
    )
    return result


def _numeric_ranges(
    corpus: Corpus,
) -> list[tuple[float, float] | None]:
    ranges: list[tuple[float, float] | None] = []
    for field_index, field in enumerate(corpus.profile_schema):
        if field.kind != "numeric":
            ranges.append(None)
            continue
        observed = [
            float(document.profile[field_index])
            for document in corpus.documents
            if document.profile[field_index] is not None
        ]
        if field.numeric_min is not None and field.numeric_max is not None:
            ranges.append((field.numeric_min, field.numeric_max))
        elif observed:
            ranges.append((min(observed), max(observed)))
        else:
            ranges.append(None)
    return ranges


def _gower_distance(
    left: tuple[ProfileValue, ...],
    right: tuple[ProfileValue, ...],
    fields: tuple[ProfileFieldConfig, ...],
    numeric_ranges: list[tuple[float, float] | None],
) -> tuple[float | None, int]:
    weighted_distance = 0.0
    available_weight = 0.0
    dimensions = 0
    for index, field in enumerate(fields):
        left_value = left[index]
        right_value = right[index]
        if left_value is None or right_value is None:
            continue
        if field.kind == "numeric":
            bounds = numeric_ranges[index]
            if bounds is None:
                continue
            scale = bounds[1] - bounds[0]
            if scale <= 0.0:
                distance = 0.0
            else:
                distance = min(abs(float(left_value) - float(right_value)) / scale, 1.0)
        elif field.kind == "set":
            left_set = set(left_value)
            right_set = set(right_value)
            union = left_set | right_set
            distance = 1.0 - len(left_set & right_set) / len(union) if union else 0.0
        else:
            distance = float(left_value != right_value)
        weighted_distance += field.weight * distance
        available_weight += field.weight
        dimensions += 1
    if available_weight == 0.0:
        return None, 0
    return weighted_distance / available_weight, dimensions


def _add_to_reservoir(
    reservoir: list[float], value: float, seen: int, capacity: int, rng: random.Random
) -> None:
    if len(reservoir) < capacity:
        reservoir.append(value)
        return
    replacement = rng.randrange(seen)
    if replacement < capacity:
        reservoir[replacement] = value


def _profile_combination_counts(
    corpus: Corpus,
) -> tuple[Counter[object], Counter[object]]:
    all_profiles = Counter(document.profile for document in corpus.documents)
    complete_profiles = Counter(
        document.profile
        for document in corpus.documents
        if all(value is not None for value in document.profile)
    )
    return all_profiles, complete_profiles


def profile_diversity_metrics(
    corpus: Corpus,
    config: ProfileDiversityConfig,
    *,
    seed: int,
) -> dict[str, object]:
    """Measure marginal diversity and exhaustively verify Gower separation."""

    if not corpus.profile_schema:
        return {
            "status": "not_configured",
            "reason": "the dataset has no configured persisted profile fields",
        }
    document_count = len(corpus.documents)
    if document_count > config.exhaustive_max_documents:
        raise ValueError(
            f"profile separation for {corpus.name!r} requires all document pairs, "
            f"but {document_count} exceeds profile_diversity."
            f"exhaustive_max_documents={config.exhaustive_max_documents}; increase "
            "the audited limit to avoid an unsupported 'no close profiles' claim"
        )

    field_reports: dict[str, object] = {}
    for field_index, field in enumerate(corpus.profile_schema):
        values = [document.profile[field_index] for document in corpus.documents]
        if field.kind == "numeric":
            field_reports[field.name] = _numeric_summary(
                values, field, seed=seed + field_index
            )
        elif field.kind == "set":
            field_reports[field.name] = _set_summary(
                values,
                field,
                maximum=config.max_reported_field_frequencies,
            )
        else:
            field_reports[field.name] = _categorical_summary(
                values,
                field,
                maximum=config.max_reported_field_frequencies,
            )

    all_combinations, complete_combinations = _profile_combination_counts(corpus)
    total_pairs = document_count * (document_count - 1) // 2
    duplicate_pairs_all = sum(
        count * (count - 1) // 2 for count in all_combinations.values()
    )
    duplicate_pairs_complete = sum(
        count * (count - 1) // 2 for count in complete_combinations.values()
    )

    numeric_ranges = _numeric_ranges(corpus)
    rng = random.Random(seed)
    reservoir: list[float] = []
    nearest: list[float | None] = [None] * document_count
    comparable_pairs = 0
    close_pairs = 0
    fields_compared: list[int] = []
    closest_heap: list[tuple[float, int, int, int]] = []
    for left_index, right_index in itertools.combinations(range(document_count), 2):
        distance, dimensions = _gower_distance(
            corpus.documents[left_index].profile,
            corpus.documents[right_index].profile,
            corpus.profile_schema,
            numeric_ranges,
        )
        if distance is None:
            continue
        comparable_pairs += 1
        fields_compared.append(dimensions)
        _add_to_reservoir(
            reservoir,
            distance,
            comparable_pairs,
            config.pair_distance_sample_size,
            rng,
        )
        for index in (left_index, right_index):
            current = nearest[index]
            nearest[index] = distance if current is None else min(current, distance)
        if distance < config.close_distance_threshold:
            close_pairs += 1
        if config.max_reported_closest_pairs:
            candidate = (-distance, -left_index, -right_index, dimensions)
            if len(closest_heap) < config.max_reported_closest_pairs:
                heapq.heappush(closest_heap, candidate)
            elif distance < -closest_heap[0][0]:
                heapq.heapreplace(closest_heap, candidate)

    nearest_observed = [value for value in nearest if value is not None]
    closest_pairs = sorted(
        (
            {
                "left_document_index": -left,
                "right_document_index": -right,
                "gower_distance": -negative_distance,
                "compared_fields": dimensions,
            }
            for negative_distance, left, right, dimensions in closest_heap
        ),
        key=lambda row: (
            row["gower_distance"],
            row["left_document_index"],
            row["right_document_index"],
        ),
    )
    all_pairs_comparable = comparable_pairs == total_pairs
    all_profiles_complete = sum(complete_combinations.values()) == document_count
    return {
        "status": "computed",
        "method": {
            "distance": "weighted Gower distance",
            "categorical_component": "0 if equal, otherwise 1",
            "numeric_component": (
                "absolute difference divided by configured or observed range, "
                "capped at 1"
            ),
            "set_component": "Jaccard distance",
            "missing_component": (
                "excluded pairwise with remaining weights renormalized"
            ),
            "pair_enumeration": "exhaustive",
            "close_pair_rule": (f"Gower distance < {config.close_distance_threshold}"),
        },
        "profile_fields": field_reports,
        "joint_profiles": {
            "documents": document_count,
            "complete_profiles": sum(complete_combinations.values()),
            "unique_combinations_including_missing": len(all_combinations),
            "unique_complete_combinations": len(complete_combinations),
            "exact_duplicate_pairs_including_missing": duplicate_pairs_all,
            "exact_duplicate_complete_profile_pairs": duplicate_pairs_complete,
        },
        "separation": {
            "total_possible_pairs": total_pairs,
            "comparable_pairs": comparable_pairs,
            "pairs_without_any_shared_observed_field": total_pairs - comparable_pairs,
            "all_pairs_comparable": all_pairs_comparable,
            "all_profiles_complete": all_profiles_complete,
            "close_distance_threshold_strict": config.close_distance_threshold,
            "pairs_below_close_distance_threshold": close_pairs,
            "close_pair_prevalence_among_comparable_pairs": (
                close_pairs / comparable_pairs if comparable_pairs else None
            ),
            "passes_no_close_profiles_check": (
                all_profiles_complete and all_pairs_comparable and close_pairs == 0
            ),
            "nearest_neighbor_distance_per_document": numeric_summary(nearest_observed),
            "compared_field_count_per_comparable_pair": numeric_summary(
                fields_compared
            ),
            "pair_distance_distribution": {
                "summary": numeric_summary(reservoir),
                "sampling": (
                    "exhaustive"
                    if comparable_pairs <= config.pair_distance_sample_size
                    else "uniform reservoir sample over exhaustive pair traversal"
                ),
                "sample_size": len(reservoir),
                "population_size": comparable_pairs,
            },
            "closest_pairs": closest_pairs,
        },
        "interpretation": (
            "Separation is conditional on the configured persisted fields and their "
            "weights. Identity fields and random seeds should not be included because "
            "they can make profiles trivially distinct."
        ),
    }
