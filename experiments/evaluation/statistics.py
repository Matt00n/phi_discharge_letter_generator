"""Deterministic descriptive statistics and percentile bootstrap intervals."""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence


def quantile(values: Sequence[float], probability: float) -> float:
    """Linear-interpolated empirical quantile (Hyndman-Fan type 7)."""

    if not values:
        raise ValueError("quantile requires at least one value")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def mean_bootstrap_interval(
    values: Sequence[float],
    *,
    samples: int,
    confidence_level: float,
    seed: int,
) -> list[float] | None:
    """Return a deterministic percentile interval for the arithmetic mean."""

    if not values or samples <= 0:
        return None
    numbers = tuple(float(value) for value in values)
    rng = random.Random(seed)
    bootstrap_means = [
        sum(rng.choice(numbers) for _ in numbers) / len(numbers) for _ in range(samples)
    ]
    tail = (1.0 - confidence_level) / 2.0
    return [
        quantile(bootstrap_means, tail),
        quantile(bootstrap_means, 1.0 - tail),
    ]


def numeric_summary(
    values: Sequence[float | int],
    *,
    bootstrap_samples: int = 0,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> dict[str, float | int | list[float] | None]:
    """Summarize values using population/sample deviations and a mean CI."""

    if not values:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "std_population": None,
            "std_sample": None,
            "min": None,
            "p05": None,
            "p25": None,
            "p75": None,
            "p95": None,
            "max": None,
            "mean_confidence_interval": None,
        }
    numbers = [float(value) for value in values]
    return {
        "n": len(numbers),
        "mean": statistics.fmean(numbers),
        "median": statistics.median(numbers),
        "std_population": statistics.pstdev(numbers),
        "std_sample": statistics.stdev(numbers) if len(numbers) > 1 else None,
        "min": min(numbers),
        "p05": quantile(numbers, 0.05),
        "p25": quantile(numbers, 0.25),
        "p75": quantile(numbers, 0.75),
        "p95": quantile(numbers, 0.95),
        "max": max(numbers),
        "mean_confidence_interval": mean_bootstrap_interval(
            numbers,
            samples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=seed,
        ),
    }


def sampled_pair_summary(
    values: Sequence[float | int],
    *,
    total_possible_pairs: int,
    confidence_level: float,
) -> dict[str, float | int | list[float] | str | None]:
    """Summarize a uniform pair sample with finite-population uncertainty."""

    result = numeric_summary(values)
    sample_size = len(values)
    if sample_size == 0:
        interval = None
    elif sample_size == total_possible_pairs:
        mean = float(result["mean"])
        interval = [mean, mean]
    elif sample_size == 1:
        interval = None
    else:
        sample_variance = statistics.variance(float(value) for value in values)
        sampling_fraction = sample_size / total_possible_pairs
        standard_error = math.sqrt(
            (1.0 - sampling_fraction) * sample_variance / sample_size
        )
        critical_value = statistics.NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
        mean = float(result["mean"])
        interval = [
            mean - critical_value * standard_error,
            mean + critical_value * standard_error,
        ]
    result["mean_confidence_interval"] = interval
    result["confidence_interval_method"] = (
        "design-based normal interval for a uniform pair sample without replacement, "
        "including finite-population correction"
    )
    result["total_possible_pairs"] = total_possible_pairs
    return result
