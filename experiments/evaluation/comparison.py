"""Cross-corpus lexical and embedding-distribution comparisons."""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter
from collections.abc import Mapping

from experiments.evaluation.corpus import Corpus
from experiments.evaluation.lexical import (
    corpus_ngram_counts,
    corpus_token_counts,
    entropy_bits,
)


def _probability[T](counter: Counter[T], key: T) -> float:
    total = sum(counter.values())
    return counter[key] / total if total else 0.0


def jensen_shannon_divergence[T](left: Counter[T], right: Counter[T]) -> float | None:
    """Base-2 JSD on empirical distributions, bounded by one bit."""

    if not left or not right:
        return None
    divergence = 0.0
    for key in left.keys() | right.keys():
        left_probability = _probability(left, key)
        right_probability = _probability(right, key)
        mixture = (left_probability + right_probability) / 2.0
        if left_probability:
            divergence += 0.5 * left_probability * math.log2(left_probability / mixture)
        if right_probability:
            divergence += (
                0.5 * right_probability * math.log2(right_probability / mixture)
            )
    return divergence


def _directional_coverage_gap(
    source: Counter[str], reference: Counter[str]
) -> dict[str, float | int | None]:
    missing_types = source.keys() - reference.keys()
    source_tokens = sum(source.values())
    missing_token_count = sum(source[token] for token in missing_types)
    return {
        "missing_types": len(missing_types),
        "type_gap_rate": len(missing_types) / len(source) if source else None,
        "token_occurrences_with_missing_type": missing_token_count,
        "token_mass_gap_rate": missing_token_count / source_tokens
        if source_tokens
        else None,
    }


def lexical_comparison(
    left: Corpus, right: Corpus, *, casefold: bool
) -> dict[str, object]:
    """Directional coverage, JSD shift, and vocabulary overlap."""

    left_unigrams = corpus_token_counts(left, casefold=casefold)
    right_unigrams = corpus_token_counts(right, casefold=casefold)
    left_bigrams = corpus_ngram_counts(left, n=2, casefold=casefold)
    right_bigrams = corpus_ngram_counts(right, n=2, casefold=casefold)
    intersection = left_unigrams.keys() & right_unigrams.keys()
    union = left_unigrams.keys() | right_unigrams.keys()
    return {
        "preprocessing": "shared Unicode NFKC word tokenization",
        "casefold": casefold,
        "lexical_coverage_gaps": {
            f"{left.name}_relative_to_{right.name}": _directional_coverage_gap(
                left_unigrams, right_unigrams
            ),
            f"{right.name}_relative_to_{left.name}": _directional_coverage_gap(
                right_unigrams, left_unigrams
            ),
        },
        "lexical_shift_jensen_shannon_bits": {
            "unigrams": jensen_shannon_divergence(left_unigrams, right_unigrams),
            "bigrams": jensen_shannon_divergence(left_bigrams, right_bigrams),
        },
        "vocabulary_jaccard_similarity": len(intersection) / len(union)
        if union
        else None,
        "vocabulary_intersection_size": len(intersection),
        "vocabulary_union_size": len(union),
    }


def subword_comparison(
    left: Counter[int], right: Counter[int]
) -> dict[str, float | int | None]:
    """Entropy levels and shift under one fixed subword tokenizer."""

    left_entropy = entropy_bits(left)
    right_entropy = entropy_bits(right)
    return {
        "left_entropy_bits": left_entropy,
        "right_entropy_bits": right_entropy,
        "right_minus_left_entropy_bits": right_entropy - left_entropy,
        "jensen_shannon_divergence_bits": jensen_shannon_divergence(left, right),
        "shared_subword_ids": len(left.keys() & right.keys()),
        "union_subword_ids": len(left.keys() | right.keys()),
    }


def _stable_seed(base_seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _sample_rows(
    values: object,
    corpus: Corpus,
    maximum: int,
    *,
    seed: int,
) -> tuple[object, dict[str, object]]:
    number_of_rows = len(corpus.documents)
    if number_of_rows <= maximum:
        indices = list(range(number_of_rows))
        method = "all documents"
    else:
        indices = sorted(random.Random(seed).sample(range(number_of_rows), maximum))
        method = "uniform documents without replacement"
    row_indices = "\n".join(str(index) for index in indices)
    return values[indices], {
        "method": method,
        "random_seed": seed,
        "sample_size": len(indices),
        "population_size": number_of_rows,
        "ordered_row_index_sha256": hashlib.sha256(
            row_indices.encode("utf-8")
        ).hexdigest(),
    }


def _squared_distances(left: object, right: object) -> object:
    import numpy as np

    distances = (
        np.sum(left * left, axis=1)[:, None]
        + np.sum(right * right, axis=1)[None, :]
        - 2.0 * (left @ right.T)
    )
    return np.maximum(distances, 0.0)


def _median_bandwidth(
    pooled: object, maximum: int, *, seed: int
) -> tuple[float, dict[str, object]]:
    import numpy as np

    population = pooled.shape[0]
    if population > maximum:
        indices = sorted(random.Random(seed).sample(range(population), maximum))
        sample = pooled[indices]
        method = "uniform pooled subsample without replacement"
    else:
        sample = pooled
        method = "all pooled documents"
    distances = _squared_distances(sample, sample)
    upper = distances[np.triu_indices(len(sample), k=1)]
    nonzero = upper[upper > np.finfo(np.float64).eps]
    if nonzero.size:
        sigma_squared = float(np.median(nonzero))
        degenerate = False
    else:
        sigma_squared = 1.0
        degenerate = True
    return math.sqrt(sigma_squared), {
        "method": method,
        "random_seed": seed,
        "sample_size": len(sample),
        "median_nonzero_pairwise_distance": math.sqrt(sigma_squared),
        "degenerate_fallback_bandwidth": degenerate,
    }


def _rbf_kernel(left: object, right: object, sigma: float) -> object:
    import numpy as np

    return np.exp(-_squared_distances(left, right) / (2.0 * sigma**2))


def mmd_rbf(left: object, right: object, *, sigma: float) -> dict[str, float | None]:
    """Biased and unbiased empirical squared MMD under an RBF kernel."""

    import numpy as np

    left_kernel = _rbf_kernel(left, left, sigma)
    right_kernel = _rbf_kernel(right, right, sigma)
    cross_kernel = _rbf_kernel(left, right, sigma)
    biased = float(left_kernel.mean() + right_kernel.mean() - 2.0 * cross_kernel.mean())
    if len(left) < 2 or len(right) < 2:
        unbiased = None
    else:
        unbiased = float(
            (left_kernel.sum() - np.trace(left_kernel)) / (len(left) * (len(left) - 1))
            + (right_kernel.sum() - np.trace(right_kernel))
            / (len(right) * (len(right) - 1))
            - 2.0 * cross_kernel.mean()
        )
    return {
        "mmd_squared_biased": max(0.0, biased),
        "mmd_biased": math.sqrt(max(0.0, biased)),
        "mmd_squared_unbiased": unbiased,
        "mmd_unbiased_clipped_for_square_root": math.sqrt(max(0.0, unbiased))
        if unbiased is not None
        else None,
        "rbf_bandwidth_sigma": sigma,
    }


def frechet_gaussian_distance(left: object, right: object) -> dict[str, float | None]:
    """2-Wasserstein/Fréchet distance between fitted Gaussian summaries."""

    import numpy as np

    if len(left) < 2 or len(right) < 2:
        return {
            "frechet_distance_squared": None,
            "frechet_distance": None,
            "covariance_regularization": None,
        }
    left_mean = np.mean(left, axis=0)
    right_mean = np.mean(right, axis=0)
    left_covariance = np.cov(left, rowvar=False, ddof=1)
    right_covariance = np.cov(right, rowvar=False, ddof=1)
    dimension = left.shape[1]
    scale = float(np.trace(left_covariance) + np.trace(right_covariance)) / (
        2.0 * dimension
    )
    regularization = max(np.finfo(np.float64).eps, 1e-6 * scale)
    identity = np.eye(dimension)
    left_covariance = left_covariance + regularization * identity
    right_covariance = right_covariance + regularization * identity
    eigenvalues, eigenvectors = np.linalg.eigh(left_covariance)
    left_sqrt = (eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))) @ (
        eigenvectors.T
    )
    covariance_product = left_sqrt @ right_covariance @ left_sqrt
    product_eigenvalues = np.linalg.eigvalsh(
        (covariance_product + covariance_product.T) / 2.0
    )
    mean_term = float(np.sum((left_mean - right_mean) ** 2))
    covariance_term = float(
        np.trace(left_covariance)
        + np.trace(right_covariance)
        - 2.0 * np.sum(np.sqrt(np.maximum(product_eigenvalues, 0.0)))
    )
    squared = max(0.0, mean_term + covariance_term)
    return {
        "frechet_distance_squared": squared,
        "frechet_distance": math.sqrt(squared),
        "mean_component": mean_term,
        "covariance_component": covariance_term,
        "covariance_regularization": regularization,
    }


def embedding_distribution_comparison(
    left_corpus: Corpus,
    right_corpus: Corpus,
    embeddings: Mapping[str, object],
    *,
    maximum_documents: int,
    bandwidth_sample_size: int,
    seed: int,
) -> dict[str, object]:
    """MMD and Gaussian Fréchet distance on fixed document embeddings."""

    import numpy as np

    left_seed = _stable_seed(seed, left_corpus.name)
    right_seed = _stable_seed(seed, right_corpus.name)
    left, left_sampling = _sample_rows(
        embeddings[left_corpus.name],
        left_corpus,
        maximum_documents,
        seed=left_seed,
    )
    right, right_sampling = _sample_rows(
        embeddings[right_corpus.name],
        right_corpus,
        maximum_documents,
        seed=right_seed,
    )
    pooled = np.concatenate([left, right], axis=0)
    bandwidth_seed = _stable_seed(seed, f"{left_corpus.name}:{right_corpus.name}")
    bandwidth, bandwidth_metadata = _median_bandwidth(
        pooled, bandwidth_sample_size, seed=bandwidth_seed
    )
    return {
        "embedding_samples": {
            left_corpus.name: left_sampling,
            right_corpus.name: right_sampling,
        },
        "mmd": {
            "estimator_note": (
                "unbiased MMD^2 may be negative at finite sample size; biased "
                "MMD^2 is nonnegative"
            ),
            "kernel": "RBF",
            "bandwidth": bandwidth_metadata,
            **mmd_rbf(left, right, sigma=bandwidth),
        },
        "frechet_gaussian": {
            "definition": (
                "2-Wasserstein distance between sample Gaussian summaries in "
                "embedding space"
            ),
            **frechet_gaussian_distance(left, right),
        },
    }
