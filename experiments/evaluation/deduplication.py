"""Exact and verified near-duplicate detection."""

from __future__ import annotations

import hashlib
import itertools
from collections import defaultdict
from collections.abc import Iterable

from experiments.evaluation.config import DeduplicationConfig
from experiments.evaluation.corpus import Corpus
from experiments.evaluation.text import ngrams, normalize_for_deduplication, word_tokens


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _clusters_from_groups(
    groups: Iterable[list[int]], max_examples: int
) -> dict[str, object]:
    duplicate_groups = [sorted(group) for group in groups if len(group) > 1]
    duplicate_groups.sort(key=lambda group: (-len(group), group))
    pair_count = sum(len(group) * (len(group) - 1) // 2 for group in duplicate_groups)
    return {
        "number_of_clusters": len(duplicate_groups),
        "documents_in_clusters": sum(len(group) for group in duplicate_groups),
        "duplicate_pairs": pair_count,
        "examples": [group for group in duplicate_groups[:max_examples]],
        "document_reference": "zero-based input row index",
    }


def _digest_groups(texts: Iterable[str]) -> list[list[int]]:
    groups: defaultdict[str, list[int]] = defaultdict(list)
    for index, text in enumerate(texts):
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        groups[digest].append(index)
    return list(groups.values())


def _shingles(text: str, n: int) -> frozenset[str]:
    tokens = word_tokens(text, casefold=True)
    if len(tokens) >= n:
        return frozenset("\x1f".join(value) for value in ngrams(tokens, n))
    if tokens:
        return frozenset(tokens)
    normalized = normalize_for_deduplication(text)
    return frozenset({normalized or "<EMPTY>"})


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _lsh_candidates(
    shingles: list[frozenset[str]], config: DeduplicationConfig, seed: int
) -> set[tuple[int, int]]:
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError as exc:
        raise RuntimeError(
            "near-duplicate LSH requires the 'experiments' dependency group"
        ) from exc

    minhash_seed = seed % (2**32)
    signatures = []
    for values in shingles:
        signature = MinHash(
            num_perm=config.minhash_permutations,
            seed=minhash_seed,
        )
        for value in sorted(values):
            signature.update(value.encode("utf-8"))
        signatures.append(signature)
    index = MinHashLSH(
        threshold=config.jaccard_threshold,
        num_perm=config.minhash_permutations,
    )
    for document_index, signature in enumerate(signatures):
        index.insert(str(document_index), signature)
    candidates: set[tuple[int, int]] = set()
    for left, signature in enumerate(signatures):
        for raw_right in index.query(signature):
            right = int(raw_right)
            if left < right:
                candidates.add((left, right))
    return candidates


def deduplication_metrics(
    corpus: Corpus, config: DeduplicationConfig, *, seed: int
) -> dict[str, object]:
    """Find raw, normalized, and high-Jaccard duplicate clusters."""

    raw_texts = [document.text for document in corpus.documents]
    normalized_texts = [normalize_for_deduplication(text) for text in raw_texts]
    exact_raw = _clusters_from_groups(
        _digest_groups(raw_texts), config.max_reported_pairs
    )
    exact_normalized = _clusters_from_groups(
        _digest_groups(normalized_texts), config.max_reported_pairs
    )

    shingles = [_shingles(text, config.ngram_size) for text in normalized_texts]
    number_of_documents = len(shingles)
    possible_pairs = number_of_documents * (number_of_documents - 1) // 2
    if number_of_documents <= config.exhaustive_max_documents:
        candidates = set(itertools.combinations(range(number_of_documents), 2))
        candidate_method = "exhaustive"
    else:
        candidates = _lsh_candidates(shingles, config, seed)
        candidate_method = "MinHash LSH candidates followed by exact Jaccard"

    qualifying: list[tuple[int, int, float]] = []
    union_find = _UnionFind(number_of_documents)
    for left, right in sorted(candidates):
        similarity = _jaccard(shingles[left], shingles[right])
        if similarity >= config.jaccard_threshold:
            qualifying.append((left, right, similarity))
            union_find.union(left, right)
    cluster_members: defaultdict[int, list[int]] = defaultdict(list)
    for index in range(number_of_documents):
        cluster_members[union_find.find(index)].append(index)
    near_clusters = _clusters_from_groups(
        cluster_members.values(), config.max_reported_pairs
    )
    normalized_exact_pairs = {
        pair
        for group in _digest_groups(normalized_texts)
        for pair in itertools.combinations(sorted(group), 2)
    }
    nonexact_qualifying = [
        item for item in qualifying if (item[0], item[1]) not in normalized_exact_pairs
    ]
    return {
        "raw_exact": exact_raw,
        "normalized_exact": exact_normalized,
        "near_duplicate": {
            "representation": f"set of within-document word {config.ngram_size}-grams",
            "jaccard_threshold": config.jaccard_threshold,
            "candidate_method": candidate_method,
            "candidate_pairs_verified": len(candidates),
            "all_possible_pairs": possible_pairs,
            "qualifying_pairs_including_normalized_exact": len(qualifying),
            "qualifying_nonexact_pairs": len(nonexact_qualifying),
            "qualifying_pair_fraction": len(qualifying) / possible_pairs
            if possible_pairs
            else 0.0,
            "clusters": near_clusters,
            "highest_similarity_nonexact_examples": [
                {
                    "left_index": left,
                    "right_index": right,
                    "jaccard": similarity,
                }
                for left, right, similarity in sorted(
                    nonexact_qualifying, key=lambda item: (-item[2], item[0], item[1])
                )[: config.max_reported_pairs]
            ],
            "document_reference": "zero-based input row index",
            "minhash_permutations": (
                config.minhash_permutations
                if candidate_method != "exhaustive"
                else None
            ),
            "minhash_seed_uint32": (
                seed % (2**32) if candidate_method != "exhaustive" else None
            ),
        },
    }
