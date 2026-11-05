"""Non-neural lexical diversity and repetition metrics."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence

from experiments.evaluation.corpus import Corpus
from experiments.evaluation.statistics import numeric_summary
from experiments.evaluation.text import ngrams, sentence_strings, word_tokens


def entropy_bits(counter: Counter[object]) -> float:
    """Shannon entropy of an empirical count distribution in bits."""

    total = sum(counter.values())
    if total == 0:
        return 0.0
    return -sum(
        (count / total) * math.log2(count / total)
        for count in counter.values()
        if count
    )


def yules_k(tokens: Sequence[str]) -> float | None:
    """Yule's K lexical concentration; lower values indicate more diversity."""

    if not tokens:
        return None
    counts = Counter(tokens)
    token_count = len(tokens)
    second_frequency_moment = sum(count * count for count in counts.values())
    return 10_000.0 * (second_frequency_moment - token_count) / token_count**2


def _mtld_one_direction(tokens: Sequence[str], threshold: float) -> float | None:
    if not tokens:
        return None
    factors = 0.0
    segment_types: set[str] = set()
    segment_length = 0
    for token in tokens:
        segment_types.add(token)
        segment_length += 1
        type_token_ratio = len(segment_types) / segment_length
        if type_token_ratio <= threshold:
            factors += 1.0
            segment_types.clear()
            segment_length = 0
    if segment_length:
        type_token_ratio = len(segment_types) / segment_length
        factors += (1.0 - type_token_ratio) / (1.0 - threshold)
    return len(tokens) / factors if factors > 0.0 else float(len(tokens))


def mtld(tokens: Sequence[str], *, threshold: float = 0.72) -> float | None:
    """Bidirectional Measure of Textual Lexical Diversity."""

    forward = _mtld_one_direction(tokens, threshold)
    backward = _mtld_one_direction(list(reversed(tokens)), threshold)
    if forward is None or backward is None:
        return None
    return (forward + backward) / 2.0


def _ngram_counter(
    documents: Sequence[Sequence[str]], n: int
) -> Counter[tuple[str, ...]]:
    counts: Counter[tuple[str, ...]] = Counter()
    for tokens in documents:
        counts.update(ngrams(tokens, n))
    return counts


def _distinct(counter: Counter[tuple[str, ...]]) -> float | None:
    total = sum(counter.values())
    return len(counter) / total if total else None


def conditional_bigram_entropy(tokenized: Sequence[Sequence[str]]) -> float | None:
    """Empirical H(next token | current token), weighted by prefix frequency."""

    continuations: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for tokens in tokenized:
        for current, following in ngrams(tokens, 2):
            continuations[current][following] += 1
    total_bigrams = sum(sum(counter.values()) for counter in continuations.values())
    if total_bigrams == 0:
        return None
    return sum(
        (sum(counter.values()) / total_bigrams) * entropy_bits(counter)
        for counter in continuations.values()
    )


def _top_ngrams(
    counter: Counter[tuple[str, ...]], top_k: int
) -> list[dict[str, object]]:
    total = sum(counter.values())
    return [
        {
            "ngram": " ".join(ngram),
            "count": count,
            "relative_frequency": count / total if total else None,
        }
        for ngram, count in counter.most_common(top_k)
    ]


def lexical_metrics(
    corpus: Corpus,
    *,
    top_k: int,
    casefold: bool,
    bootstrap_samples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, object]:
    """Compute corpus-level and document-level lexical diversity statistics."""

    tokenized = [
        word_tokens(document.text, casefold=casefold) for document in corpus.documents
    ]
    unigram_counts: Counter[str] = Counter(
        token for tokens in tokenized for token in tokens
    )
    ngram_counts = {n: _ngram_counter(tokenized, n) for n in range(1, 5)}
    document_mtld = [
        value for tokens in tokenized if (value := mtld(tokens)) is not None
    ]
    document_yule = [
        value for tokens in tokenized if (value := yules_k(tokens)) is not None
    ]
    document_repetition: list[float] = []
    repeated_fourgrams = 0
    total_fourgrams = 0
    for tokens in tokenized:
        counts = Counter(ngrams(tokens, 4))
        total = sum(counts.values())
        repeated = sum(count - 1 for count in counts.values() if count > 1)
        if total:
            document_repetition.append(repeated / total)
        repeated_fourgrams += repeated
        total_fourgrams += total

    # Document-specific trigram novelty: the share of a document's distinct
    # trigrams that occur in no other document in the corpus.
    trigram_document_frequency: Counter[tuple[str, ...]] = Counter()
    document_trigram_sets: list[set[tuple[str, ...]]] = []
    for tokens in tokenized:
        trigrams = set(ngrams(tokens, 3))
        document_trigram_sets.append(trigrams)
        trigram_document_frequency.update(trigrams)
    document_trigram_novelty = [
        sum(1 for trigram in trigrams if trigram_document_frequency[trigram] == 1)
        / len(trigrams)
        for trigrams in document_trigram_sets
        if trigrams
    ]
    document_ttr = [len(set(tokens)) / len(tokens) for tokens in tokenized if tokens]
    hapax_unigram_types = sum(1 for count in unigram_counts.values() if count == 1)

    sentence_lengths = [
        len(word_tokens(sentence, casefold=casefold))
        for document in corpus.documents
        for sentence in sentence_strings(document.text)
    ]
    all_tokens = [token for tokens in tokenized for token in tokens]
    summary_kwargs = {
        "bootstrap_samples": bootstrap_samples,
        "confidence_level": confidence_level,
    }
    return {
        "tokenization": {
            "method": "Unicode word regex after NFKC normalization",
            "casefold": casefold,
            "document_boundaries_crossed_by_ngrams": False,
        },
        "vocabulary_size": len(unigram_counts),
        "token_count": len(all_tokens),
        "type_token_ratio": len(unigram_counts) / len(all_tokens)
        if all_tokens
        else None,
        "type_token_ratio_per_document": numeric_summary(
            document_ttr, seed=seed + 15, **summary_kwargs
        ),
        "corpus_hapax_ratio": (
            hapax_unigram_types / len(unigram_counts) if unigram_counts else None
        ),
        "trigram_novelty_per_document": numeric_summary(
            document_trigram_novelty, seed=seed + 16, **summary_kwargs
        ),
        "vocabulary_entropy_bits": entropy_bits(unigram_counts),
        "vocabulary_entropy_normalized": (
            entropy_bits(unigram_counts) / math.log2(len(unigram_counts))
            if len(unigram_counts) > 1
            else 0.0
        ),
        "conditional_bigram_entropy_bits": conditional_bigram_entropy(tokenized),
        "mtld_threshold": 0.72,
        "mtld_per_document": numeric_summary(
            document_mtld, seed=seed + 11, **summary_kwargs
        ),
        "yules_k_corpus": yules_k(all_tokens),
        "yules_k_per_document": numeric_summary(
            document_yule, seed=seed + 12, **summary_kwargs
        ),
        "distinct_1": _distinct(ngram_counts[1]),
        "distinct_2": _distinct(ngram_counts[2]),
        "fourgram_self_repetition": {
            "definition": (
                "repeated occurrences beyond the first / all within-document "
                "4-gram occurrences"
            ),
            "micro_rate": repeated_fourgrams / total_fourgrams
            if total_fourgrams
            else None,
            "per_document": numeric_summary(
                document_repetition, seed=seed + 13, **summary_kwargs
            ),
        },
        "sentence_length_word_tokens_fallback": numeric_summary(
            sentence_lengths,
            bootstrap_samples=0,
            confidence_level=confidence_level,
            seed=seed + 14,
        ),
        "ngram_frequencies": {
            str(n): {
                "total_occurrences": sum(counter.values()),
                "unique_types": len(counter),
                "hapax_types": sum(count == 1 for count in counter.values()),
                "top": _top_ngrams(counter, top_k),
            }
            for n, counter in ngram_counts.items()
        },
    }


def corpus_token_counts(corpus: Corpus, *, casefold: bool) -> Counter[str]:
    """Expose identical preprocessing for cross-corpus comparisons."""

    return Counter(
        token
        for document in corpus.documents
        for token in word_tokens(document.text, casefold=casefold)
    )


def corpus_ngram_counts(
    corpus: Corpus, *, n: int, casefold: bool
) -> Counter[tuple[str, ...]]:
    """Count n-grams without crossing document boundaries."""

    return _ngram_counter(
        [word_tokens(doc.text, casefold=casefold) for doc in corpus.documents], n
    )
