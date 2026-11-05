"""Document embeddings, subword distributions, and linguistic analysis."""

from __future__ import annotations

import importlib.metadata
import logging
from collections import Counter
from dataclasses import dataclass

from experiments.evaluation.config import ModelConfig
from experiments.evaluation.corpus import Corpus
from experiments.evaluation.similarity import _model_snapshot
from experiments.evaluation.statistics import numeric_summary, sampled_pair_summary

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class EmbeddingResult:
    values: object
    metadata: dict[str, object]


class DocumentEmbedder:
    """Chunk long documents, mean-pool chunks, and L2-normalize documents."""

    def __init__(self, config: ModelConfig) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "embedding metrics require the 'experiments' dependency group"
            ) from exc
        resolved_model = _model_snapshot(
            config.embedding_model, config.embedding_revision
        )
        self.model = SentenceTransformer(
            resolved_model,
            device=config.device,
            revision=None,
        )
        self.config = config
        auto_model = getattr(self.model[0], "auto_model", None)
        model_config = getattr(auto_model, "config", None)
        self.metadata = {
            "model": config.embedding_model,
            "requested_revision": config.embedding_revision,
            "resolved_commit": getattr(model_config, "_commit_hash", None),
            "sentence_transformers_version": importlib.metadata.version(
                "sentence-transformers"
            ),
            "pooling": (
                "token-count-weighted mean of chunk embeddings followed by L2 "
                "normalization"
            ),
        }

    def encode(self, corpus: Corpus) -> EmbeddingResult:
        import numpy as np

        tokenizer = self.model.tokenizer
        model_limit = int(getattr(self.model, "max_seq_length", 512))
        chunk_size = min(self.config.embedding_chunk_tokens, max(16, model_limit - 2))
        chunks: list[str] = []
        owners: list[int] = []
        weights: list[int] = []
        chunks_per_document: list[int] = []
        for document_index, document in enumerate(corpus.documents):
            token_ids = tokenizer.encode(document.text, add_special_tokens=False)
            if not token_ids:
                token_ids = tokenizer.encode(" ", add_special_tokens=False) or [0]
            document_chunks = [
                token_ids[start : start + chunk_size]
                for start in range(0, len(token_ids), chunk_size)
            ]
            chunks_per_document.append(len(document_chunks))
            for chunk in document_chunks:
                chunks.append(tokenizer.decode(chunk, skip_special_tokens=True))
                owners.append(document_index)
                weights.append(len(chunk))
        show_progress = LOGGER.isEnabledFor(logging.INFO)
        LOGGER.info(
            "Embedding %s: encoding %d chunks from %d documents",
            corpus.name,
            len(chunks),
            len(corpus.documents),
        )
        chunk_embeddings = self.model.encode(
            chunks,
            batch_size=self.config.embedding_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=show_progress,
        )
        document_embeddings = np.zeros(
            (len(corpus.documents), chunk_embeddings.shape[1]), dtype=np.float64
        )
        document_weights = np.zeros(len(corpus.documents), dtype=np.float64)
        for embedding, owner, weight in zip(
            chunk_embeddings, owners, weights, strict=True
        ):
            document_embeddings[owner] += embedding.astype(np.float64) * weight
            document_weights[owner] += weight
        document_embeddings /= document_weights[:, None]
        norms = np.linalg.norm(document_embeddings, axis=1, keepdims=True)
        document_embeddings /= np.maximum(norms, np.finfo(np.float64).eps)
        metadata = {
            **self.metadata,
            "model_max_sequence_length": model_limit,
            "configured_chunk_tokens": self.config.embedding_chunk_tokens,
            "effective_chunk_tokens": chunk_size,
            "embedding_dimension": int(document_embeddings.shape[1]),
            "documents_with_multiple_chunks": sum(
                count > 1 for count in chunks_per_document
            ),
            "total_chunks": len(chunks),
        }
        return EmbeddingResult(values=document_embeddings, metadata=metadata)


def embedding_pair_similarity(
    embeddings: object,
    pairs: list[tuple[int, int]],
    *,
    confidence_level: float,
    total_possible_pairs: int,
) -> dict[str, object]:
    """Cosine scores; embeddings are normalized by DocumentEmbedder."""

    similarities = [
        float(embeddings[left] @ embeddings[right]) for left, right in pairs
    ]
    return {
        "definition": "cosine similarity between L2-normalized document embeddings",
        "score": sampled_pair_summary(
            similarities,
            total_possible_pairs=total_possible_pairs,
            confidence_level=confidence_level,
        ),
    }


def subword_counts(
    corpus: Corpus, config: ModelConfig
) -> tuple[Counter[int], dict[str, object]]:
    """Count fixed-tokenizer subwords for entropy-shift comparisons."""

    try:
        from transformers import AutoConfig, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "subword metrics require the 'experiments' dependency group"
        ) from exc
    model_name = config.subword_tokenizer_model or config.embedding_model
    revision = config.subword_tokenizer_revision or config.embedding_revision
    resolved_model = _model_snapshot(model_name, revision)
    tokenizer = AutoTokenizer.from_pretrained(resolved_model)
    model_metadata = AutoConfig.from_pretrained(resolved_model)
    counts: Counter[int] = Counter()
    for document in corpus.documents:
        counts.update(tokenizer.encode(document.text, add_special_tokens=False))
    return counts, {
        "tokenizer": model_name,
        "requested_revision": revision,
        "resolved_commit": getattr(model_metadata, "_commit_hash", None),
        "transformers_version": importlib.metadata.version("transformers"),
        "special_tokens_included": False,
        "total_subword_tokens": sum(counts.values()),
        "unique_subword_ids": len(counts),
    }


def linguistic_metrics(
    corpus: Corpus,
    *,
    bootstrap_samples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, object]:
    """Compute sentence length and Universal-POS distributions with spaCy."""

    if corpus.spacy_model is None:
        raise ValueError(f"spaCy model is disabled for corpus {corpus.name!r}")
    try:
        import spacy
    except ImportError as exc:
        raise RuntimeError(
            "linguistic metrics require the 'experiments' dependency group"
        ) from exc
    try:
        nlp = spacy.load(corpus.spacy_model)
    except OSError as exc:
        raise RuntimeError(
            f"spaCy pipeline {corpus.spacy_model!r} is not installed; see the "
            "experiments/evaluation README for the pinned model installation step"
        ) from exc
    maximum_length = max(len(document.text) for document in corpus.documents) + 1
    nlp.max_length = max(nlp.max_length, maximum_length)
    pos_counts: Counter[str] = Counter()
    sentence_lengths: list[int] = []
    document_mean_sentence_lengths: list[float] = []
    for parsed in nlp.pipe(
        (document.text for document in corpus.documents),
        batch_size=32,
    ):
        document_lengths: list[int] = []
        for sentence in parsed.sents:
            length = sum(
                not token.is_space and not token.is_punct for token in sentence
            )
            if length:
                sentence_lengths.append(length)
                document_lengths.append(length)
        if document_lengths:
            document_mean_sentence_lengths.append(
                sum(document_lengths) / len(document_lengths)
            )
        pos_counts.update(
            token.pos_ for token in parsed if not token.is_space and not token.is_punct
        )
    tagged_tokens = sum(pos_counts.values())
    distribution = {
        tag: count / tagged_tokens for tag, count in sorted(pos_counts.items())
    }
    summary_kwargs = {
        "bootstrap_samples": bootstrap_samples,
        "confidence_level": confidence_level,
    }
    return {
        "model": corpus.spacy_model,
        "model_version": nlp.meta.get("version"),
        "spacy_version": importlib.metadata.version("spacy"),
        "sentence_length_definition": "non-space, non-punctuation spaCy tokens",
        "sentence_length_tokens": numeric_summary(
            sentence_lengths,
            bootstrap_samples=0,
            confidence_level=confidence_level,
            seed=seed + 51,
        ),
        "mean_sentence_length_per_document": numeric_summary(
            document_mean_sentence_lengths, seed=seed + 52, **summary_kwargs
        ),
        "universal_pos": {
            "tagged_token_count": tagged_tokens,
            "distribution": distribution,
            "requested_groups": {
                "nouns_NOUN": distribution.get("NOUN", 0.0),
                "proper_nouns_PROPN": distribution.get("PROPN", 0.0),
                "nouns_including_proper": (pos_counts["NOUN"] + pos_counts["PROPN"])
                / tagged_tokens
                if tagged_tokens
                else None,
                "verbs_VERB": distribution.get("VERB", 0.0),
                "auxiliaries_AUX": distribution.get("AUX", 0.0),
                "verbs_including_auxiliaries": (pos_counts["VERB"] + pos_counts["AUX"])
                / tagged_tokens
                if tagged_tokens
                else None,
                "numerals_NUM": distribution.get("NUM", 0.0),
            },
        },
    }
