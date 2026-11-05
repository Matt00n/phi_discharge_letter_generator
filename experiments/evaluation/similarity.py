"""Uniform pair sampling and within-corpus overlap metrics."""

from __future__ import annotations

import importlib.metadata
import logging
import random
from collections.abc import Sequence

from experiments.evaluation.config import ModelConfig
from experiments.evaluation.corpus import Corpus
from experiments.evaluation.statistics import sampled_pair_summary
from experiments.evaluation.text import word_tokens

LOGGER = logging.getLogger(__name__)


class _UnicodeRougeTokenizer:
    def tokenize(self, text: str) -> list[str]:
        return word_tokens(text, casefold=True)


def _pair_from_rank(rank: int, number_of_documents: int) -> tuple[int, int]:
    """Map a lexicographically ordered pair rank to an unordered index pair."""

    def row_start(left_index: int) -> int:
        return left_index * (2 * number_of_documents - left_index - 1) // 2

    low = 0
    high = number_of_documents - 1
    while low + 1 < high:
        middle = (low + high) // 2
        if row_start(middle) <= rank:
            low = middle
        else:
            high = middle
    left = low
    right = left + 1 + (rank - row_start(left))
    return left, right


def sample_document_pairs(
    number_of_documents: int, sample_size: int, *, seed: int
) -> tuple[list[tuple[int, int]], dict[str, object]]:
    """Sample unordered document pairs uniformly without replacement."""

    possible_pairs = number_of_documents * (number_of_documents - 1) // 2
    if possible_pairs == 0:
        return [], {
            "sampling": "not applicable",
            "random_seed": seed,
            "all_possible_pairs": possible_pairs,
            "sampled_pairs": 0,
        }
    selected_size = min(sample_size, possible_pairs)
    if selected_size == possible_pairs:
        ranks: Sequence[int] = range(possible_pairs)
        sampling = "exhaustive"
    else:
        ranks = sorted(random.Random(seed).sample(range(possible_pairs), selected_size))
        sampling = "uniform without replacement"
    pairs = [_pair_from_rank(rank, number_of_documents) for rank in ranks]
    return pairs, {
        "sampling": sampling,
        "random_seed": seed,
        "all_possible_pairs": possible_pairs,
        "sampled_pairs": len(pairs),
    }


def overlap_metrics(
    corpus: Corpus,
    pairs: Sequence[tuple[int, int]],
    *,
    confidence_level: float,
) -> dict[str, object]:
    """Compute symmetric ROUGE and sentence-BLEU on sampled document pairs."""

    try:
        from rouge_score.rouge_scorer import RougeScorer
        from sacrebleu.metrics import BLEU
    except ImportError as exc:
        raise RuntimeError(
            "ROUGE/BLEU metrics require the 'experiments' dependency group"
        ) from exc

    rouge = RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=False,
        tokenizer=_UnicodeRougeTokenizer(),
    )
    bleu = BLEU(tokenize="none", smooth_method="exp", effective_order=True)
    scores: dict[str, list[float]] = {
        "rouge1_f1": [],
        "rouge2_f1": [],
        "rougeL_f1": [],
        "symmetric_sentence_bleu": [],
    }
    tokenized_texts = [
        " ".join(word_tokens(document.text, casefold=True))
        for document in corpus.documents
    ]
    for left, right in pairs:
        left_text = corpus.documents[left].text
        right_text = corpus.documents[right].text
        rouge_scores = rouge.score(left_text, right_text)
        for metric_name in ("rouge1", "rouge2", "rougeL"):
            scores[f"{metric_name}_f1"].append(
                float(rouge_scores[metric_name].fmeasure)
            )
        left_bleu = bleu.sentence_score(
            tokenized_texts[left], [tokenized_texts[right]]
        ).score
        right_bleu = bleu.sentence_score(
            tokenized_texts[right], [tokenized_texts[left]]
        ).score
        scores["symmetric_sentence_bleu"].append(
            (float(left_bleu) + float(right_bleu)) / 200.0
        )
    total_possible_pairs = len(corpus.documents) * (len(corpus.documents) - 1) // 2
    return {
        "definitions": {
            "rouge": (
                "ROUGE-1/2/L F1 with a shared Unicode, casefolded word tokenizer "
                "and no stemming"
            ),
            "bleu": (
                "mean of both sentence-BLEU directions; SacreBLEU exponential "
                "smoothing, effective order, pretokenized input"
            ),
        },
        "package_versions": {
            "rouge-score": importlib.metadata.version("rouge-score"),
            "sacrebleu": importlib.metadata.version("sacrebleu"),
        },
        "scores": {
            name: sampled_pair_summary(
                values,
                total_possible_pairs=total_possible_pairs,
                confidence_level=confidence_level,
            )
            for name, values in sorted(scores.items())
        },
    }


def _model_snapshot(model: str, revision: str | None) -> str:
    if revision is None:
        return model
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "model revision resolution requires huggingface-hub"
        ) from exc
    return snapshot_download(repo_id=model, revision=revision)


def _head_tail_truncate(
    text: str, tokenizer: object, max_tokens: int
) -> tuple[str, bool]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return text, False
    head_size = max_tokens // 2
    retained = token_ids[:head_size] + token_ids[-(max_tokens - head_size) :]
    return tokenizer.decode(retained, skip_special_tokens=True), True


def bertscore_metrics(
    corpus: Corpus,
    pairs: Sequence[tuple[int, int]],
    config: ModelConfig,
    *,
    confidence_level: float,
    seed: int,
) -> dict[str, object]:
    """Compute symmetric BERTScore F1 with explicit head-tail truncation."""

    try:
        from bert_score import score as bert_score
        from transformers import AutoConfig, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "BERTScore requires the 'experiments' dependency group"
        ) from exc

    total_possible_pairs = len(corpus.documents) * (len(corpus.documents) - 1) // 2
    if not pairs:
        return {
            "definition": "mean BERTScore F1 over both candidate/reference directions",
            "sampled_pairs": 0,
            "score": sampled_pair_summary(
                [],
                total_possible_pairs=total_possible_pairs,
                confidence_level=confidence_level,
            ),
        }

    if len(pairs) > config.bertscore_pair_sample_size:
        model_pairs = random.Random(seed).sample(
            list(pairs), config.bertscore_pair_sample_size
        )
        sampling = "uniform subsample of the corpus pair sample"
    else:
        model_pairs = list(pairs)
        sampling = "all corpus pair-sample pairs"
    resolved_model = _model_snapshot(config.bertscore_model, config.bertscore_revision)
    tokenizer = AutoTokenizer.from_pretrained(resolved_model)
    model_metadata = AutoConfig.from_pretrained(resolved_model)
    tokenizer_limit = getattr(tokenizer, "model_max_length", 0)
    if isinstance(tokenizer_limit, int) and 2 < tokenizer_limit < 1_000_000:
        effective_max_tokens = min(config.bertscore_max_tokens, tokenizer_limit - 2)
    else:
        effective_max_tokens = config.bertscore_max_tokens
    prepared: dict[int, str] = {}
    truncated: set[int] = set()
    for index in {item for pair in model_pairs for item in pair}:
        prepared[index], was_truncated = _head_tail_truncate(
            corpus.documents[index].text,
            tokenizer,
            effective_max_tokens,
        )
        if was_truncated:
            truncated.add(index)

    candidates: list[str] = []
    references: list[str] = []
    for left, right in model_pairs:
        candidates.extend([prepared[left], prepared[right]])
        references.extend([prepared[right], prepared[left]])
    # Surface bert_score's own batch progress bar whenever INFO logging is on
    # (the CLI default); silent under --log-level WARNING. Each pair scores in
    # both directions, so the bar advances over 2 * len(model_pairs) candidates.
    show_progress = LOGGER.isEnabledFor(logging.INFO)
    LOGGER.info(
        "BERTScore for %s: scoring %d pairs (%d directional candidates) on %s",
        corpus.name,
        len(model_pairs),
        len(candidates),
        config.device or "auto device",
    )
    _, _, f1 = bert_score(
        candidates,
        references,
        model_type=resolved_model,
        num_layers=config.bertscore_num_layers,
        lang=corpus.language,
        batch_size=config.bertscore_batch_size,
        device=config.device,
        rescale_with_baseline=False,
        verbose=show_progress,
    )
    directional = [float(value) for value in f1.tolist()]
    symmetric = [
        (directional[index] + directional[index + 1]) / 2.0
        for index in range(0, len(directional), 2)
    ]
    return {
        "definition": "mean BERTScore F1 over both candidate/reference directions",
        "model": config.bertscore_model,
        "requested_revision": config.bertscore_revision,
        "resolved_commit": getattr(model_metadata, "_commit_hash", None),
        "package_version": importlib.metadata.version("bert-score"),
        "rescale_with_baseline": False,
        "num_layers": config.bertscore_num_layers,
        "maximum_document_subword_tokens": effective_max_tokens,
        "truncation": "equal-size head and tail retention",
        "unique_documents_truncated": len(truncated),
        "sampled_pairs": len(model_pairs),
        "sampling": sampling,
        "random_seed": seed,
        "score": sampled_pair_summary(
            symmetric,
            total_possible_pairs=total_possible_pairs,
            confidence_level=confidence_level,
        ),
    }
