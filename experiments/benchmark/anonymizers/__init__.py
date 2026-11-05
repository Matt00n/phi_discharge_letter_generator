"""Anonymizer registry.

Maps the ``anonymizer.name`` from a benchmark configuration to a concrete
:class:`~experiments.benchmark.anonymizers.base.Anonymizer`. Register additional
methods here as they are added.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from experiments.benchmark.anonymizers.base import Anonymizer
from experiments.benchmark.anonymizers.flair_ner import FlairAnonymizer
from experiments.benchmark.anonymizers.gemini import GeminiAnonymizer
from experiments.benchmark.anonymizers.lmstudio import LMStudioAnonymizer
from experiments.benchmark.anonymizers.openmed_pii import OpenMedPIIAnonymizer
from experiments.benchmark.anonymizers.presidio_spacy import PresidioAnonymizer
from experiments.benchmark.anonymizers.rule_based import RuleBasedAnonymizer

if TYPE_CHECKING:
    from experiments.benchmark.config import AnonymizerConfig

#: Supported anonymizer identifiers (values usable for ``anonymizer.name``).
REGISTERED = ("gemini", "rule_based", "presidio", "flair", "openmed", "lmstudio")


def build_anonymizer(config: AnonymizerConfig) -> Anonymizer:
    """Instantiate the anonymizer named by ``config``."""
    if config.name == "gemini":
        return GeminiAnonymizer(
            model_name=config.model,
            cache_db=str(config.cache_db),
            temperature=config.temperature,
            retry_attempts=config.retry_attempts,
            retry_delay_seconds=config.retry_delay_seconds,
        )
    if config.name == "rule_based":
        return RuleBasedAnonymizer()
    if config.name == "presidio":
        return PresidioAnonymizer(
            spacy_model=config.spacy_model,
            language=config.language,
            score_threshold=config.score_threshold,
        )
    if config.name == "flair":
        return FlairAnonymizer(
            flair_model=config.flair_model,
            spacy_model=config.spacy_model,
            language=config.language,
            score_threshold=config.score_threshold,
        )
    if config.name == "openmed":
        return OpenMedPIIAnonymizer(
            model=config.pii_model,
            score_threshold=config.score_threshold,
        )
    if config.name == "lmstudio":
        return LMStudioAnonymizer(
            model=config.lmstudio_model,
            base_url=config.lmstudio_base_url,
            cache_db=str(config.cache_db),
            temperature=config.temperature,
            retry_attempts=config.retry_attempts,
            retry_delay_seconds=config.retry_delay_seconds,
            timeout=config.lmstudio_timeout,
            max_tokens=config.lmstudio_max_tokens,
        )
    raise ValueError(
        f"Unknown anonymizer '{config.name}'. Registered: {', '.join(REGISTERED)}."
    )


__all__ = [
    "Anonymizer",
    "FlairAnonymizer",
    "GeminiAnonymizer",
    "LMStudioAnonymizer",
    "OpenMedPIIAnonymizer",
    "PresidioAnonymizer",
    "RuleBasedAnonymizer",
    "build_anonymizer",
    "REGISTERED",
]
