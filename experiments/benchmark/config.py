"""Validated configuration for anonymizer benchmark runs."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class AnonymizerConfig(BaseModel):
    """Which anonymizer to run and how."""

    model_config = ConfigDict(extra="forbid")

    name: Literal[
        "gemini", "rule_based", "presidio", "flair", "openmed", "lmstudio"
    ] = "gemini"
    model: str = "gemini-3.7-flash"
    cache_db: Path = Path("gemini_anon_cache.db")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    retry_attempts: int = Field(default=3, ge=1)
    retry_delay_seconds: int = Field(default=5, ge=0)
    workers: int = Field(default=4, ge=1)

    # Presidio/Flair/OpenMed-specific settings (ignored by other methods).
    spacy_model: str = "de_core_news_lg"
    flair_model: str = "flair/ner-german-large"
    pii_model: str = "OpenMed/OpenMed-PII-German-ClinicalE5-Base-109M-v1"
    language: str = "de"
    score_threshold: float = Field(default=0.35, ge=0.0, le=1.0)

    # LM Studio settings (ignored by other methods). ``lmstudio_model`` is the
    # model key LM Studio serves (see ``GET /v1/models``).
    lmstudio_model: str = "local-model"
    lmstudio_base_url: str = "http://localhost:1234/v1"
    lmstudio_timeout: float = Field(default=600.0, gt=0.0)
    #: Upper bound on generated tokens. The model rewrites the whole document, so
    #: output length ~ input length; set this high enough to avoid truncating long
    #: letters. With ``parallel N`` in LM Studio each request gets ctx/N tokens, so
    #: prompt + max_tokens must fit that per-slot budget for a full rewrite.
    lmstudio_max_tokens: int = Field(default=16384, ge=1)


class EvaluationSettings(BaseModel):
    """How predictions are scored against the gold spans (text mode)."""

    model_config = ConfigDict(extra="forbid")

    matching: Literal["strict", "lenient", "token"] = "lenient"
    iou_threshold: float = Field(default=0.5, gt=0.0, le=1.0)
    verbose: bool = False


class BenchmarkConfig(BaseModel):
    """Complete, serializable anonymizer-benchmark specification.

    The evaluation is fixed to the text-replacement path: the anonymizer returns
    full PHI-replaced text, which is aligned back to offsets and scored against
    the gold spans by :mod:`experiments.benchmark.scoring` (``--mode text``).
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    #: Gold-standard dataset (pipeline output JSONL with doc_id/text/spans).
    dataset_path: Path
    #: Directory for predictions and the evaluation report.
    output_dir: Path = Path("experiments/results/benchmark")
    #: Process only the first N documents (handy for smoke tests).
    limit: int | None = Field(default=None, ge=1)
    anonymizer: AnonymizerConfig = Field(default_factory=AnonymizerConfig)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)


def load_config(path: str | Path) -> BenchmarkConfig:
    """Load and validate a YAML benchmark configuration.

    Relative ``dataset_path``, ``output_dir``, and ``anonymizer.cache_db`` are
    resolved against the configuration file's directory, mirroring the corpus
    evaluator's behaviour.
    """
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"configuration must be a YAML mapping: {config_path}")
    config = BenchmarkConfig.model_validate(raw)

    base = config_path.resolve().parent
    if not config.dataset_path.is_absolute():
        config.dataset_path = (base / config.dataset_path).resolve()
    if not config.output_dir.is_absolute():
        config.output_dir = (base / config.output_dir).resolve()
    if not config.anonymizer.cache_db.is_absolute():
        config.anonymizer.cache_db = (base / config.anonymizer.cache_db).resolve()
    return config
