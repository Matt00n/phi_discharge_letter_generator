"""Validated configuration for corpus evaluation runs."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ProfileFieldConfig(BaseModel):
    """One persisted sampling dimension used in mixed-type profile distances."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    path: str = Field(min_length=1)
    kind: Literal["categorical", "numeric", "set"]
    weight: float = Field(default=1.0, gt=0.0)
    numeric_min: float | None = None
    numeric_max: float | None = None
    casefold: bool = True
    include_value_labels_in_report: bool = False

    @model_validator(mode="after")
    def numeric_bounds_are_valid(self) -> ProfileFieldConfig:
        bounds = (self.numeric_min, self.numeric_max)
        if self.kind != "numeric" and any(value is not None for value in bounds):
            raise ValueError(
                "numeric_min/numeric_max are only valid for numeric fields"
            )
        if (self.numeric_min is None) != (self.numeric_max is None):
            raise ValueError("numeric_min and numeric_max must be provided together")
        if (
            self.numeric_min is not None
            and self.numeric_max is not None
            and self.numeric_min >= self.numeric_max
        ):
            raise ValueError("numeric_min must be smaller than numeric_max")
        return self


class DatasetConfig(BaseModel):
    """How to load one corpus without coupling the evaluator to its schema."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    path: Path
    format: Literal["auto", "jsonl", "json", "csv", "txt"] = "auto"
    encoding: str = "utf-8"
    records_field: str | None = None
    text_field: str = "text"
    id_field: str | None = "doc_id"
    phi_spans_field: str | None = "spans"
    span_start_field: str = "start"
    span_end_field: str = "end"
    span_label_field: str | None = "label"
    span_form_field: str | None = "form"
    validate_phi_offsets: bool = True
    negatives_field: str | None = None
    section_sequence_field: str | None = None
    icd_fields: list[str] = Field(default_factory=list)
    extract_icd_from_text: bool = True
    extract_ops_from_text: bool = False
    icd_catalog_path: Path | None = None
    language: str = "de"
    spacy_model: str | None = "de_core_news_lg"
    profile_fields: list[ProfileFieldConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def profile_names_are_unique(self) -> DatasetConfig:
        names = [field.name for field in self.profile_fields]
        if len(names) != len(set(names)):
            raise ValueError("profile field names must be unique within a dataset")
        return self


class DeduplicationConfig(BaseModel):
    """Exact and lexical near-duplicate detection settings."""

    model_config = ConfigDict(extra="forbid")

    ngram_size: int = Field(default=5, ge=1)
    jaccard_threshold: float = Field(default=0.80, gt=0.0, le=1.0)
    exhaustive_max_documents: int = Field(default=2_000, ge=2)
    minhash_permutations: int = Field(default=256, ge=16)
    max_reported_pairs: int = Field(default=100, ge=0)


class ModelConfig(BaseModel):
    """Pinned model choices for model-based metrics."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    embedding_revision: str | None = None
    embedding_batch_size: int = Field(default=32, ge=1)
    embedding_chunk_tokens: int = Field(default=384, ge=16)
    bertscore_model: str = "FacebookAI/xlm-roberta-large"
    bertscore_revision: str | None = None
    bertscore_num_layers: int = Field(default=17, ge=1)
    bertscore_batch_size: int = Field(default=16, ge=1)
    bertscore_max_tokens: int = Field(default=510, ge=16)
    bertscore_pair_sample_size: int = Field(default=1_000, ge=1)
    subword_tokenizer_model: str | None = None
    subword_tokenizer_revision: str | None = None
    device: str | None = None


class ProfileDiversityConfig(BaseModel):
    """Exact mixed-type separation checks for persisted sampling profiles."""

    model_config = ConfigDict(extra="forbid")

    close_distance_threshold: float = Field(default=0.10, ge=0.0, le=1.0)
    exhaustive_max_documents: int = Field(default=5_000, ge=2)
    pair_distance_sample_size: int = Field(default=100_000, ge=1)
    max_reported_closest_pairs: int = Field(default=100, ge=0)
    max_reported_field_frequencies: int = Field(default=100, ge=0)


class ReadabilityConfig(BaseModel):
    """Formula-based readability evaluation settings."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True


class LLMJudgeConfig(BaseModel):
    """Cost- and privacy-explicit configuration for the Gemini realism judge."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    allow_external_api_calls: bool = False
    model: str = "gemini-3.7-flash"
    thinking_level: Literal["low", "medium", "high"] = "medium"
    backend: Literal["vertex_ai", "google_ai"] = "vertex_ai"
    api_key_environment_variable: str = "GEMINI_API_KEY"
    project: str | None = None
    project_environment_variable: str = "GCP_PROJECT_ID"
    location: str | None = None
    location_environment_variable: str = "GCP_LOCATION"
    cache_path: Path = Path("../results/gemini_realism_judge.sqlite3")
    sample_size: int = Field(default=200, ge=1)
    repetitions: int = Field(default=1, ge=1)
    max_document_characters: int = Field(default=100_000, ge=1)
    max_output_tokens: int = Field(default=8_192, ge=512)
    max_concurrent_requests: int = Field(default=12, ge=1)
    max_retries: int = Field(default=3, ge=1)
    retry_initial_seconds: float = Field(default=2.0, ge=0.0)
    passing_score: int = Field(default=4, ge=1, le=5)
    include_rationales_in_report: bool = False


class EvaluationConfig(BaseModel):
    """Complete, serializable experiment specification."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[2] = 2
    datasets: list[DatasetConfig] = Field(min_length=1)
    output_path: Path = Path("experiments/results/diversity_report.json")
    overwrite: bool = False
    random_seed: int = 1729
    pair_sample_size: int = Field(default=10_000, ge=1)
    bootstrap_samples: int = Field(default=2_000, ge=0)
    confidence_level: float = Field(default=0.95, gt=0.0, lt=1.0)
    top_k_ngrams: int = Field(default=50, ge=1)
    max_distribution_documents: int = Field(default=2_000, ge=2)
    bandwidth_sample_size: int = Field(default=2_000, ge=2)
    lexical_casefold: bool = True
    strict_optional_dependencies: bool = True
    deduplication: DeduplicationConfig = Field(default_factory=DeduplicationConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    profile_diversity: ProfileDiversityConfig = Field(
        default_factory=ProfileDiversityConfig
    )
    readability: ReadabilityConfig = Field(default_factory=ReadabilityConfig)
    llm_judge: LLMJudgeConfig = Field(default_factory=LLMJudgeConfig)

    @model_validator(mode="after")
    def names_are_unique(self) -> EvaluationConfig:
        names = [dataset.name for dataset in self.datasets]
        if len(names) != len(set(names)):
            raise ValueError("dataset names must be unique")
        return self


def load_config(path: str | Path) -> EvaluationConfig:
    """Load and validate a YAML experiment configuration."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"configuration must be a YAML mapping: {config_path}")
    config = EvaluationConfig.model_validate(raw)

    base = config_path.resolve().parent
    for dataset in config.datasets:
        if not dataset.path.is_absolute():
            dataset.path = (base / dataset.path).resolve()
        if dataset.icd_catalog_path and not dataset.icd_catalog_path.is_absolute():
            dataset.icd_catalog_path = (base / dataset.icd_catalog_path).resolve()
    if not config.llm_judge.cache_path.is_absolute():
        config.llm_judge.cache_path = (base / config.llm_judge.cache_path).resolve()
    if not config.output_path.is_absolute():
        config.output_path = (base / config.output_path).resolve()
    return config
