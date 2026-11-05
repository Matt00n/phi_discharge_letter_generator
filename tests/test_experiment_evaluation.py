from __future__ import annotations

import json
import math
from collections import Counter

import numpy as np
import pytest

from experiments.evaluation.aggregate import aggregate_metrics, icd_metrics, ops_metrics
from experiments.evaluation.comparison import (
    frechet_gaussian_distance,
    jensen_shannon_divergence,
    lexical_comparison,
    mmd_rbf,
)
from experiments.evaluation.composite import aggregate_diversity_index
from experiments.evaluation.config import (
    DatasetConfig,
    DeduplicationConfig,
    LLMJudgeConfig,
    ProfileDiversityConfig,
    ProfileFieldConfig,
)
from experiments.evaluation.corpus import Corpus, Document, SpanSurface, load_corpus
from experiments.evaluation.deduplication import deduplication_metrics
from experiments.evaluation.lexical import entropy_bits, lexical_metrics, mtld, yules_k
from experiments.evaluation.llm_judge import (
    ClinicalJudgeResponse,
    JudgeCallResult,
    JudgmentRecord,
    aggregate_judgments,
    llm_judge_metrics,
)
from experiments.evaluation.phi_surface import (
    negative_trap_metrics,
    phi_surface_metrics,
)
from experiments.evaluation.profiles import profile_diversity_metrics
from experiments.evaluation.realism import readability_metrics
from experiments.evaluation.similarity import sample_document_pairs
from experiments.evaluation.statistics import sampled_pair_summary
from experiments.evaluation.structural import structural_metrics


def _document(
    doc_id: str,
    text: str,
    *,
    spans: int | None = 0,
    icd_text: tuple[str, ...] = (),
    icd_fields: tuple[str, ...] = (),
    profile: tuple[str | float | tuple[str, ...] | None, ...] = (),
    phi_spans: tuple[SpanSurface, ...] = (),
    negative_spans: tuple[SpanSurface, ...] = (),
    section_sequence: tuple[str, ...] | None = None,
    ops_text_codes: tuple[str, ...] = (),
) -> Document:
    return Document(
        doc_id,
        text,
        spans,
        icd_text,
        icd_fields,
        profile,
        phi_spans,
        negative_spans,
        section_sequence,
        ops_text_codes,
    )


def _corpus(
    name: str,
    documents: list[Document],
    *,
    profile_schema: tuple[ProfileFieldConfig, ...] = (),
) -> Corpus:
    return Corpus(
        name,
        tuple(documents),
        "fixture.jsonl",
        "de",
        None,
        None,
        profile_schema,
    )


def test_load_corpus_validates_spans_and_extracts_icd(tmp_path) -> None:
    source = tmp_path / "data.jsonl"
    record = {
        "id": "a",
        "body": "Diagnose I25.1, Name Max",
        "annotations": [{"begin": 20, "finish": 23}],
        "clinical": {"codes": ["E11.9"]},
    }
    source.write_text(json.dumps(record) + "\n", encoding="utf-8")
    config = DatasetConfig(
        name="fixture",
        path=source,
        text_field="body",
        id_field="id",
        phi_spans_field="annotations",
        span_start_field="begin",
        span_end_field="finish",
        icd_fields=["clinical.codes"],
        spacy_model=None,
    )

    corpus = load_corpus(config)

    assert corpus.documents[0].phi_span_count == 1
    assert corpus.documents[0].icd_text_codes == ("I25.1",)
    assert corpus.documents[0].icd_field_codes == ("E11.9",)


def test_load_corpus_rejects_overlapping_offsets(tmp_path) -> None:
    source = tmp_path / "data.jsonl"
    record = {
        "text": "abcdef",
        "spans": [{"start": 0, "end": 3}, {"start": 2, "end": 5}],
    }
    source.write_text(json.dumps(record) + "\n", encoding="utf-8")
    config = DatasetConfig(name="fixture", path=source, id_field=None)

    with pytest.raises(ValueError, match="overlapping PHI spans"):
        load_corpus(config)


def test_aggregate_does_not_report_partial_phi_total() -> None:
    corpus = _corpus(
        "fixture",
        [_document("a", "abc", spans=2), _document("b", "de", spans=None)],
    )

    result = aggregate_metrics(
        corpus,
        bootstrap_samples=0,
        confidence_level=0.95,
        seed=7,
        casefold=True,
    )

    assert result["number_of_documents"] == 2
    assert result["total_characters"] == 5
    assert result["total_phi_spans"] is None
    assert result["document_length_characters"]["std_population"] == 0.5
    assert result["document_length_characters"]["std_sample"] == pytest.approx(
        math.sqrt(0.5)
    )
    assert result["standard_deviations_across_documents"] == {
        "definition": "population standard deviation across documents",
        "document_length_characters": 0.5,
        "phi_spans_per_document": 0.0,
    }


def test_icd_combinations_and_catalog_coverage(tmp_path) -> None:
    catalog = tmp_path / "icd.txt"
    catalog.write_text("I25.1 One\nE11.9 Two\n", encoding="utf-8")
    corpus = Corpus(
        "fixture",
        (
            _document("a", "", icd_text=("I25.1",)),
            _document("b", "", icd_fields=("E11.9",)),
            _document("c", ""),
        ),
        "fixture.jsonl",
        "de",
        None,
        str(catalog),
    )

    result = icd_metrics(corpus, top_k=10)

    assert result["unique_codes_observed"] == 2
    assert result["unique_document_code_combinations_including_empty"] == 3
    assert result["unique_nonempty_document_code_combinations"] == 2
    assert result["catalogue_coverage"]["exact_code_coverage"] == 1.0


def test_lexical_formulae() -> None:
    tokens = ["a", "a", "b"]

    assert entropy_bits({"a": 2, "b": 1}) == pytest.approx(0.918295834)
    assert yules_k(tokens) == pytest.approx(10_000 * 2 / 9)
    assert mtld(["a", "b", "c"]) == pytest.approx(3.0)


def test_pair_sampling_is_deterministic_and_exhaustive() -> None:
    exhaustive, metadata = sample_document_pairs(4, 100, seed=3)
    first, _ = sample_document_pairs(100, 20, seed=42)
    second, _ = sample_document_pairs(100, 20, seed=42)

    assert exhaustive == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    assert metadata["sampling"] == "exhaustive"
    assert first == second
    assert len(first) == len(set(first)) == 20
    assert all(left < right < 100 for left, right in first)


def test_exhaustive_pair_summary_has_zero_sampling_uncertainty() -> None:
    result = sampled_pair_summary(
        [0.1, 0.2, 0.3], total_possible_pairs=3, confidence_level=0.95
    )

    assert result["mean_confidence_interval"] == pytest.approx([0.2, 0.2])


def test_deduplication_separates_raw_normalized_and_near() -> None:
    corpus = _corpus(
        "fixture",
        [
            _document("a", "Alpha beta gamma delta epsilon"),
            _document("b", "  ALPHA beta gamma delta epsilon  "),
            _document("c", "completely unrelated words occur here"),
        ],
    )
    config = DeduplicationConfig(
        ngram_size=2,
        jaccard_threshold=0.8,
        exhaustive_max_documents=10,
    )

    result = deduplication_metrics(corpus, config, seed=9)

    assert result["raw_exact"]["duplicate_pairs"] == 0
    assert result["normalized_exact"]["duplicate_pairs"] == 1
    assert result["near_duplicate"]["qualifying_pairs_including_normalized_exact"] == 1


def test_lexical_comparison_has_directional_gaps_and_bounded_jsd() -> None:
    left = _corpus("left", [_document("a", "alpha beta")])
    right = _corpus("right", [_document("b", "beta gamma")])

    result = lexical_comparison(left, right, casefold=True)

    assert result["vocabulary_jaccard_similarity"] == pytest.approx(1 / 3)
    gaps = result["lexical_coverage_gaps"]
    assert gaps["left_relative_to_right"]["type_gap_rate"] == pytest.approx(0.5)
    assert 0.0 <= result["lexical_shift_jensen_shannon_bits"]["unigrams"] <= 1.0
    assert jensen_shannon_divergence(
        Counter({"a": 1}),
        Counter({"b": 1}),
    ) == pytest.approx(1.0)


def test_embedding_distances_are_zero_for_identical_samples() -> None:
    values = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])

    mmd = mmd_rbf(values, values, sigma=1.0)
    frechet = frechet_gaussian_distance(values, values)

    assert mmd["mmd_squared_biased"] == pytest.approx(0.0, abs=1e-12)
    assert frechet["frechet_distance_squared"] == pytest.approx(0.0, abs=1e-10)
    assert math.isfinite(frechet["covariance_regularization"])


def test_profile_extraction_supports_mixed_types(tmp_path) -> None:
    source = tmp_path / "profiles.jsonl"
    source.write_text(
        json.dumps(
            {
                "text": "Brief.",
                "spans": [],
                "metadata": {
                    "age": 42,
                    "department": "Innere Medizin",
                    "sections": ["Diagnosen", "Verlauf", "Diagnosen"],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = DatasetConfig(
        name="profiles",
        path=source,
        id_field=None,
        spacy_model=None,
        profile_fields=[
            ProfileFieldConfig(
                name="age",
                path="metadata.age",
                kind="numeric",
                numeric_min=0,
                numeric_max=110,
            ),
            ProfileFieldConfig(
                name="department",
                path="metadata.department",
                kind="categorical",
            ),
            ProfileFieldConfig(name="sections", path="metadata.sections", kind="set"),
        ],
    )

    document = load_corpus(config).documents[0]

    assert document.profile == (
        42.0,
        "innere medizin",
        ("diagnosen", "verlauf"),
    )


def test_profile_diversity_detects_close_and_duplicate_profiles() -> None:
    schema = (
        ProfileFieldConfig(
            name="age",
            path="age",
            kind="numeric",
            numeric_min=0,
            numeric_max=100,
        ),
        ProfileFieldConfig(name="department", path="department", kind="categorical"),
        ProfileFieldConfig(name="sections", path="sections", kind="set"),
    )
    corpus = _corpus(
        "profiles",
        [
            _document("a", "", profile=(10.0, "a", ("x",))),
            _document("b", "", profile=(90.0, "b", ("y",))),
            _document("c", "", profile=(10.0, "a", ("x",))),
        ],
        profile_schema=schema,
    )

    result = profile_diversity_metrics(
        corpus,
        ProfileDiversityConfig(close_distance_threshold=0.1),
        seed=11,
    )

    assert result["joint_profiles"]["exact_duplicate_complete_profile_pairs"] == 1
    separation = result["separation"]
    assert separation["comparable_pairs"] == 3
    assert separation["pairs_below_close_distance_threshold"] == 1
    assert separation["passes_no_close_profiles_check"] is False
    assert separation["nearest_neighbor_distance_per_document"]["min"] == 0.0


def test_profile_diversity_can_certify_no_close_pairs() -> None:
    field = ProfileFieldConfig(name="group", path="group", kind="categorical")
    corpus = _corpus(
        "separated",
        [
            _document("a", "", profile=("a",)),
            _document("b", "", profile=("b",)),
            _document("c", "", profile=("c",)),
        ],
        profile_schema=(field,),
    )

    result = profile_diversity_metrics(
        corpus,
        ProfileDiversityConfig(close_distance_threshold=0.1),
        seed=5,
    )

    assert result["separation"]["passes_no_close_profiles_check"] is True
    assert result["separation"]["nearest_neighbor_distance_per_document"]["min"] == 1.0


def test_readability_separates_smog_norming_eligibility() -> None:
    long_document = " ".join(["Die Behandlung verlief komplikationslos."] * 30)
    corpus = _corpus(
        "readability",
        [_document("long", long_document), _document("short", "Kurzer Satz.")],
    )

    result = readability_metrics(
        corpus,
        bootstrap_samples=0,
        confidence_level=0.95,
        seed=3,
    )

    assert (
        result["method"]["grade_level_interpretation_valid_for_configured_language"]
        is False
    )
    assert (
        result["eligibility"][
            "documents_meeting_smog_30_sentence_norming_recommendation"
        ]
        == 1
    )
    assert result["document_level"]["flesch_kincaid_grade_level"]["n"] == 2
    assert result["document_level"]["smog_index_at_least_30_sentences"]["n"] == 1


def _judge_response(score: int, *, major_error: bool = False) -> ClinicalJudgeResponse:
    return ClinicalJudgeResponse(
        structure_and_genre=score,
        clinical_plausibility=score,
        internal_consistency=score,
        temporal_consistency=score,
        care_trajectory_coherence=score,
        discharge_plan_completeness=score,
        linguistic_and_register_realism=score,
        overall_realism=score,
        has_major_error=major_error,
        appears_synthetic=major_error,
        failure_tags=["clinical_impossibility"] if major_error else [],
        concise_rationale="Concise audit rationale.",
    )


def test_llm_judge_aggregation_uses_document_as_primary_unit() -> None:
    records = [
        JudgmentRecord(
            document_index=document_index,
            repetition=repetition,
            result=JudgeCallResult(
                response=_judge_response(score, major_error=score == 2),
                model_version="gemini-test-001",
                usage={"total_token_count": 10},
                cache_hit=False,
            ),
        )
        for document_index, scores in enumerate(((4, 5), (2, 4)))
        for repetition, score in enumerate(scores)
    ]
    config = LLMJudgeConfig(repetitions=2, passing_score=4)

    result = aggregate_judgments(
        records,
        config,
        bootstrap_samples=0,
        confidence_level=0.95,
        seed=9,
    )

    overall = result["dimensions"]["overall_realism"]
    assert overall["document_mean_score"]["n"] == 2
    assert overall["document_mean_score"]["mean"] == pytest.approx(3.75)
    assert overall["documents_at_or_above_passing_score"] == 1
    assert result["within_document_repeat_stability"]["status"] == "computed"
    assert result["requests"]["api_calls_this_run"] == 4
    assert (
        result["document_results"][0]["repetitions"][0].get("concise_rationale") is None
    )


def test_llm_judge_requires_explicit_external_api_acknowledgement() -> None:
    corpus = _corpus("guarded", [_document("a", "Brief.")])
    config = LLMJudgeConfig(enabled=True, allow_external_api_calls=False)

    with pytest.raises(ValueError, match="allow_external_api_calls"):
        llm_judge_metrics(
            corpus,
            config,
            bootstrap_samples=0,
            confidence_level=0.95,
            seed=1,
        )

    assert config.model == "gemini-3.7-flash"


def test_load_corpus_extracts_surfaces_negatives_sections_and_ops(tmp_path) -> None:
    source = tmp_path / "data.jsonl"
    record = {
        "text": "Herr Dr. Meier, OP 5-470.",
        "spans": [{"start": 5, "end": 14, "label": "STAFF_NAME", "form": "hon_last"}],
        "negatives": [{"start": 19, "end": 24, "label": "LOOKALIKE", "form": None}],
        "metadata": {"sections": ["Anamnese", "Epikrise", "Anamnese"]},
    }
    source.write_text(json.dumps(record) + "\n", encoding="utf-8")
    config = DatasetConfig(
        name="fixture",
        path=source,
        id_field=None,
        negatives_field="negatives",
        section_sequence_field="metadata.sections",
        extract_ops_from_text=True,
        spacy_model=None,
    )

    document = load_corpus(config).documents[0]

    assert document.phi_spans == (SpanSurface("Dr. Meier", "STAFF_NAME", "hon_last"),)
    assert document.negative_spans == (SpanSurface("5-470", "LOOKALIKE", None),)
    assert document.section_sequence == ("Anamnese", "Epikrise", "Anamnese")
    assert document.ops_text_codes == ("5-470",)


def test_phi_surface_reports_surface_and_mention_form_diversity() -> None:
    corpus = _corpus(
        "phi",
        [
            _document(
                "a",
                "",
                spans=2,
                phi_spans=(
                    SpanSurface("Meier", "STAFF_NAME", "last"),
                    SpanSurface("Dr. Meier", "STAFF_NAME", "hon_last"),
                ),
            ),
            _document(
                "b",
                "",
                spans=1,
                phi_spans=(SpanSurface("Schmidt", "PATIENT_NAME", "last"),),
            ),
        ],
    )

    result = phi_surface_metrics(corpus, max_reported_labels=10)

    assert result["total_annotations"] == 3
    assert result["unique_surface_strings"] == 3
    assert result["labels"]["normalized_entropy"] == pytest.approx(0.9182958340)
    assert result["mention_forms"]["unique_mention_forms"] == 2
    staff = next(row for row in result["per_label"] if row["label"] == "STAFF_NAME")
    assert staff["unique_surface_strings"] == 2
    assert staff["mention_form_normalized_entropy"] == pytest.approx(1.0)


def test_phi_surface_reports_not_configured_without_annotations() -> None:
    corpus = _corpus("no_phi", [_document("a", "Brief.", spans=None)])

    assert phi_surface_metrics(corpus, max_reported_labels=10)["status"] == (
        "not_configured"
    )


def test_negative_trap_diversity_counts_unique_surfaces_per_label() -> None:
    corpus = _corpus(
        "traps",
        [
            _document(
                "a",
                "",
                negative_spans=(
                    SpanSurface("Bell", "EPONYM", None),
                    SpanSurface("Bell", "EPONYM", None),
                ),
            ),
            _document(
                "b",
                "",
                negative_spans=(SpanSurface("XY-1", "NONPII_ID", None),),
            ),
        ],
    )

    result = negative_trap_metrics(corpus, max_reported_labels=10)

    assert result["total_traps"] == 3
    assert result["unique_surface_strings"] == 2
    eponym = next(row for row in result["per_label"] if row["label"] == "EPONYM")
    assert eponym["traps"] == 2
    assert eponym["unique_surface_strings"] == 1
    assert result["traps_per_document"]["mean"] == pytest.approx(1.5)


def test_structural_diversity_uses_ordered_section_patterns() -> None:
    corpus = _corpus(
        "structure",
        [
            _document("a", "", section_sequence=("A", "B", "A")),
            _document("b", "", section_sequence=("A", "B", "A")),
            _document("c", "", section_sequence=("B", "A")),
        ],
    )

    result = structural_metrics(corpus, max_reported_patterns=10)

    assert result["unique_ordered_section_patterns"] == 2
    assert result["section_pattern_normalized_entropy"] == pytest.approx(0.9182958340)
    assert result["unique_section_labels"] == 2
    assert result["sections_per_document"]["mean"] == pytest.approx(8 / 3)


def test_ops_metrics_group_codes_by_chapter() -> None:
    corpus = _corpus(
        "ops",
        [
            _document("a", "", ops_text_codes=("5-470", "8-831")),
            _document("b", "", ops_text_codes=("5-470",)),
        ],
    )

    result = ops_metrics(corpus, top_k=10)

    assert result["unique_codes_observed_in_text"] == 2
    assert result["documents_with_at_least_one_code"] == 2
    assert result["top_codes"][0] == {"code": "5-470", "document_count": 2}
    chapters = {row["chapter"] for row in result["top_chapters"]}
    assert chapters == {"5", "8"}


def test_lexical_adds_hapax_ratio_ttr_and_trigram_novelty() -> None:
    corpus = _corpus(
        "lexical",
        [_document("a", "a a b"), _document("b", "c c d")],
    )

    result = lexical_metrics(
        corpus,
        top_k=5,
        casefold=True,
        bootstrap_samples=0,
        confidence_level=0.95,
        seed=1,
    )

    assert result["corpus_hapax_ratio"] == pytest.approx(0.5)
    assert result["type_token_ratio_per_document"]["mean"] == pytest.approx(2 / 3)
    assert result["trigram_novelty_per_document"]["mean"] == pytest.approx(1.0)


def test_aggregate_diversity_index_is_harmonic_and_omits_missing() -> None:
    result = aggregate_diversity_index({"origin": 1.0, "style": 0.5, "surname": None})

    assert result["harmonic_mean_normalized_entropy"] == pytest.approx(2 / 3)
    assert result["dimensions_included"] == ["origin", "style"]
    assert result["dimensions_omitted"] == ["surname"]


def test_aggregate_diversity_index_collapses_on_zero_dimension() -> None:
    result = aggregate_diversity_index({"a": 0.9, "b": 0.0})

    assert result["harmonic_mean_normalized_entropy"] == 0.0
    assert result["zero_entropy_dimensions"] == ["b"]

    empty = aggregate_diversity_index({"a": None})
    assert empty["status"] == "not_computed"
