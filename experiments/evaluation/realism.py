"""Formula-based textual realism metrics for clinical documents."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from experiments.evaluation.corpus import Corpus
from experiments.evaluation.statistics import numeric_summary

SMOG_IMPLEMENTATION_MINIMUM_SENTENCES = 3
SMOG_NORMING_SENTENCES = 30


@dataclass(frozen=True, slots=True)
class ReadabilityCounts:
    """Primitive counts and scores for one text."""

    sentences: int
    words: int
    syllables: int
    polysyllabic_words: int
    flesch_kincaid_grade_level: float | None
    smog_index: float | None
    smog_meets_30_sentence_recommendation: bool


def _readability_counts(text: str, textstat_module: object) -> ReadabilityCounts:
    sentences = int(textstat_module.sentence_count(text))
    words = int(textstat_module.lexicon_count(text))
    syllables = int(textstat_module.syllable_count(text))
    polysyllables = int(textstat_module.polysyllabcount(text))
    flesch_kincaid = None
    if sentences > 0 and words > 0:
        flesch_kincaid = 0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59
    smog = None
    if sentences >= SMOG_IMPLEMENTATION_MINIMUM_SENTENCES:
        smog = 1.043 * (polysyllables * (30.0 / sentences)) ** 0.5 + 3.1291
    return ReadabilityCounts(
        sentences=sentences,
        words=words,
        syllables=syllables,
        polysyllabic_words=polysyllables,
        flesch_kincaid_grade_level=flesch_kincaid,
        smog_index=smog,
        smog_meets_30_sentence_recommendation=sentences >= SMOG_NORMING_SENTENCES,
    )


def readability_metrics(
    corpus: Corpus,
    *,
    bootstrap_samples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, object]:
    """Compute per-document and pooled Flesch--Kincaid and SMOG proxies."""

    try:
        import textstat
    except ImportError as exc:
        raise ImportError(
            "readability evaluation requires the 'textstat' experiment dependency"
        ) from exc

    language = corpus.language.replace("-", "_")
    textstat.set_lang(language)
    per_document = [
        _readability_counts(document.text, textstat) for document in corpus.documents
    ]
    pooled = _readability_counts(
        "\n".join(document.text for document in corpus.documents), textstat
    )
    flesch_kincaid_values = [
        result.flesch_kincaid_grade_level
        for result in per_document
        if result.flesch_kincaid_grade_level is not None
    ]
    smog_values = [
        result.smog_index for result in per_document if result.smog_index is not None
    ]
    normed_smog_values = [
        result.smog_index
        for result in per_document
        if result.smog_index is not None
        and result.smog_meets_30_sentence_recommendation
    ]
    summary_kwargs = {
        "bootstrap_samples": bootstrap_samples,
        "confidence_level": confidence_level,
    }
    is_english = language.casefold().startswith("en")
    return {
        "status": "computed",
        "method": {
            "counting_implementation": "textstat",
            "syllabification_language": language,
            "flesch_kincaid_formula": (
                "0.39*(words/sentences) + 11.8*(syllables/words) - 15.59"
            ),
            "smog_formula": ("1.043*sqrt(polysyllabic_words*(30/sentences)) + 3.1291"),
            "smog_polysyllabic_definition": "at least three syllables",
            "calibration_language": "English",
            "grade_level_interpretation_valid_for_configured_language": is_english,
        },
        "eligibility": {
            "documents": len(per_document),
            "documents_with_flesch_kincaid_score": len(flesch_kincaid_values),
            "documents_with_smog_score_at_implementation_minimum_3_sentences": len(
                smog_values
            ),
            "documents_meeting_smog_30_sentence_norming_recommendation": len(
                normed_smog_values
            ),
        },
        "document_level": {
            "flesch_kincaid_grade_level": numeric_summary(
                flesch_kincaid_values, seed=seed + 1, **summary_kwargs
            ),
            "smog_index_at_least_3_sentences": numeric_summary(
                smog_values, seed=seed + 2, **summary_kwargs
            ),
            "smog_index_at_least_30_sentences": numeric_summary(
                normed_smog_values, seed=seed + 3, **summary_kwargs
            ),
            "sentence_count": numeric_summary(
                [result.sentences for result in per_document],
                seed=seed + 4,
                **summary_kwargs,
            ),
        },
        "pooled_corpus": asdict(pooled),
        "interpretation": (
            "Flesch--Kincaid Grade Level and SMOG are English-calibrated proxies. "
            "For non-English corpora, language-aware syllable estimates do not make "
            "the resulting values local educational grade levels. SMOG values for "
            "documents below 30 sentences are reported separately because the "
            "formula was normed on 30-sentence samples."
        ),
    }
