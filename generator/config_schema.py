"""Single source of truth for all tunable dataset-pipeline hyperparameters.

Every numeric knob that steers sampling, generation post-processing, and
red-team QC lives here as a typed dataclass field. The defaults reproduce the
pipeline's historical hardcoded behaviour *exactly* — instantiating
:class:`PipelineConfig` with no overrides yields the same distributions the code
used before this module existed (the sole intentional change is the calendar
reference date, see :class:`DatesConfig.reference_date`).

Overrides are supplied as a nested dict (typically parsed from ``config.yaml``)
via :meth:`PipelineConfig.from_dict`; only the keys present are overridden, so a
YAML file may set a single probability without restating the rest.

Scope: "Tier 1" — probabilities, weights, ranges, and distributions. The large
*content* pools (name lists, street/city tables, institution templates, section
names, eponym lists) remain in their data modules; only their attached numeric
weights/ranges are surfaced here where cleanly separable.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, is_dataclass
from typing import Any

from generator.name_data import ORIGIN_WEIGHTS
from generator.profile_data import (
    AGE_BINS,
    MARITAL_STATUS_BY_AGE,
    TITEL_BY_ROLE,
)

# ---------------------------------------------------------------------------
# Sampling sub-configs (Stage 1)
# ---------------------------------------------------------------------------


def _difficulty_profiles_default() -> dict[str, dict[str, Any]]:
    return {
        "easy": dict(
            relatives=(0, 1),
            eponyms=(0, 0),
            lookalikes=(0, 0),
            shared_surname=False,
            needle=False,
            quasi=False,
            initials=False,
        ),
        "medium": dict(
            relatives=(0, 1),
            eponyms=(1, 2),
            lookalikes=(0, 1),
            shared_surname=False,
            needle=False,
            quasi=True,
            initials=True,
        ),
        "hard": dict(
            relatives=(1, 2),
            eponyms=(2, 3),
            lookalikes=(1, 2),
            shared_surname=0.10,
            needle=0.6,
            quasi=True,
            initials=True,
        ),
        "adversarial": dict(
            relatives=(1, 3),
            eponyms=(3, 5),
            lookalikes=(2, 4),
            shared_surname=0.20,
            needle=True,
            quasi=True,
            initials=True,
        ),
    }


@dataclass
class DifficultyConfig:
    # Weights over (easy, medium, hard, adversarial) when difficulty is unforced.
    distribution: list[float] = field(default_factory=lambda: [0.29, 0.29, 0.29, 0.13])
    # Per-tier entity counts and hard-case flags (see _DIFFICULTY_PROFILE).
    profiles: dict[str, dict[str, Any]] = field(
        default_factory=_difficulty_profiles_default
    )


@dataclass
class PiiDensityConfig:
    # Beta(alpha, beta) shape per (style -> length). Controls optional-field density.
    beta: dict[str, dict[str, list[float]]] = field(
        default_factory=lambda: {
            "Telegrammstil-Kompakt": {
                "kurz": [2.0, 4.0],
                "mittel": [3.0, 3.5],
                "ausführlich": [3.5, 3.0],
                "sehr_lang": [4.0, 2.5],
            },
            "Standard-Klinik": {
                "kurz": [3.0, 3.0],
                "mittel": [4.0, 3.0],
                "ausführlich": [5.0, 2.5],
                "sehr_lang": [5.5, 2.0],
            },
            "Akademisch-Ausführlich": {
                "kurz": [3.5, 2.5],
                "mittel": [5.0, 2.5],
                "ausführlich": [6.0, 2.0],
                "sehr_lang": [7.0, 1.5],
            },
        }
    )
    default_beta: list[float] = field(default_factory=lambda: [4.0, 3.0])
    # Optional-field gate: base_prob * (floor + scale * pii_density).
    include_floor: float = 0.4
    include_scale: float = 0.6


@dataclass
class AdversarialConfig:
    # Weights over (eponym_collision, word_collision, deep_needle, typo_text).
    mode_weights: list[float] = field(default_factory=lambda: [0.25, 0.30, 0.20, 0.25])
    typo_text_density_range: list[float] = field(default_factory=lambda: [0.01, 0.10])
    # For non-typo modes: probability of any typos, and their density range.
    other_typo_prob: float = 0.30
    other_typo_density_range: list[float] = field(default_factory=lambda: [0.005, 0.02])


@dataclass
class PersonsConfig:
    maiden_name_prob: float = 0.65
    relative_shared_surname_prob: float = 0.6
    shared_surname_remark_prob: float = 0.30
    # Extra-person count: per difficulty, (values, weights).
    extras_weights: dict[str, dict[str, list[float]]] = field(
        default_factory=lambda: {
            "easy": {"values": [0, 1], "weights": [0.70, 0.30]},
            "medium": {"values": [0, 1, 2], "weights": [0.40, 0.45, 0.15]},
            "hard": {"values": [0, 1, 2, 3], "weights": [0.30, 0.35, 0.25, 0.10]},
            "adversarial": {
                "values": [0, 1, 2, 3],
                "weights": [0.30, 0.35, 0.25, 0.10],
            },
        }
    )


@dataclass
class DatesConfig:
    # Calendar anchor for all sampled dates. Historically ``date.today()`` (a
    # non-reproducible input); pinned here so absolute dates are deterministic.
    # ISO ``YYYY-MM-DD`` string, or ``"today"`` to fall back to the run date.
    reference_date: str = "2026-09-03"
    birth_month_range: list[int] = field(default_factory=lambda: [1, 12])
    birth_day_range: list[int] = field(default_factory=lambda: [1, 28])
    years_back_values: list[int] = field(
        default_factory=lambda: [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15]
    )
    years_back_weights: list[float] = field(
        default_factory=lambda: [30, 12, 8, 8, 6, 6, 5, 5, 5, 4, 4]
    )
    years_back_days: int = 365
    admission_offset_range: list[int] = field(default_factory=lambda: [5, 365])
    # Weights over (ambulant, acute, long).
    stay_type_weights: list[float] = field(default_factory=lambda: [0.08, 0.78, 0.14])
    stay_days_ambulant: list[int] = field(default_factory=lambda: [0, 1])
    stay_days_acute: list[int] = field(default_factory=lambda: [2, 18])
    stay_days_long: list[int] = field(default_factory=lambda: [21, 90])
    # Number of extra OTHER_DATE entities: weights over (0, 1, 2, 3).
    other_dates_count_weights: list[float] = field(
        default_factory=lambda: [0.1, 0.3, 0.4, 0.2]
    )
    other_dates_density_mult: int = 3
    # Follow-up appointment offset after discharge, in days.
    wiedervorstellung_offset_range: list[int] = field(default_factory=lambda: [7, 84])


@dataclass
class AgeConfig:
    # ((lo, hi), weight) buckets — the hospital-weighted German age pyramid.
    bins: list = field(default_factory=lambda: copy.deepcopy(AGE_BINS))
    # age_min -> {status: probability}.
    marital_by_age: dict = field(
        default_factory=lambda: copy.deepcopy(MARITAL_STATUS_BY_AGE)
    )
    include_prob: float = 0.75
    extreme_low_max: int = 5  # age <= this is always mentioned
    extreme_high_min: int = 90  # age >= this is always mentioned
    inline_bracket_prob: float = 0.1  # hard/adversarial only


@dataclass
class NamesConfig:
    origin_weights: dict[str, float] = field(
        default_factory=lambda: dict(ORIGIN_WEIGHTS)
    )
    surreal_prob_hard: float = 0.10
    # Per-difficulty probabilities (with fallback for unlisted tiers).
    cross_origin_p: dict[str, float] = field(
        default_factory=lambda: {
            "easy": 0.00,
            "medium": 0.00,
            "hard": 0.40,
            "adversarial": 0.40,
        }
    )
    double_given_p: dict[str, float] = field(
        default_factory=lambda: {
            "easy": 0.00,
            "medium": 0.12,
            "hard": 0.22,
            "adversarial": 0.22,
        }
    )
    double_surname_p: dict[str, float] = field(
        default_factory=lambda: {
            "easy": 0.00,
            "medium": 0.08,
            "hard": 0.15,
            "adversarial": 0.15,
        }
    )
    nobility_prob: float = 0.03
    # role -> [(title, weight), ...]
    titel_by_role: dict = field(default_factory=lambda: copy.deepcopy(TITEL_BY_ROLE))


@dataclass
class QuasiConfig:
    # Probability the patient is a physician (drives collegial address).
    medical_prob: dict[str, float] = field(
        default_factory=lambda: {"medium": 0.06, "hard": 0.15, "adversarial": 0.15}
    )
    employer_prob: float = 0.6
    nationality_prob_de: float = 0.05
    nationality_prob_nonde: float = 0.40


@dataclass
class NegativesConfig:
    # Probability of injecting non-PII numeric ID traps (hard/adversarial).
    nonpii_prob: dict[str, float] = field(
        default_factory=lambda: {"hard": 0.3, "adversarial": 0.40}
    )
    nonpii_count_range: list[int] = field(default_factory=lambda: [1, 3])
    colliding_count_range: list[int] = field(default_factory=lambda: [1, 3])
    # Collision strength weights over (exact, substring, padded).
    colliding_mode_weights: list[float] = field(
        default_factory=lambda: [0.40, 0.35, 0.25]
    )


@dataclass
class PlacementConfig:
    # Probability a document receives non-standard PII placement hints.
    prob: dict[str, float] = field(
        default_factory=lambda: {"medium": 0.15, "hard": 0.30, "adversarial": 0.50}
    )
    # Number of hints per difficulty (lo, hi).
    count: dict[str, list[int]] = field(
        default_factory=lambda: {
            "medium": [1, 2],
            "hard": [1, 3],
            "adversarial": [2, 4],
        }
    )


@dataclass
class SectionsConfig:
    # Target section count per (length -> difficulty) as (lo, hi).
    count_ranges: dict[str, dict[str, list[int]]] = field(
        default_factory=lambda: {
            "kurz": {
                "easy": [3, 5],
                "medium": [3, 5],
                "hard": [4, 6],
                "adversarial": [4, 6],
            },
            "mittel": {
                "easy": [5, 7],
                "medium": [5, 8],
                "hard": [6, 9],
                "adversarial": [6, 10],
            },
            "ausführlich": {
                "easy": [6, 8],
                "medium": [7, 10],
                "hard": [8, 12],
                "adversarial": [9, 14],
            },
            "sehr_lang": {
                "easy": [8, 12],
                "medium": [10, 14],
                "hard": [12, 18],
                "adversarial": [14, 20],
            },
        }
    )
    default_count_range: list[int] = field(default_factory=lambda: [5, 8])
    # Format-choice biases for styles with multiple format options.
    telegramm_nonfliess_prob: float = 0.7
    akademisch_fliess_prob: float = 0.7


@dataclass
class PresentationConfig:
    # ICD/OPS code-citation register: weights over (inline, primary_only, none).
    code_style_weights: dict[str, list[float]] = field(
        default_factory=lambda: {
            "Telegrammstil-Kompakt": [0.55, 0.25, 0.20],
            "Akademisch-Ausführlich": [0.25, 0.30, 0.45],
            "default": [0.40, 0.30, 0.30],
        }
    )
    # Opening register: weights over (bericht, aufnahme, diagnose_zuerst,
    # epikrise_kurz, frei).
    opening_style_weights: list[float] = field(
        default_factory=lambda: [0.30, 0.20, 0.20, 0.15, 0.15]
    )
    has_medications_prob: float = 0.80
    # Framing element base rates per style: (briefkopf, anrede,
    # patientenstammdaten, schlussformel).
    framing_rates: dict[str, list[float]] = field(
        default_factory=lambda: {
            "Standard-Klinik": [0.50, 0.60, 0.35, 0.55],
            "Akademisch-Ausführlich": [0.70, 0.80, 0.50, 0.65],
            "Telegrammstil-Kompakt": [0.30, 0.40, 0.20, 0.40],
            "default": [0.50, 0.60, 0.35, 0.55],
        }
    )
    # Multiplier applied to briefkopf/patientenstammdaten rates when unstructured.
    framing_unstructured_mult: float = 0.5


@dataclass
class AddressConfig:
    tricky_prob_hard: float = 0.35
    house_number_range: list[int] = field(default_factory=lambda: [1, 180])
    house_suffix_prob: float = 0.2


@dataclass
class MentionConfig:
    # Number of patient mentions per difficulty: (values, weights). Drives how
    # often the patient is named (and thus how many form variants appear).
    patient_counts: dict[str, dict[str, list[float]]] = field(
        default_factory=lambda: {
            "easy": {"values": [1, 2, 3], "weights": [0.15, 0.50, 0.35]},
            "medium": {
                "values": [2, 3, 4, 5, 6],
                "weights": [0.15, 0.30, 0.30, 0.15, 0.10],
            },
            "hard": {
                "values": [2, 3, 4, 5, 6, 7],
                "weights": [0.10, 0.20, 0.30, 0.20, 0.15, 0.05],
            },
            "adversarial": {
                "values": [2, 3, 4, 5, 6, 7, 8],
                "weights": [0.05, 0.10, 0.20, 0.25, 0.20, 0.15, 0.05],
            },
        }
    )


@dataclass
class SamplingConfig:
    difficulty: DifficultyConfig = field(default_factory=DifficultyConfig)
    # Length weights per difficulty: over (kurz, mittel, ausführlich, sehr_lang).
    length_weights: dict[str, list[float]] = field(
        default_factory=lambda: {
            "easy": [0.30, 0.45, 0.20, 0.05],
            "medium": [0.20, 0.35, 0.30, 0.15],
            "hard": [0.10, 0.25, 0.35, 0.30],
            "adversarial": [0.05, 0.15, 0.35, 0.45],
        }
    )
    # Structure weights per difficulty: over (structured, mixed, unstructured).
    structure_weights: dict[str, list[float]] = field(
        default_factory=lambda: {
            "easy": [0.90, 0.08, 0.02],
            "medium": [0.70, 0.20, 0.10],
            "hard": [0.45, 0.30, 0.25],
            "adversarial": [0.30, 0.30, 0.40],
        }
    )
    pii_density: PiiDensityConfig = field(default_factory=PiiDensityConfig)
    adversarial: AdversarialConfig = field(default_factory=AdversarialConfig)
    persons: PersonsConfig = field(default_factory=PersonsConfig)
    dates: DatesConfig = field(default_factory=DatesConfig)
    age: AgeConfig = field(default_factory=AgeConfig)
    names: NamesConfig = field(default_factory=NamesConfig)
    quasi: QuasiConfig = field(default_factory=QuasiConfig)
    # Base inclusion probability for each optional field (pre density scaling).
    field_inclusion: dict[str, float] = field(
        default_factory=lambda: {
            "adresse": 0.85,
            "mobil": 0.35,
            "email": 0.30,
            "kvnr": 0.75,
            "fallnr": 0.80,
            "patid": 0.65,
            "lanr": 0.40,
            "bsnr": 0.30,
            "ik_nr": 0.20,
            "klinik_url": 0.50,
            "adresse_klinik": 0.85,
            "praxis_einweiser": 0.70,
            "klinik_vorbehandlung": 0.30,
            "reha_klinik": 0.20,
            "versicherer": 0.70,
        }
    )
    negatives: NegativesConfig = field(default_factory=NegativesConfig)
    placement: PlacementConfig = field(default_factory=PlacementConfig)
    sections: SectionsConfig = field(default_factory=SectionsConfig)
    presentation: PresentationConfig = field(default_factory=PresentationConfig)
    address: AddressConfig = field(default_factory=AddressConfig)
    mention: MentionConfig = field(default_factory=MentionConfig)
    # Probability a referring practice is named after the physician.
    praxis_named_prob: float = 0.55


# ---------------------------------------------------------------------------
# Generation post-processing (Stages 2-3)
# ---------------------------------------------------------------------------


@dataclass
class GenerationConfig:
    # Repetition guard on the LLM skeleton.
    skeleton_min_sentence_len: int = 15
    skeleton_repeat_threshold: int = 3
    # Entity-collapse predicate: flag when ratio * missing > planned.
    entity_collapse_ratio: int = 2
    # Typo injection (post-render).
    pii_typo_prob: float = 0.6
    n_pii_corruptions_range: list[int] = field(default_factory=lambda: [1, 3])
    # Character-mutation mode weights: over (keyboard, ocr, double, drop).
    typo_mode_weights: list[float] = field(
        default_factory=lambda: [0.40, 0.25, 0.20, 0.15]
    )
    # Max word-collision context forms shown to the LLM in the prompt.
    word_collision_context_max: int = 5


# ---------------------------------------------------------------------------
# Red-team QC (Stage 4)
# ---------------------------------------------------------------------------


@dataclass
class RedTeamConfig:
    # Minimum length of a reported substring for it to count as a finding.
    finding_min_len: int = 2
    # NOTE: sampling temperature is intentionally NOT configured here — the
    # red-team call uses the LLM client's default temperature.


# ---------------------------------------------------------------------------
# Top-level config + override merging
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    red_team: RedTeamConfig = field(default_factory=RedTeamConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> PipelineConfig:
        """Build a config, overlaying only the keys present in ``data``.

        ``data`` mirrors the dataclass structure (``{"sampling": {...},
        "generation": {...}, "red_team": {...}}``). Unknown keys raise, so typos
        in a YAML file surface immediately rather than being silently ignored.
        """
        cfg = cls()
        if data:
            _apply_overrides(cfg, data)
        return cfg


def _apply_overrides(obj: Any, data: dict[str, Any]) -> None:
    for key, val in data.items():
        if not hasattr(obj, key):
            raise ValueError(f"Unknown config key {key!r} on {type(obj).__name__}")
        cur = getattr(obj, key)
        if is_dataclass(cur) and isinstance(val, dict):
            _apply_overrides(cur, val)
        elif isinstance(cur, dict) and isinstance(val, dict):
            _merge_dict(cur, val)
        else:
            setattr(obj, key, val)


def _merge_dict(cur: dict[str, Any], val: dict[str, Any]) -> None:
    for k, v in val.items():
        if isinstance(cur.get(k), dict) and isinstance(v, dict):
            _merge_dict(cur[k], v)
        else:
            cur[k] = v
