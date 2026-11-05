"""Pipeline orchestrator: Stage 1 -> 2 -> 3 -> 4 for a single document.

Stage 2 (LLM authoring) is injectable so the deterministic stages can be tested
without any API access.
"""

from __future__ import annotations

import hashlib
import logging
import re as _re
from dataclasses import dataclass
from typing import Any, Callable, Optional

from generator.clinical_seeds import ClinicalSeedsManager
from generator.config_schema import PipelineConfig
from generator.labels import PERSON_LABELS
from generator.profile_sampler import GenerationSpec, ProfileSampler
from generator.red_team import (
    RED_TEAM_SYSTEM,
    RedTeamResponse,
    build_red_team_user,
    cache_key_for,
    compare,
)
from generator.render import (
    Catalog,
    RenderResult,
    render,
    scan_leaks,
    scan_negatives,
    validate_spans,
)

logger = logging.getLogger(__name__)
from generator.skeleton_prompt import (
    SkeletonResponse,
    build_system_instruction,
    build_user_content,
)
from generator.profile_data import EMBEDDED_FORMAT_INSTRUCTIONS
from generator.typo_injection import apply_typo_injection

# A skeleton provider turns (spec, catalog) into skeleton text.
SkeletonProvider = Callable[[GenerationSpec, Catalog], str]

# Format hints (e.g. "tabellarisch") reach the LLM as bracketed tags on section
# headings ("Diagnostik [tabellarisch]"); under the "use exactly these headings"
# instruction the model sometimes copies the tag into the heading. The tag is an
# authoring directive, not content.
_FORMAT_TAG_RE = _re.compile(
    r"[ \t]*\[(?:"
    + "|".join(_re.escape(k) for k in EMBEDDED_FORMAT_INSTRUCTIONS)
    + r")\]"
)
# The inline age variant already renders with its own parentheses ("(85)"); the
# model occasionally wraps a redundant outer pair around the placeholder, giving
# "((85))". Strip any parens the model added around the inline placeholder.
_INLINE_AGE_PARENS_RE = _re.compile(
    r"\(+\s*(\[\[\s*alter\b[^\]]*variant=inline[^\]]*\]\])\s*\)+"
)


def _scrub_skeleton(skeleton: str) -> str:
    """Remove authoring artifacts the LLM leaks verbatim into the skeleton.

    Runs before ``render()`` so gold-span offsets (assigned by construction)
    stay correct without any re-alignment.
    """
    skeleton = _FORMAT_TAG_RE.sub("", skeleton)
    skeleton = _INLINE_AGE_PARENS_RE.sub(r"\1", skeleton)
    return skeleton


# Sentence-split pattern: period/exclamation/question followed by whitespace or end.
_SENT_SPLIT = _re.compile(r"(?<=[.!?])\s+")

# Minimum sentence length to consider for repetition detection (avoids
# matching very short phrases like "Ja." or "s.o.").
_MIN_SENT_LEN = 15

# How many consecutive repeats trigger a drop.
_REPEAT_THRESHOLD = 3


class SkeletonDegenerationError(RuntimeError):
    """Raised when the LLM skeleton contains degenerate repetition."""


def _check_repetition(
    skeleton: str,
    min_sent_len: int = _MIN_SENT_LEN,
    repeat_threshold: int = _REPEAT_THRESHOLD,
) -> None:
    """Raise SkeletonDegenerationError if the skeleton has degenerate repetition."""
    sentences = _SENT_SPLIT.split(skeleton)
    run_count = 1
    prev = None
    for sent in sentences:
        sent_stripped = sent.strip()
        if len(sent_stripped) < min_sent_len:
            prev = None
            run_count = 1
            continue
        if sent_stripped == prev:
            run_count += 1
            if run_count >= repeat_threshold:
                raise SkeletonDegenerationError(
                    f"Repetition detected ({run_count}× consecutive identical "
                    f"sentence, total skeleton length={len(skeleton)} chars, "
                    f"repeated: {sent_stripped[:60]!r}...)"
                )
        else:
            prev = sent_stripped
            run_count = 1


# String values of the labels whose surface forms are persons.
_PERSON_LABEL_VALUES = frozenset(label.value for label in PERSON_LABELS)


def _missing_planned_persons(result: RenderResult, spec: GenerationSpec) -> list[str]:
    """Return planned person entities that never appear in the rendered spans.

    The profile's ``mention_plan`` lists every person the document intends to
    feature. A person the plan named but that is absent from every rendered
    person span was dropped or collapsed by the LLM skeleton. This is recorded
    for *every* document as an informational signal; it does not by itself
    flag the document (the LLM legitimately omits peripheral persons — a second
    relative, an optional consultant — in a large fraction of normal letters).
    See :func:`_is_entity_collapse` for the flagging predicate.
    """
    used = {
        span.entity_id for span in result.spans if span.label in _PERSON_LABEL_VALUES
    }
    return [ref for ref in spec.mention_plan if ref not in used]


def _is_entity_collapse(
    missing: list[str], spec: GenerationSpec, collapse_ratio: int = 2
) -> bool:
    """Decide whether missing persons indicate a degenerate *entity collapse*.

    Empirically omission is bimodal: normal letters drop at most a single
    peripheral person, while degenerate skeletons that funnel every role onto
    one identity (typically the patient) lose the *majority* of the planned
    cast at once — with a wide empty gap in between.

    The predicate is therefore "more planned persons went missing than were
    actually rendered" (``2 * missing > planned``). It is purely relative to
    the document's own plan: a single-entity document (plan == {patient}) only
    trips if the patient itself fails to render, which is a genuine defect. No
    absolute count threshold is used, so future single- or few-entity cases are
    handled correctly.
    """
    n_planned = len(spec.mention_plan)
    if n_planned == 0:
        return False
    return collapse_ratio * len(missing) > n_planned


@dataclass
class Record:
    doc_id: str
    text: str
    spans: list[dict]
    negatives: list[dict]
    metadata: dict
    qc: dict

    def as_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "text": self.text,
            "spans": self.spans,
            "negatives": self.negatives,
            "metadata": self.metadata,
            "qc": self.qc,
        }


class Pipeline:
    def __init__(
        self,
        seeds_manager: ClinicalSeedsManager,
        llm_client: Any | None = None,
        *,
        skeleton_provider: Optional[SkeletonProvider] = None,
        red_team: bool = False,
        config: PipelineConfig | None = None,
    ):
        self.config = config or PipelineConfig()
        self.sampler = ProfileSampler(seeds_manager, config=self.config)
        self.llm = llm_client
        self.skeleton_provider = skeleton_provider
        self.red_team = red_team

    # --- Stage 2 -----------------------------------------------------------
    def _author_skeleton(self, spec: GenerationSpec, catalog: Catalog) -> str:
        if self.skeleton_provider is not None:
            return _scrub_skeleton(self.skeleton_provider(spec, catalog))
        if self.llm is None:
            raise RuntimeError("No skeleton_provider and no llm_client configured.")
        system = build_system_instruction()
        user = build_user_content(
            spec,
            catalog,
            word_collision_context_max=(
                self.config.generation.word_collision_context_max
            ),
        )
        cache_key = (
            "skeleton|"
            + hashlib.sha256(
                f"{spec.doc_id}|{spec.seed}|{user}".encode("utf-8")
            ).hexdigest()
        )
        resp = self.llm.generate_structured(
            system_instruction=system,
            user_content=user,
            response_schema=SkeletonResponse,
            cache_key=cache_key,
        )
        skeleton = resp["skeleton"]
        # Fix: LLM sometimes returns literal "\n" (escaped) instead of real newlines
        # in the JSON string value. Normalize to actual newlines.
        if "\\n" in skeleton:
            skeleton = skeleton.replace("\\n", "\n")
        return _scrub_skeleton(skeleton)

    # --- Stage 4 -----------------------------------------------------------
    def _red_team(self, result: RenderResult) -> dict | None:
        if not self.red_team or self.llm is None:
            return None
        user_content = build_red_team_user(result)
        try:
            resp = self.llm.generate_structured(
                system_instruction=RED_TEAM_SYSTEM,
                user_content=user_content,
                response_schema=RedTeamResponse,
                cache_key=cache_key_for(result.text),
            )
        except Exception as exc:  # noqa: BLE001
            # A red-team call that exhausts its retries must NOT drop the whole
            # document silently. Record the failure so the fail-safe exclusion
            # (never ship an unverified doc) is auditable and distinguishable
            # from a finding-based exclusion in the provenance chain.
            logger.warning("red-team call failed: %s", exc)
            return {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "agreement": None,
                "review_items": [],
            }
        agreement = compare(
            result,
            resp.get("findings", []),
            finding_min_len=self.config.red_team.finding_min_len,
        )
        return {
            "status": "ok",
            "agreement": agreement.as_dict(),
            "review_items": [vars(ri) for ri in agreement.review_items],
        }

    # --- Orchestration -----------------------------------------------------
    def generate_record(
        self,
        doc_id: str,
        seed: int,
        *,
        style: str | None = None,
        difficulty: str | None = None,
        length: str | None = None,
    ) -> Record:
        catalog, spec = self.sampler.sample(
            doc_id, seed, style=style, difficulty=difficulty, length=length
        )
        gen_cfg = self.config.generation
        skeleton = self._author_skeleton(spec, catalog)
        _check_repetition(
            skeleton,
            min_sent_len=gen_cfg.skeleton_min_sentence_len,
            repeat_threshold=gen_cfg.skeleton_repeat_threshold,
        )
        result = render(skeleton, catalog)

        # --- Adversarial typo injection (post-render) ----------------------
        if spec.typo_density > 0 or spec.adversarial_mode == "typo_text":
            import random as _random

            typo_rng = _random.Random(seed)  # deterministic from doc seed
            result = apply_typo_injection(
                result,
                typo_rng,
                typo_density=spec.typo_density,
                pii_typo_prob=(
                    gen_cfg.pii_typo_prob
                    if spec.adversarial_mode == "typo_text"
                    else 0.0
                ),
                n_pii_corruptions=typo_rng.randint(*gen_cfg.n_pii_corruptions_range)
                if spec.adversarial_mode == "typo_text"
                else 0,
                mode_weights=gen_cfg.typo_mode_weights,
            )

        # --- Hard invariant check (D+E) — must pass or the doc is broken ---
        validate_spans(result, catalog)

        leaks = scan_leaks(result, catalog)
        neg_findings = scan_negatives(result, catalog)

        reasons: list[str] = []
        if leaks:
            reasons.append(f"{len(leaks)} potential PII leak(s) outside gold spans")
        if neg_findings:
            reasons.append(f"{len(neg_findings)} eponym/negative integrity issue(s)")

        missing_persons = _missing_planned_persons(result, spec)
        if missing_persons and _is_entity_collapse(
            missing_persons, spec, collapse_ratio=gen_cfg.entity_collapse_ratio
        ):
            reasons.append(
                f"entity collapse: {len(missing_persons)}/{len(spec.mention_plan)} "
                f"planned persons never rendered "
                f"({', '.join(missing_persons)})"
            )

        red_team_result = self._red_team(result)
        if red_team_result and red_team_result["review_items"]:
            reasons.append(
                f"{len(red_team_result['review_items'])} red-team finding(s)"
            )
        if red_team_result and red_team_result.get("status") == "error":
            reasons.append("red-team verification unavailable (call failed)")

        qc = {
            "flagged": bool(reasons),
            "reasons": reasons,
            "leaks": [vars(f) for f in leaks],
            "negative_findings": [vars(f) for f in neg_findings],
            "missing_planned_persons": missing_persons,
            "red_team": red_team_result,
        }

        metadata = spec.to_metadata()
        metadata["n_spans"] = len(result.spans)
        metadata["n_negatives"] = len(result.negative_spans)
        metadata["quasi"] = spec.quasi
        # Full pre-generation sampling record: the complete Stage-1 output
        # (every spec field and the entire entity catalog, including atomic
        # variants and any entities the skeleton did not reference).
        metadata["sampled_profile"] = {
            "spec": spec.to_dict(),
            "catalog": catalog.to_dict(),
        }

        return Record(
            doc_id=doc_id,
            text=result.text,
            spans=[s.as_dict() for s in result.spans],
            negatives=[s.as_dict() for s in result.negative_spans],
            metadata=metadata,
            qc=qc,
        )
