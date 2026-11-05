"""Shared harness for the ablation experiments.

Everything privacy-critical (QC masking, leak scan, red-team resolution,
diversity) is imported straight from ``generator`` so every ablation arm runs
the *same* code path as the real pipeline — the ablations only swap out the
stage under test.
"""

from __future__ import annotations

import io
import json
import os
import sys
from typing import Any

from generator.labels import Label, tier_of
from generator.morphology import Case, PersonForm, render_person
from generator.red_team import (
    RED_TEAM_SYSTEM,
    RedTeamResponse,
    build_red_team_user,
    cache_key_for,
    compare,
)
from generator.render import Catalog, RenderResult, Span

# --- Paths --------------------------------------------------------------------

ABLATIONS_DIR = os.path.dirname(os.path.abspath(__file__))
# A dedicated cache DB keeps ablation LLM calls out of the main run's cache so
# the two never collide on keys and the production cache stays pristine.
ABLATION_CACHE_DB = os.path.join(ABLATIONS_DIR, "ablations_cache.db")
DEFAULT_BASELINE_DIR = os.path.join(ABLATIONS_DIR, "..", "datasets", "sparcling-de")


def utf8_stdout() -> None:
    """Force UTF-8 on the Windows console so German output does not crash."""
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")


# --- LLM client ---------------------------------------------------------------


def make_llm(model_name: str = "gemini-3.7-flash") -> Any:
    """Build the same cached Gemini client the pipeline uses (separate cache DB)."""
    from generator.llm_client import GeminiCachedClient

    return GeminiCachedClient(
        db_path=ABLATION_CACHE_DB,
        model_name=model_name,
        retry_attempts=5,
        retry_delay_seconds=5,
    )


# --- Label coercion -----------------------------------------------------------

# Maps the German red-team / self-annotation category vocabulary onto our Label
# enum. The tier (via ``tier_of``) is what the QC masker actually needs; the
# exact person/date/id label within a tier is not privacy-relevant here.
CATEGORY_TO_LABEL: dict[str, Label] = {
    "PERSON_NAME": Label.PATIENT_NAME,
    "DATUM": Label.DATE_OTHER,
    "ADRESSE": Label.ADDRESS,
    "PLZ": Label.ZIP,
    "ORT": Label.CITY,
    "TELEFON": Label.PHONE,
    "FAX": Label.FAX,
    "EMAIL": Label.EMAIL,
    "URL": Label.URL,
    "VERSICHERTENNR": Label.ID_INSURANCE,
    "FALLNR": Label.ID_CASE,
    "PATIENTEN_ID": Label.ID_PATIENT,
    "SONSTIGE_ID": Label.ID_OTHER,
    "INSTITUTION": Label.INSTITUTION,
    "VERSICHERER": Label.INSURER,
    "ALTER": Label.AGE,
    "BERUF": Label.PROFESSION,
    "ARBEITGEBER": Label.EMPLOYER,
    "NATIONALITAET": Label.NATIONALITY,
}


def coerce_label(category: str) -> Label:
    """Resolve a self-annotation category to a Label (unknown → DIRECT id)."""
    cat = (category or "").strip().upper()
    if cat in CATEGORY_TO_LABEL:
        return CATEGORY_TO_LABEL[cat]
    # Also accept a raw Label value (e.g. "PATIENT_NAME") directly.
    try:
        return Label(cat)
    except ValueError:
        return Label.ID_OTHER  # unknown but still masked (DIRECT tier)


# --- Concrete-value extraction (ablation 2) -----------------------------------


def concrete_profile(catalog: Catalog, spec: Any) -> list[dict[str, str]]:
    """Flatten a sampled catalog into concrete (role, value, category) records.

    This is the exact real PII the LLM is handed to write free text from in the
    no-templating arm. Persons are given as their full name core; atomics as
    their default surface.
    """
    by_ref = {b["ref"]: b for b in spec.entity_brief}
    out: list[dict[str, str]] = []
    for ref, person in catalog.persons.items():
        mention = render_person(person, PersonForm.HON_FIRST_LAST, Case.NOM)
        role = by_ref.get(ref, {}).get("role", person.label.value)
        out.append(
            {
                "ref": ref,
                "role": str(role),
                "value": mention.core,
                "label": person.label.value,
            }
        )
    for ref, atomic in catalog.atomics.items():
        role = by_ref.get(ref, {}).get("role", atomic.label.value)
        out.append(
            {
                "ref": ref,
                "role": str(role),
                "value": atomic.surface(None),
                "label": atomic.label.value,
            }
        )
    return out


# --- RenderResult assembly from self-annotations ------------------------------


def build_render_result(text: str, span_dicts: list[dict]) -> RenderResult:
    """Build a RenderResult (for QC) from a list of ``{start,end,label}`` dicts.

    Overlapping spans are dropped (first-wins by start offset) so the masker and
    ``validate``-style invariants stay consistent.
    """
    spans: list[Span] = []
    occupied: list[tuple[int, int]] = []
    for d in sorted(span_dicts, key=lambda s: (s["start"], -(s["end"] - s["start"]))):
        start, end = d["start"], d["end"]
        if start >= end:
            continue
        if any(max(start, a) < min(end, b) for a, b in occupied):
            continue  # overlaps an already-accepted span
        label = coerce_label(d.get("label", d.get("category", "")))
        spans.append(
            Span(
                start=start,
                end=end,
                label=label.value,
                tier=tier_of(label).value,
                entity_id=d.get("entity_id", f"selfann_{len(spans)}"),
            )
        )
        occupied.append((start, end))
    spans.sort(key=lambda s: s.start)
    return RenderResult(text=text, spans=spans, negative_spans=[])


# --- Stage-4 red-team QC (identical to the pipeline) --------------------------


def run_red_team_qc(llm: Any, result: RenderResult, finding_min_len: int = 2) -> dict:
    """Run the exact Stage-4 red-team pass and return the pipeline-shaped dict."""
    user_content = build_red_team_user(result)
    try:
        resp = llm.generate_structured(
            system_instruction=RED_TEAM_SYSTEM,
            user_content=user_content,
            response_schema=RedTeamResponse,
            cache_key=cache_key_for(result.text),
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "agreement": None,
            "review_items": [],
        }
    agreement = compare(
        result, resp.get("findings", []), finding_min_len=finding_min_len
    )
    return {
        "status": "ok",
        "agreement": agreement.as_dict(),
        "review_items": [vars(ri) for ri in agreement.review_items],
    }


# --- I/O ----------------------------------------------------------------------


def load_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_kept_slice(run_dir: str, n: int | None = None) -> list[dict]:
    """Load kept (QC-passed) records from a run, sorted by doc_id, first ``n``."""
    recs = load_jsonl(os.path.join(run_dir, "dataset.jsonl"))
    recs.sort(key=lambda r: r["doc_id"])
    return recs[:n] if n else recs


def load_pre_qc_corpus(run_dir: str) -> list[dict]:
    """Reconstruct the full pre-QC corpus: kept records ∪ excluded review items.

    Kept records carry ``qc.flagged == False``; review-queue entries are the
    excluded docs, reshaped to the record schema with ``qc.flagged == True`` so
    both halves feed the same provenance aggregator.
    """
    kept = load_jsonl(os.path.join(run_dir, "dataset.jsonl"))
    review = load_jsonl(os.path.join(run_dir, "review_queue.jsonl"))
    corpus: list[dict] = list(kept)
    for item in review:
        corpus.append(
            {
                "doc_id": item.get("doc_id"),
                "text": item.get("text", ""),
                "spans": [],
                "negatives": [],
                "metadata": {
                    "difficulty": item.get("difficulty"),
                    "style": item.get("style"),
                },
                "qc": {
                    "flagged": True,
                    "reasons": item.get("reasons", []),
                    "leaks": item.get("leaks", []),
                    "negative_findings": item.get("negative_findings", []),
                    "missing_planned_persons": item.get("missing_planned_persons", []),
                    "red_team": item.get("red_team"),
                },
            }
        )
    corpus.sort(key=lambda r: r["doc_id"] or "")
    return corpus
