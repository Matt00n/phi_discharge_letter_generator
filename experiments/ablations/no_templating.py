"""Ablation 2 — NO TEMPLATING.

Instead of authoring a placeholder skeleton that a deterministic renderer fills
(spans correct by construction), the LLM is handed the *concrete* sampled PII and
writes the letter directly as free text, then self-annotates the PII it used —
all in one structured call.

We then run the byte-identical Stage-4 QC:
  * deterministic ``scan_leaks`` against the known sampled values, and
  * the red-team auditor on the self-annotated text.

Expected result: QC fires heavily. Free-text authoring inflects names,
paraphrases, drops or invents identifiers, and self-annotation misses spans — so
both the deterministic leak scan and the red-team light up far above the
templated baseline. This quantifies what the placeholder discipline buys.

Same seeds as the main run (base_seed 1,000,000) so each ablation doc is the
free-text counterpart of the same sampled profile.

Usage:
    python -m experiments.ablations.no_templating [--num 100] [--baseline-dir DIR]
"""

from __future__ import annotations

import argparse
import os

from pydantic import BaseModel, Field

from experiments.ablations._common import (
    DEFAULT_BASELINE_DIR,
    build_render_result,
    concrete_profile,
    load_pre_qc_corpus,
    make_llm,
    run_red_team_qc,
    utf8_stdout,
)
from generator.clinical_seeds import ClinicalSeedsManager
from generator.config_schema import PipelineConfig
from generator.exporter import Exporter
from generator.profile_sampler import ProfileSampler
from generator.red_team import findings_to_positions
from generator.render import scan_leaks
from generator.skeleton_prompt import _clinical_block

BASE_SEED = 1_000_000  # identical to sparcling-de
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT_DIR = os.path.join(_HERE, "..", "results", "ablation_no_templating")


# --- Structured response ------------------------------------------------------


class SelfAnnPII(BaseModel):
    exact_string: str = Field(
        description="Exakter Substring aus dem Brief, der ein PII-Datum ist."
    )
    category: str = Field(
        description=(
            "Kategorie: PERSON_NAME | DATUM | ADRESSE | PLZ | ORT | TELEFON | FAX | "
            "EMAIL | URL | VERSICHERTENNR | FALLNR | PATIENTEN_ID | SONSTIGE_ID | "
            "INSTITUTION | VERSICHERER | ALTER | BERUF | ARBEITGEBER | NATIONALITAET"
        )
    )


class DirectLetterResponse(BaseModel):
    letter: str = Field(description="Der vollständige Entlassungsbrief als Freitext.")
    pii: list[SelfAnnPII] = Field(
        default_factory=list,
        description="Alle personenbezogenen Substrings, die im Brief vorkommen.",
    )


SYSTEM = (
    "Du bist eine erfahrene Oberärztin / ein erfahrener Oberarzt und verfasst "
    "klinisch authentische, grammatikalisch einwandfreie deutsche "
    "Entlassungsbriefe (Arztbriefe).\n\n"
    "Du erhältst die ECHTEN Patienten- und Falldaten. Schreibe einen "
    "vollständigen, natürlichen Entlassungsbrief und verwende dabei diese Daten "
    "DIREKT im Text (keine Platzhalter, keine Maskierung). Beuge Namen "
    "grammatikalisch korrekt und formuliere flüssig.\n\n"
    "ZUSÄTZLICH: Liste im Feld 'pii' JEDEN personenbezogenen Substring auf, den "
    "du im Brief verwendet hast (Namen, Daten, Adressen, Nummern, Alter, Beruf "
    "usw.), jeweils mit exaktem Substring (copy-paste) und Kategorie. Erfasse "
    "ALLE Vorkommen, auch gebeugte Namensformen und Wiederholungen.\n\n"
    "Antworte ausschließlich im geforderten JSON-Format."
)


def _profile_block(profile: list[dict]) -> str:
    lines = ["ECHTE PATIENTEN- UND FALLDATEN (direkt im Brief verwenden):"]
    for p in profile:
        lines.append(f"  - {p['role']}: {p['value']}")
    return "\n".join(lines)


def _build_user(spec, profile: list[dict]) -> str:
    return (
        f"Erstelle einen deutschen Entlassungsbrief.\n\n"
        f"STIL: {spec.style}\n"
        f"LÄNGE: {spec.length_hint}\n\n"
        f"KLINISCHER FALL:\n{_clinical_block(spec)}\n\n"
        f"{_profile_block(profile)}\n\n"
        f"Schreibe nun den vollständigen Brief als Freitext und annotiere alle PII."
    )


def main() -> None:
    utf8_stdout()
    ap = argparse.ArgumentParser(description="Ablation 2: no-templating.")
    ap.add_argument("--num", type=int, default=100)
    ap.add_argument("--baseline-dir", default=DEFAULT_BASELINE_DIR)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--model", default="gemini-3.7-flash")
    args = ap.parse_args()

    out_dir = args.out_dir
    cfg = PipelineConfig()  # sparcling-de uses all defaults
    sampler = ProfileSampler(ClinicalSeedsManager(data_dir="data"), config=cfg)
    llm = make_llm(args.model)
    min_len = cfg.red_team.finding_min_len

    records: list[dict] = []
    print("=" * 64)
    print(f"  ABLATION 2 — NO TEMPLATING   n={args.num}  model={args.model}")
    print("=" * 64)

    for i in range(args.num):
        doc_id = f"doc_{i:05d}"
        seed = BASE_SEED + i
        catalog, spec = sampler.sample(doc_id, seed)
        profile = concrete_profile(catalog, spec)

        try:
            resp = llm.generate_structured(
                system_instruction=SYSTEM,
                user_content=_build_user(spec, profile),
                response_schema=DirectLetterResponse,
                cache_key=f"notempl|{doc_id}|{seed}",
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[{i + 1}/{args.num}] {doc_id}  AUTHOR-ERROR: {exc}")
            continue

        letter = resp.get("letter", "")
        if "\\n" in letter:
            letter = letter.replace("\\n", "\n")

        # Resolve self-annotations to spans, then run the identical Stage-4 QC.
        positions = findings_to_positions(letter, resp.get("pii", []), min_len=2)
        span_dicts = [
            {"start": s, "end": e, "label": cat} for (_txt, s, e, cat) in positions
        ]
        result = build_render_result(letter, span_dicts)

        leaks = scan_leaks(result, catalog)
        red_team = run_red_team_qc(llm, result, finding_min_len=min_len)

        reasons: list[str] = []
        if leaks:
            reasons.append(f"{len(leaks)} potential PII leak(s) outside gold spans")
        if red_team["review_items"]:
            reasons.append(f"{len(red_team['review_items'])} red-team finding(s)")
        if red_team.get("status") == "error":
            reasons.append("red-team verification unavailable (call failed)")

        metadata = spec.to_metadata()
        metadata["n_spans"] = len(result.spans)
        metadata["n_self_annotations"] = len(resp.get("pii", []))
        qc = {
            "flagged": bool(reasons),
            "reasons": reasons,
            "leaks": [vars(f) for f in leaks],
            "negative_findings": [],
            "missing_planned_persons": [],
            "red_team": red_team,
        }
        records.append(
            {
                "doc_id": doc_id,
                "text": letter,
                "spans": [s.as_dict() for s in result.spans],
                "negatives": [],
                "metadata": metadata,
                "qc": qc,
            }
        )
        flag = "FLAG" if qc["flagged"] else "ok"
        print(
            f"[{i + 1}/{args.num}] {doc_id}  {spec.difficulty:<6} "
            f"spans={len(result.spans):<3} leaks={len(leaks):<2} "
            f"rt={len(red_team['review_items']):<2} {flag}"
        )

    if not records:
        raise SystemExit("No records generated.")

    exporter = Exporter(output_dir=out_dir)
    exporter.write_jsonl(records, "dataset.jsonl")
    exporter.export_review_queue(records)
    _, prov = exporter.export_provenance(records, [], auto_mode=True)

    n = len(records)
    flagged = prov["yield"]["flagged"]
    print("-" * 64)
    print(f"no-templating flag rate: {flagged}/{n} = {flagged / n:.1%}")

    # N-matched baseline (templated pipeline) flag rate, if the run is available.
    base_ds = os.path.join(args.baseline_dir, "dataset.jsonl")
    if os.path.exists(base_ds):
        base_corpus = load_pre_qc_corpus(args.baseline_dir)
        ids = {r["doc_id"] for r in records}
        base_slice = [r for r in base_corpus if r["doc_id"] in ids]
        b_flag = sum(1 for r in base_slice if r["qc"]["flagged"])
        print(
            f"baseline (templated) flag rate on same {len(base_slice)} docs: "
            f"{b_flag}/{len(base_slice)} = "
            f"{b_flag / len(base_slice):.1%}"
            if base_slice
            else "baseline slice empty"
        )
    else:
        print(f"(baseline {base_ds} not found — run comparison once it finishes)")
    print(f"wrote outputs to {out_dir}/")
    print("=" * 64)


if __name__ == "__main__":
    main()
