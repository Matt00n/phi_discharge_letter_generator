"""Ablation 3 — NO SAMPLING.

The Stage-1 deterministic sampler (which draws grounded, high-entropy names,
ages, cities, dates, IDs from curated pools) is removed. Instead the LLM does
everything in two passes:

  1. authors a discharge-letter *template* with ``[[ref]]`` placeholders drawn
     from a fixed field schema, inventing its own clinical scenario, and
  2. fills every placeholder with a value it invents itself.

Values are substituted by the same deterministic renderer, so spans are still
correct by construction (QC would pass) — but the PII *distribution* is now the
LLM's, not the sampler's. We run the identical Stage-5 diversity audit and
compare to an N-matched slice of the main run.

Expected result: diversity collapses — surname/city/age entropy and PII-surface
variety drop sharply (LLMs fall back on "Max Mustermann", "Musterstadt", round
numbers), while the templated pipeline stays high.

Usage:
    python -m experiments.ablations.no_sampling [--num 100] [--baseline-dir DIR]
"""

from __future__ import annotations

import argparse
import os
import re

from pydantic import BaseModel, Field

from experiments.ablations._common import (
    DEFAULT_BASELINE_DIR,
    load_kept_slice,
    make_llm,
    utf8_stdout,
)
from generator.diversity import _entropy, audit
from generator.exporter import Exporter
from generator.labels import Label
from generator.render import (
    _PLACEHOLDER_RE,
    AtomicEntity,
    Catalog,
    _parse_placeholder_tokens,
    render,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT_DIR = os.path.join(_HERE, "..", "results", "ablation_no_sampling")

# Fixed field schema the LLM authors against: ref -> (Label, German description).
SCHEMA: dict[str, tuple[Label, str]] = {
    "patient": (Label.PATIENT_NAME, "Voller Name des Patienten/der Patientin"),
    "alter": (Label.AGE, "Alter des Patienten (z.B. '72 Jahre')"),
    "geburtsdatum": (Label.DATE_BIRTH, "Geburtsdatum (TT.MM.JJJJ)"),
    "aufnahmedatum": (Label.DATE_ADMISSION, "Aufnahmedatum (TT.MM.JJJJ)"),
    "entlassungsdatum": (Label.DATE_DISCHARGE, "Entlassungsdatum (TT.MM.JJJJ)"),
    "adresse": (Label.ADDRESS, "Straße und Hausnummer des Patienten"),
    "plz": (Label.ZIP, "Postleitzahl"),
    "ort": (Label.CITY, "Wohnort"),
    "telefon": (Label.PHONE, "Telefonnummer"),
    "arzt_station": (Label.STAFF_NAME, "Name des Stationsarztes/-ärztin"),
    "arzt_ober": (Label.STAFF_NAME, "Name des Oberarztes/-ärztin"),
    "arzt_einweiser": (Label.STAFF_NAME, "Name des einweisenden Arztes/Ärztin"),
    "institution": (Label.INSTITUTION, "Name der Klinik/des Krankenhauses"),
    "fallnr": (Label.ID_CASE, "Fallnummer"),
    "versicherer": (Label.INSURER, "Name der Krankenkasse"),
    "beruf": (Label.PROFESSION, "Beruf des Patienten"),
}

_REFS_DOC = "\n".join(f"  - [[{ref}]]: {desc}" for ref, (_lbl, desc) in SCHEMA.items())


class TemplateResponse(BaseModel):
    fachabteilung: str = Field(description="Gewählte Fachabteilung.")
    template: str = Field(
        description="Entlassungsbrief-Vorlage mit [[ref]]-Platzhaltern."
    )


class FillItem(BaseModel):
    ref: str
    value: str


class FillResponse(BaseModel):
    fills: list[FillItem] = Field(default_factory=list)


TEMPLATE_SYSTEM = (
    "Du bist eine erfahrene Oberärztin / ein erfahrener Oberarzt. Verfasse die "
    "VORLAGE eines realistischen deutschen Entlassungsbriefs (Arztbrief). "
    "Erfinde selbst einen plausiblen klinischen Fall (Diagnosen, Verlauf, "
    "Medikation) und schreibe einen vollständigen, natürlichen Brief.\n\n"
    "Für alle personenbezogenen Daten verwendest du AUSSCHLIESSLICH Platzhalter "
    "der Form [[ref]] aus der folgenden festen Liste (keine anderen Platzhalter, "
    "keine echten Namen/Daten/Nummern im Text):\n"
    f"{_REFS_DOC}\n\n"
    "Nutze die Platzhalter, die zum Brief passen (nicht zwingend alle). "
    "Antworte im JSON-Format mit 'fachabteilung' und 'template'."
)

FILL_SYSTEM = (
    "Du hast einen Entlassungsbrief mit Platzhaltern verfasst. Gib nun für jeden "
    "aufgeführten Platzhalter einen konkreten, realistischen deutschen Wert an "
    "(erfinde ihn passend zum Brief). Antworte im JSON-Format als Liste 'fills' "
    "mit Objekten {ref, value}."
)


def _used_refs(template: str) -> list[str]:
    """Refs that appear in the template AND are in the schema (dedup, in order)."""
    seen: list[str] = []
    for m in _PLACEHOLDER_RE.finditer(template):
        ref, _attrs = _parse_placeholder_tokens(m.group(1), m.group(2) or "")
        if ref in SCHEMA and ref not in seen:
            seen.append(ref)
    return seen


def _scrub_unknown(template: str, keep: set[str]) -> str:
    """Drop any placeholder whose ref is not in ``keep`` (unknown / unfilled)."""

    def _sub(m: re.Match) -> str:
        ref, _attrs = _parse_placeholder_tokens(m.group(1), m.group(2) or "")
        return m.group(0) if ref in keep else ""

    return _PLACEHOLDER_RE.sub(_sub, template)


def _age_entropy_ablation(records: list[dict]) -> tuple[int, float]:
    from collections import Counter

    ages: Counter = Counter()
    for r in records:
        for sp in r["spans"]:
            if sp["label"] == "AGE":
                nums = re.findall(r"\d{1,3}", r["text"][sp["start"] : sp["end"]])
                if nums:
                    ages[nums[0]] += 1
    return len(ages), round(_entropy(ages), 3)


def _age_entropy_baseline(records: list[dict]) -> tuple[int, float]:
    from collections import Counter

    ages: Counter = Counter()
    for r in records:
        a = r.get("metadata", {}).get("patient_age")
        if a:
            ages[str(a)] += 1
    return len(ages), round(_entropy(ages), 3)


def _comparison_md(abl: dict, base: dict, abl_age, base_age, baseline_dir: str) -> str:
    def g(d: dict, *path):
        for p in path:
            d = d.get(p, {})
        return d

    rows = [
        ("n documents", abl["n_documents"], base["n_documents"]),
        ("unique surnames", abl["unique_surnames"], base["unique_surnames"]),
        (
            "surname entropy (bits)",
            abl["surname_entropy_bits"],
            base["surname_entropy_bits"],
        ),
        ("unique ages", abl_age[0], base_age[0]),
        ("age entropy (bits)", abl_age[1], base_age[1]),
        (
            "unique fachabteilungen",
            g(abl, "clinical", "unique_fachabteilungen"),
            g(base, "clinical", "unique_fachabteilungen"),
        ),
        (
            "fachabteilung entropy (bits)",
            g(abl, "clinical", "fachabteilung_entropy_bits"),
            g(base, "clinical", "fachabteilung_entropy_bits"),
        ),
        (
            "unique PII surfaces",
            g(abl, "pii_surface", "unique_pii_surfaces"),
            g(base, "pii_surface", "unique_pii_surfaces"),
        ),
        (
            "surface/entity ratio",
            g(abl, "pii_surface", "surface_to_entity_ratio"),
            g(base, "pii_surface", "surface_to_entity_ratio"),
        ),
        (
            "spans/doc (mean)",
            g(abl, "pii_surface", "spans_per_doc", "mean"),
            g(base, "pii_surface", "spans_per_doc", "mean"),
        ),
        (
            "TTR (median)",
            g(abl, "lexical", "ttr", "median"),
            g(base, "lexical", "ttr", "median"),
        ),
        (
            "trigram novelty (mean)",
            g(abl, "lexical", "trigram_novelty", "mean"),
            g(base, "lexical", "trigram_novelty", "mean"),
        ),
        (
            "inter-doc Jaccard (mean)",
            g(abl, "lexical", "inter_document_jaccard", "mean"),
            g(base, "lexical", "inter_document_jaccard", "mean"),
        ),
    ]

    lines = [
        "# Ablation 3 — No Sampling (diversity collapse)",
        "",
        f"Baseline (templated + sampled): `{baseline_dir}`, N-matched slice.",
        "",
        "| Metric | No-sampling (LLM) | Baseline (sampled) |",
        "| --- | ---: | ---: |",
    ]
    for name, a, b in rows:
        lines.append(f"| {name} | {a} | {b} |")
    lines.append("")
    lines.append(
        "> Note: `patient_origin`, `style`, `difficulty` and section structure "
        "are sampler concepts not applicable to the no-sampling arm, so the "
        "single aggregate diversity score is not directly comparable; the "
        "per-dimension PII/lexical metrics above are the fair comparison."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    utf8_stdout()
    ap = argparse.ArgumentParser(description="Ablation 3: no-sampling.")
    ap.add_argument("--num", type=int, default=100)
    ap.add_argument("--baseline-dir", default=DEFAULT_BASELINE_DIR)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--model", default="gemini-3.7-flash")
    args = ap.parse_args()

    out_dir = args.out_dir
    llm = make_llm(args.model)
    records: list[dict] = []
    print("=" * 64)
    print(f"  ABLATION 3 — NO SAMPLING   n={args.num}  model={args.model}")
    print("=" * 64)

    for i in range(args.num):
        doc_id = f"doc_{i:05d}"
        try:
            t = llm.generate_structured(
                system_instruction=TEMPLATE_SYSTEM,
                user_content="Erstelle die Entlassungsbrief-Vorlage.",
                response_schema=TemplateResponse,
                cache_key=f"nosample-tmpl|{doc_id}",
                temperature=0.9,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[{i + 1}/{args.num}] {doc_id}  TEMPLATE-ERROR: {exc}")
            continue

        template = t.get("template", "")
        if "\\n" in template:
            template = template.replace("\\n", "\n")
        used = _used_refs(template)
        if not used:
            print(f"[{i + 1}/{args.num}] {doc_id}  skipped (no schema refs used)")
            continue

        refs_doc = "\n".join(f"  - {ref}: {SCHEMA[ref][1]}" for ref in used)
        try:
            f = llm.generate_structured(
                system_instruction=FILL_SYSTEM,
                user_content=(
                    f"Fülle folgende Platzhalter dieses Briefs:\n{refs_doc}\n\n"
                    f"BRIEF:\n{template}"
                ),
                response_schema=FillResponse,
                cache_key=f"nosample-fill|{doc_id}",
                temperature=0.9,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[{i + 1}/{args.num}] {doc_id}  FILL-ERROR: {exc}")
            continue

        fills = {
            item["ref"]: item["value"]
            for item in f.get("fills", [])
            if item.get("ref") in SCHEMA and item.get("value")
        }
        if not fills:
            print(f"[{i + 1}/{args.num}] {doc_id}  skipped (no fills)")
            continue

        catalog = Catalog()
        for ref, value in fills.items():
            catalog.atomics[ref] = AtomicEntity(ref, SCHEMA[ref][0], {"default": value})
        scrubbed = _scrub_unknown(template, set(fills))
        try:
            result = render(scrubbed, catalog)
        except Exception as exc:  # noqa: BLE001
            print(f"[{i + 1}/{args.num}] {doc_id}  RENDER-ERROR: {exc}")
            continue

        patient_val = fills.get("patient", "")
        surname = patient_val.strip().split()[-1] if patient_val.strip() else "?"
        metadata = {
            "doc_id": doc_id,
            "patient_surname": surname,
            "patient_origin": "?",  # not a concept in the no-sampling arm
            "fachabteilung": t.get("fachabteilung", "?"),
            "style": "?",
            "difficulty": "?",
            "difficulty_tags": [],
            "sections": [],
            "n_spans": len(result.spans),
        }
        records.append(
            {
                "doc_id": doc_id,
                "text": result.text,
                "spans": [s.as_dict() for s in result.spans],
                "negatives": [],
                "metadata": metadata,
                "qc": {"flagged": False, "reasons": []},
            }
        )
        print(
            f"[{i + 1}/{args.num}] {doc_id}  {metadata['fachabteilung']:<16} "
            f"spans={len(result.spans):<3} surname={surname}"
        )

    if not records:
        raise SystemExit("No records generated.")

    exporter = Exporter(output_dir=out_dir)
    exporter.write_jsonl(records, "dataset.jsonl")
    abl_report = audit(records)
    exporter.write_json(abl_report, "diversity.json")

    base_ds = os.path.join(args.baseline_dir, "dataset.jsonl")
    print("-" * 64)
    if os.path.exists(base_ds):
        base_slice = load_kept_slice(args.baseline_dir, len(records))
        base_report = audit(base_slice)
        md = _comparison_md(
            abl_report,
            base_report,
            _age_entropy_ablation(records),
            _age_entropy_baseline(base_slice),
            args.baseline_dir,
        )
        with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as fh:
            fh.write(md)
        print(md)
    else:
        print(f"(baseline {base_ds} not found — run comparison once it finishes)")
        print(
            f"no-sampling surname entropy: {abl_report['surname_entropy_bits']} bits "
            f"({abl_report['unique_surnames']} unique)"
        )
    print(f"wrote outputs to {out_dir}/")
    print("=" * 64)


if __name__ == "__main__":
    main()
