"""Offline skeleton provider.

Builds a plausible Arztbrief skeleton deterministically from the catalog/spec,
without calling an LLM. Used for tests and for running the full pipeline
(render + QC + export + eval) without API access. It deliberately exercises many
placeholder forms/cases so the deterministic core is stressed.
"""

from __future__ import annotations

from generator.profile_sampler import GenerationSpec
from generator.render import Catalog


def _has(catalog: Catalog, ref: str) -> bool:
    return ref in catalog.atomics or ref in catalog.persons


def template_skeleton(spec: GenerationSpec, catalog: Catalog) -> str:
    c = spec.clinical_case
    # Build patient mention sequence from the mention plan.
    # Each entry is consumed in order; extras cycle back.
    pat_forms = list(
        spec.mention_plan.get("patient", ["hon_first_last", "hon_last", "last"])
    )
    _pat_idx = 0

    def _pat(case: str = "nom") -> str:
        """Next patient placeholder using the mention plan's form sequence."""
        nonlocal _pat_idx
        form = pat_forms[_pat_idx % len(pat_forms)]
        _pat_idx += 1
        return f"[[patient form={form} case={case}]]"

    def _person(ref: str, case: str = "nom", idx: int = 0) -> str:
        """Person placeholder using the mention plan for that ref."""
        forms = spec.mention_plan.get(ref, ["hon_first_last"])
        form = forms[idx % len(forms)]
        return f"[[{ref} form={form} case={case}]]"

    lines: list[str] = []

    framing = getattr(spec, "framing", {})

    # --- Briefkopf (letterhead) ---
    if framing.get("briefkopf", True):
        lines.append("[[institution]]")
        if _has(catalog, "adresse_klinik"):
            lines.append("[[adresse_klinik]], [[plz_klinik]] [[ort_klinik]]")
        if _has(catalog, "klinik_url"):
            lines.append("[[klinik_url]]")
        lines.append("")

    # --- Anrede (greeting) ---
    if framing.get("anrede", True):
        lines.append("Sehr geehrte Kolleginnen und Kollegen,")
        lines.append("")
    birth = "[[geburtsdatum variant=long]]"
    adm = "[[aufnahmedatum]]"
    dis = "[[entlassungsdatum]]"
    # Patient intro — address only if present
    intro = f"wir berichten über {_pat('akk')}, geb. {birth}"
    if _has(catalog, "adresse"):
        intro += ", wohnhaft [[adresse]], [[plz]] [[ort]]"
    intro += (
        f", der/die sich vom {adm} bis {dis} in unserer stationären Behandlung befand."
    )
    lines.append(intro)
    if _has(catalog, "alter") or _has(catalog, "nationalitaet"):
        demo_parts = []
        if _has(catalog, "alter"):
            _gender = (
                catalog.persons["patient"].gender
                if "patient" in catalog.persons
                else "m"
            )
            # "Es handelt sich um eine/n ..." is accusative: masculine needs the
            # weak "-en" form, feminine stays "-e".
            _age_var = "adjective_f" if _gender == "f" else "adjective_m_akk"
            demo_parts.append(f"[[alter variant={_age_var}]]")
        if _has(catalog, "nationalitaet"):
            demo_parts.append("[[nationalitaet]]")
        lines.append("Es handelt sich um eine/n " + ", ".join(demo_parts) + ".")
    if _has(catalog, "beruf"):
        beruf_line = (
            f"Anamnestisch ist {_pat('gen')} Tätigkeit als [[beruf]] zu erwähnen"
        )
        if "arbeitgeber" in catalog.atomics:
            beruf_line += ", Arbeitgeber [[arbeitgeber]]."
        else:
            beruf_line += "."
        lines.append(beruf_line)
        if spec.collegial_address and _has(catalog, "beruf_kollegial"):
            lines.append(
                f"Als unser [[beruf_kollegial]] ist {_pat('dat')} die Diagnose und das "
                "geplante Procedere ausführlich erläutert worden."
            )

    lines.append("")
    lines.append(f"Aufnahmeanlass: {c.get('anlass')}.")
    lines.append(f"Hauptdiagnose: {c.get('hauptdiagnose')}.")
    if c.get("nebendiagnosen"):
        lines.append("Nebendiagnosen: " + "; ".join(c["nebendiagnosen"]) + ".")

    # weave in negatives (eponyms) as plain text
    neg_surfaces = [n.surface for n in catalog.negatives.values()]
    if neg_surfaces:
        lines.append("")
        lines.append("Vorbekannt sind " + ", ".join(neg_surfaces) + ".")

    lines.append("")
    if _has(catalog, "praxis_einweiser"):
        lines.append(
            f"Die stationäre Aufnahme erfolgte auf Einweisung durch "
            f"{_person('arzt_einweiser', 'akk', 0)} "
            f"([[praxis_einweiser]], [[adresse_einweiser]], [[plz_einweiser]] [[ort_einweiser]]). "
            f"Die Betreuung übernahm "
            f"{_person('arzt_station', 'nom', 0)} unter oberärztlicher Leitung von "
            f"{_person('arzt_ober', 'dat', 0)}."
        )
    else:
        lines.append(
            f"Die stationäre Aufnahme erfolgte auf Einweisung durch "
            f"{_person('arzt_einweiser', 'akk', 0)}. "
            f"Die Betreuung übernahm "
            f"{_person('arzt_station', 'nom', 0)} unter oberärztlicher Leitung von "
            f"{_person('arzt_ober', 'dat', 0)}."
        )

    if _has(catalog, "klinik_vorbehandlung"):
        lines.append("Bereits vorbehandelt in [[klinik_vorbehandlung]].")

    if c.get("prozeduren"):
        proc_line = "Durchgeführte Prozeduren: " + "; ".join(c["prozeduren"]) + "."
        if _has(catalog, "datum_prozedur"):
            proc_line = f"Am [[datum_prozedur]] " + proc_line[0].lower() + proc_line[1:]
        lines.append(proc_line)

    # Clinical course
    if c.get("verlauf"):
        lines.append("")
        lines.append(f"Verlauf: {c['verlauf']}.")
        # Additional patient mentions to exhaust the mention plan
        if _pat_idx < len(pat_forms) - 1:  # reserve 1 for closing
            lines.append(
                f"Bei {_pat('dat')} zeigten sich im Verlauf keine weiteren Komplikationen."
            )
        if _pat_idx < len(pat_forms) - 1:
            lines.append(f"Die Vitalparameter von {_pat('dat')} waren im Normbereich.")
        if _pat_idx < len(pat_forms) - 1:
            lines.append(
                f"Im interdisziplinären Konsens wurde {_pat('akk')} zur Mobilisation freigegeben."
            )

    if _has(catalog, "datum_konsil"):
        konsil_line = "Ein konsiliarisches Mitbeurteilen erfolgte am [[datum_konsil]]"
        if "konsiliararzt" in catalog.persons:
            konsil_line += f" durch {_person('konsiliararzt', 'akk', 0)}"
        konsil_line += "."
        lines.append(konsil_line)
    elif "konsiliararzt" in catalog.persons:
        lines.append(
            "Eine konsiliarische Mitbeurteilung durch "
            f"{_person('konsiliararzt', 'akk', 0)} ergab keinen weiteren Handlungsbedarf."
        )

    if "anaesthesist" in catalog.persons:
        lines.append(
            "Die anästhesiologische Betreuung erfolgte durch "
            f"{_person('anaesthesist', 'akk', 0)}."
        )

    if _has(catalog, "datum_befund"):
        lines.append(
            "Die laborchemische Kontrolle vom [[datum_befund]] zeigte rückläufige Entzündungswerte."
        )

    if "angehoeriger_0" in catalog.persons:
        lines.append(
            f"Im Aufnahmegespräch war {_person('angehoeriger_0', 'nom', 0)} anwesend."
        )
        # Second mention of relative in hard cases
        rel_forms = spec.mention_plan.get("angehoeriger_0", [])
        if len(rel_forms) > 1:
            lines.append(
                f"Telefonische Rücksprache mit {_person('angehoeriger_0', 'dat', 1)} am Folgetag."
            )
    # Further relatives (angehoeriger_1, angehoeriger_2, ...) planned for hard /
    # adversarial documents — render each so no planned person is dropped.
    _rel_idx = 1
    while f"angehoeriger_{_rel_idx}" in catalog.persons:
        _rel_ref = f"angehoeriger_{_rel_idx}"
        lines.append(
            f"Unterstützend steht {_person(_rel_ref, 'nom', 0)} zur Verfügung."
        )
        _rel_forms = spec.mention_plan.get(_rel_ref, [])
        if len(_rel_forms) > 1:
            lines.append(
                f"Auch mit {_person(_rel_ref, 'dat', 1)} wurde das weitere Vorgehen besprochen."
            )
        _rel_idx += 1
    if "betreuer" in catalog.persons:
        lines.append(
            "Die gesetzliche Betreuung wird durch "
            f"{_person('betreuer', 'akk', 0)} wahrgenommen."
        )
    if "notarzt" in catalog.persons:
        lines.append(
            "Die notärztliche Erstversorgung erfolgte durch "
            f"{_person('notarzt', 'akk', 0)}."
        )
    if "physiotherapeut" in catalog.persons:
        lines.append(
            "Physiotherapeutische Mobilisation durch "
            f"{_person('physiotherapeut', 'akk', 0)} ab dem 2. postoperativen Tag."
        )
    if "sozialarbeiter" in catalog.persons:
        lines.append(
            f"Der Sozialdienst ({_person('sozialarbeiter', 'nom', 0)}) "
            "organisierte die poststationäre Versorgung."
        )
    if "pflegekraft" in catalog.persons:
        lines.append(f"Bezugspflege: {_person('pflegekraft', 'nom', 0)}.")

    if c.get("medikation_entlassung"):
        lines.append("")
        med_line = (
            "Entlassungsmedikation: " + "; ".join(c["medikation_entlassung"]) + "."
        )
        if _has(catalog, "datum_medikation"):
            med_line = (
                f"Umstellung am [[datum_medikation]]: "
                + med_line[0].lower()
                + med_line[1:]
            )
        lines.append(med_line)

    # Rehabilitation referral
    if _has(catalog, "reha_klinik"):
        lines.append("Anschlussheilbehandlung in [[reha_klinik]] wird empfohlen.")

    # Discharge disposition
    if c.get("entlassung"):
        lines.append(c["entlassung"] + ".")

    if "hausarzt" in catalog.persons:
        lines.append(
            "Die hausärztliche Weiterbetreuung erfolgt durch "
            f"{_person('hausarzt', 'akk', 0)}."
        )

    # contact / identifiers
    contact = []
    if _has(catalog, "kvnr"):
        contact.append("Versichertennr. [[kvnr]]")
    if _has(catalog, "fallnr"):
        contact.append("Fallnr. [[fallnr]]")
    if _has(catalog, "patid"):
        contact.append("Pat.-ID [[patid]]")
    if "telefon" in catalog.atomics:
        contact.append("Tel. [[telefon]]")
    if "fax" in catalog.atomics:
        contact.append("Fax [[fax]]")
    if "mobil" in catalog.atomics:
        contact.append("Mobil [[mobil]]")
    if "email" in catalog.atomics:
        contact.append("E-Mail [[email]]")
    if "lanr" in catalog.atomics:
        contact.append("LANR [[lanr]]")
    if "bsnr" in catalog.atomics:
        contact.append("BSNR [[bsnr]]")
    if "ik_nr" in catalog.atomics:
        contact.append("IK [[ik_nr]]")
    if contact:
        lines.append("")
        lines.append("Administrative Daten: " + ", ".join(contact) + ".")

    # needle-in-haystack: late, short mention (only if plan has remaining entries)
    lines.append("")
    followup = "Wir bitten um Wiedervorstellung"
    if _has(catalog, "datum_wiedervorstellung"):
        followup += " am [[datum_wiedervorstellung]]"
    if _pat_idx < len(pat_forms):
        followup += f". Für Rückfragen zu {_pat('dat')} stehen wir zur Verfügung. "
    else:
        followup += ". Für Rückfragen stehen wir jederzeit zur Verfügung. "

    # --- Schlussformel (closing + signature) ---
    if framing.get("schlussformel", True):
        lines.append(
            followup + f"\n\nMit freundlichen kollegialen Grüßen\n\n"
            f"{_person('arzt_station', 'nom', 1)}\n"
            f"{_person('arzt_ober', 'nom', 1)}"
        )
    else:
        lines.append(followup)
    return "\n".join(lines)
