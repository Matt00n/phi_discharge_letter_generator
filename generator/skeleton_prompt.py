"""Stage 2: build the skeleton-authoring prompt and parse the response.

The LLM receives the entity *catalog* (refs + roles + allowed forms) and the
generation spec (style, sections, difficulty), and returns a **skeleton**: the
full Arztbrief written with typed placeholders instead of real PII. It never
sees or writes literal identifiers, so it cannot leak ungoverned PII and cannot
mislabel - labelling happens deterministically in Stage 3.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from generator.morphology import Case, PersonForm
from generator.profile_data import (
    EMBEDDED_FORMAT_INSTRUCTIONS,
    LENGTH_TARGETS,
    STYLE_DESCRIPTIONS,
)
from generator.profile_sampler import GenerationSpec
from generator.render import Catalog


class SkeletonResponse(BaseModel):
    skeleton: str = Field(
        description=(
            "Der vollständige deutsche Entlassungsbrief als Skelett-Text. PII "
            "ausschließlich über Platzhalter [[ref ...]]. Eponyme und "
            "Fachbegriffe als normalen Text ausschreiben."
        )
    )


_PERSON_FORMS = ", ".join(f.value for f in PersonForm)
_CASES = ", ".join(c.value for c in Case)


# How ICD-10 / OPS codes should appear in the running text.
_CODE_STYLE_INSTRUCTIONS = {
    "inline": (
        "KODIERUNG: Nenne zu Diagnosen und Prozeduren – sofern im klinischen "
        "Fall ein Kode angegeben ist – den ICD-10- bzw. OPS-Kode in Klammern "
        "direkt im Text (z.B. 'arterielle Hypertonie (I10.90)')."
    ),
    "primary_only": (
        "KODIERUNG: Kodiere ausschließlich die Hauptdiagnose mit ICD-10-Kode "
        "in Klammern. Nebendiagnosen und Prozeduren OHNE Kodes als reinen "
        "klinischen Klartext ausschreiben."
    ),
    "none": (
        "KODIERUNG: Verwende KEINE ICD-10- oder OPS-Kodes im Text. Schreibe "
        "alle Diagnosen und Prozeduren ausschließlich als klinischen Klartext "
        "(wie in vielen narrativen oder älteren Arztbriefen üblich). Die im "
        "klinischen Fall genannten Kodes dienen dir nur zur Orientierung und "
        "dürfen NICHT im Brieftext erscheinen."
    ),
}


# Opening-sentence register, to break the "Wir berichten über ..." monotony.
_OPENING_STYLE_INSTRUCTIONS = {
    "bericht": (
        "EINLEITUNG: Beginne in klassischer Berichtsform "
        "(z.B. 'Wir berichten über ...')."
    ),
    "aufnahme": (
        "EINLEITUNG: Beginne mit dem Aufnahmevorgang bzw. der Zuweisung "
        "(z.B. 'Die stationäre Aufnahme von ... erfolgte am ...' oder "
        "'... wurde uns am ... zugewiesen'). Vermeide die Formel "
        "'Wir berichten über'."
    ),
    "diagnose_zuerst": (
        "EINLEITUNG: Beginne mit der führenden Diagnose / dem Grund des "
        "Aufenthalts vorangestellt, bevor die Person genannt wird "
        "(z.B. 'Bei stationärer Aufnahme bestand der dringende Verdacht "
        "auf ...'). Vermeide die Formel 'Wir berichten über'."
    ),
    "epikrise_kurz": (
        "EINLEITUNG: Beginne mit einer knappen zusammenfassenden Feststellung "
        "zum Aufenthalt (Aufnahmedatum, Hauptdiagnose), bevor Details folgen. "
        "Vermeide die Formel 'Wir berichten über'."
    ),
    "frei": (
        "EINLEITUNG: Wähle einen natürlichen, abwechslungsreichen Einstieg, "
        "der NICHT mit 'Wir berichten über' beginnt."
    ),
}


def _build_framing_instructions(spec: GenerationSpec, catalog: Catalog) -> str:
    """Build instructions for document framing elements (letterhead, greeting, etc.)."""
    framing = getattr(spec, "framing", {})
    if not any(framing.values()):
        return ""

    parts: list[str] = []
    parts.append("RAHMENELEMENTE DES BRIEFES:")

    if framing.get("briefkopf"):
        # Determine which entities are available for the letterhead
        has_klinik_addr = "adresse_klinik" in catalog.atomics
        addr_hint = (
            "Adresse ([[adresse_klinik]], [[plz_klinik]] [[ort_klinik]]), "
            if has_klinik_addr
            else ""
        )
        has_url = "klinik_url" in catalog.atomics
        url_hint = "Website ([[klinik_url]]), " if has_url else ""
        parts.append(
            "  • BRIEFKOPF: Beginne den Brief mit einem formalen Briefkopf "
            "(Klinik-Header), der typischerweise enthält: "
            f"[[institution]], {addr_hint}"
            f"Telefon ([[telefon]]), Fax ([[fax]]), {url_hint}"
            "ggf. Abteilungsbezeichnung. "
            "Formatiere ihn wie einen echten Briefkopf (z.B. zentriert oder "
            "linksbündig, wie auf Klinik-Briefpapier üblich)."
        )

    if framing.get("anrede"):
        # Choose between generic and addressed greeting
        has_einweiser = "arzt_einweiser" in catalog.persons
        if has_einweiser:
            parts.append(
                "  • ANREDE: Füge nach dem Briefkopf/vor dem Inhalt eine "
                "Grußzeile ein. Varianten:\n"
                "    – 'Sehr geehrte Kolleginnen und Kollegen,' (generisch)\n"
                "    – 'Sehr geehrte Frau Kollegin,' / 'Sehr geehrter Herr Kollege,' (an Einweiser)\n"
                "    – '[[arzt_einweiser form=salutation case=nom]],' (personalisiert)\n"
                "    Wähle eine passende Variante. Die generische Form ist KEIN "
                "Platzhalter und enthält kein PII."
            )
        else:
            parts.append(
                "  • ANREDE: Füge vor dem Inhalt eine Grußzeile ein: "
                "'Sehr geehrte Kolleginnen und Kollegen,'"
            )

    if framing.get("patientenstammdaten"):
        # Patient info block — structured header with demographics
        fields: list[str] = []
        fields.append("Name: [[patient form=hon_first_last case=nom]]")
        if "geburtsdatum" in catalog.atomics:
            fields.append("geb.: [[geburtsdatum]]")
        if "adresse" in catalog.atomics:
            fields.append("Anschrift: [[adresse]], [[plz]] [[ort]]")
        if "kvnr" in catalog.atomics:
            fields.append("Vers.-Nr.: [[kvnr]]")
        if "fallnr" in catalog.atomics:
            fields.append("Fallnr.: [[fallnr]]")
        if "patid" in catalog.atomics:
            fields.append("Pat.-ID: [[patid]]")
        if "versicherer" in catalog.atomics:
            fields.append("Kasse: [[versicherer]]")
        parts.append(
            "  • PATIENTENSTAMMDATEN-BLOCK: Füge einen strukturierten "
            "Kopfbereich mit Patientendaten ein (vor dem Fließtext, nach "
            "Briefkopf/Anrede). Mögliche Felder:\n"
            "    " + "\n    ".join(fields)
        )

    if framing.get("schlussformel"):
        parts.append(
            "  • SCHLUSSFORMEL & UNTERSCHRIFT: Beende den Brief mit einer "
            "Grußformel und Unterschriftenblock:\n"
            "    – 'Mit freundlichen Grüßen' / 'Mit kollegialen Grüßen' / "
            "'Mit freundlichen kollegialen Grüßen'\n"
            "    – Danach die unterschreibenden Ärzte mit Platzhaltern, z.B.:\n"
            "      [[arzt_station form=hon_first_last case=nom]]\n"
            "      [[arzt_ober form=hon_first_last case=nom]]\n"
            "    – Optional mit Funktionsbezeichnung (Stationsarzt, Oberarzt/Oberärztin, "
            "Chefarzt etc.) unter dem Namen."
        )

    return "\n".join(parts)


def _serialize_catalog(catalog: Catalog, spec: GenerationSpec) -> str:
    lines: list[str] = []
    lines.append("PERSONEN (Platzhalter: [[ref form=<form> case=<case>]]):")
    by_ref = {b["ref"]: b for b in spec.entity_brief}
    for ref, person in catalog.persons.items():
        b = by_ref.get(ref, {})
        role = b.get("role", person.label.value)
        extra = f", Titel: {person.titel}" if person.titel else ""
        forms = spec.mention_plan.get(ref, [])
        form_hint = f", Formen: {' → '.join(forms)}" if forms else ""
        lines.append(f"  - {ref}: {role} (Anrede: {person.anrede}{extra}{form_hint})")

    lines.append("")
    lines.append("ATOMARE DATEN (Platzhalter: [[ref]] oder [[ref variant=<variant>]]):")
    for ref, atomic in catalog.atomics.items():
        b = by_ref.get(ref, {})
        role = b.get("role", atomic.label.value)
        variants = ", ".join(sorted(atomic.variants))
        lines.append(f"  - {ref}: {role} [Varianten: {variants}]")

    if catalog.negatives:
        # Split negatives into eponyms/lookalikes vs non-PII IDs
        eponym_negs = [
            n
            for n in catalog.negatives.values()
            if n.label.value in ("EPONYM", "LOOKALIKE")
        ]
        id_negs = [
            n for n in catalog.negatives.values() if n.label.value == "NONPII_ID"
        ]

        if eponym_negs:
            lines.append("")
            lines.append(
                "EPONYME / FACHBEGRIFFE (KEIN Platzhalter – als normalen Text einbauen, "
                "exakt wie angegeben):"
            )
            for neg in eponym_negs:
                lines.append(f"  - {neg.surface}")

        if id_negs:
            lines.append("")
            lines.append(
                "NICHT-PII NUMMERN (KEIN Platzhalter – als normalen Text einbauen, "
                "exakt wie angegeben; diese sind KEINE personenbezogenen Daten):"
            )
            for neg in id_negs:
                lines.append(f"  - {neg.surface}")

    return "\n".join(lines)


def _clean_desc(desc: str | None) -> str | None:
    """Strip ICD-10/OPS catalog notation from a diagnosis/procedure description.

    Removes bracketed inclusion lists ("[Humerus, Ellenbogengelenk]") and
    "not otherwise specified" catalog tails ("N.n.bez.", "nicht näher
    bezeichnet"), which the model otherwise copies verbatim into the prose. Only
    the human-readable description is touched; the separate ICD/OPS code token is
    never passed through here, so code extraction downstream is unaffected.
    """
    if not desc:
        return desc
    desc = re.sub(r"\s*\[[^\]]*\]", "", desc)  # inclusion/exclusion term lists
    # OPS transfusion/apheresis quantity-range enumeration, e.g.
    # "294 bis unter 314 Apherese-Thrombozytenkonzentrate" / "64 bis unter 70 TKs";
    # keep the clinical noun phrase, drop the catalog counting scaffold.
    desc = re.sub(r"\d+\s+bis\s+unter\s+\d+\s+", "", desc, flags=re.IGNORECASE)
    desc = re.sub(
        r"[,:]\s*(?:n\.?\s*n\.?\s*bez\.?|nicht näher bezeichnet)\.?\s*$",
        "",
        desc,
        flags=re.IGNORECASE,
    )
    desc = re.sub(r"\s{2,}", " ", desc).strip(" :;,")
    return desc


def _clinical_block(spec: GenerationSpec) -> str:
    c = spec.clinical_case
    code_style = getattr(spec, "code_style", "inline")

    def _fmt(code: str, desc: str, *, primary: bool = False) -> str:
        # "inline" codes everything; "primary_only" codes just the main
        # diagnosis; "none" never shows a code.
        desc = _clean_desc(desc)
        if code_style == "inline" or (code_style == "primary_only" and primary):
            return f"{desc} ({code})"
        return desc

    if code_style == "none":
        haupt = _clean_desc(c.get("hauptdiagnose_text") or c.get("hauptdiagnose"))
    else:
        ht, hc = _clean_desc(c.get("hauptdiagnose_text")), c.get("hauptdiagnose_code")
        haupt = f"{ht} ({hc})" if ht and hc else _clean_desc(c.get("hauptdiagnose"))

    parts = [
        f"Fachabteilung: {c.get('fachabteilung')}",
        f"Aufnahmeanlass: {c.get('anlass')}",
        f"Hauptdiagnose: {haupt}",
    ]

    # Demographic coherence: give the model a *qualitative* age band as a hard
    # constraint so it does not invent adult biographies for children (jobs,
    # spouses, own children, occupational accidents) or obstetric content for
    # post-reproductive patients — contradictions that code-level gating alone
    # cannot reach because they live in the free-text narrative.
    #
    # IMPORTANT: we deliberately do NOT state the exact age here. Injecting the
    # literal number made the model echo it verbatim as free text ("57-jährige
    # Patientin" im Aufnahmebefund), producing un-annotated AGE leaks that the
    # red-team flags (~10x the baseline leak rate). All coherence constraints
    # below are band-level and need no exact number; the exact age reaches the
    # text only via the [[alter]] placeholder.
    age = getattr(spec, "patient_age", 0) or 0
    if 0 < age < 15:
        parts.insert(
            0,
            "WICHTIG – Patient ist ein Kind. Alle Inhalte "
            "altersgerecht: KEINE Berufstätigkeit, KEIN Arbeitsplatz/Arbeitsunfall, "
            "KEINE Ehe/Partnerschaft, KEINE eigenen Kinder, keine Militär-/"
            "Auslandseinsätze. Bezugs-/Kontaktpersonen sind Eltern bzw. "
            "Sorgeberechtigte. Anamnese und Verlauf pädiatrisch plausibel halten.",
        )
    elif 15 <= age < 18:
        parts.insert(
            0,
            "WICHTIG – Patient ist ein:e Jugendliche:r. Inhalte "
            "altersgerecht: Schule/Ausbildung statt etablierter Berufskarriere, "
            "in der Regel keine Ehe und keine eigenen Kinder; Bezugspersonen "
            "meist die Eltern.",
        )
    elif age >= 55:
        parts.insert(
            0,
            "WICHTIG – Patient ist im höheren Lebensalter. KEINE Schwangerschafts-, "
            "Geburtshilfe- oder Wochenbett-Inhalte.",
        )

    neben = c.get("nebendiagnosen_struct")
    if neben:
        parts.append(
            "Nebendiagnosen: " + "; ".join(_fmt(code, desc) for code, desc in neben)
        )
    elif c.get("nebendiagnosen"):
        parts.append(
            "Nebendiagnosen: " + "; ".join(_clean_desc(d) for d in c["nebendiagnosen"])
        )

    proz = c.get("prozeduren_struct")
    if proz:
        parts.append(
            "Prozeduren: " + "; ".join(_fmt(code, desc) for code, desc in proz)
        )
    elif c.get("prozeduren"):
        parts.append(
            "Prozeduren: " + "; ".join(_clean_desc(p) for p in c["prozeduren"])
        )

    if c.get("medikation_entlassung"):
        parts.append("Entlassungsmedikation: " + "; ".join(c["medikation_entlassung"]))

    # Temporal frame: bound the narrative so invented day-references (e.g. "am
    # 7. postoperativen Tag") cannot exceed the actual length of stay.
    stay = getattr(spec, "stay_days", 0)
    if stay <= 0:
        parts.append(
            "Aufenthaltsdauer: ambulant (Aufnahme und Entlassung am selben Tag)"
        )
    else:
        parts.append(f"Aufenthaltsdauer: {stay} Tage")

    if c.get("scenario_hint"):
        parts.append(f"Klinischer Hinweis: {c['scenario_hint']}")

    return "\n".join(parts)


def build_system_instruction() -> str:
    return (
        "Du bist eine erfahrene Oberärztin / ein erfahrener Oberarzt und verfasst "
        "klinisch authentische, grammatikalisch einwandfreie deutsche "
        "Entlassungsbriefe (Arztbriefe).\n\n"
        "WICHTIG – PLATZHALTER-SYSTEM:\n"
        "Du schreibst KEINE echten personenbezogenen Daten (Namen, Daten, Adressen, "
        "Nummern) direkt in den Text. Stattdessen verwendest du AUSSCHLIESSLICH "
        "Platzhalter der Form [[ref ...]]. Ein nachgelagertes System ersetzt diese "
        "Platzhalter deterministisch durch echte Werte.\n\n"
        "Platzhalter-Syntax:\n"
        "  • Personen: [[ref form=<form> case=<case>]]\n"
        f"      form ∈ {{{_PERSON_FORMS}}}\n"
        f"      case ∈ {{{_CASES}}}  (grammatischer Kasus im Satz!)\n"
        "      Beispiel: '... wurde [[patient form=hon_first_last case=nom]] "
        "aufgenommen. Wir berichten über [[patient form=hon_last case=akk]].'\n"
        "  • Atomare Daten: [[ref]] oder [[ref variant=<variant>]]\n"
        "      Beispiel: 'geb. [[geburtsdatum variant=long]]'\n\n"
        "REGELN:\n"
        "1. Wähle den grammatisch korrekten 'case' passend zur Satzstellung "
        "(Nominativ/Genitiv/Dativ/Akkusativ). Wähle die 'form' passend zum Kontext "
        "(volle Nennung bei Erstnennung, danach gern kürzer: hon_last, last, "
        "initial_last, hon_initial).\n"
        "2. Erfinde NIEMALS zusätzliche Namen, Adressen, Telefonnummern, "
        "Befundnummern oder Institutionsnamen — AUCH NICHT in eingebetteten "
        "Konsilberichten, Pathologiebefunden, Laborbefunden oder zitierten "
        "Vorbefunden. Statt erfundener Namen verwende:\n"
        "   • Für Ärzte: einen verfügbaren Platzhalter aus dem Katalog "
        "(z.B. [[konsiliararzt form=hon_last case=nom]]) ODER eine generische "
        "Bezeichnung ('der Pathologe', 'die Konsiliarärztin', 'die untersuchende "
        "Ärztin').\n"
        "   • Für Institutionen: [[institution]] oder eine generische Bezeichnung "
        "('das auswärtige Labor', 'die zuweisende Klinik').\n"
        "   • Für Befundnummern: weglassen oder generisch ('s. Befundbericht').\n"
        "   • Für Telefon-/Faxnummern: [[telefon]], [[fax]] oder weglassen.\n"
        "   Der Katalog kann neben den Kernpersonen (Patient, Angehörige, Einweiser, "
        "Stations-/Oberarzt) auch weitere Personen enthalten (Konsiliararzt, Hausarzt, "
        "Physiotherapeut, Sozialarbeiter etc.) — verwende deren Platzhalter in "
        "passenden Kontexten.\n"
        "3. Eponyme/Fachbegriffe aus der Liste werden als normaler Text exakt "
        "ausgeschrieben – niemals als Platzhalter.\n"
        "4. Variiere die Erwähnungsformen realistisch über den Brief hinweg. "
        "Zu jeder Person ist eine Sequenz von 'Formen' angegeben (z.B. "
        "hon_first_last → hon_last → initial_last). Verwende diese Formen der "
        "Reihe nach bei aufeinanderfolgenden Erwähnungen derselben Person. "
        "Den grammatischen Kasus (case) bestimmst du weiterhin selbst.\n"
        "5. Antworte ausschließlich mit dem Skelett-Text im JSON-Feld 'skeleton'.\n"
        "6. Alle im Katalog aufgeführten atomaren Quasi-Identifikatoren müssen über "
        "ihren Platzhalter erscheinen – niemals als Freitext oder Umschreibung. "
        "Verwende ausschließlich Platzhalter, die im Katalog stehen; "
        "erfinde keine weiteren.\n"
        "7. Formuliere Diagnosen und Prozeduren in natürlicher klinischer Sprache; "
        "übernimm keine Katalog-Schreibweise wörtlich (keine eckigen Klammern, keine "
        "'n.n.bez.'-Zusätze). Halte zeitliche Verlaufsangaben innerhalb der "
        "angegebenen Aufenthaltsdauer.\n\n"
        "FALSCH (niemals so schreiben!):\n"
        '  "Untersucher: Dr. med. Weber"\n'
        '  "Urologische Gemeinschaftspraxis Dr. Schwarz & Partner"\n'
        '  "Tel: 040-99887766"\n'
        '  "Befund-Nr. H-2023-9842"\n'
        '  "Prof. Dr. med. Müller"\n'
        '  "Der Patient war als Buchhalter tätig."\n\n'
        "RICHTIG:\n"
        '  "Untersucher: [[konsiliararzt form=hon_last case=nom]]"\n'
        '  "Untersucher: die zuständige Fachärztin"\n'
        '  "[[institution]]"\n'
        '  "Tel: [[telefon]]"\n'
        "  \"Der Patient war als [[beruf]] tätig.\"  (nur wenn 'beruf' im Katalog steht)\n"
        "  (Befundnummern weglassen)"
    )


def build_user_content(
    spec: GenerationSpec, catalog: Catalog, word_collision_context_max: int = 5
) -> str:
    diff_instructions = {
        "easy": "Einfacher Fall: überwiegend vollständige Namensnennungen, klare Struktur.",
        "medium": "Mittlerer Fall: variiere Erwähnungsformen, einige Kurz-/Initialformen, "
        "baue Quasi-Identifikatoren (Beruf/Arbeitgeber) natürlich ein – "
        "stets per Platzhalter (z.B. '[[beruf]]'), nie als Freitext.",
        "hard": "Schwerer Fall: nutze Initial- und Kurzformen, platziere mindestens eine "
        "PII-Nennung weit hinten im Text (needle-in-haystack), baue ggf. "
        "gleiche Nachnamen verschiedener Personen natürlich ein.",
        "adversarial": "Adversarialer Fall: maximiere die Schwierigkeit für automatische "
        "Anonymisierung. Verwende möglichst kurze/ambigue Erwähnungsformen "
        "(nur Nachname, keine Titel, Initialformen). PII soll an unerwarteten "
        "Stellen auftauchen und leicht übersehen werden können.",
    }[spec.difficulty]

    extra = []
    if "needle_in_haystack" in spec.difficulty_tags:
        extra.append(
            "- Verstecke mindestens eine Patienten-Nennung in einem späten Abschnitt."
        )
    if "shared_surname" in spec.difficulty_tags:
        hint = (
            "- Eine behandelnde Ärztin/ein Arzt trägt denselben Nachnamen wie der Patient; "
            "nutze beide Platzhalter konsequent, damit sie unterscheidbar bleiben."
        )
        if "shared_surname_remark" in spec.difficulty_tags:
            hint += " Mache keine explizite Bemerkung über diese Namensgleichheit im Brieftext."
        extra.append(hint)
    if spec.collegial_address:
        extra.append(
            "- Der Patient / die Patientin ist selbst Mediziner:in. Verwende bei jeder "
            "kollegialen Erwähnung im Brief den Platzhalter [[beruf_kollegial]] "
            "als kollegiale Anrede anstelle von 'der Patient / die Patientin'. "
            "Schreibe den Artikel/das Possessivpronomen VOR den Platzhalter "
            "(z.B. 'der [[beruf_kollegial]]', 'unser [[beruf_kollegial]]', "
            "'als [[beruf_kollegial]]'). Der Platzhalter enthält NUR das "
            "Substantiv (Kollege/Kollegin). "
            "ACHTUNG: Die Standardbegrüßung 'Sehr geehrte Kolleginnen und "
            "Kollegen' am Briefanfang ist KEIN Platzhalter — sie bleibt normaler "
            "Text und bezieht sich auf den Empfänger, nicht den Patienten."
        )
    if spec.placement_hints:
        extra.append(
            "- PLATZIERUNG: Die folgenden PII-Platzhalter sollen an "
            "ungewöhnlichen/nicht-standard Stellen im Text erscheinen:"
        )
        for hint in spec.placement_hints:
            extra.append(f"  · {hint}")

    # --- Adversarial sub-challenge instructions ---
    if spec.adversarial_mode == "eponym_collision":
        extra.append(
            "- EPONYM-KOLLISION: Der Nachname des Patienten ist identisch mit "
            "einem medizinischen Eponym (z.B. 'Crohn', 'Parkinson'). Baue im "
            "klinischen Text die namensgleiche Erkrankung/den namensgleichen "
            "Befund ein (als normalen Text, KEIN Platzhalter), sodass der Nachname "
            "des Patienten (Platzhalter) und der Krankheitsname (normaler Text) "
            "im selben Dokument vorkommen. Das erzeugt eine echte "
            "Disambiguierungsaufgabe."
        )
    elif spec.adversarial_mode == "word_collision":
        extra.append(
            "- WORT-KOLLISION: Der Nachname des Patienten ist ein häufiges "
            "deutsches Wort. Verwende dieses Wort im klinischen Text mehrfach "
            "in seiner natürlichen Bedeutung (als normalen Text ohne Platzhalter). "
            "Beispiel: Patient 'Klein' → 'kleine Läsion', 'kleinzellig', "
            "'Kleinhirn' etc. Der Anonymisierer muss den Namen-Platzhalter vom "
            "normalen Wortgebrauch unterscheiden."
        )
        if spec.word_surname_context:
            extra.append(
                f"  Verwende folgende Wortformen natürlich im Text: "
                f"{', '.join(spec.word_surname_context[:word_collision_context_max])}"
            )
    elif spec.adversarial_mode == "deep_needle":
        extra.append(
            "- DEEP NEEDLE: Schreibe einen langen, detaillierten Brief. "
            "Verstecke GENAU EINE einzige Patienten-Nennung (Platzhalter) "
            "tief im Text — in einem späten, unauffälligen Abschnitt "
            "(z.B. mitten in einem Laborbefund, einer Medikamentenliste, "
            "oder einem Nebensatz im Verlauf). Alle anderen Stellen sollen "
            "den Patienten nur als 'der Patient' / 'die Patientin' "
            "oder mit Pronomen bezeichnen."
        )
    elif spec.adversarial_mode == "typo_text":
        extra.append(
            "- TIPPFEHLER/GRAMMATIK: Simuliere einen hastig geschriebenen "
            "oder schlecht OCR-gescannten Brief. Baue gelegentlich "
            "Tippfehler, fehlende Buchstaben, Buchstabendreher und "
            "grammatikalische Fehler in den NORMALEN TEXT ein "
            "(NICHT in die Platzhalter!). Ca. 2-5% der Wörter sollen "
            "kleine Fehler enthalten. Die Platzhalter [[...]] müssen "
            "EXAKT korrekt bleiben."
        )

    extra_block = ("\nBesondere Hinweise:\n" + "\n".join(extra)) if extra else ""

    # --- Medication suppression instruction ---
    if not getattr(spec, "has_medications", True):
        extra_block += (
            "\nMEDIKATION: Dieser Brief enthält KEINE dedizierten "
            "Medikamentenlisten (keine Aufnahme- oder Entlassungsmedikation). "
            "Erwähne keine konkreten Medikamentennamen, Dosierungen oder "
            "Einnahmeschemata."
        )

    # --- Build section instructions with format hints ---
    section_lines: list[str] = []
    for sec_name, fmt in spec.section_formats:
        fmt_instr = EMBEDDED_FORMAT_INSTRUCTIONS.get(fmt, "")
        if fmt_instr:
            section_lines.append(f"  • {sec_name} [{fmt}] — {fmt_instr}")
        else:
            section_lines.append(f"  • {sec_name}")
    section_block = "\n".join(section_lines)

    # --- Structure mode instruction ---
    _STRUCTURE_INSTRUCTIONS = {
        "structured": (
            "GLIEDERUNG (verwende genau diese Abschnitte mit Überschriften in "
            "dieser Reihenfolge):"
        ),
        "mixed": (
            "GLIEDERUNG (mische strukturierte und unstrukturierte Teile: "
            "einige Abschnitte mit Überschrift, andere fließend ineinander "
            "übergehend ohne klare Trennung. Nicht alle Themen brauchen eine "
            "eigene Überschrift):"
        ),
        "unstructured": (
            "INHALTLICHE THEMEN (schreibe den Brief als zusammenhängenden "
            "Fließtext OHNE Abschnitts-Überschriften. Die folgenden Themen "
            "sollen inhaltlich abgedeckt werden, aber fließend ineinander "
            "übergehen):"
        ),
    }
    structure_instruction = _STRUCTURE_INSTRUCTIONS[spec.structure_mode]

    # Style and length descriptions
    style_desc = STYLE_DESCRIPTIONS.get(spec.style, "")
    length_desc = LENGTH_TARGETS.get(spec.length_hint, spec.length_hint)

    code_instruction = _CODE_STYLE_INSTRUCTIONS.get(
        getattr(spec, "code_style", "inline"), _CODE_STYLE_INSTRUCTIONS["inline"]
    )
    opening_instruction = _OPENING_STYLE_INSTRUCTIONS.get(
        getattr(spec, "opening_style", "bericht"),
        _OPENING_STYLE_INSTRUCTIONS["bericht"],
    )

    framing_block = _build_framing_instructions(spec, catalog)
    framing_section = f"\n\n{framing_block}" if framing_block else ""

    return (
        f"Erstelle einen Entlassungsbrief.\n\n"
        f"STIL: {spec.style}\n  → {style_desc}\n"
        f"LÄNGE: {spec.length_hint} ({length_desc})\n"
        f"SCHWIERIGKEIT: {diff_instructions}\n"
        f"{opening_instruction}\n"
        f"{code_instruction}\n\n"
        f"{structure_instruction}\n"
        f"{section_block}\n\n"
        f"KLINISCHER FALL:\n{_clinical_block(spec)}\n\n"
        f"ENTITÄTEN-KATALOG:\n{_serialize_catalog(catalog, spec)}\n"
        f"{extra_block}{framing_section}\n\n"
        f"Schreibe nun den vollständigen Brief als Skelett mit Platzhaltern."
    )
