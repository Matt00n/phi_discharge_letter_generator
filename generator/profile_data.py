"""Data pools and sampling helpers for profile generation.

Separated from profile_sampler.py for clarity. Contains:
- Demographic distributions (age, marital status, clinician titles)
- Institution/practice name templates and samplers
- Document structure constants (styles, sections, professions, employers)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Realistic German age pyramid (Statistisches Bundesamt 2023, simplified).
# Bins: 0-4, 5-14, 15-24, 25-34, 35-44, 45-54, 55-64, 65-74, 75-84, 85+
# Weights approximate the actual hospital inpatient age mix (DRG-Statistik):
# hospitals see disproportionately more elderly and fewer children.
# ---------------------------------------------------------------------------
AGE_BINS = [
    ((0, 4), 0.02),
    ((5, 14), 0.03),
    ((15, 24), 0.05),
    ((25, 34), 0.07),
    ((35, 44), 0.09),
    ((45, 54), 0.13),
    ((55, 64), 0.17),
    ((65, 74), 0.19),
    ((75, 84), 0.17),
    ((85, 99), 0.08),
]

# ---------------------------------------------------------------------------
# Marital status by age (simplified from Mikrozensus 2022).
# ---------------------------------------------------------------------------
MARITAL_STATUS_BY_AGE = {
    # age_min: {status: probability}
    0: {"ledig": 1.0},
    18: {"ledig": 0.90, "verheiratet": 0.10},
    25: {"ledig": 0.60, "verheiratet": 0.35, "geschieden": 0.05},
    35: {"ledig": 0.25, "verheiratet": 0.55, "geschieden": 0.15, "verwitwet": 0.05},
    50: {"ledig": 0.15, "verheiratet": 0.55, "geschieden": 0.20, "verwitwet": 0.10},
    65: {"ledig": 0.08, "verheiratet": 0.50, "geschieden": 0.17, "verwitwet": 0.25},
    80: {"ledig": 0.05, "verheiratet": 0.30, "geschieden": 0.10, "verwitwet": 0.55},
}

# ---------------------------------------------------------------------------
# Role-correlated clinician title pools.
# ---------------------------------------------------------------------------
TITEL_BY_ROLE = {
    "arzt_einweiser": [
        ("Dr. med.", 0.60),
        (None, 0.25),  # niedergelassener Arzt without title
        ("Prof. Dr. med.", 0.05),
        ("Dr. med. univ.", 0.10),
    ],
    "arzt_station": [
        ("Dr. med.", 0.40),
        (None, 0.40),  # Assistenzarzt often has no title
        ("Dr. med. univ.", 0.15),
        ("Dr. rer. nat.", 0.05),
    ],
    "arzt_ober": [
        ("Prof. Dr. med.", 0.25),
        ("PD Dr. med.", 0.20),
        ("Dr. med.", 0.45),
        ("Dr. med. habil.", 0.10),
    ],
}

# ---------------------------------------------------------------------------
# Institution name templates.
# Pattern strings use {city} and {dept} for interpolation.
# Real German hospital names fall into distinct naming patterns.
# ---------------------------------------------------------------------------
INSTITUTION_TEMPLATES = [
    # University hospitals
    "Universitätsklinikum {city}",
    "Universitätsklinikum {city} – Klinik für {dept}",
    "Universitätsmedizin {city}",
    # Large public clinics
    "Klinikum {city}",
    "Klinikum {city} – {dept}",
    "Städtisches Klinikum {city}",
    "Krankenhaus {city}-Mitte",
    # Religious / charitable
    "Kliniken Maria Hilf GmbH",
    "St. Elisabeth-Krankenhaus {city}",
    "Ev. Krankenhaus Bethesda {city}",
    "Diakonissenkrankenhaus {city}",
    "Caritas-Klinik {city}",
    # Named chains / foundations
    "Sana Kliniken {city}",
    "Asklepios Klinik {city}",
    "Helios Klinikum {city}",
    "Schön Klinik {city}",
    "Vivantes Klinikum {city}",
    # Rehab / specialized
    "Rehaklinik {city}",
    "Neurologisches Rehabilitationszentrum {city}",
    "Fachklinik {city} für {dept}",
    "Reha-Zentrum {city}",
]

PRAXIS_TEMPLATES_NAMED = [
    # Named after the physician (classic)
    "Praxis {arzt}",
    "Gemeinschaftspraxis {arzt}",
    "Hausarztpraxis {arzt}",
    "Praxis für Allgemeinmedizin {arzt}",
    "Facharztpraxis für Innere Medizin {arzt}",
    "Internistische Praxis {arzt}",
    "Praxis für Kardiologie {arzt}",
    "Orthopädische Praxis {arzt}",
    "Chirurgische Praxis {arzt}",
]

PRAXIS_TEMPLATES_GENERIC = [
    # Location-based / creative names (not named after owner)
    "Praxis am Marktplatz, {city}",
    "Hausarztzentrum am Ring, {city}",
    "Praxis im Ärztehaus {city}",
    "Gesundheitszentrum Mitte, {city}",
    "Praxis an der Alten Post, {city}",
    "Ärztehaus am Bahnhof {city}",
    "Praxis am Stadtpark {city}",
    "Hausarztpraxis am Rathaus, {city}",
    "Praxis im Gesundheitszentrum {city}",
    "Internistische Gemeinschaftspraxis am Dom, {city}",
    "MVZ {city}",
    "Medizinisches Versorgungszentrum {city}",
    "MVZ am Klinikum {city}",
    "MVZ Zentrum {city}",
    "Ärztezentrum Süd, {city}",
    "Hausärzte am Schlosspark, {city}",
    "MedPraxis {city}-Mitte",
    "Praxis an der Linde, {city}",
    "Gesundheitshaus am Brunnen, {city}",
]

# Cities for "other" institutions (not the patient's city).
# Includes mid-size cities not in atom_samplers.CITY_PLZ to avoid collisions.
OTHER_CITIES = [
    "Essen",
    "Düsseldorf",
    "Nürnberg",
    "Hannover",
    "Stuttgart",
    "Dresden",
    "Mannheim",
    "Bonn",
    "Aachen",
    "Würzburg",
    "Freiburg",
    "Rostock",
    "Augsburg",
    "Wiesbaden",
    "Kiel",
    "Regensburg",
    "Ulm",
    "Heidelberg",
    "Lübeck",
    "Münster",
    "Göttingen",
    "Erlangen",
    "Mainz",
    "Jena",
    "Darmstadt",
    "Tübingen",
    "Marburg",
    "Gießen",
    "Greifswald",
    "Bamberg",
    "Passau",
    "Ingolstadt",
    "Hildesheim",
    "Osnabrück",
    "Paderborn",
    "Siegen",
    "Solingen",
    "Leverkusen",
    "Reutlingen",
    "Bochum",
    "Wuppertal",
    "Gelsenkirchen",
    "Oberhausen",
    "Krefeld",
    "Pforzheim",
    "Offenbach",
    "Wolfsburg",
    "Salzgitter",
    "Plauen",
    "Zwickau",
    "Fulda",
    "Neubrandenburg",
    "Stralsund",
    "Dessau",
]

# Document style and structure constants
STYLES = ["Standard-Klinik", "Akademisch-Ausführlich", "Telegrammstil-Kompakt"]

# Style descriptions for the LLM prompt — concrete writing instructions.
STYLE_DESCRIPTIONS: dict[str, str] = {
    "Standard-Klinik": (
        "Üblicher Klinikarztbrief: vollständige Sätze, sachlicher Ton, "
        "klare Gliederung mit Überschriften, angemessene Detailtiefe."
    ),
    "Akademisch-Ausführlich": (
        "Ausführlicher akademischer Stil: lange, differenzierte Sätze, "
        "detaillierte Befundbeschreibungen, Literaturverweise/Leitlinien möglich, "
        "vollständige Differentialdiagnostik, umfangreiche Epikrise."
    ),
    "Telegrammstil-Kompakt": (
        "Telegrafischer Kurzstil: unvollständige Sätze, Stichpunkte, "
        "Abkürzungen (Z.n., ED, V.a., St.p., AZ, EZ, BSG, CRP ...), "
        "kein Fließtext in Labor/Diagnostik, minimale Formulierungen."
    ),
}

# Length calibration — word-count target ranges for the LLM.
LENGTH_TARGETS: dict[str, str] = {
    "kurz": "150–350 Wörter (ca. ½–1 Seite)",
    "mittel": "400–800 Wörter (ca. 1–2 Seiten)",
    "ausführlich": "800–1500 Wörter (ca. 2–4 Seiten)",
    "sehr_lang": "1500–3500 Wörter (ca. 5–10 Seiten)",
}

# --- Section pool with per-section metadata --------------------------------
# Each section has:
#   name: heading text
#   weight: base sampling probability (higher = more common)
#   core: if True, always included regardless of sampling
#   format_options: possible formatting styles the LLM can be told to use
#   requires: other sections that should be present if this one is (soft dep)


@dataclass
class SectionDef:
    name: str
    weight: float
    core: bool = False
    format_options: tuple[str, ...] = ("fließtext",)
    requires: tuple[str, ...] = ()


# fmt: off
SECTION_POOL: list[SectionDef] = [
    # Core sections (almost always present in real Arztbriefe)
    SectionDef("Anamnese", 1.0, core=True),
    SectionDef("Diagnosen", 0.80, format_options=("fließtext", "stichpunkte", "tabellarisch")),
    SectionDef("Aufnahmebefund", 0.85),
    SectionDef("Körperliche Untersuchung", 0.70),
    SectionDef("Diagnostik", 0.80, format_options=("fließtext", "stichpunkte", "tabellarisch")),
    SectionDef("Verlauf", 0.90),
    SectionDef("Therapie", 0.75),
    SectionDef("Interventionen", 0.45, format_options=("fließtext", "stichpunkte")),
    SectionDef("Epikrise", 0.70),
    SectionDef("Procedere / Empfehlungen", 0.85),
    # Medication sections
    SectionDef("Aufnahmemedikation", 0.50, format_options=("fließtext", "tabellarisch", "stichpunkte")),
    SectionDef("Entlassungsmedikation", 0.70, format_options=("fließtext", "tabellarisch", "stichpunkte")),
    # Clinical sub-sections (variably present)
    SectionDef("Labor", 0.65, format_options=("fließtext", "tabellarisch", "stichpunkte")),
    SectionDef("Bildgebung", 0.45, format_options=("fließtext", "befund_zitat")),
    SectionDef("Mikrobiologie", 0.20, format_options=("fließtext", "tabellarisch")),
    SectionDef("Histologie", 0.25, format_options=("fließtext", "befund_zitat")),
    SectionDef("Konsil", 0.35, format_options=("fließtext", "eingebetteter_befund")),
    SectionDef("Pathologie", 0.15, format_options=("fließtext", "befund_zitat")),
    # Embedded sub-documents (PII inside foreign-format blocks — hard cases)
    SectionDef("Zitierter Vorbefund", 0.20, format_options=("eingebetteter_befund",)),
    SectionDef("Konsilbericht (eingebettet)", 0.25, format_options=("eingebetteter_befund",),
               requires=("Konsil",)),
    SectionDef("OP-Bericht (Auszug)", 0.30, format_options=("fließtext", "eingebetteter_befund")),
    # Rare / specialty-specific
    SectionDef("Funktionsdiagnostik", 0.25, format_options=("fließtext", "tabellarisch")),
    SectionDef("Sozialanamnese", 0.40),
    SectionDef("Pflegerische Übergabe", 0.15),
    SectionDef("Psychologischer Befund", 0.15),
]
# fmt: on

# Embedded sub-document format instructions for the LLM.
EMBEDDED_FORMAT_INSTRUCTIONS: dict[str, str] = {
    "fließtext": "",  # default, no extra instruction
    "stichpunkte": "Formatiere diesen Abschnitt als Stichpunktliste (- Punkt).",
    "tabellarisch": (
        "Formatiere diesen Abschnitt tabellarisch/strukturiert "
        "(z.B. Parameter: Wert [Einheit] oder Medikament – Dosis – Häufigkeit)."
    ),
    "befund_zitat": (
        "Zitiere in diesem Abschnitt einen Originalbefund wörtlich (z.B. "
        "radiologischer Befund mit eigenem Kopf: Datum, Untersucher, Befundtext). "
        "WICHTIG: Verwende für den Untersucher einen Katalog-Platzhalter "
        "(z.B. [[konsiliararzt ...]]) oder eine generische Bezeichnung "
        "('die Radiologin'). KEINE erfundenen Namen oder Befundnummern!"
    ),
    "eingebetteter_befund": (
        "Bette einen vollständigen Fremdbefund ein — mit eigenem Briefkopf "
        "(Klinik/Praxis, Arzt, Datum), Befundtext und Unterschrift. "
        "WICHTIG: Auch im eingebetteten Befund KEINE erfundenen Namen, Nummern "
        "oder Institutionsnamen! Für den Autor verwende einen Katalog-Platzhalter "
        "(z.B. [[konsiliararzt form=hon_last case=nom]], [[arzt_ober ...]]) oder "
        "eine generische Rolle ('Die Untersucherin'). Für die Institution verwende "
        "[[institution]] oder 'das externe Institut'. Befundnummern weglassen."
    ),
}


def sample_sections(
    rng: "random.Random",
    difficulty: str,
    style: str,
    length_hint: str,
    cfg: "Any | None" = None,
) -> list[tuple[str, str]]:
    """Sample a diverse set of sections with format hints.

    Returns list of (section_name, format_instruction) tuples.
    Difficulty/style/length influence how many sections and which formats.
    ``cfg`` (a ``SectionsConfig``) overrides the count ranges and format
    biases; when ``None`` the historical literals are used.
    """
    # Target section count by length and difficulty
    if cfg is not None:
        lo, hi = cfg.count_ranges.get(length_hint, {}).get(
            difficulty, cfg.default_count_range
        )
        telegramm_nonfliess_prob = cfg.telegramm_nonfliess_prob
        akademisch_fliess_prob = cfg.akademisch_fliess_prob
    else:
        _count_ranges = {
            ("kurz", "easy"): (3, 5),
            ("kurz", "medium"): (3, 5),
            ("kurz", "hard"): (4, 6),
            ("kurz", "adversarial"): (4, 6),
            ("mittel", "easy"): (5, 7),
            ("mittel", "medium"): (5, 8),
            ("mittel", "hard"): (6, 9),
            ("mittel", "adversarial"): (6, 10),
            ("ausführlich", "easy"): (6, 8),
            ("ausführlich", "medium"): (7, 10),
            ("ausführlich", "hard"): (8, 12),
            ("ausführlich", "adversarial"): (9, 14),
            ("sehr_lang", "easy"): (8, 12),
            ("sehr_lang", "medium"): (10, 14),
            ("sehr_lang", "hard"): (12, 18),
            ("sehr_lang", "adversarial"): (14, 20),
        }
        lo, hi = _count_ranges.get((length_hint, difficulty), (5, 8))
        telegramm_nonfliess_prob = 0.7
        akademisch_fliess_prob = 0.7
    target_n = rng.randint(lo, hi)

    # Always include core sections
    chosen: list[SectionDef] = [s for s in SECTION_POOL if s.core]
    remaining = [s for s in SECTION_POOL if not s.core]

    # Weighted sampling without replacement for non-core sections
    while len(chosen) < target_n and remaining:
        weights = [s.weight for s in remaining]
        idx = rng.choices(range(len(remaining)), weights=weights, k=1)[0]
        pick = remaining.pop(idx)
        # Check soft dependencies — include required sections if missing
        for req in pick.requires:
            if req not in [c.name for c in chosen]:
                # Find and add the required section
                for j, s in enumerate(remaining):
                    if s.name == req:
                        chosen.append(remaining.pop(j))
                        break
        chosen.append(pick)

    # Sort by a canonical ordering (roughly clinical timeline)
    _ORDER = {s.name: i for i, s in enumerate(SECTION_POOL)}
    chosen.sort(key=lambda s: _ORDER.get(s.name, 99))

    # Assign format to each section
    result: list[tuple[str, str]] = []
    for sec in chosen:
        if len(sec.format_options) == 1:
            fmt = sec.format_options[0]
        else:
            # Telegrammstil prefers stichpunkte/tabellarisch; Akademisch prefers fließtext
            if style == "Telegrammstil-Kompakt":
                # Prefer non-fließtext if available
                non_fliess = [f for f in sec.format_options if f != "fließtext"]
                fmt = (
                    rng.choice(non_fliess)
                    if non_fliess and rng.random() < telegramm_nonfliess_prob
                    else rng.choice(sec.format_options)
                )
            elif style == "Akademisch-Ausführlich":
                # Prefer fließtext
                fmt = (
                    "fließtext"
                    if rng.random() < akademisch_fliess_prob
                    else rng.choice(sec.format_options)
                )
            else:
                fmt = rng.choice(sec.format_options)
        result.append((sec.name, fmt))

    return result


PROFESSIONS = [
    # Manual / trade
    "Busfahrer",
    "Elektroinstallateur",
    "Maurer",
    "Bäckermeister",
    "KFZ-Mechatroniker",
    "Schornsteinfeger",
    "Landwirt",
    "Dachdecker",
    "Maler und Lackierer",
    "Tischler",
    "Schlosser",
    "Schweißer",
    "Straßenbauer",
    "Fliesenleger",
    "Zimmermann",
    "Industriemechaniker",
    "Werkzeugmacher",
    "Lagerarbeiter",
    "LKW-Fahrer",
    "Gärtner",
    "Fleischer",
    "Klempner",
    # Service / care
    "Grundschullehrerin",
    "Altenpflegerin",
    "Krankenschwester",
    "Friseurin",
    "Erzieherin",
    "Verkäuferin",
    "Arzthelferin",
    "Kellnerin",
    "Reinigungskraft",
    "Hauswirtschafterin",
    "Tagesmutter",
    # Office / white collar
    "Steuerberater",
    "Bibliothekarin",
    "Kaufmännischer Angestellter",
    "Sachbearbeiterin",
    "Verwaltungsfachangestellte",
    "Bankkaufmann",
    "Rechtsanwaltsfachangestellte",
    "Versicherungskaufmann",
    "Buchhalter",
    "Personalreferentin",
    # Academic / professional
    "Gymnasiallehrer",
    "Ingenieur",
    "Architektin",
    "Apotheker",
    "Informatiker",
    "Betriebswirt",
    "Sozialarbeiterin",
    "Chemielaborant",
    "Journalistin",
    "Diplom-Psychologin",
    # Self-employed / other
    "Taxifahrer",
    "Berufskraftfahrer",
    "Hausfrau",
    "Rentner",
    "Beamter im Ruhestand",
    "Selbständiger Handelsvertreter",
]

# Medical professions — sampled separately with difficulty-gated probability.
# When the patient is a physician, collegial address forms apply (hard cases).
MEDICAL_PROFESSIONS_M = [
    "Arzt",
    "Facharzt für Innere Medizin",
    "Allgemeinmediziner",
    "Zahnarzt",
    "Kinderarzt",
    "niedergelassener Internist",
    "Chirurg",
    "Orthopäde",
    "Radiologe",
    "Anästhesist",
    "Neurologe",
    "Urologe",
]
MEDICAL_PROFESSIONS_F = [
    "Ärztin",
    "Fachärztin für Chirurgie",
    "Zahnärztin",
    "Assistenzärztin",
    "Kinderärztin",
    "niedergelassene Internistin",
    "Chirurgin",
    "Orthopädin",
    "Radiologin",
    "Anästhesistin",
    "Neurologin",
    "Urologin",
]

# Age-appropriate occupations for patients outside typical working age (18-66).
# Gender-aware to avoid introducing new contradictions. Used to replace the
# general PROFESSIONS draw for children, adolescents, and retirees so that a
# 6-year-old is not documented as a taxi driver or a 90-year-old as an active
# roofer. The pre-school band is split from the school band so that a 12-year-old
# is not labelled "Vorschulkind".
KLEINKIND_OCCUPATIONS_M = ["Kindergartenkind", "Vorschulkind"]
KLEINKIND_OCCUPATIONS_F = ["Kindergartenkind", "Vorschulkind"]
SCHULKIND_OCCUPATIONS_M = ["Schüler", "Schulkind"]
SCHULKIND_OCCUPATIONS_F = ["Schülerin", "Schulkind"]
ADOLESCENT_OCCUPATIONS_M = ["Schüler", "Auszubildender", "Student"]
ADOLESCENT_OCCUPATIONS_F = ["Schülerin", "Auszubildende", "Studentin"]
RETIRED_OCCUPATIONS_M = ["Rentner", "Pensionär", "im Ruhestand"]
RETIRED_OCCUPATIONS_F = ["Rentnerin", "Pensionärin", "im Ruhestand"]

EMPLOYERS = [
    # Public
    "Stadtwerke Cottbus",
    "Universitätsklinikum Leipzig",
    "Grundschule am Sonnenhang",
    "Stadtverwaltung Erfurt",
    "Bezirksamt Berlin-Neukölln",
    "Finanzamt Düsseldorf-Mitte",
    "Polizeipräsidium München",
    "Landratsamt Esslingen",
    # Industry / manufacturing
    "VW-Werk Wolfsburg",
    "ThyssenKrupp Steel",
    "BASF SE Ludwigshafen",
    "Siemens AG München",
    "Bosch GmbH Stuttgart",
    "Bayer AG Leverkusen",
    "Continental AG Hannover",
    "Henkel AG Düsseldorf",
    "Airbus Operations GmbH Hamburg",
    "Daimler Truck AG Wörth",
    # Logistics / transport
    "Deutsche Bahn AG",
    "Deutsche Post DHL Group Bonn",
    "Logistik Nord AG",
    "Hermes Germany GmbH Hamburg",
    "DB Schenker AG Essen",
    # Trade / small business
    "Bäckerei Steinmann GmbH",
    "Metzgerei Braun oHG",
    "Autohaus Krämer GmbH",
    "Schreinerei Hoffmann",
    "Hotel Kaiserhof GmbH",
    "Restaurant Lindenhof",
    # Retail / services
    "EDEKA Süd",
    "Rossmann GmbH",
    "Aldi Süd KG",
    "dm-Drogeriemarkt GmbH",
    "Galeria Karstadt Kaufhof GmbH",
    "Commerzbank AG Frankfurt",
    "Allianz SE München",
    "Sparkasse Köln-Bonn",
    "R+V Versicherung Wiesbaden",
    "Caritas-Verband Freiburg",
]

INSURERS = [
    # GKV (public)
    "AOK Bayern",
    "AOK Niedersachsen",
    "AOK Nordost",
    "AOK Rheinland/Hamburg",
    "AOK Baden-Württemberg",
    "Techniker Krankenkasse",
    "Barmer",
    "DAK-Gesundheit",
    "IKK classic",
    "IKK gesund plus",
    "Knappschaft",
    "hkk Krankenkasse",
    "SBK Siemens-Betriebskrankenkasse",
    "HEK – Hanseatische Krankenkasse",
    "KKH Kaufmännische Krankenkasse",
    # PKV (private)
    "Allianz Private Krankenversicherung",
    "Debeka",
    "DKV Deutsche Krankenversicherung",
    "Signal Iduna",
    "AXA Krankenversicherung",
]


# ---------------------------------------------------------------------------
# Sampling helper functions
# ---------------------------------------------------------------------------


def sample_age(rng: random.Random, bins: list | None = None) -> int:
    """Draw an age from the hospital-weighted German age distribution."""
    bins = bins if bins is not None else AGE_BINS
    ranges = [b[0] for b in bins]
    weights = [b[1] for b in bins]
    lo, hi = rng.choices(ranges, weights=weights, k=1)[0]
    return rng.randint(lo, hi)


def sample_marital_status(
    age: int, rng: random.Random, table: dict | None = None
) -> str:
    """Pick a marital status appropriate for the given age."""
    table = table if table is not None else MARITAL_STATUS_BY_AGE
    thresholds = sorted(table.keys())
    chosen = thresholds[0]
    for t in thresholds:
        if t <= age:
            chosen = t
    dist = table[chosen]
    return rng.choices(list(dist.keys()), weights=list(dist.values()), k=1)[0]


def sample_titel(
    role: str, rng: random.Random, pool_map: dict | None = None
) -> str | None:
    """Pick a title appropriate for the clinician's role."""
    pool_map = pool_map if pool_map is not None else TITEL_BY_ROLE
    pool = pool_map.get(role, [("Dr. med.", 1.0)])
    titles = [t for t, _ in pool]
    weights = [w for _, w in pool]
    return rng.choices(titles, weights=weights, k=1)[0]


def sample_institution(rng: random.Random, city: str, dept: str) -> str:
    """Generate a diverse institution name for the current clinic."""
    tmpl = rng.choice(INSTITUTION_TEMPLATES)
    return tmpl.format(city=city, dept=dept)


def sample_praxis(
    rng: random.Random, arzt_name: str, city: str, named_prob: float = 0.55
) -> str:
    """Generate a referring physician's practice name."""
    # ~55% named after the physician, ~45% generic/location-based
    if rng.random() < named_prob:
        tmpl = rng.choice(PRAXIS_TEMPLATES_NAMED)
    else:
        tmpl = rng.choice(PRAXIS_TEMPLATES_GENERIC)
    return tmpl.format(arzt=arzt_name, city=city)


def sample_other_institution(rng: random.Random, exclude_city: str) -> str:
    """Generate a previous/external clinic name (different city)."""
    pool = [c for c in OTHER_CITIES if c != exclude_city]
    other_city = rng.choice(pool)
    clinic_templates = [
        "Universitätsklinikum {city}",
        "Klinikum {city}",
        "Städtisches Klinikum {city}",
        "Krankenhaus {city}-Süd",
        "St. Marien-Hospital {city}",
        "Kliniken {city}",
        "Helios Klinikum {city}",
        "Ev. Krankenhaus {city}",
    ]
    tmpl = rng.choice(clinic_templates)
    return tmpl.format(city=other_city)


def sample_rehab(rng: random.Random, exclude_city: str) -> str:
    """Generate a rehabilitation clinic name."""
    pool = [c for c in OTHER_CITIES if c != exclude_city]
    other_city = rng.choice(pool)
    rehab_templates = [
        "Rehaklinik {city}",
        "Reha-Zentrum {city}",
        "Klinik am Kurpark {city}",
        "Neurologisches Rehabilitationszentrum {city}",
        "Rehabilitationsklinik {city}",
        "Fachklinik Bad {city}",
        "MediClin Reha-Zentrum {city}",
    ]
    tmpl = rng.choice(rehab_templates)
    return tmpl.format(city=other_city)


# ---------------------------------------------------------------------------
# Nationality strings (origin code -> natural-language nationality mentions).
# Multiple variants per origin for diversity in how nationality surfaces in text.
NATIONALITY_BY_ORIGIN: dict[str, list[str]] = {
    "de": ["deutsch", "deutscher Staatsangehörigkeit"],
    "tr": ["türkisch", "türkischer Herkunft", "türkischstämmig"],
    "ar": [
        "syrisch",
        "irakisch",
        "libanesisch",
        "arabischer Herkunft",
        "syrischer Herkunft",
    ],
    "pl": ["polnisch", "polnischer Herkunft"],
    "ru": ["russisch", "russischsprachig", "kasachisch", "ukrainisch"],
    "ex_yu": ["serbisch", "kroatisch", "bosnisch", "kosovarisch"],
    "it": ["italienisch", "italienischer Herkunft"],
    "gr": ["griechisch", "griechischer Herkunft"],
    "vn": ["vietnamesisch", "vietnamesischer Herkunft"],
    "other_eu": ["rumänisch", "bulgarisch", "portugiesisch", "spanisch"],
}
