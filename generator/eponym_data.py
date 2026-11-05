"""Eponym and look-alike negatives (over-redaction traps).

German clinical language is saturated with person-derived terms that *look* like
PII but must NOT be redacted. Including them as explicit negatives lets us
measure over-redaction - a contribution under-explored by existing German de-id
datasets. Placement is controlled by the pipeline, so coverage is guaranteed.
"""

from __future__ import annotations

import random

# Eponymous diseases, syndromes, signs, scales, classifications, maneuvers.
EPONYMS: list[str] = [
    # Diseases / syndromes
    "Morbus Crohn",
    "Morbus Parkinson",
    "Morbus Bechterew",
    "Morbus Basedow",
    "Cushing-Syndrom",
    "Guillain-Barré-Syndrom",
    "Wegener-Granulomatose",
    "Alzheimer-Demenz",
    "Hashimoto-Thyreoiditis",
    "Ramsay-Hunt-Syndrom",
    "Horner-Syndrom",
    "Sjögren-Syndrom",
    "Raynaud-Syndrom",
    "Marfan-Syndrom",
    "Addison-Krise",
    "Hodgkin-Lymphom",
    "Non-Hodgkin-Lymphom",
    "Kaposi-Sarkom",
    "Bowen-Karzinom",
    "Burkitt-Lymphom",
    "Paget-Krankheit",
    "Behçet-Syndrom",
    "Reiter-Syndrom",
    "Felty-Syndrom",
    "Korsakow-Syndrom",
    "Wernicke-Enzephalopathie",
    "Turner-Syndrom",
    "Klinefelter-Syndrom",
    "Kawasaki-Syndrom",
    "Goodpasture-Syndrom",
    # Signs / tests / maneuvers
    "Bell-Parese",
    "Murphy-Zeichen",
    "Babinski-Reflex",
    "Homans-Zeichen",
    "Trendelenburg-Test",
    "Tinel-Zeichen",
    "Lasègue-Zeichen",
    "Lachman-Test",
    "Romberg-Stehversuch",
    "Phalen-Test",
    "Brudzinski-Zeichen",
    "Kernig-Zeichen",
    "McBurney-Punkt",
    "Lanz-Punkt",
    "Rovsing-Zeichen",
    "Courvoisier-Zeichen",
    # Scales / scores / classifications
    "Glasgow Coma Scale",
    "Killip-Klassifikation",
    "Child-Pugh-Score",
    "Rankin-Skala",
    "Barthel-Index",
    "Apgar-Score",
    "Norton-Skala",
    "Braden-Skala",
    "NYHA-Klassifikation",
    "Karnofsky-Index",
    # Eponymous anatomy / structures
    "Charcot-Trias",
    "Virchow-Trias",
    "His-Bündel",
    "Langerhans-Inseln",
    "Douglas-Raum",
]

# Drugs / devices / units / anatomy whose names contain person- or place-like
# tokens, plus institution-shaped phrases that are generic (not a named clinic).
LOOKALIKES: list[str] = [
    # Drugs that look like surnames
    "Marcumar",
    "Tamoxifen",
    "Rituximab",
    "Adalimumab",
    "Infliximab",
    "Pembrolizumab",
    "Nivolumab",
    # Person-named devices / procedures
    "Sengstaken-Sonde",
    "Swan-Ganz-Katheter",
    "Esmarch-Handgriff",
    "Fowler-Lagerung",
    "Hartmann-Operation",
    "Billroth-II-Resektion",
    "Foley-Katheter",
    "Port-a-Cath",
    "Kirschner-Draht",
    "Bülau-Drainage",
    "Weber-Fraktur",
    "Colles-Fraktur",
    "Jackson-Pratt-Drainage",
    "Redon-Drainage",
    # Place-derived medical terms
    "Mainz-Pouch",
    "Breslau-Handgriff",
    "Marburg-Virus",
    "Ebola-Virus",
    "Lyme-Borreliose",
    "Coxsackie-Virus",
    # Generic institution-shaped phrases (not PII)
    "Kreiskrankenhaus",
    "Ärztlicher Bereitschaftsdienst",
    "Kassenärztliche Vereinigung",
    "Gesundheitsamt",
    "Berufsgenossenschaftliche Unfallklinik",
]


def sample_eponyms(rng: random.Random, n: int) -> list[str]:
    n = max(0, min(n, len(EPONYMS)))
    return rng.sample(EPONYMS, n)


def sample_lookalikes(rng: random.Random, n: int) -> list[str]:
    n = max(0, min(n, len(LOOKALIKES)))
    return rng.sample(LOOKALIKES, n)
