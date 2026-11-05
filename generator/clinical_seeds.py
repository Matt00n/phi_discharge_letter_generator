"""Clinical case assembly from ICD/OPS catalogues and specialty metadata.

Loads ICD-10-GM and OPS code files from the data/ directory and combines them
with specialty-specific clinical details (complaints, medications, department
names) to produce complete clinical case dicts for each document.
"""

import os
import random
import re
from typing import Any, Dict, List, Tuple

from generator.specialty_data import (
    CASE_FLAVOUR_WEIGHTS,
    COMPLICATION_CODES,
    GENERAL_MEDICATIONS,
    GENERAL_SECONDARY_DIAGNOSES,
    SPECIALTIES_METADATA,
    SPECIALTY_CONSULT_MAP,
)

_ICD_CHAPTER = re.compile(r"^([A-Z])(\d+)")
_OPS_CHAPTER = re.compile(r"^5-(\d{3})")

# Sex-specific ICD-10-GM codes that fall outside the genital/obstetric chapters
# handled by the range logic below. Matched by longest code prefix. Only codes
# whose *description* is intrinsically sex-bound are listed; e.g. only the
# postmenopausal osteoporosis subcodes (M80.0-/M81.0-), not the whole M80/M81
# block, which also covers sex-neutral (drug-induced, idiopathic) osteoporosis.
_ICD_SEX_PREFIX = {
    # female-specific
    "M80.0": "f",  # postmenopausale Osteoporose mit pathologischer Fraktur
    "M81.0": "f",  # postmenopausale Osteoporose
    "E28": "f",  # ovarielle Dysfunktion
    "F53": "f",  # psychische Störungen im Wochenbett (Puerperium)
    "R87": "f",  # abnorme Befunde, weibliche Genitalorgane
    "Q50": "f",  # angeborene Fehlbildungen: Ovar, Tube, Lig. latum
    "Q51": "f",  # angeborene Fehlbildungen: Uterus, Cervix
    "Q52": "f",  # angeborene Fehlbildungen: sonstige weibliche Genitalorgane
    # male-specific
    "R86": "m",  # abnorme Befunde, männliche Genitalorgane
    "Q53": "m",  # Nondescensus testis
    "Q54": "m",  # Hypospadie
    "Q55": "m",  # sonstige angeborene Fehlbildungen der männlichen Genitalorgane
}


def _icd_sex(code: str) -> str | None:
    """Sex applicability of an ICD-10 code per the WHO/ICD-10-GM sex edits.

    Returns 'f' (female-only), 'm' (male-only), or None (applies to both). Only
    the unambiguous genital/obstetric ranges are gated; ambiguous codes such as
    C50 (breast — male breast cancer exists) are deliberately left as None.
    """
    for prefix in sorted(_ICD_SEX_PREFIX, key=len, reverse=True):
        if code.startswith(prefix):
            return _ICD_SEX_PREFIX[prefix]
    m = _ICD_CHAPTER.match(code)
    if not m:
        return None
    ch, num = m.group(1), int(m.group(2))
    if ch == "O":  # pregnancy, childbirth, and the puerperium
        return "f"
    if ch == "C":
        if 51 <= num <= 58:  # female genital organs
            return "f"
        if 60 <= num <= 63:  # male genital organs
            return "m"
    if ch == "D":
        if num == 6 or 25 <= num <= 28:  # in-situ cervix / benign female genital
            return "f"
    if ch == "N":
        if 40 <= num <= 51:  # male genital organs (prostate, testis, penis, ...)
            return "m"
        if 70 <= num <= 98:  # female pelvic/genital disorders
            return "f"
    if ch == "Z":
        if 32 <= num <= 37:  # pregnancy-related encounters
            return "f"
    return None


def _ops_sex(code: str) -> str | None:
    """Sex applicability of an OPS procedure code (genital/obstetric surgery)."""
    m = _OPS_CHAPTER.match(code)
    if not m:
        return None
    n = int(m.group(1))
    if 600 <= n <= 649:  # operations on the male genital organs
        return "m"
    if 650 <= n <= 759:  # female genital organs + obstetric operations
        return "f"
    return None


def _sex_compatible(code: str, gender: str | None) -> bool:
    """True unless ``code`` is sex-specific and contradicts ``gender``."""
    if gender not in ("m", "f"):
        return True
    applicable = _icd_sex(code) if code[:1].isalpha() else _ops_sex(code)
    return applicable is None or applicable == gender


# Age-bound ICD-10-GM codes, matched by longest code prefix. Deliberately
# minimal: only codes whose *description* is biologically or definitionally
# bound to an age band (reproductive, post-menopausal, adult-degenerative) are
# gated, so that overall diagnosis diversity is preserved. Each value is the
# inclusive ``(min_age, max_age)`` in years for which the code is plausible.
_ICD_AGE_PREFIX: dict[str, tuple[int, int]] = {
    # post-menopausal — definitionally post-reproductive
    "M80.0": (45, 120),  # postmenopausale Osteoporose mit pathologischer Fraktur
    "M81.0": (45, 120),  # postmenopausale Osteoporose
    # other primary osteoporosis (idiopathic/unspecified) — an adult condition;
    # a pathological osteoporotic fracture in a young adult is implausible. The
    # more specific postmenopausal subcodes above win via longest-prefix match.
    "M80": (40, 120),  # Osteoporose mit pathologischer Fraktur
    "M81": (40, 120),  # Osteoporose ohne pathologische Fraktur
    # adult-degenerative neurological — implausible in young children
    "G20": (30, 120),  # Parkinson-Krankheit
    "G30": (40, 120),  # Alzheimer-Krankheit
    "F00": (40, 120),  # Demenz bei Alzheimer-Krankheit
    "F01": (40, 120),  # vaskuläre Demenz
    "F03": (40, 120),  # nicht näher bezeichnete Demenz
    # acquired chronic ischaemic heart disease
    "I25": (35, 120),  # chronische ischämische Herzkrankheit
}


# Minimum patient age for inherently age-bound specialties. Geriatric medicine
# treats older adults; there is no paediatric specialty in the catalogue, so no
# upper-bound gating is needed.
_SPECIALTY_MIN_AGE: dict[str, int] = {"Geriatrie": 65}


def _icd_age(code: str) -> tuple[int, int] | None:
    """Plausible ``(min_age, max_age)`` for an ICD-10 code, or None if unbound."""
    for prefix in sorted(_ICD_AGE_PREFIX, key=len, reverse=True):
        if code.startswith(prefix):
            return _ICD_AGE_PREFIX[prefix]
    m = _ICD_CHAPTER.match(code)
    if not m:
        return None
    ch, num = m.group(1), int(m.group(2))
    if ch == "O":  # pregnancy, childbirth, and the puerperium
        return (12, 55)
    if ch == "Z" and 32 <= num <= 37:  # pregnancy-related encounters
        return (12, 55)
    return None


def _age_compatible(code: str, age: int | None) -> bool:
    """True unless ``code`` is age-bound and ``age`` falls outside its band.

    Only ICD diagnosis codes are gated; OPS procedure codes are left ungated
    (their obstetric subset is already excluded via the age-gated diagnoses).
    """
    if age is None or not code[:1].isalpha():
        return True
    band = _icd_age(code)
    if band is None:
        return True
    lo, hi = band
    return lo <= age <= hi


# Drug names whose indication is essentially sex-bound. Matched case-insensitively
# as a substring of the medication string. Deliberately conservative: only drugs
# that are almost exclusively prescribed to one sex in practice. Agents used in
# both sexes (e.g. Tamoxifen — male breast cancer exists, consistent with C50
# being left ungated) are intentionally omitted.
_MED_SEX = {
    # male: benign prostatic hyperplasia / androgens
    "finasterid": "m",
    "dutasterid": "m",
    "tamsulosin": "m",
    "alfuzosin": "m",
    "silodosin": "m",
    "testosteron": "m",
    # female: hormone replacement / gynaecological
    "estradiol": "f",
    "estriol": "f",
    "östrogen": "f",
    "oestrogen": "f",
    "clomifen": "f",
    "tibolon": "f",
    "raloxifen": "f",
    "norethisteron": "f",
}


def _med_compatible(med: str, gender: str | None) -> bool:
    """True unless ``med`` names a sex-bound drug that contradicts ``gender``."""
    if gender not in ("m", "f"):
        return True
    low = med.lower()
    return all(not (name in low and sex != gender) for name, sex in _MED_SEX.items())


class ClinicalSeedsManager:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.icd_codes: List[Tuple[str, str]] = []
        self.ops_codes: List[Tuple[str, str]] = []
        self._load_catalogues()

    def _load_catalogues(self):
        icd_path = os.path.join(self.data_dir, "icd10_codes.txt")
        ops_path = os.path.join(self.data_dir, "ops_codes.txt")

        # Parse ICD codes
        if os.path.exists(icd_path):
            with open(icd_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(" ", 1)
                    if len(parts) == 2:
                        code, desc = parts
                        code = code.strip()
                        desc = desc.strip()
                        # Skip category headers ending with ".-" and reserved
                        # placeholder keys ("Nicht belegte Schlüsselnummer"),
                        # which are administrative slots, not real diagnoses.
                        if code.endswith(".-"):
                            continue
                        if desc.startswith("Nicht belegte Schlüsselnummer"):
                            continue
                        self.icd_codes.append((code, desc))
        else:
            print(f"Warning: ICD-10 file not found at {icd_path}. Using fallback.")
            self.icd_codes = GENERAL_SECONDARY_DIAGNOSES[:]

        # Parse OPS codes
        if os.path.exists(ops_path):
            with open(ops_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(" ", 1)
                    if len(parts) == 2:
                        code, desc = parts
                        code = code.strip()
                        desc = desc.strip()
                        # Skip major chapter headers (e.g. codes that are just short numbers or end in range)
                        if len(code) > 2:
                            self.ops_codes.append((code, desc))
        else:
            print(f"Warning: OPS file not found at {ops_path}. Using fallback.")
            self.ops_codes = [
                ("1-275.1", "Koronarangiographie und Druckmessung im linken Ventrikel"),
                ("3-200.0", "Native Magnetresonanztomographie des Gehirns"),
                ("5-820.01", "Implantation einer Gelenkendoprothese am Hüftgelenk"),
                ("1-440.a", "1 bis 5 Biopsien am oberen Verdauungstrakt"),
                ("8-854.10", "Hämodialyse, intermittierend"),
            ]

    def get_specialty_by_icd(self, icd_code: str) -> str:
        """Find matching specialty key based on ICD prefix."""
        for spec, meta in SPECIALTIES_METADATA.items():
            for prefix in meta["icd_prefixes"]:
                if icd_code.startswith(prefix):
                    # Edge check: Oncology has specific overlap codes
                    if spec == "Onkologie" and icd_code.startswith("C"):
                        return "Onkologie"
                    if spec == "Kardiologie" and icd_code.startswith("I"):
                        return "Kardiologie"
                    return spec
        return random.choice(list(SPECIALTIES_METADATA.keys()))

    def sample_clinical_case(
        self,
        rng: random.Random | None = None,
        patient_gender: str | None = None,
        patient_age: int | None = None,
    ) -> Dict[str, Any]:
        """Samples a complete clinical case consisting of specialty, diagnoses, procedures, and medication.

        Pass ``rng`` (a seeded ``random.Random``) for reproducible sampling; if
        omitted, the global ``random`` module is used.
        Pass ``patient_gender`` ('m' or 'f') to weight specialty selection towards
        that gender's typical patient population.
        Pass ``patient_age`` (years) to exclude diagnoses that are biologically
        implausible for that age (e.g. obstetrics in a 64-year-old).
        """
        r = rng or random
        # 1. Select a specialty stochastically, optionally weighted by patient gender
        specialties = list(SPECIALTIES_METADATA.keys())
        # Age-gate inherently age-bound specialties (Geriatrie is for the elderly;
        # a young patient on a geriatric ward is demographically incoherent).
        if patient_age is not None:
            specialties = [
                s for s in specialties if patient_age >= _SPECIALTY_MIN_AGE.get(s, 0)
            ]
        if patient_gender is not None:
            weights = [
                SPECIALTIES_METADATA[s].get(
                    "gender_distribution", {"m": 0.5, "f": 0.5}
                )[patient_gender]
                for s in specialties
            ]
            specialty = r.choices(specialties, weights=weights, k=1)[0]
        else:
            specialty = r.choice(specialties)
        meta = SPECIALTIES_METADATA[specialty]

        # 1b. Case complexity tier (drives number of comorbidities, procedures, meds)
        complexity = r.choices(
            ["simple", "moderate", "complex"],
            weights=[0.25, 0.50, 0.25],
            k=1,
        )[0]

        # 2. Sample Primary Diagnosis matching specialty (sex-compatible only)
        primary_candidates = []
        for code, desc in self.icd_codes:
            for prefix in meta["icd_prefixes"]:
                if (
                    code.startswith(prefix)
                    and _sex_compatible(code, patient_gender)
                    and _age_compatible(code, patient_age)
                ):
                    primary_candidates.append((code, desc))
                    break

        if primary_candidates:
            primary_code, primary_desc = r.choice(primary_candidates)
        else:
            # Fallback if no matching code parsed
            primary_code, primary_desc = (
                "I21.9",
                "Akuter Myokardinfarkt, nicht näher bezeichnet",
            )
            specialty = "Kardiologie"
            meta = SPECIALTIES_METADATA[specialty]

        # 3. Sample Secondary Diagnoses (complexity-driven, three-tier)
        # Tier A (40 %): GENERAL_SECONDARY_DIAGNOSES — common chronic comorbidities
        # Tier B (40 %): adjacent ICD chapters for this specialty — clinically plausible
        # Tier C (20 %): full catalogue — wild card for genuine diversity
        adj_chapters = set(meta.get("secondary_chapter_adjacency", []))
        adj_icd_pool = (
            [pair for pair in self.icd_codes if pair[0][0] in adj_chapters]
            if adj_chapters
            else self.icd_codes
        )
        num_secs_range = {"simple": (1, 2), "moderate": (2, 4), "complex": (4, 12)}
        num_secs = r.randint(*num_secs_range[complexity])
        secondary_diagnoses: List[Tuple[str, str]] = []
        drawn = 0
        attempts = 0
        while drawn < num_secs and attempts < 40:
            attempts += 1
            roll = r.random()
            if roll < 0.40:
                sec_code, sec_desc = r.choice(GENERAL_SECONDARY_DIAGNOSES)
            elif roll < 0.80:
                pool = adj_icd_pool if adj_icd_pool else self.icd_codes
                sec_code, sec_desc = r.choice(pool)
            else:
                sec_code, sec_desc = r.choice(self.icd_codes)

            # Avoid duplicates, primary overlap, and sex-incompatible codes
            if (
                sec_code != primary_code
                and sec_code not in [c for c, _ in secondary_diagnoses]
                and _sex_compatible(sec_code, patient_gender)
                and _age_compatible(sec_code, patient_age)
            ):
                secondary_diagnoses.append((sec_code, sec_desc))
                drawn += 1

        # 4. Sample Procedures (complexity-driven, sex-compatible only)
        ops_candidates = []
        for code, desc in self.ops_codes:
            for prefix in meta["ops_prefixes"]:
                if code.startswith(prefix) and _sex_compatible(code, patient_gender):
                    ops_candidates.append((code, desc))
                    break

        num_procs_range = {"simple": (1, 1), "moderate": (1, 2), "complex": (2, 3)}
        target_procs = r.randint(*num_procs_range[complexity])
        procedures: List[Tuple[str, str]] = []
        if ops_candidates:
            for _ in range(min(target_procs, len(ops_candidates))):
                cand = r.choice(ops_candidates)
                if cand[0] not in [c for c, _ in procedures]:
                    procedures.append(cand)
        if not procedures:
            procedures.append(
                ("1-275.1", "Koronarangiographie und Druckmessung im linken Ventrikel")
            )

        # 5. Assemble Medication Regimen (complexity-driven)
        num_spec_meds = {"simple": (1, 3), "moderate": (3, 5), "complex": (3, 8)}
        num_gen_meds = {"simple": (1, 2), "moderate": (2, 4), "complex": (2, 7)}
        medication_list = []
        # Draw from specialty specific meds (sex-incompatible drugs filtered out)
        spec_pool = [
            m for m in meta["medications"] if _med_compatible(m, patient_gender)
        ]
        spec_meds = r.sample(
            spec_pool,
            min(len(spec_pool), r.randint(*num_spec_meds[complexity])),
        )
        medication_list.extend(spec_meds)

        # Draw from general meds
        gen_pool = [
            m for m in GENERAL_MEDICATIONS if _med_compatible(m, patient_gender)
        ]
        gen_meds = r.sample(
            gen_pool,
            min(len(gen_pool), r.randint(*num_gen_meds[complexity])),
        )
        medication_list.extend(gen_meds)

        # 6. Select main complaint
        complaint = r.choice(meta["complaints"])

        # 7. Sample clinical course hint (gives LLM narrative variation)
        course_pool = [
            "komplikationsloser stationärer Verlauf",
            "zunächst verzögerter Heilungsverlauf, im weiteren Verlauf Besserung",
            "passagere Verschlechterung mit Verlegung auf die Intermediate Care Station",
            "rasches Ansprechen auf die eingeleitete Therapie",
            "prolongierter Verlauf mit Notwendigkeit einer Therapieeskalation",
            "komplikationsloser postoperativer Verlauf mit zügiger Mobilisation",
            "kurzzeitige intensivmedizinische Überwachung bei hämodynamischer Instabilität",
            "Auftreten einer nosokomialen Infektion im Verlauf",
            "gutes Ansprechen auf die medikamentöse Neueinstellung",
            "stabile Befunde unter konservativer Therapie",
            "Besserung der Symptomatik nach interdisziplinärer Therapieanpassung",
            "regelhafter Verlauf ohne Komplikationen, Entlassung in stabilem Allgemeinzustand",
        ]
        course = r.choice(course_pool)

        # 8. Sample discharge disposition
        disposition_pool = [
            "Entlassung nach Hause in stabilem Zustand",
            "Entlassung in die ambulante Weiterbehandlung",
            "Verlegung in die Anschlussheilbehandlung",
            "Entlassung in die hausärztliche Weiterbetreuung",
            "Verlegung in eine weiterbehandelnde Einrichtung",
            "Entlassung auf eigenen Wunsch gegen ärztlichen Rat",
            "Entlassung mit ambulanter Pflegeversorgung",
        ]
        disposition = r.choice(disposition_pool)

        # 9. Case flavour — optionally augment secondaries / procedures with
        #    clinically interesting structure.
        flavour_keys = list(CASE_FLAVOUR_WEIGHTS.keys())
        flavour_weights = [CASE_FLAVOUR_WEIGHTS[k] for k in flavour_keys]
        case_flavour = r.choices(flavour_keys, weights=flavour_weights, k=1)[0]
        scenario_hint = ""

        existing_sec_codes = {c for c, _ in secondary_diagnoses}
        existing_proc_codes = {c for c, _ in procedures}

        if case_flavour == "cross_specialty":
            # 75 % bias towards clinically adjacent specialties; 25 % full breadth
            if r.random() < 0.75 and specialty in SPECIALTY_CONSULT_MAP:
                cross_spec = r.choice(SPECIALTY_CONSULT_MAP[specialty])
            else:
                other_specialties = [s for s in specialties if s != specialty]
                cross_spec = r.choice(other_specialties)
            cross_meta = SPECIALTIES_METADATA[cross_spec]
            # Add one OPS from the cross-specialty
            cross_ops = [
                (c, d)
                for c, d in self.ops_codes
                for p in cross_meta["ops_prefixes"]
                if c.startswith(p) and _sex_compatible(c, patient_gender)
            ]
            if cross_ops:
                cp = r.choice(cross_ops)
                if cp[0] not in existing_proc_codes:
                    procedures.append(cp)
                    existing_proc_codes.add(cp[0])
            # Add one secondary ICD from the cross-specialty
            cross_icd = [
                (c, d)
                for c, d in self.icd_codes
                for p in cross_meta["icd_prefixes"]
                if c.startswith(p)
                and _sex_compatible(c, patient_gender)
                and _age_compatible(c, patient_age)
            ]
            if cross_icd:
                ci = r.choice(cross_icd)
                if ci[0] != primary_code and ci[0] not in existing_sec_codes:
                    secondary_diagnoses.append(ci)
                    existing_sec_codes.add(ci[0])
            dep = cross_meta["department_name"].replace("Klinik für ", "")
            scenario_hint = f"Im Verlauf war ein interdisziplinäres Konsil mit der {dep} erforderlich"

        elif case_flavour == "complication_chain":
            comp = r.choice(COMPLICATION_CODES)
            if (
                comp[0] != primary_code
                and comp[0] not in existing_sec_codes
                and _sex_compatible(comp[0], patient_gender)
                and _age_compatible(comp[0], patient_age)
            ):
                secondary_diagnoses.insert(0, comp)
            scenario_hint = (
                "Der stationäre Aufenthalt ist wesentlich durch eine Komplikation "
                "oder Folgeerkrankung einer früheren Behandlung geprägt"
            )

        return {
            "fachabteilung": specialty,
            "klinik_name": f"Klinikum und {meta['department_name']}",
            "hauptdiagnose": f"{primary_desc} ({primary_code})",
            "hauptdiagnose_code": primary_code,
            "hauptdiagnose_text": primary_desc,
            "nebendiagnosen": [
                f"{desc} ({code})" for code, desc in secondary_diagnoses
            ],
            "nebendiagnosen_struct": [
                (code, desc) for code, desc in secondary_diagnoses
            ],
            "prozeduren": [f"{desc} ({code})" for code, desc in procedures],
            "prozeduren_struct": [(code, desc) for code, desc in procedures],
            "medikation_entlassung": medication_list,
            "anlass": complaint,
            "complexity": complexity,
            "verlauf": course,
            "entlassung": disposition,
            "case_flavour": case_flavour,
            "scenario_hint": scenario_hint,
        }
