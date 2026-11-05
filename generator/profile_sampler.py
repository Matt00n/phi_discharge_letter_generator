"""Stage 1: grounded profile sampler.

Produces, per document, a :class:`~generator.render.Catalog` of real entity
values plus a :class:`GenerationSpec` that controls style, structure, difficulty
tier and which hard cases are injected. Because placement and difficulty are
decided here (not left to the LLM), hard cases are *guaranteed*, not hoped for.

No LLM is used in this stage - it is fully deterministic given a seed, which
makes the whole dataset reproducible.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from typing import Any

from generator.atom_samplers import (
    TRICKY_LOCATIONS,
    address,
    bsnr,
    case_number,
    date_variants,
    email,
    fax,
    ik_nummer,
    kv_number,
    lanr,
    mobile,
    patient_id,
    phone,
    sample_colliding_nonpii_id,
    sample_nonpii_ids,
    url,
)
from generator.clinical_seeds import ClinicalSeedsManager
from generator.config_schema import PipelineConfig
from generator.eponym_data import sample_eponyms, sample_lookalikes
from generator.labels import Label
from generator.morphology import Person
from generator.name_data import (
    EPONYM_SURNAMES,
    GIVEN_NAMES_F,
    GIVEN_NAMES_M,
    WORD_SURNAME_CONTEXTS,
    WORD_SURNAMES,
    SampledName,
    _weighted_origin,
    sample_eponym_surname,
    sample_name,
    sample_surreal_name,
    sample_word_surname,
)
from generator.profile_data import (
    ADOLESCENT_OCCUPATIONS_F,
    ADOLESCENT_OCCUPATIONS_M,
    EMPLOYERS,
    INSURERS,
    KLEINKIND_OCCUPATIONS_F,
    KLEINKIND_OCCUPATIONS_M,
    LENGTH_TARGETS,
    MEDICAL_PROFESSIONS_F,
    MEDICAL_PROFESSIONS_M,
    NATIONALITY_BY_ORIGIN,
    PROFESSIONS,
    RETIRED_OCCUPATIONS_F,
    RETIRED_OCCUPATIONS_M,
    SCHULKIND_OCCUPATIONS_F,
    SCHULKIND_OCCUPATIONS_M,
    STYLE_DESCRIPTIONS,
    STYLES,
    sample_age,
    sample_institution,
    sample_marital_status,
    sample_other_institution,
    sample_praxis,
    sample_rehab,
    sample_sections,
    sample_titel,
)
from generator.render import AtomicEntity, Catalog, Negative


def _relative_terms_for(age: int, gender: str) -> list[str]:
    """Age-plausible relation terms for a relative of the given gender.

    Prevents demographically impossible pairings such as a 12-year-old with a
    "Tochter" or a 90-year-old whose "Vater" is a living contact person. Siblings
    are plausible at any age; spouses/partners require an adult patient; a
    named (adult) child implies the patient is middle-aged or older; a living
    parent is implausible once the patient is elderly.
    """
    m = gender == "m"
    sibling = "Bruder" if m else "Schwester"
    spouse = "Ehemann" if m else "Ehefrau"
    partner = "Lebenspartner" if m else "Lebenspartnerin"
    child = "Sohn" if m else "Tochter"
    parent = "Vater" if m else "Mutter"
    terms = [sibling]
    if age >= 18:
        terms += [spouse, partner]
    if age >= 45:
        terms.append(child)
    if age <= 75:
        terms.append(parent)
    return terms


@dataclass
class GenerationSpec:
    doc_id: str
    seed: int
    style: str
    difficulty: str  # "easy" | "medium" | "hard" | "adversarial"
    sections: list[str]
    length_hint: str
    clinical_case: dict[str, Any]
    entity_brief: list[dict[str, Any]]  # human/LLM-readable description of each ref
    section_formats: list[tuple[str, str]] = field(
        default_factory=list
    )  # (name, format_hint)
    structure_mode: str = "structured"  # "structured" | "mixed" | "unstructured"
    placement_hints: list[str] = field(
        default_factory=list
    )  # non-standard PII placement
    difficulty_tags: list[str] = field(default_factory=list)
    # Adversarial sub-challenges
    adversarial_mode: str = (
        ""  # "eponym_collision" | "word_collision" | "deep_needle" | "typo_text" | ""
    )
    word_surname_context: list[str] = field(
        default_factory=list
    )  # words that appear naturally & match surname
    typo_density: float = 0.0  # 0.0 = no typos, 0.01-0.05 = varying density
    quasi: dict[str, str] = field(default_factory=dict)
    patient_surname: str = ""
    patient_origin: str = ""
    patient_age: int = 0
    include_age_mention: bool = False
    marital_status: str = ""
    collegial_address: bool = False  # patient is a physician → "Kollege/Kollegin"
    pii_density: float = 0.5  # [0,1] controls optional PII field inclusion
    # How ICD-10 / OPS codes are presented in the clinical text:
    #   "inline"       — code in parentheses after every diagnosis/procedure
    #   "primary_only" — only the main diagnosis is coded; rest as plain text
    #   "none"         — no codes at all (narrative / older-letter register)
    code_style: str = "inline"
    # Opening-sentence register, to break the "Wir berichten über..." monotony.
    opening_style: str = "bericht"
    # Per-person mention plan: ref -> list of required forms (PersonForm values)
    mention_plan: dict[str, list[str]] = field(default_factory=dict)
    # Document framing elements (letterhead, greeting, patient block, closing)
    framing: dict[str, bool] = field(default_factory=dict)
    # Whether the document includes medication lists
    has_medications: bool = True
    # Length of the inpatient stay in days (0 = same-day/ambulant); passed to the
    # skeleton prompt so narrative day-references stay within the actual stay.
    stay_days: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Complete JSON-native serialization of the sampled spec.

        Unlike :meth:`to_metadata` (a curated summary), this exports *every*
        field of the spec, so the full pre-generation sampling decision is
        preserved in the dataset. Using ``asdict`` keeps it exhaustive if new
        fields are added. The spec holds only JSON-native values (no enums), so
        the result serializes directly; ``section_formats`` tuples become lists.
        """
        return asdict(self)

    def to_metadata(self) -> dict[str, Any]:
        meta = {
            "doc_id": self.doc_id,
            "seed": self.seed,
            "style": self.style,
            "difficulty": self.difficulty,
            "difficulty_tags": self.difficulty_tags,
            "sections": self.sections,
            "fachabteilung": self.clinical_case.get("fachabteilung"),
            "patient_surname": self.patient_surname,
            "patient_origin": self.patient_origin,
            "patient_age": self.patient_age,
            "marital_status": self.marital_status,
            "pii_density": round(self.pii_density, 3),
        }
        if self.adversarial_mode:
            meta["adversarial_mode"] = self.adversarial_mode
            meta["typo_density"] = round(self.typo_density, 4)
        meta["code_style"] = self.code_style
        meta["opening_style"] = self.opening_style
        meta["framing"] = self.framing
        meta["has_medications"] = self.has_medications
        return meta


def _resolve_reference_date(reference_date: str) -> date:
    """Resolve the config reference-date string to a ``date``.

    ``"today"`` falls back to the run date (the historical behaviour); any other
    value is parsed as an ISO ``YYYY-MM-DD`` calendar anchor.
    """
    if reference_date == "today":
        return date.today()
    return date.fromisoformat(reference_date)


class ProfileSampler:
    def __init__(
        self,
        seeds_manager: ClinicalSeedsManager,
        config: PipelineConfig | None = None,
    ):
        self.seeds = seeds_manager
        self.config = config or PipelineConfig()

    def sample(
        self,
        doc_id: str,
        seed: int,
        *,
        style: str | None = None,
        difficulty: str | None = None,
        length: str | None = None,
    ) -> tuple[Catalog, GenerationSpec]:
        rng = random.Random(seed)
        cfg = self.config.sampling
        style = style or rng.choice(STYLES)
        difficulty = (
            difficulty
            or rng.choices(
                ["easy", "medium", "hard", "adversarial"],
                weights=cfg.difficulty.distribution,
                k=1,
            )[0]
        )
        prof = cfg.difficulty.profiles[difficulty]
        tags: list[str] = []

        # --- PII density ---------------------------------------------------
        # Controls how many optional PII fields appear. Style/length bias the
        # mean but don't determine it — short+dense and long+sparse both occur.
        # Beta distribution gives natural [0,1] with tunable shape.
        length_hint = (
            length
            or rng.choices(
                ["kurz", "mittel", "ausführlich", "sehr_lang"],
                weights=cfg.length_weights[difficulty],
                k=1,
            )[0]
        )
        alpha, beta = cfg.pii_density.beta.get(style, {}).get(
            length_hint, cfg.pii_density.default_beta
        )
        pii_density = rng.betavariate(alpha, beta)  # float in [0, 1]
        _density_floor = cfg.pii_density.include_floor
        _density_scale = cfg.pii_density.include_scale

        def _include(base_prob: float) -> bool:
            """Gate a field: base_prob scaled by pii_density."""
            return rng.random() < base_prob * (
                _density_floor + _density_scale * pii_density
            )

        _addr_kwargs = dict(
            tricky_prob=cfg.address.tricky_prob_hard,
            house_number_range=cfg.address.house_number_range,
            house_suffix_prob=cfg.address.house_suffix_prob,
        )

        catalog = Catalog()
        brief: list[dict[str, Any]] = []

        # --- Adversarial sub-challenge selection ---------------------------
        adversarial_mode = ""
        word_surname_context: list[str] = []
        typo_density = 0.0

        if difficulty == "adversarial":
            adversarial_mode = rng.choices(
                ["eponym_collision", "word_collision", "deep_needle", "typo_text"],
                weights=cfg.adversarial.mode_weights,
                k=1,
            )[0]
            tags.append(f"adversarial_{adversarial_mode}")

            # Typo density: always some for typo_text, sometimes for others
            if adversarial_mode == "typo_text":
                typo_density = rng.uniform(*cfg.adversarial.typo_text_density_range)
            elif rng.random() < cfg.adversarial.other_typo_prob:
                typo_density = rng.uniform(*cfg.adversarial.other_typo_density_range)

        # --- Patient -------------------------------------------------------
        if adversarial_mode == "eponym_collision":
            # Force patient surname to be an eponym name
            pat_gender = rng.choice(["m", "f"])
            given_pool = GIVEN_NAMES_M if pat_gender == "m" else GIVEN_NAMES_F
            given_origin = _weighted_origin(rng, cfg.names.origin_weights)
            pat = SampledName(
                vorname=rng.choice(given_pool[given_origin]),
                nachname=sample_eponym_surname(rng),
                gender=pat_gender,
                origin=given_origin,
            )
        elif adversarial_mode == "word_collision":
            # Force patient surname to be a common word
            pat_gender = rng.choice(["m", "f"])
            given_pool = GIVEN_NAMES_M if pat_gender == "m" else GIVEN_NAMES_F
            given_origin = _weighted_origin(rng, cfg.names.origin_weights)
            word_surname = sample_word_surname(rng)
            pat = SampledName(
                vorname=rng.choice(given_pool[given_origin]),
                nachname=word_surname,
                gender=pat_gender,
                origin=given_origin,
            )
            # Get context words that naturally use this word
            word_surname_context = WORD_SURNAME_CONTEXTS.get(word_surname, [])
        elif adversarial_mode == "deep_needle":
            # Use a surreal/very-short name for the needle
            pat = sample_surreal_name(rng)
        else:
            pat = sample_name(rng, difficulty=difficulty, cfg=cfg.names)
        anrede = "Herr" if pat.gender == "m" else "Frau"
        catalog.persons["patient"] = Person(
            entity_id="patient",
            label=Label.PATIENT_NAME,
            vorname=pat.vorname,
            nachname=pat.nachname,
            gender=pat.gender,
            anrede=anrede,
        )
        brief.append(
            {
                "ref": "patient",
                "type": "person",
                "role": "Patient:in",
                "gender": pat.gender,
                "anrede": anrede,
            }
        )

        # --- Marital status & maiden name (Geburtsname) --------------------
        age = sample_age(rng, bins=cfg.age.bins)
        marital = sample_marital_status(age, rng, table=cfg.age.marital_by_age)
        # Women who are/were married often have a Geburtsname (maiden name).
        # Easy docs skip this: a maiden name adds an extra identity to track.
        has_maiden = (
            difficulty != "easy"
            and pat.gender == "f"
            and marital in ("verheiratet", "geschieden", "verwitwet")
            # not all take the spouse's name
            and rng.random() < cfg.persons.maiden_name_prob
        )
        if has_maiden:
            maiden = sample_name(rng, gender="f", difficulty=difficulty, cfg=cfg.names)
            catalog.persons["patient_geburtsname"] = Person(
                entity_id="patient_geburtsname",
                label=Label.PATIENT_NAME,
                vorname=pat.vorname,
                nachname=maiden.nachname,
                gender=pat.gender,
                anrede=anrede,
            )
            brief.append(
                {
                    "ref": "patient_geburtsname",
                    "type": "person",
                    "role": "Geburtsname der Patientin",
                    "gender": pat.gender,
                    "anrede": anrede,
                }
            )
            tags.append("maiden_name")

        # --- Relatives -----------------------------------------------------
        n_rel = rng.randint(*prof["relatives"])
        for i in range(n_rel):
            rel = sample_name(rng, difficulty=difficulty, cfg=cfg.names)
            # spouses/children often share the surname
            if rng.random() < cfg.persons.relative_shared_surname_prob:
                rel = type(rel)(rel.vorname, pat.nachname, rel.gender, rel.origin)
            rel_anrede = "Herr" if rel.gender == "m" else "Frau"
            ref = f"angehoeriger_{i}"
            catalog.persons[ref] = Person(
                entity_id=ref,
                label=Label.RELATIVE_NAME,
                vorname=rel.vorname,
                nachname=rel.nachname,
                gender=rel.gender,
                anrede=rel_anrede,
            )
            # Relation term must agree with the relative's sampled gender
            # (otherwise "der Sohn, Frau Müller") and stay demographically
            # plausible for the patient's age (no child's "Tochter", no
            # nonagenarian's living "Vater").
            beziehung = rng.choice(_relative_terms_for(age, rel.gender))
            brief.append(
                {
                    "ref": ref,
                    "type": "person",
                    "role": f"Angehörige:r ({beziehung})",
                    "gender": rel.gender,
                    "anrede": rel_anrede,
                }
            )

        # --- Clinicians ----------------------------------------------------
        clinical = self.seeds.sample_clinical_case(
            rng, patient_gender=pat.gender, patient_age=age
        )
        roles = [
            ("arzt_einweiser", "einweisende:r Ärztin/Arzt"),
            ("arzt_station", "Stationsärztin/-arzt"),
            ("arzt_ober", "Oberärztin/-arzt"),
        ]
        # Probabilistic shared-surname: 10 % for hard, 20 % for adversarial.
        _do_shared_surname = rng.random() < prof["shared_surname"]
        for idx, (ref, role) in enumerate(roles):
            doc = sample_name(rng, difficulty=difficulty, cfg=cfg.names)
            # Hard case: a clinician deliberately shares the patient's surname.
            if _do_shared_surname and idx == 1:
                doc = type(doc)(doc.vorname, pat.nachname, doc.gender, doc.origin)
                tags.append("shared_surname")
                if rng.random() < cfg.persons.shared_surname_remark_prob:
                    tags.append("shared_surname_remark")
            d_anrede = "Herr" if doc.gender == "m" else "Frau"
            titel = sample_titel(ref, rng, pool_map=cfg.names.titel_by_role)
            catalog.persons[ref] = Person(
                entity_id=ref,
                label=Label.STAFF_NAME,
                vorname=doc.vorname,
                nachname=doc.nachname,
                gender=doc.gender,
                anrede=d_anrede,
                titel=titel,
            )
            brief.append(
                {
                    "ref": ref,
                    "type": "person",
                    "role": role,
                    "gender": doc.gender,
                    "titel": catalog.persons[ref].titel,
                }
            )

        # --- Konsiliararzt (always present — needed for sub-documents) -----
        konsil_doc = sample_name(rng, difficulty=difficulty, cfg=cfg.names)
        konsil_anrede = "Herr" if konsil_doc.gender == "m" else "Frau"
        konsil_titel = sample_titel(
            "konsiliararzt", rng, pool_map=cfg.names.titel_by_role
        )
        catalog.persons["konsiliararzt"] = Person(
            entity_id="konsiliararzt",
            label=Label.STAFF_NAME,
            vorname=konsil_doc.vorname,
            nachname=konsil_doc.nachname,
            gender=konsil_doc.gender,
            anrede=konsil_anrede,
            titel=konsil_titel,
        )
        brief.append(
            {
                "ref": "konsiliararzt",
                "type": "person",
                "role": "Konsiliararzt/-ärztin (für eingebettete Befunde/Konsile)",
                "gender": konsil_doc.gender,
                "titel": konsil_titel,
            }
        )

        # --- Extra persons (consulting, GP, therapist, etc.) ---------------
        # 0-3 additional named persons, weighted towards fewer.
        _EXTRA_PERSON_POOL = [
            # (ref, role_description, label, probability_weight, needs_titel)
            ("anaesthesist", "Anästhesist:in", Label.STAFF_NAME, 2, True),
            (
                "hausarzt",
                "Hausärztin/-arzt (weiterbetreuend)",
                Label.OTHER_PERSON_NAME,
                3,
                True,
            ),
            ("notarzt", "Notärztin/-arzt", Label.OTHER_PERSON_NAME, 1, True),
            (
                "betreuer",
                "Gesetzliche:r Betreuer:in",
                Label.OTHER_PERSON_NAME,
                1,
                False,
            ),
            (
                "physiotherapeut",
                "Physiotherapeut:in",
                Label.OTHER_PERSON_NAME,
                2,
                False,
            ),
            (
                "sozialarbeiter",
                "Sozialarbeiter:in (Sozialdienst)",
                Label.OTHER_PERSON_NAME,
                2,
                False,
            ),
            ("pflegekraft", "Bezugspflegekraft", Label.OTHER_PERSON_NAME, 1, False),
        ]
        _extras_cfg = cfg.persons.extras_weights[difficulty]
        _ew_vals, _ew_probs = _extras_cfg["values"], _extras_cfg["weights"]
        n_extras = rng.choices(_ew_vals, weights=_ew_probs, k=1)[0]
        if n_extras > 0:
            # Weighted selection without replacement
            pool = list(_EXTRA_PERSON_POOL)
            chosen_extras = []
            for _ in range(min(n_extras, len(pool))):
                weights = [entry[3] for entry in pool]
                idx_chosen = rng.choices(range(len(pool)), weights=weights, k=1)[0]
                chosen_extras.append(pool.pop(idx_chosen))

            for ref, role, label, _w, needs_titel in chosen_extras:
                ep = sample_name(rng, difficulty=difficulty, cfg=cfg.names)
                ep_anrede = "Herr" if ep.gender == "m" else "Frau"
                ep_titel = (
                    sample_titel(ref, rng, pool_map=cfg.names.titel_by_role)
                    if needs_titel
                    else None
                )
                catalog.persons[ref] = Person(
                    entity_id=ref,
                    label=label,
                    vorname=ep.vorname,
                    nachname=ep.nachname,
                    gender=ep.gender,
                    anrede=ep_anrede,
                    titel=ep_titel,
                )
                brief.append(
                    {
                        "ref": ref,
                        "type": "person",
                        "role": role,
                        "gender": ep.gender,
                        "titel": ep_titel,
                    }
                )
            tags.append("extra_persons")

        # --- Dates ---------------------------------------------------------
        _dcfg = cfg.dates
        today = _resolve_reference_date(_dcfg.reference_date)
        # Encounter ("letter") date FIRST: a recency-weighted choice of how many
        # years back, plus a random day-of-year offset so month AND day are
        # randomized too (not just the year). This date -- not the global
        # reference date -- anchors the patient's age and birthdate below.
        years_back = rng.choices(
            _dcfg.years_back_values,
            weights=_dcfg.years_back_weights,
            k=1,
        )[0]
        adm = today - timedelta(
            days=years_back * _dcfg.years_back_days
            + rng.randint(*_dcfg.admission_offset_range)
        )
        # Birthdate is derived from the encounter date so the age stated in the
        # letter is the patient's age AT admission. (Previously birth was pinned
        # to the global reference year, so a back-dated letter stated the age the
        # patient would only reach at the reference date -- an age/date contradiction.)
        _birth_month = rng.randint(*_dcfg.birth_month_range)
        _birth_day = rng.randint(*_dcfg.birth_day_range)
        _birthday_reached = (adm.month, adm.day) >= (_birth_month, _birth_day)
        birth = date(
            adm.year - age - (0 if _birthday_reached else 1),
            _birth_month,
            _birth_day,
        )
        # Stay length: mostly acute (2-18 days), with occasional long
        # psych/rehab stays (21-90 days) and same-day ambulant cases (0-1 days).
        stay_type = rng.choices(
            ["ambulant", "acute", "long"],
            weights=_dcfg.stay_type_weights,
            k=1,
        )[0]
        if stay_type == "ambulant":
            stay_days = rng.randint(*_dcfg.stay_days_ambulant)
        elif stay_type == "long":
            stay_days = rng.randint(*_dcfg.stay_days_long)
        else:
            stay_days = rng.randint(*_dcfg.stay_days_acute)
        dis = adm + timedelta(days=stay_days)

        catalog.atomics["geburtsdatum"] = AtomicEntity(
            "geburtsdatum", Label.DATE_BIRTH, date_variants(birth)
        )
        catalog.atomics["aufnahmedatum"] = AtomicEntity(
            "aufnahmedatum", Label.DATE_ADMISSION, date_variants(adm)
        )
        catalog.atomics["entlassungsdatum"] = AtomicEntity(
            "entlassungsdatum", Label.DATE_DISCHARGE, date_variants(dis)
        )
        for ref, lbl in [
            ("geburtsdatum", "Geburtsdatum"),
            ("aufnahmedatum", "Aufnahmedatum"),
            ("entlassungsdatum", "Entlassungsdatum"),
        ]:
            brief.append({"ref": ref, "type": "date", "role": lbl})

        # --- OTHER_DATE entities (procedure / follow-up / medication) ------
        # 0-3 extra dates, density-scaled (low-density docs may have none).
        _max_other_dates = rng.choices(
            [0, 1, 2, 3], weights=_dcfg.other_dates_count_weights, k=1
        )[0]
        n_other_dates = min(
            _max_other_dates, round(_dcfg.other_dates_density_mult * pii_density)
        )
        other_date_roles = [
            ("datum_prozedur", "OP-/Prozedurdatum"),
            ("datum_wiedervorstellung", "Wiedervorstellungsdatum"),
            ("datum_medikation", "Medikationsänderung"),
            ("datum_befund", "Befunddatum"),
            ("datum_konsil", "Konsildatum"),
        ]
        chosen_roles = rng.sample(
            other_date_roles, k=min(n_other_dates, len(other_date_roles))
        )
        for ref, role in chosen_roles:
            if ref == "datum_wiedervorstellung":
                # Follow-up: 1-12 weeks after discharge
                d = dis + timedelta(
                    days=rng.randint(*_dcfg.wiedervorstellung_offset_range)
                )
            elif ref == "datum_prozedur":
                # Procedure during stay (or on admission day for ambulant)
                d = adm + timedelta(days=rng.randint(0, max(stay_days, 1)))
            elif ref == "datum_medikation":
                # Medication change: during stay or at discharge
                d = adm + timedelta(days=rng.randint(0, max(stay_days, 1)))
            elif ref == "datum_befund":
                # Lab/radiology result: during stay
                d = adm + timedelta(days=rng.randint(1, max(stay_days, 2)))
            else:  # datum_konsil
                # Consult during stay
                d = adm + timedelta(days=rng.randint(1, max(stay_days, 2)))
            catalog.atomics[ref] = AtomicEntity(ref, Label.DATE_OTHER, date_variants(d))
            brief.append({"ref": ref, "type": "date", "role": role})

        # --- Address & contact --------------------------------------------
        # Patient address: most letters include it, but not all (e.g. brief
        # Konsile, follow-up notes). Density-gated with high base prob.
        street_v, zip_v, city_v, area_code = address(
            rng, difficulty=difficulty, **_addr_kwargs
        )
        if _include(cfg.field_inclusion["adresse"]):
            catalog.atomics["adresse"] = AtomicEntity(
                "adresse", Label.ADDRESS, street_v
            )
            catalog.atomics["plz"] = AtomicEntity("plz", Label.ZIP, zip_v)
            catalog.atomics["ort"] = AtomicEntity("ort", Label.CITY, city_v)
            # Tag tricky location names (look like person names / common words)
            _tricky_names = {name for name, _ in TRICKY_LOCATIONS}
            if city_v["default"] in _tricky_names:
                tags.append("tricky_location")
        # telefon and fax are always in catalog (LLM may need them for sub-documents)
        catalog.atomics["telefon"] = AtomicEntity(
            "telefon", Label.PHONE, phone(rng, area_code=area_code)
        )
        catalog.atomics["fax"] = AtomicEntity(
            "fax", Label.FAX, fax(rng, area_code=area_code)
        )
        if _include(cfg.field_inclusion["mobil"]):
            catalog.atomics["mobil"] = AtomicEntity("mobil", Label.PHONE, mobile(rng))
        if _include(cfg.field_inclusion["email"]):
            catalog.atomics["email"] = AtomicEntity(
                "email", Label.EMAIL, email(rng, pat.vorname, pat.nachname)
            )
        for ref in ("adresse", "plz", "ort", "telefon", "mobil", "email", "fax"):
            if ref in catalog.atomics:
                brief.append({"ref": ref, "type": "atomic", "role": ref})

        # --- Identifiers ---------------------------------------------------
        if _include(cfg.field_inclusion["kvnr"]):
            catalog.atomics["kvnr"] = AtomicEntity(
                "kvnr", Label.ID_INSURANCE, kv_number(rng)
            )
        if _include(cfg.field_inclusion["fallnr"]):
            catalog.atomics["fallnr"] = AtomicEntity(
                "fallnr", Label.ID_CASE, case_number(rng)
            )
        if _include(cfg.field_inclusion["patid"]):
            catalog.atomics["patid"] = AtomicEntity(
                "patid", Label.ID_PATIENT, patient_id(rng)
            )
        # LANR (Arztnummer)
        if _include(cfg.field_inclusion["lanr"]):
            catalog.atomics["lanr"] = AtomicEntity("lanr", Label.ID_OTHER, lanr(rng))
        # BSNR (Betriebsstättennummer)
        if _include(cfg.field_inclusion["bsnr"]):
            catalog.atomics["bsnr"] = AtomicEntity("bsnr", Label.ID_OTHER, bsnr(rng))
        # IK-Nummer
        if _include(cfg.field_inclusion["ik_nr"]):
            catalog.atomics["ik_nr"] = AtomicEntity(
                "ik_nr", Label.ID_OTHER, ik_nummer(rng)
            )
        for ref, role in [
            ("kvnr", "Versichertennummer"),
            ("fallnr", "Fallnummer"),
            ("patid", "Patienten-ID"),
            ("lanr", "Arztnummer (LANR)"),
            ("bsnr", "Betriebsstättennummer (BSNR)"),
            ("ik_nr", "Institutionskennzeichen (IK)"),
        ]:
            if ref in catalog.atomics:
                brief.append({"ref": ref, "type": "atomic", "role": role})

        # --- Institution & insurer ----------------------------------------
        city = city_v["default"]
        dept = clinical["klinik_name"].split("und ")[-1]
        institution = sample_institution(rng, city, dept)
        catalog.atomics["institution"] = AtomicEntity(
            "institution", Label.INSTITUTION, {"default": institution}
        )
        brief.append({"ref": "institution", "type": "atomic", "role": "Klinikname"})

        # URL (hospital website) — density-gated
        if _include(cfg.field_inclusion["klinik_url"]):
            catalog.atomics["klinik_url"] = AtomicEntity(
                "klinik_url", Label.URL, url(rng, institution)
            )
            brief.append(
                {"ref": "klinik_url", "type": "atomic", "role": "Klinik-Website"}
            )

        # Hospital letterhead address — most letters have it, but short notes
        # (Konsil, Befundbericht) might omit it
        klinik_street_v, _, _, _ = address(rng, difficulty=difficulty, **_addr_kwargs)
        if _include(cfg.field_inclusion["adresse_klinik"]):
            catalog.atomics["adresse_klinik"] = AtomicEntity(
                "adresse_klinik", Label.ADDRESS, klinik_street_v
            )
            catalog.atomics["plz_klinik"] = AtomicEntity(
                "plz_klinik",
                Label.ZIP,
                zip_v,  # same PLZ pool as patient's city
            )
            catalog.atomics["ort_klinik"] = AtomicEntity(
                "ort_klinik",
                Label.CITY,
                city_v,  # same city
            )
            for ref in ("adresse_klinik", "plz_klinik", "ort_klinik"):
                brief.append({"ref": ref, "type": "atomic", "role": ref})

        # Referring physician's practice — not all letters include full
        # Einweiser details (e.g. self-referral, emergency admission)
        einweiser_p = catalog.persons["arzt_einweiser"]
        einweiser_name = (
            f"{einweiser_p.titel} " if einweiser_p.titel else ""
        ) + f"{einweiser_p.vorname[0]}. {einweiser_p.nachname}"
        praxis = sample_praxis(
            rng, einweiser_name, city, named_prob=cfg.praxis_named_prob
        )
        einw_street_v, einw_zip_v, einw_city_v, _ = address(
            rng, difficulty=difficulty, **_addr_kwargs
        )
        if _include(cfg.field_inclusion["praxis_einweiser"]):
            catalog.atomics["praxis_einweiser"] = AtomicEntity(
                "praxis_einweiser", Label.INSTITUTION, {"default": praxis}
            )
            catalog.atomics["adresse_einweiser"] = AtomicEntity(
                "adresse_einweiser", Label.ADDRESS, einw_street_v
            )
            catalog.atomics["plz_einweiser"] = AtomicEntity(
                "plz_einweiser", Label.ZIP, einw_zip_v
            )
            catalog.atomics["ort_einweiser"] = AtomicEntity(
                "ort_einweiser", Label.CITY, einw_city_v
            )
            brief.append(
                {
                    "ref": "praxis_einweiser",
                    "type": "atomic",
                    "role": "Praxis des Einweisers",
                }
            )
            for ref in ("adresse_einweiser", "plz_einweiser", "ort_einweiser"):
                brief.append({"ref": ref, "type": "atomic", "role": ref})

        # Previous clinic (prior hospitalization)
        if _include(cfg.field_inclusion["klinik_vorbehandlung"]):
            prev_clinic = sample_other_institution(rng, city)
            catalog.atomics["klinik_vorbehandlung"] = AtomicEntity(
                "klinik_vorbehandlung", Label.INSTITUTION, {"default": prev_clinic}
            )
            brief.append(
                {
                    "ref": "klinik_vorbehandlung",
                    "type": "atomic",
                    "role": "Vorbehandelnde Klinik",
                }
            )
            tags.append("multiple_institutions")

        # Rehabilitation / transfer destination
        if _include(cfg.field_inclusion["reha_klinik"]):
            rehab = sample_rehab(rng, city)
            catalog.atomics["reha_klinik"] = AtomicEntity(
                "reha_klinik", Label.INSTITUTION, {"default": rehab}
            )
            brief.append(
                {
                    "ref": "reha_klinik",
                    "type": "atomic",
                    "role": "Rehabilitationsklinik",
                }
            )
            tags.append("multiple_institutions")

        # Insurer — most Entlassbriefe mention it, but not all
        if _include(cfg.field_inclusion["versicherer"]):
            insurer = rng.choice(INSURERS)
            catalog.atomics["versicherer"] = AtomicEntity(
                "versicherer", Label.INSURER, {"default": insurer}
            )
            brief.append(
                {"ref": "versicherer", "type": "atomic", "role": "Krankenkasse"}
            )

        # --- Quasi-identifiers (medium/hard) ------------------------------
        quasi: dict[str, str] = {}
        collegial = False
        if prof["quasi"]:
            # Age gates the occupation so demographics stay coherent: children and
            # adolescents are pupils/trainees, the elderly are retired, and only
            # working-age patients can be assigned a job (incl. the physician path).
            working_age = 18 <= age < 70
            # Only working occupations get an employer attached; children and
            # retirees must not be documented as "beschäftigt bei <Firma>".
            working = False
            # Medical profession probability: higher on hard
            medical_prob = cfg.quasi.medical_prob[difficulty]
            if working_age and rng.random() < medical_prob:
                # Patient is a physician
                working = True
                med_pool = (
                    MEDICAL_PROFESSIONS_M
                    if pat.gender == "m"
                    else MEDICAL_PROFESSIONS_F
                )
                profession = rng.choice(med_pool)
                # Collegial address: hard / adversarial only
                if difficulty in ("hard", "adversarial"):
                    collegial = True
                    tags.append("collegial_address")
                    # Labeled collegial forms (must be found by anonymizer)
                    if pat.gender == "f":
                        kollege = "Kollegin"
                    else:
                        kollege = "Kollege"
                    catalog.atomics["beruf_kollegial"] = AtomicEntity(
                        "beruf_kollegial",
                        Label.PROFESSION,
                        {
                            "default": kollege,
                        },
                    )
                    brief.append(
                        {
                            "ref": "beruf_kollegial",
                            "type": "quasi",
                            "role": f"Kollegiale Anrede ({kollege})",
                        }
                    )
            elif age < 6:
                pool = (
                    KLEINKIND_OCCUPATIONS_M
                    if pat.gender == "m"
                    else KLEINKIND_OCCUPATIONS_F
                )
                profession = rng.choice(pool)
            elif age < 15:
                pool = (
                    SCHULKIND_OCCUPATIONS_M
                    if pat.gender == "m"
                    else SCHULKIND_OCCUPATIONS_F
                )
                profession = rng.choice(pool)
            elif age < 18:
                pool = (
                    ADOLESCENT_OCCUPATIONS_M
                    if pat.gender == "m"
                    else ADOLESCENT_OCCUPATIONS_F
                )
                profession = rng.choice(pool)
            elif age >= 70:
                pool = (
                    RETIRED_OCCUPATIONS_M
                    if pat.gender == "m"
                    else RETIRED_OCCUPATIONS_F
                )
                profession = rng.choice(pool)
            else:
                working = True
                profession = rng.choice(PROFESSIONS)

            catalog.atomics["beruf"] = AtomicEntity(
                "beruf", Label.PROFESSION, {"default": profession}
            )
            # "im Ruhestand" is an Erwerbsstatus phrase, not a job noun: without
            # this hint the model produces "tätig als im Ruhestand".
            if profession.startswith("im "):
                beruf_role = (
                    "Erwerbsstatus des Patienten (Statusangabe, KEIN Beruf). "
                    "Prädikativ einbetten, z.B. 'Der Patient ist [[beruf]].' "
                    "oder 'Die Patientin befindet sich [[beruf]].' – NIEMALS "
                    "mit 'tätig als', 'von Beruf' oder 'arbeitet als' kombinieren."
                )
            else:
                beruf_role = "Beruf"
            brief.append({"ref": "beruf", "type": "quasi", "role": beruf_role})
            quasi["beruf"] = profession
            tags.append("quasi_identifier")

            if working and rng.random() < cfg.quasi.employer_prob:
                employer = rng.choice(EMPLOYERS)
                catalog.atomics["arbeitgeber"] = AtomicEntity(
                    "arbeitgeber", Label.EMPLOYER, {"default": employer}
                )
                brief.append(
                    {"ref": "arbeitgeber", "type": "quasi", "role": "Arbeitgeber"}
                )
                quasi["arbeitgeber"] = employer

            # Nationality — 40% for non-German origins, 5% for German origin
            origin = pat.origin
            nat_prob = (
                cfg.quasi.nationality_prob_de
                if origin == "de"
                else cfg.quasi.nationality_prob_nonde
            )
            if rng.random() < nat_prob:
                nat_pool = NATIONALITY_BY_ORIGIN.get(origin, ["unbekannter Herkunft"])
                nationality = rng.choice(nat_pool)
                catalog.atomics["nationalitaet"] = AtomicEntity(
                    "nationalitaet", Label.NATIONALITY, {"default": nationality}
                )
                brief.append(
                    {"ref": "nationalitaet", "type": "quasi", "role": "Nationalität"}
                )
                quasi["nationalitaet"] = nationality

        # --- Age (QUASI-identifier, all difficulty tiers) -----------------
        # 40% baseline; extremes (≤5 or ≥90) are always included.
        # AGE is QUASI, not DIRECT: a bare age only re-identifies in combination
        # with other quasi-identifiers.
        age_extreme = age <= cfg.age.extreme_low_max or age >= cfg.age.extreme_high_min
        include_age = age_extreme or rng.random() < cfg.age.include_prob
        if include_age:
            # Style-varied surface forms so anonymizers cannot memorize one pattern.
            _age_forms: dict[str, str] = {
                "default": f"{age}-jährig",  # stem: gender-neutral header use
                # Attributive adjective declines for gender, case AND article.
                # Masculine nominative differs by article: weak "-e" after the
                # definite article ("der 72-jährige Patient") but strong "-er"
                # after "ein"/no article ("ein 72-jähriger Patient"). Accusative/
                # dative is "-en" ("den/dem 72-jährigen Patienten"). A single
                # form produced errors like "den 72-jähriger" or "ein
                # 72-jährige". Feminine is "-e" in nom/acc.
                "adjective_m": f"{age}-jährige",  # "der 72-jährige Patient" (def., Nom.)
                "adjective_m_indef": f"{age}-jähriger",  # "ein 72-jähriger Patient"
                "adjective_m_akk": f"{age}-jährigen",  # "den 72-jährigen Patienten"
                "adjective_f": f"{age}-jährige",  # "die 72-jährige Patientin"
                "noun": f"{age} Jahre alt",  # "Alter: 72 Jahre alt"
                "compact": f"{age}J.",  # Telegrammstil
            }
            _age_role = (
                "Alter des Patienten. Attributiv direkt vor 'Patient'/'Patientin' "
                "die zu Geschlecht, Fall UND Artikel passende Adjektiv-Variante "
                "wählen: männlich Nominativ mit bestimmtem Artikel 'adjective_m' "
                "(z.B. 'der [[alter variant=adjective_m]] Patient'); männlich "
                "Nominativ mit 'ein'/ohne Artikel 'adjective_m_indef' (z.B. 'ein "
                "[[alter variant=adjective_m_indef]] Patient'); männlich "
                "Akkusativ/Dativ 'adjective_m_akk' (z.B. 'über den "
                "[[alter variant=adjective_m_akk]] Patienten', 'bei dem "
                "[[alter variant=adjective_m_akk]] Patienten'); weiblich "
                "'adjective_f' (z.B. 'die [[alter variant=adjective_f]] "
                "Patientin', 'eine [[alter variant=adjective_f]] Patientin'). "
                "'noun' prädikativ oder nach 'Alter:' (z.B. 'Alter: "
                "[[alter variant=noun]]'); 'default' nur ohne unmittelbar "
                "folgendes Nomen; 'compact' im Telegrammstil. Verwende 'default' "
                "NIE direkt vor 'Patient'/'Patientin' (grammatikalisch falsch)."
            )
            if (
                difficulty in ("hard", "adversarial")
                and rng.random() < cfg.age.inline_bracket_prob
            ):
                # Inline bracket form: 'Patient Stein (85)' — looks like a
                # clarifying aside and is easy to miss or misparse.
                _age_forms["inline"] = f"({age})"
                _age_role = (
                    "Alter des Patienten – verwende mindestens einmal die "
                    "Variante 'inline' direkt nach einer Nennung des "
                    "Patientennamens (der Platzhalter-Wert enthält bereits die "
                    "Klammern), z.B. "
                    "'[[patient form=hon_last case=nom]] [[alter variant=inline]]'"
                )
            catalog.atomics["alter"] = AtomicEntity("alter", Label.AGE, _age_forms)
            brief.append({"ref": "alter", "type": "quasi", "role": _age_role})
            if age_extreme:
                tags.append("age_extreme")

        # --- Negatives (eponyms / look-alikes) ----------------------------
        for i, ep in enumerate(sample_eponyms(rng, rng.randint(*prof["eponyms"]))):
            catalog.negatives[f"eponym_{i}"] = Negative(f"eponym_{i}", Label.EPONYM, ep)
        for i, la in enumerate(
            sample_lookalikes(rng, rng.randint(*prof["lookalikes"]))
        ):
            catalog.negatives[f"lookalike_{i}"] = Negative(
                f"lookalike_{i}", Label.LOOKALIKE, la
            )

        # In eponym_collision mode, ALWAYS register eponyms containing the
        # patient's surname. Without this, "Morbus Parkinson" in the text
        # would be flagged as a leak when patient surname = "Parkinson".
        if adversarial_mode == "eponym_collision":
            from generator.eponym_data import EPONYMS

            existing_surfaces = {n.surface for n in catalog.negatives.values()}
            n_existing = len(catalog.negatives)
            for ep in EPONYMS:
                if pat.nachname in ep and ep not in existing_surfaces:
                    ref_id = f"eponym_{n_existing}"
                    catalog.negatives[ref_id] = Negative(ref_id, Label.EPONYM, ep)
                    n_existing += 1

        if catalog.negatives:
            tags.append("eponym_trap")

        # --- Non-PII numeric IDs (over-redaction traps for numbers) --------
        # Hard: random non-PII IDs (PZN, ATC, DRG, etc.) that look like
        # identifiers but aren't. Adversarial: same but digit patterns
        # deliberately collide with actual PII number strings.
        if difficulty in ("hard", "adversarial"):
            _nonpii_prob = cfg.negatives.nonpii_prob[difficulty]
            if rng.random() < _nonpii_prob:
                if difficulty == "adversarial":
                    # Colliding mode: generate IDs whose digits overlap PII
                    pii_surfaces = [
                        surf
                        for surf, _eid in catalog.pii_surfaces()
                        if any(c.isdigit() for c in surf)
                    ]
                    n_colliding = rng.randint(*cfg.negatives.colliding_count_range)
                    n_neg = len(catalog.negatives)
                    for i in range(n_colliding):
                        if pii_surfaces:
                            pii_surf = rng.choice(pii_surfaces)
                            type_lbl, surface = sample_colliding_nonpii_id(
                                rng,
                                pii_surf,
                                mode_weights=cfg.negatives.colliding_mode_weights,
                            )
                        else:
                            # Fallback to random
                            ids = sample_nonpii_ids(rng, 1)
                            type_lbl, surface = ids[0]
                        ref_id = f"nonpii_id_{n_neg + i}"
                        catalog.negatives[ref_id] = Negative(
                            ref_id, Label.NONPII_ID, surface
                        )
                    tags.append("nonpii_id_collision")
                else:
                    # Hard: random non-PII IDs
                    n_ids = rng.randint(*cfg.negatives.nonpii_count_range)
                    ids = sample_nonpii_ids(rng, n_ids)
                    n_neg = len(catalog.negatives)
                    for i, (type_lbl, surface) in enumerate(ids):
                        ref_id = f"nonpii_id_{n_neg + i}"
                        catalog.negatives[ref_id] = Negative(
                            ref_id, Label.NONPII_ID, surface
                        )
                    tags.append("nonpii_id_trap")

        # ``needle`` is a probability: False -> never, 0.6 -> 60% (hard),
        # True -> always (adversarial). rng.random() < bool works since
        # True == 1 (always) and False == 0 (never).
        if rng.random() < prof["needle"]:
            tags.append("needle_in_haystack")
        if prof["initials"]:
            tags.append("initial_mentions")

        sections_with_format = sample_sections(
            rng, difficulty, style, length_hint, cfg=cfg.sections
        )
        sections = [name for name, _fmt in sections_with_format]

        # --- Structure mode (headings vs blob) -----------------------------
        structure_mode = rng.choices(
            ["structured", "mixed", "unstructured"],
            weights=cfg.structure_weights[difficulty],
            k=1,
        )[0]
        if structure_mode != "structured":
            tags.append("unstructured_text")

        # --- Placement hints (PII in non-standard positions) ---------------
        # Medium/hard/adversarial cases occasionally get instructions to place
        # atomics in unusual but realistic spots within the body text. The
        # deep_needle adversarial mode is excluded: it deliberately minimizes
        # context (one buried mention, no framing), so re-injecting PII into the
        # body would contradict its design.
        placement_hints: list[str] = []
        if difficulty in ("medium", "hard", "adversarial") and (
            adversarial_mode != "deep_needle"
        ):
            # Pool of possible hints — only applicable if the entity exists
            _PLACEMENT_POOL: list[tuple[str, str]] = [
                (
                    "telefon",
                    "Erw\u00e4hne die Telefonnummer des Patienten im Abschnitt Procedere/Empfehlungen (z.B. 'telefonisch erreichbar unter [[telefon]]').",
                ),
                (
                    "adresse",
                    "Wiederhole die Patientenadresse einmal im Flie\u00dftext (z.B. im Verlauf: 'R\u00fcckkehr nach [[adresse]]' oder bei Sozialanamnese).",
                ),
                (
                    "email",
                    "Platziere die E-Mail-Adresse im Procedere oder Verlauf (z.B. 'Befund wird an [[email]] \u00fcbermittelt').",
                ),
                (
                    "geburtsdatum",
                    "Erw\u00e4hne das Geburtsdatum noch einmal im Flie\u00dftext au\u00dferhalb des Kopfbereichs (z.B. im Aufnahmebefund oder Verlauf).",
                ),
                (
                    "kvnr",
                    "Platziere die KV-Nummer im Flie\u00dftext (z.B. 'Versichertennr. [[kvnr]]' in einem Verwaltungsabschnitt oder Konsil).",
                ),
                (
                    "fallnr",
                    "Erw\u00e4hne die Fallnummer im Flie\u00dftext (z.B. im Konsil oder eingebetteten Befund).",
                ),
                (
                    "patid",
                    "Erw\u00e4hne die Patienten-ID im Brieftext (z.B. in einem zitierten Laborbefund-Kopf oder Pathologiebefund).",
                ),
                (
                    "aufnahmedatum",
                    "Verwende das Aufnahmedatum im Verlauf oder Epikrise erneut (nicht nur im Kopf).",
                ),
                (
                    "entlassungsdatum",
                    "Erw\u00e4hne das Entlassungsdatum im Procedere-Abschnitt (z.B. 'ab [[entlassungsdatum]] ...').",
                ),
                (
                    "mobil",
                    "Platziere die Mobilnummer im Procedere (z.B. 'Patient mobil erreichbar: [[mobil]]').",
                ),
                (
                    "institution",
                    "Erw\u00e4hne den Klinik-/Praxisnamen im Flie\u00dftext (z.B. im Verlauf: '\u00dcbernahme in die [[institution]]').",
                ),
            ]
            # Filter to entities actually in catalog
            available = [
                (ref, hint) for ref, hint in _PLACEMENT_POOL if ref in catalog.atomics
            ]
            # Gate: only a fraction of docs get placement hints
            _hint_prob = cfg.placement.prob[difficulty]
            if available and rng.random() < _hint_prob:
                n_hints = cfg.placement.count[difficulty]
                k = min(rng.randint(*n_hints), len(available))
                if k > 0:
                    chosen_hints = rng.sample(available, k)
                    placement_hints = [hint for _, hint in chosen_hints]
                    tags.append("nonstandard_placement")

        # --- Mention plan (form variation per person) ----------------------
        # Controls how many times and in which surface forms each person
        # appears. Difficulty drives form diversity (harder = more short/
        # ambiguous forms). Case is NOT prescribed — author picks it to fit
        # grammar.
        _FULL_FORMS = ["hon_first_last", "first_last"]
        _MEDIUM_FORMS = ["hon_last", "last", "initial_last"]
        _HARD_FORMS = ["first", "hon_initial", "initial_last"]

        mention_plan: dict[str, list[str]] = {}

        # Patient mention plan — stochastic count + form selection
        _pat_counts = cfg.mention.patient_counts[difficulty]
        n_pat = rng.choices(_pat_counts["values"], weights=_pat_counts["weights"], k=1)[
            0
        ]
        if difficulty == "easy":
            form_pool = list(_FULL_FORMS)  # full forms only: hon_first_last, first_last
        elif difficulty == "medium":
            form_pool = _FULL_FORMS + _MEDIUM_FORMS
        elif difficulty == "hard":
            form_pool = _FULL_FORMS + _MEDIUM_FORMS + _HARD_FORMS
        else:  # adversarial
            # Adversarial: heavily favor ambiguous forms (just last, just first)
            form_pool = _MEDIUM_FORMS + _HARD_FORMS + ["last", "last", "first"]

        # First mention is always a full form (realistic)
        pat_forms = [rng.choice(_FULL_FORMS)]
        for _ in range(n_pat - 1):
            pat_forms.append(rng.choice(form_pool))
        mention_plan["patient"] = pat_forms

        # Clinician mention plans (shorter — typically 1-2 mentions each)
        for ref in ("arzt_einweiser", "arzt_station", "arzt_ober"):
            if ref not in catalog.persons:
                continue
            if difficulty == "easy":
                mention_plan[ref] = ["hon_first_last"]
            elif difficulty == "medium":
                mention_plan[ref] = [
                    rng.choice(["hon_first_last", "hon_last"]),
                    rng.choice(["hon_last", "last"]),
                ]
            elif difficulty == "hard":
                mention_plan[ref] = [
                    rng.choice(["hon_first_last", "hon_last"]),
                    rng.choice(["last", "initial_last", "hon_last"]),
                ]
            else:  # adversarial — only last name, no title
                mention_plan[ref] = [
                    rng.choice(["last", "initial_last"]),
                    rng.choice(["last", "first", "initial_last"]),
                ]

        # Relatives — 1-2 mentions
        for ref in catalog.persons:
            if ref.startswith("angehoeriger_"):
                if difficulty in ("hard", "adversarial"):
                    mention_plan[ref] = [
                        "hon_first_last" if difficulty == "hard" else "last",
                        rng.choice(["first", "hon_last", "last"]),
                    ]
                else:
                    mention_plan[ref] = ["hon_first_last"]

        # Extra persons — single mention, form varies by difficulty
        for ref in catalog.persons:
            if ref in mention_plan:
                continue
            if ref in ("patient", "patient_geburtsname"):
                continue
            mention_plan[ref] = [
                "hon_first_last"
                if difficulty == "easy"
                else rng.choice(["hon_first_last", "hon_last", "initial_last"])
            ]

        # --- Adversarial overrides to mention plan -------------------------
        if adversarial_mode == "deep_needle":
            # Force exactly 1 patient mention in a minimal form
            mention_plan["patient"] = [rng.choice(["last", "initial_last"])]

        # --- Presentation diversity (codes + opening register) -------------
        # Real Arztbriefe vary widely in whether they cite ICD-10/OPS codes in
        # the running text. Telegrammstil tends to code densely; narrative /
        # academic letters often omit codes entirely. Spread the register so the
        # benchmark does not teach "a code in parentheses is always present".
        _code_weights = cfg.presentation.code_style_weights
        code_style = rng.choices(
            ["inline", "primary_only", "none"],
            weights=_code_weights.get(style, _code_weights["default"]),
            k=1,
        )[0]

        # Opening register — break the "Wir berichten über ..." monotony.
        opening_style = rng.choices(
            ["bericht", "aufnahme", "diagnose_zuerst", "epikrise_kurz", "frei"],
            weights=cfg.presentation.opening_style_weights,
            k=1,
        )[0]

        # --- Medication inclusion (80% of docs) ---
        # Some real-world letters omit dedicated medication sections.
        has_medications = rng.random() < cfg.presentation.has_medications_prob
        if not has_medications:
            # Remove medication sections from the section list
            _med_sections = {"Aufnahmemedikation", "Entlassungsmedikation"}
            sections = [s for s in sections if s not in _med_sections]
            sections_with_format = [
                (s, f) for s, f in sections_with_format if s not in _med_sections
            ]
            # Clear the medication list so prompt/dry_run won't include meds
            clinical["medikation_entlassung"] = []

        # --- Framing elements (letterhead, greeting, patient block, closing) ---
        # Real Arztbriefe often have a formal letterhead, greeting line,
        # patient info block, and closing signature block. We sample these
        # per-doc. deep_needle mode disables all framing (minimal context).
        if adversarial_mode == "deep_needle":
            framing = {
                "briefkopf": False,
                "anrede": False,
                "patientenstammdaten": False,
                "schlussformel": False,
            }
        else:
            # Base rates adjusted by style
            _framing_rates = cfg.presentation.framing_rates
            bk_rate, an_rate, ps_rate, sf_rate = _framing_rates.get(
                style, _framing_rates["default"]
            )
            # Unstructured mode reduces letterhead probability
            if structure_mode == "unstructured":
                bk_rate *= cfg.presentation.framing_unstructured_mult
                ps_rate *= cfg.presentation.framing_unstructured_mult
            framing = {
                "briefkopf": rng.random() < bk_rate,
                "anrede": rng.random() < an_rate,
                "patientenstammdaten": rng.random() < ps_rate,
                "schlussformel": rng.random() < sf_rate,
            }

        spec = GenerationSpec(
            doc_id=doc_id,
            seed=seed,
            style=style,
            difficulty=difficulty,
            sections=sections,
            section_formats=sections_with_format,
            structure_mode=structure_mode,
            placement_hints=placement_hints,
            length_hint=length_hint,
            clinical_case=clinical,
            entity_brief=brief,
            difficulty_tags=sorted(set(tags)),
            adversarial_mode=adversarial_mode,
            word_surname_context=word_surname_context,
            typo_density=typo_density,
            quasi=quasi,
            include_age_mention=include_age,
            collegial_address=collegial,
            pii_density=pii_density,
            code_style=code_style,
            opening_style=opening_style,
            patient_surname=pat.nachname,
            patient_origin=pat.origin,
            patient_age=age,
            marital_status=marital,
            mention_plan=mention_plan,
            framing=framing,
            has_medications=has_medications,
            stay_days=stay_days,
        )
        return catalog, spec
