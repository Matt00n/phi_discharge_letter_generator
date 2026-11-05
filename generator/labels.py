"""PII / non-PII label taxonomy for the anonymization benchmark.

The schema is hierarchical and aligned with recognized de-identification
standards (i2b2 / PhysioNet), extended for German clinical text. Every label
belongs to exactly one *tier*:

* ``DIRECT``  - direct identifiers; an anonymizer MUST redact these. Gold must
  never contain a false negative here (the privacy-critical headline metric is
  direct-identifier recall / "leakage").
* ``QUASI``   - quasi-identifiers; identifying only in combination. Reported as
  a separate tier so metrics can be computed with and without them.
* ``NONPII``  - explicit negatives (e.g. eponymous medical terms) that look like
  PII but MUST NOT be redacted. Used to measure over-redaction.

Labels are plain strings so they serialize cleanly to JSONL and interoperate
with external NLP tooling.
"""

from __future__ import annotations

from enum import Enum


class Tier(str, Enum):
    DIRECT = "DIRECT"
    QUASI = "QUASI"
    NONPII = "NONPII"


class Label(str, Enum):
    # --- DIRECT identifiers -------------------------------------------------
    PATIENT_NAME = "PATIENT_NAME"
    RELATIVE_NAME = "RELATIVE_NAME"
    STAFF_NAME = "STAFF_NAME"
    OTHER_PERSON_NAME = "OTHER_PERSON_NAME"

    DATE_BIRTH = "DATE_BIRTH"
    DATE_ADMISSION = "DATE_ADMISSION"
    DATE_DISCHARGE = "DATE_DISCHARGE"
    DATE_OTHER = "DATE_OTHER"

    ADDRESS = "ADDRESS"
    ZIP = "ZIP"
    CITY = "CITY"

    PHONE = "PHONE"
    FAX = "FAX"
    EMAIL = "EMAIL"
    URL = "URL"

    ID_INSURANCE = "ID_INSURANCE"  # KV-Nummer / Versichertennummer
    ID_CASE = "ID_CASE"  # Fallnummer
    ID_PATIENT = "ID_PATIENT"  # Patienten-ID
    ID_OTHER = "ID_OTHER"

    INSTITUTION = "INSTITUTION"  # named clinic / hospital / practice
    INSURER = "INSURER"  # named insurance company

    # --- QUASI identifiers --------------------------------------------------
    AGE = "AGE"  # age (quasi-identifier; re-identifying only at extremes)
    PROFESSION = "PROFESSION"
    EMPLOYER = "EMPLOYER"
    NATIONALITY = "NATIONALITY"

    # --- NON-PII negatives --------------------------------------------------
    EPONYM = "EPONYM"  # person-derived medical term (trap)
    LOOKALIKE = "LOOKALIKE"  # other name-like non-PII trap
    NONPII_ID = "NONPII_ID"  # non-PII number (PZN, ATC, etc.) that looks like an ID


# Tier membership ----------------------------------------------------------------
TIER_OF: dict[Label, Tier] = {
    # DIRECT
    Label.PATIENT_NAME: Tier.DIRECT,
    Label.RELATIVE_NAME: Tier.DIRECT,
    Label.STAFF_NAME: Tier.DIRECT,
    Label.OTHER_PERSON_NAME: Tier.DIRECT,
    Label.DATE_BIRTH: Tier.DIRECT,
    Label.DATE_ADMISSION: Tier.DIRECT,
    Label.DATE_DISCHARGE: Tier.DIRECT,
    Label.DATE_OTHER: Tier.DIRECT,
    Label.ADDRESS: Tier.DIRECT,
    # ZIP, CITY, INSTITUTION, INSURER are quasi-identifiers: they narrow down a
    # person only in combination with other attributes, never alone.
    Label.ZIP: Tier.QUASI,
    Label.CITY: Tier.QUASI,
    Label.PHONE: Tier.DIRECT,
    Label.FAX: Tier.DIRECT,
    Label.EMAIL: Tier.DIRECT,
    Label.URL: Tier.DIRECT,
    Label.ID_INSURANCE: Tier.DIRECT,
    Label.ID_CASE: Tier.DIRECT,
    Label.ID_PATIENT: Tier.DIRECT,
    Label.ID_OTHER: Tier.DIRECT,
    Label.INSTITUTION: Tier.QUASI,
    Label.INSURER: Tier.QUASI,
    # QUASI
    Label.AGE: Tier.QUASI,
    Label.PROFESSION: Tier.QUASI,
    Label.EMPLOYER: Tier.QUASI,
    Label.NATIONALITY: Tier.QUASI,
    # NON-PII
    Label.EPONYM: Tier.NONPII,
    Label.LOOKALIKE: Tier.NONPII,
    Label.NONPII_ID: Tier.NONPII,
}


# Labels whose surface forms are persons (rendered via the morphology engine).
PERSON_LABELS: frozenset[Label] = frozenset(
    {
        Label.PATIENT_NAME,
        Label.RELATIVE_NAME,
        Label.STAFF_NAME,
        Label.OTHER_PERSON_NAME,
    }
)


def tier_of(label: Label | str) -> Tier:
    """Return the tier for a label (accepts the enum or its string value)."""
    if isinstance(label, str) and not isinstance(label, Label):
        label = Label(label)
    return TIER_OF[label]


def is_direct(label: Label | str) -> bool:
    return tier_of(label) is Tier.DIRECT


def all_labels() -> list[str]:
    return [label.value for label in Label]
