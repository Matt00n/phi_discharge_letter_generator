"""Shared de-identification prompt for the text-replacement benchmark.

The label definitions are aligned precisely with the gold-standard schema used by
the generator, so an anonymizer is scored against the same notion of PII the
dataset was built with. Only the *text* output format is supported here — the
model returns the full document with every PII entity replaced by a marker.
"""

from __future__ import annotations

from pydantic import BaseModel

#: Marker an anonymizer must emit in place of each detected PII entity. Kept in
#: sync with the token the :mod:`experiments.benchmark.scoring` aligner treats as redacted.
REDACTION_MARKER = "[REDACTED]"

_LABEL_DEFINITIONS = """\
You are an expert medical document de-identification system for German clinical text.
Your task is to identify ALL personally identifying information (PII) in the text.

## What to redact (MUST identify):

### DIRECT identifiers (identify a person on their own):
- **Person names**: Patient names, relative/family member names, staff/doctor names, any other person names. This includes first names, last names, titles with names (but NOT the title prefix itself — "Dr." or "Herr" alone is not PII, but "Dr. Müller" → "Müller" is PII).
- **Dates**: Birth dates, admission dates, discharge dates, surgery/procedure dates, document dates, any other specific dates. Relative expressions like "vor 3 Wochen" are NOT dates.
- **Street addresses**: Street name and house number (e.g. "Hauptstraße 12"). ZIP codes and city names are quasi-identifiers — see below.
- **Contact**: Phone numbers, fax numbers, email addresses, URLs.
- **IDs**: Insurance numbers (KV-Nummer), case numbers (Fallnummer), patient IDs, any other identifying numbers.

### QUASI identifiers (identifying only in combination, never alone):
- **ZIP codes**: Postal codes (e.g. "50931").
- **City names**: Town/city of residence or treatment.
- **Named institutions**: Named hospitals, clinics, practices, or departments when they identify a specific institution.
- **Named insurers**: Named insurance companies.
- **Age**: Specific age mentions (e.g. "72-jährig", "im Alter von 45 Jahren").
- **Profession**: Specific occupation of the patient (e.g. "Schreiner", "Lehrerin").
- **Employer**: Named employer.
- **Nationality**: Specific nationality of the patient.

## What NOT to redact (must PRESERVE unchanged):
- Medical eponyms: Parkinson, Alzheimer, Crohn, Hashimoto, Basedow, Hodgkin, Ewing, Guillain-Barré, etc. — these are disease/procedure names, NOT person names.
- ICD-10 and OPS codes (e.g. "I10.90", "5-820.00").
- Drug names, even if person-derived.
- Anatomical terms.
- Generic role descriptions without a name ("der behandelnde Arzt", "die Pflegekraft").
- Relative time expressions ("vor 2 Wochen", "seit 3 Monaten").
- Weekdays without specific dates.
- Hospital department names when generic ("Kardiologie", "Station 3B") — only redact when they uniquely identify an institution.
- "Sehr geehrte Damen und Herren" or similar generic salutations.
- Medical device names or product names.

## Critical rules:
1. Be thorough — missing a real PII span is worse than over-redacting.
2. Do NOT redact medical eponyms. "Morbus Parkinson" → keep. "Patient Parkinson" → redact "Parkinson".
3. Titles/honorifics ("Herr", "Frau", "Dr.", "Prof.") are NOT PII by themselves. Only the actual name following them is PII.
4. Numbers that are clearly clinical measurements (blood pressure, lab values, dosages) are NOT PII.
"""

#: System instruction for the full-text replacement mode.
TEXT_INSTRUCTION = (
    _LABEL_DEFINITIONS
    + f"""
## Output format:
Return the FULL text with every PII span replaced by the marker {REDACTION_MARKER}.
Keep ALL other text exactly as-is — same whitespace, same punctuation, same line breaks.
Do not summarize, rephrase, or omit any non-PII text.
Each separate PII entity gets its own {REDACTION_MARKER} marker.
"""
)


class TextResponse(BaseModel):
    """Structured schema for the full de-identified text."""

    text: str
