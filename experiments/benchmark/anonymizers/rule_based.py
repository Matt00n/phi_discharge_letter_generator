"""Rule-based (REGEX-GAZ) de-identification baseline.

A deterministic anonymizer combining regular expressions, external gazetteers,
and document-structure cues — the classic first line of defence in German
clinical de-identification (Richter-Pechanski et al.; the University Hospital
Zurich lexical/rule pipeline). It is intentionally *not* tuned to our dataset:

* **Regex rules** catch strongly patterned identifiers — dates, phone/fax,
  email, URLs, ZIP+city blocks, context-triggered IDs, insurance numbers,
  street addresses, and ages.
* **Gazetteer rules** flag person names and cities using external Faker lexica
  (see :mod:`~experiments.benchmark.anonymizers.gazetteers`), never lists picked
  against our gold data.
* **Structural rules** use high-precision document cues — field labels
  (``Name:``), honorific/title prefixes (``Dr.``, ``Frau``), and the closing
  signature block — to recover names that are absent from the gazetteers.

The benchmark scores in text mode (detection only), so the anonymizer just needs
to replace the correct character regions with :data:`REDACTION_MARKER`; it does
not emit labels. Recall is deliberately honest: names outside the gazetteers and
without a structural cue (e.g. many migrant-origin surnames, bare initials) are
missed — which is the expected behaviour of a rule baseline and the gap a learned
model is meant to close.
"""

from __future__ import annotations

import re

from experiments.benchmark.anonymizers.base import Anonymizer
from experiments.benchmark.anonymizers.gazetteers import (
    city_gazetteer,
    person_gazetteers,
    street_suffixes,
)
from experiments.benchmark.anonymizers.spans import merge_spans, redact

# --- Name token building blocks ------------------------------------------------

#: A capitalised word (umlauts included). Lower body allows the German ``ß``.
_CAP = r"[A-ZÄÖÜ][a-zäöüß]+"
#: A single name token: an initial ("P.") or a (possibly hyphenated) capitalised
#: word ("Lebedew-Jakobsson", "Thorben-Florian").
_NAME_TOKEN = rf"(?:[A-ZÄÖÜ]\.|{_CAP}(?:-{_CAP})*)"
#: A run of 1–4 name tokens separated by single spaces.
_NAME_SEQ = rf"{_NAME_TOKEN}(?:[ ]{_NAME_TOKEN}){{0,3}}"

_NAME_TOKEN_RE = re.compile(_NAME_TOKEN)
_NAME_SEQ_RE = re.compile(_NAME_SEQ)

#: Role/qualifier words that mark an adjacent line/token as a person in a
#: signature block. Deliberately generic (no dataset-specific vocabulary).
_ROLE_WORDS = frozenset(
    {
        "arzt",
        "ärztin",
        "oberarzt",
        "oberärztin",
        "chefarzt",
        "chefärztin",
        "assistenzarzt",
        "assistenzärztin",
        "stationsarzt",
        "stationsärztin",
        "facharzt",
        "fachärztin",
        "hausarzt",
        "hausärztin",
        "direktor",
        "direktorin",
        "professor",
        "professorin",
        "doktor",
        "pflegekraft",
        "pfleger",
        "pflegerin",
        "psychologe",
        "psychologin",
        "therapeut",
        "therapeutin",
    }
)

# --- Regex rules (structured identifiers) --------------------------------------

_MONTHS = (
    "Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember"
)

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_URL_RE = re.compile(r"(?:https?://|www\.)[\w.\-/?=&%#~+]+", re.IGNORECASE)
_DATE_NUM_RE = re.compile(r"\b\d{1,2}\.\s?\d{1,2}\.\s?(?:\d{4}|\d{2})\b")
_DATE_TEXT_RE = re.compile(rf"\b\d{{1,2}}\.\s?(?:{_MONTHS})\s+\d{{4}}\b")
_DATE_ISO_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_PHONE_RE = re.compile(
    r"(?:Tel(?:efon)?|Telefonnummer|Fax|Telefax|Mobil|Handy)\.?\s*[:.]?\s*"
    r"(\+?\d[\d\s/()\-]{5,}\d)",
    re.IGNORECASE,
)
_ZIP_CITY_RE = re.compile(rf"\b\d{{5}}\s+({_CAP}(?:[ -]{_CAP})*)")
_ID_RE = re.compile(
    r"(?:Fallnummer|Fallnr|Fall-?Nr|Patienten-?(?:nummer|ID|Nr)|Pat\.?-?Nr|"
    r"Aufnahmenummer|Versicherten-?(?:nummer|Nr)|Versicherungsnummer|KV-?Nr|"
    r"VersNr)\.?\s*[:.]?\s*([A-Z]?\d[\w./\-]{2,})",
    re.IGNORECASE,
)
_INSURANCE_NO_RE = re.compile(r"\b[A-Z]\d{9}\b")
_AGE_RE = re.compile(
    r"\b\d{1,3}\s?-?\s?(?:jährige[rn]?|jährig|Jahre alt)\b"
    r"|\bim Alter von \d{1,3} Jahren\b",
    re.IGNORECASE,
)
_INSTITUTION_RE = re.compile(
    r"(?:Universitätsklinikum|Klinikum|Uniklinik|Klinik|Krankenhaus|Praxis|"
    r"MVZ|Facharztzentrum|Gesundheitszentrum|Ambulanz|Pflegeheim)"
    rf"[ \t]+{_CAP}(?:[ \t]+{_CAP})*(?:[ \t]*[–\-][ \t]*{_CAP}(?:[ \t]+{_CAP})*)*"
)
_INSURER_RE = re.compile(
    r"(?m)^[ \t]*(?:Kasse|Krankenkasse|Krankenversicherung|Versicherung|"
    r"Versicherer)[ \t]*:[ \t]*(.+?)[ \t]*$",
    re.IGNORECASE,
)

# --- Structural name rules -----------------------------------------------------

_FIELD_NAME_RE = re.compile(
    rf"(?m)^[ \t]*(?:Name|Patient(?:in)?|Vorname|Nachname)[ \t]*:[ \t]*({_NAME_SEQ})",
    re.IGNORECASE,
)
_TITLE_NAME_RE = re.compile(
    r"(?:Prof\.(?:[ \t]*Dr\.)?|Dr\.[ \t]*med\.|Dr\.|Priv\.-Doz\.|PD|"
    r"Dipl\.-?[A-Za-zäöü]*|Herrn?|Frau)[ \t]+"
    rf"({_NAME_SEQ})"
)
_CLOSING_RE = re.compile(
    r"(?:mit\s+)?(?:freundlichen|kollegialen|besten)\s+Gr(?:üßen|uessen|üssen)",
    re.IGNORECASE,
)
_LOCATION_CTX_RE = re.compile(
    rf"(?:wohnhaft in|Wohnort|in|aus|nach|bei)[ \t]+({_CAP}(?:[ -]{_CAP})?)",
)


class RuleBasedAnonymizer(Anonymizer):
    """Deterministic REGEX-GAZ de-identifier emitting PHI-replaced text."""

    def __init__(self) -> None:
        self.name = "rule_based"
        self._given, self._surnames = person_gazetteers()
        self._cities = city_gazetteer()
        suffixes = "|".join(re.escape(s) for s in street_suffixes())
        # Street either fused to a suffix ("Hauptstraße") or a spaced suffix
        # ("Platz der Republik"), optionally followed by a house number.
        self._street_re = re.compile(
            rf"\b{_CAP}(?:{suffixes})\b(?:[ \t]+\d+[a-z]?)?"
            rf"|\b{_CAP}[ \t]+(?:{suffixes})\b(?:[ \t]+\d+[a-z]?)?",
            re.IGNORECASE,
        )

    # -- Public API ------------------------------------------------------------

    def anonymize_text(self, text: str) -> str:
        """Return ``text`` with detected PII regions replaced by the marker."""
        spans = self.detect_spans(text)
        return redact(text, spans)

    def detect_spans(self, text: str) -> list[tuple[int, int]]:
        """Return merged ``(start, end)`` character spans to redact."""
        spans: list[tuple[int, int]] = []
        self._collect_regex_spans(text, spans)
        self._collect_gazetteer_names(text, spans)
        self._collect_structural_names(text, spans)
        self._collect_cities(text, spans)
        return merge_spans(spans)

    # -- Regex-driven identifiers ---------------------------------------------

    def _collect_regex_spans(self, text: str, spans: list[tuple[int, int]]) -> None:
        # Whole-match rules.
        for pattern in (
            _EMAIL_RE,
            _URL_RE,
            _DATE_TEXT_RE,
            _DATE_NUM_RE,
            _DATE_ISO_RE,
            _INSURANCE_NO_RE,
            _AGE_RE,
            _INSTITUTION_RE,
            self._street_re,
        ):
            for m in pattern.finditer(text):
                spans.append((m.start(), _rstrip_end(text, m.start(), m.end())))
        # Group-capture rules (context word stays, value is redacted).
        for pattern in (_PHONE_RE, _ID_RE, _INSURER_RE, _ZIP_CITY_RE):
            for m in pattern.finditer(text):
                spans.append((m.start(1), _rstrip_end(text, m.start(1), m.end(1))))

    # -- Gazetteer names -------------------------------------------------------

    def _collect_gazetteer_names(self, text: str, spans: list[tuple[int, int]]) -> None:
        for seq in _NAME_SEQ_RE.finditer(text):
            tokens = list(_NAME_TOKEN_RE.finditer(seq.group()))
            hit = None
            for tok in tokens:
                # Check the token and, for hyphenated compounds ("Kamil-
                # Maksymilian"), each part against the gazetteers.
                folded = tok.group().casefold()
                parts = [folded, *folded.split("-")] if "-" in folded else [folded]
                if any(p in self._given or p in self._surnames for p in parts):
                    hit = tok
                    break
            if hit is None:
                continue
            # Redact from the first gazetteer hit to the end of the sequence,
            # dropping any leading non-name capitalised words ("Der Patient …").
            spans.append((seq.start() + hit.start(), seq.end()))

    # -- Structural names ------------------------------------------------------

    def _collect_structural_names(
        self, text: str, spans: list[tuple[int, int]]
    ) -> None:
        for pattern in (_FIELD_NAME_RE, _TITLE_NAME_RE):
            for m in pattern.finditer(text):
                spans.append((m.start(1), m.end(1)))
        self._collect_signature_names(text, spans)

    def _collect_signature_names(self, text: str, spans: list[tuple[int, int]]) -> None:
        closings = list(_CLOSING_RE.finditer(text))
        region_start = closings[-1].end() if closings else max(0, len(text) - 400)
        region = text[region_start:]
        lines = region.split("\n")
        offset = region_start
        prev_role = bool(closings)
        for i, line in enumerate(lines):
            stripped = line.strip()
            line_start = offset
            offset += len(line) + 1  # +1 for the split "\n"
            if not stripped:
                continue
            m = re.fullmatch(rf"\s*({_NAME_SEQ})\s*", line)
            if m is None:
                prev_role = _is_role_line(stripped)
                continue
            next_is_role = _next_nonempty_is_role(lines, i)
            multi_token = len(_NAME_TOKEN_RE.findall(m.group(1))) >= 2
            if multi_token or next_is_role or prev_role:
                spans.append((line_start + m.start(1), line_start + m.end(1)))
            prev_role = False

    # -- Cities (gazetteer, context-guarded) -----------------------------------

    def _collect_cities(self, text: str, spans: list[tuple[int, int]]) -> None:
        # Only redact a bare city name when a location preposition precedes it
        # and the phrase is a known city — bare gazetteer matches over-redact
        # common nouns that happen to be place names ("Essen", "Halle").
        for m in _LOCATION_CTX_RE.finditer(text):
            if m.group(1).casefold() in self._cities:
                spans.append((m.start(1), m.end(1)))


# --- Span helpers --------------------------------------------------------------


def _rstrip_end(text: str, start: int, end: int) -> int:
    """Return ``end`` pulled back past trailing whitespace/punctuation."""
    while end > start and text[end - 1] in " \t.,;:)]":
        end -= 1
    return end


def _is_role_line(line: str) -> bool:
    """Whether a stripped line is (only) role/qualifier words."""
    words = re.findall(r"[A-Za-zÄÖÜäöüß]+", line)
    return bool(words) and all(w.casefold() in _ROLE_WORDS for w in words)


def _next_nonempty_is_role(lines: list[str], index: int) -> bool:
    for line in lines[index + 1 :]:
        if line.strip():
            return _is_role_line(line.strip())
    return False
