"""Stage 4: red-team verification for measured label quality.

A "Datenschutz-Auditor" LLM receives the rendered text with all gold PII spans
masked (replaced by ``[REDACTED]``). Its job: find any *remaining* sensitive
data that was NOT masked. This is strictly harder than re-annotation because
the model cannot simply agree with existing labels — it must independently
discover gaps.

Categories searched are **exactly** those in our label schema (DIRECT + QUASI),
and the prompt explicitly excludes eponyms/look-alikes. Any finding the model
reports is a *potential* annotation gap: routed to human review, never silently
corrected.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from generator.labels import Label, Tier, tier_of
from generator.render import RenderResult


# --- Structured response schema -----------------------------------------------


class RedTeamFinding(BaseModel):
    exact_string: str = Field(
        description="Exakter Substring aus dem Text, der noch sensible Daten enthält."
    )
    category: str = Field(
        description=(
            "Kategorie: PERSON_NAME | DATUM | ADRESSE | PLZ | ORT | TELEFON | FAX | "
            "EMAIL | URL | VERSICHERTENNR | FALLNR | PATIENTEN_ID | SONSTIGE_ID | "
            "INSTITUTION | VERSICHERER | ALTER | BERUF | ARBEITGEBER | NATIONALITAET"
        ),
    )
    reasoning: str = Field(
        description="Kurze Begründung, warum dieser Substring sensibel ist."
    )


class RedTeamResponse(BaseModel):
    findings: list[RedTeamFinding] = Field(
        default_factory=list,
        description="Alle noch im Text verbleibenden sensiblen Daten nach Maskierung.",
    )


# --- Prompt construction -------------------------------------------------------

RED_TEAM_SYSTEM = """\
Du bist ein extrem kritischer Datenschutz-Auditor (Red Team) für deutsche Arztbriefe.

AUFGABE: Der folgende Text wurde bereits anonymisiert — alle erkannten sensiblen \
Stellen wurden durch [REDACTED] ersetzt. Deine Aufgabe ist es, den verbleibenden \
Text auf NOCH VORHANDENE sensible Daten zu prüfen, die übersehen wurden.

GESUCHE KATEGORIEN (nur diese!):
• PERSON_NAME — Vor-/Nachnamen, Initialen von Patienten, Angehörigen, Personal, sonstigen Personen
• DATUM — Konkrete kalendarische Daten (TT.MM.JJJJ, MM/JJJJ, etc.)
• ADRESSE — Straße + Hausnummer
• PLZ — Postleitzahlen
• ORT — Städte-/Ortsnamen (wenn identifizierend)
• TELEFON — Telefon-/Mobilnummern
• FAX — Faxnummern
• EMAIL — E-Mail-Adressen
• URL — Webseiten-URLs
• VERSICHERTENNR — Krankenversichertennummern (KV-Nummer)
• FALLNR — Fallnummern
• PATIENTEN_ID — Patienten-IDs
• SONSTIGE_ID — LANR, BSNR, IK-Nummer oder andere Identifikatoren
• INSTITUTION — Namentliche Kliniken, Praxen, Krankenhäuser
• VERSICHERER — Namentliche Krankenkassen
• ALTER — Exaktes Alter (besonders identifizierend bei Extremwerten)
• BERUF — Beruf des Patienten
• ARBEITGEBER — Arbeitgeber des Patienten
• NATIONALITAET — Nationalität/Herkunft des Patienten

KEINE sensiblen Daten (ignorieren!):
• Medizinische Eponyme: Morbus Crohn, Parkinson-Syndrom, Bell-Parese, Glasgow Coma Scale, etc.
• Fachbegriffe mit Personennamen: Babinski-Reflex, Killip-Klassifikation, Braden-Skala, etc.
• Medikamentennamen: Marcumar, Rituximab, etc.
• Geräte-/Verfahrensnamen: Swan-Ganz-Katheter, Redon-Drainage, Fowler-Lagerung, etc.
• Generische Einrichtungsbezeichnungen: Kreiskrankenhaus, Gesundheitsamt, Kassenärztliche Vereinigung
• ICD-10/OPS-Codes (z.B. I25.1, 5-361.0)
• Laborparameter, Vitalwerte, Medikamentendosen
• Diagnosen jeglicher Art — auch seltene Diagnosen (z.B. Milzbrandsepsis, Dense-deposit-Krankheit, Kaschin-Beck-Krankheit). Diagnosen sind KEINE PII.
• Relative Zeitangaben: "vor 2 Jahren", "vor 12 Monaten", "seit 3 Wochen", "Erstdiagnose vor 1 Jahr" etc. Nur KONKRETE Kalenderdaten melden!

REGELN:
1. Melde NUR tatsächlich sensible Daten, die eine Person identifizieren könnten.
2. Gib den EXAKTEN Substring aus dem Text an (copy-paste).
3. Im Zweifel: lieber melden (false positive) als übersehen (false negative).
4. [REDACTED] Platzhalter selbst NICHT melden.
"""


def _mask_text(text: str, result: RenderResult) -> str:
    """Replace all gold PII + QUASI spans with [REDACTED], preserving offsets for reporting."""
    # Sort spans by start position descending so replacements don't shift earlier offsets
    all_spans = sorted(
        [s for s in result.spans if tier_of(s.label) in (Tier.DIRECT, Tier.QUASI)],
        key=lambda s: s.start,
        reverse=True,
    )
    masked = text
    for span in all_spans:
        masked = masked[: span.start] + "[REDACTED]" + masked[span.end :]
    return masked


def _negatives_context(result: RenderResult) -> str:
    """List negatives so the model knows to ignore them."""
    if not result.negative_spans:
        return ""
    surfaces = sorted({result.text[s.start : s.end] for s in result.negative_spans})
    return (
        "\n\nFolgende medizinische Fachbegriffe kommen im Text vor und sind "
        "KEINE sensiblen Daten (nicht melden):\n• " + "\n• ".join(surfaces)
    )


def build_red_team_user(result: RenderResult) -> str:
    masked = _mask_text(result.text, result)
    neg_ctx = _negatives_context(result)
    return (
        f"Prüfe den folgenden bereits anonymisierten Arztbrief auf übersehene "
        f"sensible Daten:\n\n{masked}{neg_ctx}"
    )


def cache_key_for(text: str) -> str:
    return "redteam|" + hashlib.sha256(text.encode("utf-8")).hexdigest()


# --- Finding resolution --------------------------------------------------------


def findings_to_positions(
    masked_text: str, findings: list[dict], min_len: int = 2
) -> list[tuple[str, int, int, str]]:
    """Resolve model-reported substrings to positions in the masked text.

    Returns list of (exact_string, start, end, category).
    """
    results: list[tuple[str, int, int, str]] = []
    for f in findings:
        s = (f.get("exact_string") or "").strip()
        if len(s) < min_len:
            continue
        cat = f.get("category", "UNKNOWN")
        for m in re.finditer(re.escape(s), masked_text):
            results.append((s, m.start(), m.end(), cat))
    return results


# --- Agreement / review logic --------------------------------------------------


@dataclass
class ReviewItem:
    kind: str  # "redteam_found_unmasked"
    text: str
    start: int
    end: int
    category: str = ""
    reasoning: str = ""


@dataclass
class AgreementResult:
    n_findings: int  # total findings by red-team
    review_items: list[ReviewItem] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "n_redteam_findings": self.n_findings,
            "n_review_items": len(self.review_items),
        }


def compare(
    result: RenderResult, model_findings: list[dict], finding_min_len: int = 2
) -> AgreementResult:
    """Process red-team findings into review items.

    Every finding the red-team reports on the masked text is potentially
    a missed annotation — route to human review.
    """
    masked_text = _mask_text(result.text, result)
    positions = findings_to_positions(
        masked_text, model_findings, min_len=finding_min_len
    )

    review: list[ReviewItem] = []
    for exact, start, end, cat in positions:
        # Skip if the model reported a [REDACTED] placeholder itself
        if "[REDACTED]" in exact:
            continue
        review.append(
            ReviewItem(
                kind="redteam_found_unmasked",
                text=exact,
                start=start,
                end=end,
                category=cat,
                reasoning=next(
                    (
                        f.get("reasoning", "")
                        for f in model_findings
                        if f.get("exact_string", "").strip() == exact
                    ),
                    "",
                ),
            )
        )

    return AgreementResult(
        n_findings=len(positions),
        review_items=review,
    )
