"""German morphology engine for person mentions.

This is the de-risking core of the typed-placeholder approach. Given a person
entity, a *mention form* and a grammatical *case*, it produces the exact surface
string, split into:

* a non-PII ``prefix`` (honorifics / titles / salutation lead - NOT redacted), and
* a PII ``core`` (the actual name span - redacted, and the gold span we emit).

Keeping PII contiguous in a single ``core`` means the renderer can record an
exact ``(start, end)`` span by construction, with no substring search.

Scope of morphology handled:
* Anrede inflection: ``Herr`` -> ``Herrn`` in oblique cases; ``Frau`` invariant.
* Saxon genitive on proper names (``Müller`` -> ``Müllers``; ``Markus`` -> ``Markus'``).
* Title-based honorifics for clinicians (``Dr. med.``, ``Prof. Dr. med.`` ...).
* Gender agreement in the salutation (``Sehr geehrter Herr`` / ``Sehr geehrte Frau``).

Adjective/article declension of *descriptor* phrases (``der Patient`` ->
``dem Patienten``) is intentionally out of scope here: those are written by the
LLM as free text and are not PII spans.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from generator.labels import Label


class Case(str, Enum):
    NOM = "nom"
    GEN = "gen"
    DAT = "dat"
    AKK = "akk"


OBLIQUE_CASES = frozenset({Case.GEN, Case.DAT, Case.AKK})


class PersonForm(str, Enum):
    """Mention variants. These are the controllable diversity / difficulty axis."""

    FIRST_LAST = "first_last"  # "Michael Müller"
    HON_FIRST_LAST = "hon_first_last"  # "Herr Michael Müller" / "Dr. med. Frank Wagner"
    HON_LAST = "hon_last"  # "Herr Müller" / "Dr. Wagner"
    LAST = "last"  # "Müller"
    FIRST = "first"  # "Michael"
    INITIAL_LAST = "initial_last"  # "M. Müller"
    HON_INITIAL = "hon_initial"  # "Herr M." (last-name initial)
    SALUTATION = "salutation"  # "Sehr geehrter Herr Müller"


@dataclass(frozen=True)
class Person:
    """A person entity that can be realized in several inflected mention forms."""

    entity_id: str
    label: Label
    vorname: str
    nachname: str
    gender: str  # "m" | "f" | "x"
    anrede: str | None = None  # "Herr" | "Frau" | None
    titel: str | None = None  # "Dr. med." | "Prof. Dr. med." | None

    @property
    def honorific_nom(self) -> str | None:
        """The honorific used in non-oblique position (title takes precedence)."""
        if self.titel:
            return self.titel
        return self.anrede

    def to_dict(self) -> dict[str, str | None]:
        """JSON-native serialization of the sampled person entity."""
        return {
            "entity_id": self.entity_id,
            "label": self.label.value,
            "vorname": self.vorname,
            "nachname": self.nachname,
            "gender": self.gender,
            "anrede": self.anrede,
            "titel": self.titel,
        }


@dataclass(frozen=True)
class Mention:
    """Result of rendering a person: a non-PII prefix followed by a PII core."""

    prefix: str  # honorific / salutation lead, never redacted
    core: str  # the name span, redacted; this is the emitted gold span
    label: Label
    entity_id: str

    @property
    def text(self) -> str:
        return f"{self.prefix}{self.core}"


# Names ending in a sibilant take an apostrophe genitive ("Markus'", "Weiß'").
_SIBILANT_ENDINGS = ("s", "ß", "x", "z", " s", "ce", "se", "ze")


def genitive(token: str) -> str:
    """Saxon genitive of a proper-name token."""
    if not token:
        return token
    low = token.lower()
    if low.endswith(("s", "ß", "x", "z")):
        return f"{token}'"
    return f"{token}s"


def _inflect_honorific(honorific: str | None, oblique: bool) -> str | None:
    """Inflect the honorific. Only ``Herr`` changes (-> ``Herrn``) in oblique cases."""
    if honorific is None:
        return None
    if honorific == "Herr" and oblique:
        return "Herrn"
    return honorific


def _join_core(tokens: list[str], gen: bool) -> str:
    """Join name tokens; in the genitive only the final token is inflected."""
    tokens = [t for t in tokens if t]
    if not tokens:
        return ""
    if gen:
        tokens = tokens[:-1] + [genitive(tokens[-1])]
    return " ".join(tokens)


def _initial(token: str) -> str:
    return f"{token[0]}." if token else ""


def render_person(person: Person, form: PersonForm | str, case: Case | str) -> Mention:
    """Render a person into an exact, inflected :class:`Mention`.

    Policy: spans contain ONLY the identifying name. Honorifics, titles, and
    salutation leads are NEVER part of a span. The ``hon_*`` forms are
    functionally identical to their non-``hon_`` counterparts in output — the
    LLM writes honorifics/titles freely in the skeleton text surrounding the
    placeholder, and those remain as plain (non-PII) text.

    The ``SALUTATION`` form is an exception: its lead ("Sehr geehrter Herr ")
    is emitted as a prefix since the LLM uses a single placeholder for the
    entire salutation phrase.

    Raises ``ValueError`` on unknown form/case so that skeleton validation can
    surface authoring typos instead of silently producing wrong spans.
    """
    form = PersonForm(form)
    case = Case(case)
    gen = case is Case.GEN

    vor = person.vorname
    nach = person.nachname

    if form in (PersonForm.FIRST_LAST, PersonForm.HON_FIRST_LAST):
        prefix, core = "", _join_core([vor, nach], gen)
    elif form in (PersonForm.HON_LAST, PersonForm.LAST):
        prefix, core = "", _join_core([nach], gen)
    elif form is PersonForm.FIRST:
        prefix, core = "", _join_core([vor], gen)
    elif form is PersonForm.INITIAL_LAST:
        prefix, core = "", _join_core([_initial(vor), nach], gen)
    elif form is PersonForm.HON_INITIAL:
        # Last-name initial only, e.g. "M."
        prefix, core = "", _initial(nach)
    elif form is PersonForm.SALUTATION:
        oblique = case in OBLIQUE_CASES
        hon = _inflect_honorific(person.honorific_nom, oblique)
        if person.gender == "m":
            lead = "Sehr geehrter "
        elif person.gender == "f":
            lead = "Sehr geehrte "
        else:
            lead = "Sehr geehrte/r "
        anr = _inflect_honorific(person.anrede or hon, oblique=False)
        prefix = f"{lead}{anr} " if anr else lead
        core = _join_core([nach], gen)
    else:  # pragma: no cover - exhaustive above
        raise ValueError(f"Unknown person form: {form}")

    return Mention(
        prefix=prefix, core=core, label=person.label, entity_id=person.entity_id
    )
