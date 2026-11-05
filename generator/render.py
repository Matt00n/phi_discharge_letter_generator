"""Deterministic renderer: skeleton + entity catalog -> letter text + exact spans.

Stage 3 of the pipeline. The LLM (Stage 2) authors a *skeleton* containing typed
placeholders but no real PII. This module substitutes real, morphologically
correct values and records gold spans **by construction** - there is no
substring search and therefore no span-disambiguation ambiguity.

Placeholder grammar (deliberately small and LLM-friendly)::

    [[ref]]
    [[ref form=hon_last case=gen]]
    [[ref variant=short]]

* ``ref``     - key into the entity catalog (e.g. ``patient``, ``arzt_1``, ``geburtsdatum``).
* ``form``    - person mention form (see :class:`~generator.morphology.PersonForm`).
* ``case``    - grammatical case for persons (nom/gen/dat/akk).
* ``variant`` - surface variant for atomic entities (e.g. date long/short).

Two quality gates run after rendering:

* :func:`scan_leaks` - confirms no real PII value appears unlabeled (would be a
  silent leak), and that no unresolved ``[[...]]`` remain.
* :func:`scan_negatives` - confirms eponym / look-alike negatives are present and
  were *not* accidentally labeled (over-redaction trap integrity).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from generator.labels import Label, Tier, tier_of
from generator.morphology import Case, Person, PersonForm, render_person

# --- Entity catalog -------------------------------------------------------------


@dataclass(frozen=True)
class AtomicEntity:
    """A non-person PII value (date, address, identifier, institution, ...).

    ``variants`` maps a variant name to its exact surface form. The default
    variant is used when a placeholder omits ``variant=``.
    """

    entity_id: str
    label: Label
    variants: dict[str, str]
    default_variant: str = "default"

    def surface(self, variant: str | None) -> str:
        key = variant or self.default_variant
        if key not in self.variants:
            raise ValueError(
                f"Atomic entity '{self.entity_id}' has no variant '{key}' "
                f"(available: {sorted(self.variants)})"
            )
        return self.variants[key]

    def all_surfaces(self) -> list[str]:
        return list(self.variants.values())

    def to_dict(self) -> dict:
        """JSON-native serialization of the sampled atomic entity."""
        return {
            "entity_id": self.entity_id,
            "label": self.label.value,
            "variants": dict(self.variants),
            "default_variant": self.default_variant,
        }


@dataclass(frozen=True)
class Negative:
    """An eponym / look-alike that MUST remain in the text and unredacted."""

    entity_id: str
    label: Label  # EPONYM or LOOKALIKE
    surface: str

    def to_dict(self) -> dict:
        """JSON-native serialization of the sampled negative (non-PII trap)."""
        return {
            "entity_id": self.entity_id,
            "label": self.label.value,
            "surface": self.surface,
        }


@dataclass
class Catalog:
    persons: dict[str, Person] = field(default_factory=dict)
    atomics: dict[str, AtomicEntity] = field(default_factory=dict)
    negatives: dict[str, Negative] = field(default_factory=dict)

    def __contains__(self, ref: str) -> bool:
        return ref in self.persons or ref in self.atomics or ref in self.negatives

    def to_dict(self) -> dict:
        """JSON-native serialization of the full sampled entity catalog.

        Captures every entity produced in Stage 1 — including atomic variants
        and entities that the LLM skeleton may not have referenced — so the
        complete pre-generation profile is preserved in the dataset.
        """
        return {
            "persons": {ref: p.to_dict() for ref, p in self.persons.items()},
            "atomics": {ref: a.to_dict() for ref, a in self.atomics.items()},
            "negatives": {ref: n.to_dict() for ref, n in self.negatives.items()},
        }

    def pii_surfaces(self) -> list[tuple[str, str]]:
        """All exact PII surface strings as ``(surface, entity_id)`` for leak scanning.

        For persons this includes every (form, case) realization of the PII core,
        so that an unlabeled inflected mention is still caught.
        """
        out: list[tuple[str, str]] = []
        for p in self.persons.values():
            for form in PersonForm:
                for case in Case:
                    try:
                        core = render_person(p, form, case).core
                    except ValueError:
                        continue
                    if core:
                        out.append((core, p.entity_id))
        for a in self.atomics.values():
            for s in a.all_surfaces():
                if s:
                    out.append((s, a.entity_id))
        return out


# --- Rendered output ------------------------------------------------------------


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    label: str
    tier: str
    entity_id: str
    form: str | None = None
    case: str | None = None

    def as_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "label": self.label,
            "tier": self.tier,
            "entity_id": self.entity_id,
            "form": self.form,
            "case": self.case,
        }


@dataclass(frozen=True)
class RenderResult:
    text: str
    spans: list[Span]
    negative_spans: list[Span]


# --- Placeholder parsing --------------------------------------------------------

_PLACEHOLDER_RE = re.compile(r"\[\[\s*([a-zA-Z0-9_]+)((?:\s+[^\]]+)?)\s*\]\]")
_ATTR_RE = re.compile(r"([a-zA-Z_]+)=([^\s\]]+)")
_RESIDUAL_RE = re.compile(r"\[\[|\]\]")


def _parse_placeholder_tokens(ref: str, raw_tail: str) -> tuple[str, dict[str, str]]:
    """Parse placeholder ref and attributes, handling LLM format variations.

    Handles:
    - Standard: [[patient form=hon_last case=nom]]
    - Spurious 'ref' prefix: [[ref patient form=hon_last case=nom]]
    - Positional variant: [[aufnahmedatum long]] → variant=long
    """
    tail = raw_tail.strip()

    # Fix 1: If ref is literally "ref", the actual ref is the first token in tail
    # (unless that token is a key=value attribute, in which case default to "patient")
    if ref == "ref" and tail:
        tokens = tail.split(None, 1)
        if "=" in tokens[0]:
            # No explicit entity ref — LLM used [[ref form=... case=...]]
            # Default to "patient" and keep the full tail for attribute parsing
            ref = "patient"
        else:
            ref = tokens[0]
            tail = tokens[1] if len(tokens) > 1 else ""

    # Parse key=value attributes
    attrs = {m.group(1): m.group(2) for m in _ATTR_RE.finditer(tail)}

    # Fix 2: If there's a bare word (no '=') and it's not already captured,
    # treat it as a positional variant for atomic refs.
    # Only do this if 'variant' and 'form' are not already set.
    if tail and "form" not in attrs and "variant" not in attrs:
        # Find bare tokens (words not part of key=value pairs)
        remaining = _ATTR_RE.sub("", tail).strip()
        if remaining and remaining.isalpha():
            attrs["variant"] = remaining

    return ref, attrs


def _parse_attrs(attr_str: str) -> dict[str, str]:
    return {m.group(1): m.group(2) for m in _ATTR_RE.finditer(attr_str)}


class RenderError(ValueError):
    """Raised when a skeleton cannot be rendered (unknown ref, bad form, ...)."""


def render(skeleton: str, catalog: Catalog) -> RenderResult:
    """Render a skeleton into final text plus exact PII and negative spans."""
    out: list[str] = []
    spans: list[Span] = []
    negative_spans: list[Span] = []
    pos = 0  # cursor into the *source* skeleton
    cursor = 0  # cursor into the *output* text (char offset)

    for m in _PLACEHOLDER_RE.finditer(skeleton):
        literal = skeleton[pos : m.start()]
        out.append(literal)
        cursor += len(literal)
        pos = m.end()

        ref, attrs = _parse_placeholder_tokens(m.group(1), m.group(2) or "")

        prefix, core, label, entity_id, form, case, is_negative = _resolve(
            ref, attrs, catalog
        )

        out.append(prefix)
        cursor += len(prefix)

        start = cursor
        out.append(core)
        cursor += len(core)
        end = cursor

        if core:
            span = Span(
                start=start,
                end=end,
                label=label.value,
                tier=tier_of(label).value,
                entity_id=entity_id,
                form=form,
                case=case,
            )
            if is_negative:
                negative_spans.append(span)
            else:
                spans.append(span)

    tail = skeleton[pos:]
    out.append(tail)

    text = "".join(out)
    # Negatives are usually written as plain text (eponyms/look-alikes), not via
    # placeholders. Locate them in the final text and record them as gold
    # negative spans, skipping any occurrence that overlaps a PII span.
    already = {(s.start, s.end) for s in negative_spans}
    for neg in catalog.negatives.values():
        for m in re.finditer(re.escape(neg.surface), text):
            if (m.start(), m.end()) in already:
                continue
            if _overlaps_any(m.start(), m.end(), spans):
                continue
            negative_spans.append(
                Span(
                    start=m.start(),
                    end=m.end(),
                    label=neg.label.value,
                    tier=tier_of(neg.label).value,
                    entity_id=neg.entity_id,
                )
            )
            already.add((m.start(), m.end()))

    negative_spans.sort(key=lambda s: s.start)
    return RenderResult(text=text, spans=spans, negative_spans=negative_spans)


def _resolve(ref: str, attrs: dict[str, str], catalog: Catalog):
    """Resolve a placeholder to (prefix, core, label, entity_id, form, case, is_negative)."""
    if ref in catalog.persons:
        person = catalog.persons[ref]
        form = attrs.get("form", PersonForm.HON_LAST.value)
        case = attrs.get("case", Case.NOM.value)
        try:
            mention = render_person(person, form, case)
        except ValueError as exc:
            raise RenderError(f"Bad person placeholder [[{ref} ...]]: {exc}") from exc
        return (
            mention.prefix,
            mention.core,
            person.label,
            person.entity_id,
            form,
            case,
            False,
        )

    if ref in catalog.atomics:
        atomic = catalog.atomics[ref]
        variant = attrs.get("variant")
        try:
            surface = atomic.surface(variant)
        except ValueError as exc:
            raise RenderError(f"Bad atomic placeholder [[{ref} ...]]: {exc}") from exc
        return "", surface, atomic.label, atomic.entity_id, variant, None, False

    if ref in catalog.negatives:
        neg = catalog.negatives[ref]
        return "", neg.surface, neg.label, neg.entity_id, None, None, True

    raise RenderError(f"Unknown placeholder ref '{ref}' (not in catalog)")


# --- Quality gates --------------------------------------------------------------


@dataclass(frozen=True)
class LeakFinding:
    surface: str
    entity_id: str
    start: int
    end: int
    label: str = "UNKNOWN"


def _covered(start: int, end: int, spans: Iterable[Span]) -> bool:
    return any(s.start <= start and end <= s.end for s in spans)


def _overlaps_any(start: int, end: int, spans: Iterable[Span]) -> bool:
    return any(max(start, s.start) < min(end, s.end) for s in spans)


# Labels whose surface forms are ambient tokens (short numeric strings, common words)
# that produce excessive false-positive leak findings. The by-construction span
# guarantee is the right safety net for these; the leak scanner skips them.
_LEAK_SCAN_SKIP_LABELS: frozenset[Label] = frozenset({Label.AGE})


def scan_leaks(result: RenderResult, catalog: Catalog) -> list[LeakFinding]:
    """Find real PII values that appear in the text but outside any gold span.

    A non-empty result means the skeleton (i.e. the LLM) leaked a literal value,
    or a rendered value collided with surrounding text. The document should be
    flagged for human review - never silently relabeled.
    """
    findings: list[LeakFinding] = []
    text = result.text

    if _RESIDUAL_RE.search(text):
        # Unresolved placeholder markers survived into the output.
        for m in _RESIDUAL_RE.finditer(text):
            findings.append(
                LeakFinding(
                    "[[...]]", "__residual__", m.start(), m.end(), "__residual__"
                )
            )

    all_spans = list(result.spans) + list(result.negative_spans)
    for surface, entity_id in catalog.pii_surfaces():
        # Skip entities whose surfaces are ambient tokens (e.g. two-digit ages).
        _atom = catalog.atomics.get(entity_id)
        if _atom is not None and _atom.label in _LEAK_SCAN_SKIP_LABELS:
            continue
        if len(surface) < 3:
            continue
        # Use word boundaries to avoid matching inside compound words
        # (e.g. surname "Koch" should not match inside "Kochsalzlösung").
        pattern = r"\b" + re.escape(surface) + r"\b"
        for m in re.finditer(pattern, text):
            # Skip matches that are part of a hyphenated medical compound
            # (e.g. "Parkinson-Syndrom", "Koch-Postulat"). A PII leak would
            # never appear hyphenated with a medical suffix.
            if (m.start() > 0 and text[m.start() - 1] == "-") or (
                m.end() < len(text) and text[m.end()] == "-"
            ):
                continue
            # A real leak is a PII surface that overlaps NO gold span at all.
            # Overlap (not containment) avoids false positives from overlapping
            # atomic variant surfaces (e.g. "Pat.-ID 123456" vs. the "123456" span).
            if not _overlaps_any(m.start(), m.end(), all_spans):
                # Resolve the PHI label of the leaked entity so provenance
                # analysis can break leaks down by label type.
                if entity_id in catalog.persons:
                    label_str = catalog.persons[entity_id].label.value
                elif entity_id in catalog.atomics:
                    label_str = catalog.atomics[entity_id].label.value
                else:
                    label_str = "UNKNOWN"
                findings.append(
                    LeakFinding(surface, entity_id, m.start(), m.end(), label_str)
                )
    return findings


@dataclass(frozen=True)
class NegativeFinding:
    entity_id: str
    surface: str
    reason: str
    label: str = "UNKNOWN"


def scan_negatives(result: RenderResult, catalog: Catalog) -> list[NegativeFinding]:
    """Validate eponym / look-alike traps.

    Each negative must (a) actually appear in the text and (b) not be covered by
    a *PII* span (which would wrongly mark a non-PII term as redactable).
    """
    findings: list[NegativeFinding] = []
    text = result.text
    for neg in catalog.negatives.values():
        if neg.surface not in text:
            findings.append(
                NegativeFinding(
                    neg.entity_id, neg.surface, "missing_in_text", neg.label.value
                )
            )
            continue
        for m in re.finditer(re.escape(neg.surface), text):
            if _covered(m.start(), m.end(), result.spans):
                findings.append(
                    NegativeFinding(
                        neg.entity_id, neg.surface, "overlaps_pii_span", neg.label.value
                    )
                )
                break
    return findings


# --- Structural integrity assertions (D+E) ------------------------------------


class SpanIntegrityError(ValueError):
    """Hard invariant violated — indicates a bug in the renderer or post-processing."""


def validate_spans(result: RenderResult, catalog: Catalog) -> None:
    """Assert structural invariants that must ALWAYS hold.

    These are not heuristic flags — a violation means a code bug. Raises
    :class:`SpanIntegrityError` on the first violation found.
    """
    text = result.text
    text_len = len(text)

    all_spans = list(result.spans) + list(result.negative_spans)

    for span in all_spans:
        # --- Bounds ---
        if span.start < 0:
            raise SpanIntegrityError(f"Span start < 0: {span.as_dict()}")
        if span.end > text_len:
            raise SpanIntegrityError(
                f"Span end ({span.end}) > text length ({text_len}): {span.as_dict()}"
            )
        if span.start >= span.end:
            raise SpanIntegrityError(
                f"Span start >= end (empty/inverted): {span.as_dict()}"
            )

        # --- No leading/trailing whitespace in span text ---
        span_text = text[span.start : span.end]
        if span_text != span_text.strip():
            raise SpanIntegrityError(
                f"Span has leading/trailing whitespace: "
                f"{span.as_dict()}, text={span_text!r}"
            )

        # --- entity_id exists in catalog ---
        eid = span.entity_id
        if not (
            eid in catalog.persons or eid in catalog.atomics or eid in catalog.negatives
        ):
            raise SpanIntegrityError(
                f"Span entity_id '{eid}' not found in catalog: {span.as_dict()}"
            )

    # --- No overlapping PII spans ---
    pii_sorted = sorted(result.spans, key=lambda s: (s.start, s.end))
    for i in range(len(pii_sorted) - 1):
        a, b = pii_sorted[i], pii_sorted[i + 1]
        if a.end > b.start:
            raise SpanIntegrityError(
                f"Overlapping PII spans: {a.as_dict()} and {b.as_dict()}"
            )

    # --- No PII span overlaps a negative span ---
    for pii in result.spans:
        for neg in result.negative_spans:
            if max(pii.start, neg.start) < min(pii.end, neg.end):
                raise SpanIntegrityError(
                    f"PII span overlaps negative span: "
                    f"pii={pii.as_dict()}, neg={neg.as_dict()}"
                )
