"""Mode B alignment: infer predicted spans from de-identified text.

Given the original text and a de-identified version where PII is replaced
with [REDACTED], reconstruct the character-offset spans that correspond to
each redaction in the original text.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass

from experiments.benchmark.scoring.matching import Span

REDACTION_MARKER = "[REDACTED]"

#: Single-character stand-in for the marker used only during diffing. It is a
#: Unicode Private-Use-Area codepoint that never occurs in clinical text, so
#: ``difflib`` cannot coincidentally align a marker character (the letters
#: R/E/D/A/C/T in "[REDACTED]") to a real character and fragment a redaction
#: (e.g. "Emily" -> leaked "mily", "Rehabilitationszentrum" -> "ehabilitations...").
_MARKER_SENTINEL = chr(0xE000)


@dataclass
class AlignmentResult:
    """Result of aligning de-identified text against original."""

    predicted_spans: list[Span]
    # Regions in original text that were changed but don't match [REDACTED]
    corruptions: list[tuple[int, int, str, str]]  # (start, end, original, replacement)
    # Whether alignment was successful
    success: bool
    alignment_errors: list[str]


def align_texts(original: str, deidentified: str) -> AlignmentResult:
    """Align original and de-identified text to extract predicted spans.

    Each [REDACTED] in the de-identified text maps back to a region in
    the original text that was removed. Non-PII text should be identical.

    Strategy: replace the multi-character marker with a single sentinel
    codepoint that never appears in clinical text, then diff. Because the
    sentinel shares no characters with the original, difflib can no longer
    match a marker letter (R/E/D/A/C/T) against a real character, so each
    redaction maps cleanly to exactly one contiguous original region and is
    never split into spurious sub-fragments (the failure mode that leaked
    "mily" from "Emily" or "ehabilitations..." from "Rehabilitationszentrum").

    Returns predicted spans (in original text coordinates) and any
    corruption findings where non-PII text was modified.
    """
    predicted_spans: list[Span] = []
    corruptions: list[tuple[int, int, str, str]] = []
    alignment_errors: list[str] = []

    # Normalise the marker to a single sentinel absent from clinical text. The
    # sentinel is a Private-Use-Area codepoint, so this guard effectively never
    # fires; if it ever did, we diff the raw text and flag it rather than
    # silently corrupt the mapping.
    deid_source = deidentified
    if _MARKER_SENTINEL in original or _MARKER_SENTINEL in deidentified:
        alignment_errors.append("sentinel codepoint present in input; not normalized")
    else:
        deid_source = deidentified.replace(REDACTION_MARKER, _MARKER_SENTINEL)

    matcher = difflib.SequenceMatcher(None, original, deid_source, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if _MARKER_SENTINEL in deid_source[j1:j2]:
            # This opcode realises one or more redactions. original[i1:i2] is
            # the region the marker(s) replaced; a pure insert (i2 == i1) is a
            # zero-width marker with no original text behind it and is skipped.
            if i2 > i1:
                predicted_spans.append(Span(start=i1, end=i2))
        elif tag == "delete":
            # Text removed without a marker: the system redacted something but
            # did not emit the marker; count it as a prediction.
            predicted_spans.append(Span(start=i1, end=i2))
        else:  # replace / insert of non-marker text: non-PII was altered
            # j1/j2 index into deid_source (post-sentinel-swap); slice the same
            # string so the recorded replacement text matches the offsets.
            corruptions.append((i1, i2, original[i1:i2], deid_source[j1:j2]))

    return AlignmentResult(
        predicted_spans=predicted_spans,
        corruptions=corruptions,
        success=not alignment_errors,
        alignment_errors=alignment_errors,
    )
