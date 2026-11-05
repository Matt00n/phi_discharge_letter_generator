"""Shared character-span utilities for text-replacement anonymizers.

Anonymizers detect PII as ``(start, end)`` character regions and then splice the
redaction marker in their place. Merging and splicing are identical across
methods, so they live here.
"""

from __future__ import annotations

from experiments.benchmark.prompts import REDACTION_MARKER


def merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Sort and merge overlapping/touching spans, dropping empty ones."""
    cleaned = sorted(s for s in spans if s[1] > s[0])
    if not cleaned:
        return []
    merged = [cleaned[0]]
    for start, end in cleaned[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def redact(text: str, spans: list[tuple[int, int]]) -> str:
    """Replace each (already merged, sorted) span with the redaction marker."""
    out: list[str] = []
    cursor = 0
    for start, end in spans:
        out.append(text[cursor:start])
        out.append(REDACTION_MARKER)
        cursor = end
    out.append(text[cursor:])
    return "".join(out)
