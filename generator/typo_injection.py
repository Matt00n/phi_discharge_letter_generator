"""Post-render typo/corruption injection for adversarial cases.

Introduces realistic typos, OCR artifacts, and grammatical noise into the
rendered text while adjusting gold spans accordingly. This tests whether
anonymizers rely on exact pattern matching or can handle noisy real-world input.

Two independently sampled modes:
1. PII-targeted typos: corrupt specific PII spans (harder to detect)
2. General text noise: typos/grammar errors throughout (simulates messy input)

Works with render.Span (frozen dataclass) — produces new Span objects with
adjusted offsets rather than mutating in place.
"""

from __future__ import annotations

import random

from generator.render import RenderResult, Span


# --- Character-level mutations (realistic German typos + OCR) ---------------

# Common German keyboard adjacency typos
_KEYBOARD_NEIGHBORS: dict[str, str] = {
    "a": "sqwed",
    "b": "vghn",
    "c": "xdfv",
    "d": "sfcxer",
    "e": "wrd",
    "f": "dgcvrt",
    "g": "fhbvty",
    "h": "gjbnzu",
    "i": "ujko",
    "j": "hknui",
    "k": "jlmio",
    "l": "köop",
    "m": "njk",
    "n": "bhjm",
    "o": "iklp",
    "p": "olü",
    "q": "wa",
    "r": "edt",
    "s": "awdxe",
    "t": "rfgz",
    "u": "zhji",
    "v": "cfgb",
    "w": "qase",
    "x": "ysc",
    "y": "xa",
    "z": "tuh",
    "ä": "öa",
    "ö": "älp",
    "ü": "op",
}

# OCR confusion pairs (char -> possible OCR misread)
_OCR_CONFUSIONS: dict[str, list[str]] = {
    "0": ["O", "o", "Q"],
    "O": ["0", "Q"],
    "o": ["0", "O"],
    "1": ["l", "I", "|"],
    "l": ["1", "I", "|"],
    "I": ["1", "l", "|"],
    "5": ["S", "s"],
    "S": ["5"],
    "8": ["B"],
    "B": ["8"],
    "m": ["rn", "nn"],
    "rn": ["m"],
    "ü": ["u", "ii"],
    "ä": ["a", "ae"],
    "ö": ["o", "oe"],
    "ß": ["ss", "B"],
}


def _keyboard_typo(char: str, rng: random.Random) -> str:
    """Replace a character with an adjacent keyboard key."""
    lower = char.lower()
    neighbors = _KEYBOARD_NEIGHBORS.get(lower, "")
    if not neighbors:
        return char
    replacement = rng.choice(neighbors)
    return replacement.upper() if char.isupper() else replacement


def _ocr_corruption(char: str, rng: random.Random) -> str:
    """Replace a character with an OCR confusion."""
    replacements = _OCR_CONFUSIONS.get(char, [])
    if not replacements:
        return char
    return rng.choice(replacements)


# --- Text-level mutations ---------------------------------------------------


def _mutate_char(
    char: str, rng: random.Random, mode_weights: list[float] | None = None
) -> tuple[str, str]:
    """Pick a random mutation for a character. Returns (new_chars, mode_used)."""
    mode = rng.choices(
        ["keyboard", "ocr", "double", "drop"],
        weights=mode_weights if mode_weights is not None else [0.40, 0.25, 0.20, 0.15],
        k=1,
    )[0]
    if mode == "keyboard":
        return _keyboard_typo(char, rng), mode
    elif mode == "ocr":
        return _ocr_corruption(char, rng), mode
    elif mode == "double":
        return char + char, mode
    else:  # drop
        return "", mode


def _shift_span(span: Span, delta: int) -> Span:
    """Create a new Span shifted by delta positions."""
    return Span(
        start=span.start + delta,
        end=span.end + delta,
        label=span.label,
        tier=span.tier,
        entity_id=span.entity_id,
        form=span.form,
        case=span.case,
    )


def _apply_mutations_rtl(
    text: str,
    mutations: list[tuple[int, str]],
    spans: list[Span],
    negative_spans: list[Span],
) -> tuple[str, list[Span], list[Span]]:
    """Apply character mutations right-to-left to preserve earlier offsets.

    mutations: list of (position, replacement_string) sorted by position desc.
    Returns (new_text, adjusted_spans, adjusted_negative_spans).
    """
    chars = list(text)
    # Collect (position, length_change) for span adjustment
    adjustments: list[tuple[int, int]] = []

    for pos, replacement in mutations:
        old_len = 1
        new_len = len(replacement)
        chars[pos : pos + old_len] = list(replacement)
        if new_len != old_len:
            adjustments.append((pos, new_len - old_len))

    new_text = "".join(chars)

    # Sort adjustments by position ascending for cumulative shift
    adjustments.sort(key=lambda x: x[0])

    def _adjust(span_list: list[Span]) -> list[Span]:
        result = []
        for sp in span_list:
            shift = 0
            end_shift = 0
            for adj_pos, adj_len in adjustments:
                if adj_pos < sp.start:
                    shift += adj_len
                elif adj_pos < sp.end:
                    # Mutation inside this span — adjust end only
                    end_shift += adj_len
            result.append(
                Span(
                    start=sp.start + shift,
                    end=sp.end + shift + end_shift,
                    label=sp.label,
                    tier=sp.tier,
                    entity_id=sp.entity_id,
                    form=sp.form,
                    case=sp.case,
                )
            )
        return result

    return new_text, _adjust(spans), _adjust(negative_spans)


def inject_pii_typos(
    text: str,
    spans: list[Span],
    negative_spans: list[Span],
    rng: random.Random,
    n_corruptions: int = 2,
    mode_weights: list[float] | None = None,
) -> tuple[str, list[Span], list[Span]]:
    """Corrupt a few PII spans with realistic typos.

    Selects up to n_corruptions spans (name-like, long enough) and introduces
    one typo each. Gold span boundaries are updated to cover the corrupted text.
    """
    # Only corrupt spans that are long enough and contain alphabetic chars
    candidates = [
        (i, sp)
        for i, sp in enumerate(spans)
        if (sp.end - sp.start) >= 3
        and any(c.isalpha() for c in text[sp.start : sp.end])
    ]
    if not candidates:
        return text, spans, negative_spans

    k = min(n_corruptions, len(candidates))
    chosen = rng.sample(candidates, k)

    # Build mutations: one per chosen span, sorted by position descending
    mutations: list[tuple[int, str]] = []
    for _idx, sp in sorted(chosen, key=lambda x: x[1].start, reverse=True):
        span_text = text[sp.start : sp.end]
        # Pick an alphabetic position to corrupt (avoid first char to keep capitalization)
        alpha_positions = [
            i
            for i in range(1 if len(span_text) > 2 else 0, len(span_text))
            if span_text[i].isalpha()
        ]
        if not alpha_positions:
            continue
        pos_in_span = rng.choice(alpha_positions)
        abs_pos = sp.start + pos_in_span
        char = text[abs_pos]
        new_char, _mode = _mutate_char(char, rng, mode_weights)
        mutations.append((abs_pos, new_char))

    return _apply_mutations_rtl(text, mutations, spans, negative_spans)


def inject_general_typos(
    text: str,
    spans: list[Span],
    negative_spans: list[Span],
    rng: random.Random,
    density: float = 0.02,
    mode_weights: list[float] | None = None,
) -> tuple[str, list[Span], list[Span]]:
    """Inject typos throughout the text OUTSIDE of gold spans.

    density: approximate fraction of non-span characters to corrupt.
    Span offsets are adjusted for any length changes.
    """
    # Identify positions inside spans (protected)
    protected = set()
    for sp in spans:
        for i in range(sp.start, sp.end):
            protected.add(i)
    for sp in negative_spans:
        for i in range(sp.start, sp.end):
            protected.add(i)

    # Candidate positions: outside spans, alphabetic
    candidates = [
        i for i in range(len(text)) if i not in protected and text[i].isalpha()
    ]
    if not candidates:
        return text, spans, negative_spans

    n_typos = max(1, int(len(candidates) * density))
    positions = rng.sample(candidates, min(n_typos, len(candidates)))

    # Build mutations sorted by position descending
    mutations: list[tuple[int, str]] = []
    for pos in sorted(positions, reverse=True):
        char = text[pos]
        new_char, _mode = _mutate_char(char, rng, mode_weights)
        mutations.append((pos, new_char))

    return _apply_mutations_rtl(text, mutations, spans, negative_spans)


def apply_typo_injection(
    result: RenderResult,
    rng: random.Random,
    *,
    typo_density: float = 0.0,
    pii_typo_prob: float = 0.0,
    n_pii_corruptions: int = 2,
    mode_weights: list[float] | None = None,
) -> RenderResult:
    """Top-level entry point: apply PII and/or general typos to a RenderResult.

    Both modes are sampled independently:
    - pii_typo_prob: probability of corrupting PII spans (0.0-1.0)
    - typo_density: density of general text typos (0.0 = none)
    - mode_weights: character-mutation mode weights (keyboard/ocr/double/drop)

    Returns a new RenderResult with corrupted text and adjusted spans.
    """
    text = result.text
    spans = list(result.spans)
    neg_spans = list(result.negative_spans)

    # 1. PII-targeted typos (independently gated)
    if pii_typo_prob > 0 and rng.random() < pii_typo_prob:
        text, spans, neg_spans = inject_pii_typos(
            text,
            spans,
            neg_spans,
            rng,
            n_corruptions=n_pii_corruptions,
            mode_weights=mode_weights,
        )

    # 2. General text typos (independently gated by density > 0)
    if typo_density > 0:
        text, spans, neg_spans = inject_general_typos(
            text,
            spans,
            neg_spans,
            rng,
            density=typo_density,
            mode_weights=mode_weights,
        )

    return RenderResult(text=text, spans=spans, negative_spans=neg_spans)
