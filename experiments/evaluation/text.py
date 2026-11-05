"""Shared Unicode-aware text preprocessing primitives."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence

WORD_PATTERN = re.compile(r"[^\W_]+(?:[-'’][^\W_]+)*", re.UNICODE)
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])(?:[\"')\]]*)\s+|\n+")
WHITESPACE = re.compile(r"\s+")


def normalize_unicode(text: str) -> str:
    """Apply canonical compatibility normalization."""

    return unicodedata.normalize("NFKC", text)


def normalize_for_deduplication(text: str) -> str:
    """Normalize case and whitespace while retaining punctuation and numbers."""

    return WHITESPACE.sub(" ", normalize_unicode(text).casefold()).strip()


def word_tokens(text: str, *, casefold: bool = True) -> list[str]:
    """Tokenize Unicode letter/number words with internal hyphens/apostrophes."""

    normalized = normalize_unicode(text)
    tokens = [match.group(0) for match in WORD_PATTERN.finditer(normalized)]
    return [token.casefold() for token in tokens] if casefold else tokens


def sentence_strings(text: str) -> list[str]:
    """Deterministic fallback segmentation used only without a spaCy pipeline."""

    return [part.strip() for part in SENTENCE_BOUNDARY.split(text) if part.strip()]


def ngrams(tokens: Sequence[str], n: int) -> Iterable[tuple[str, ...]]:
    """Yield contiguous n-grams without crossing document boundaries."""

    if n <= 0:
        raise ValueError("n must be positive")
    return zip(*(tokens[offset:] for offset in range(n)), strict=False)
