"""Span matching logic: strict, lenient (IoU), and token-level.

All matching is detection-only (no label classification by the prediction
system). Gold labels are preserved on matched pairs for downstream analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class Span:
    """A character-offset span."""

    start: int
    end: int
    label: str = ""  # Gold label (empty for predictions)
    entity_id: str = ""
    tier: str = ""
    form: str = ""


@dataclass
class MatchResult:
    """Result of matching predictions against gold for a single document."""

    doc_id: str
    true_positives: list[tuple[Span, Span]] = field(
        default_factory=list
    )  # (gold, pred)
    false_negatives: list[Span] = field(default_factory=list)  # missed gold
    false_positives: list[Span] = field(default_factory=list)  # spurious pred


def _iou(a: Span, b: Span) -> float:
    """Character-level Intersection over Union."""
    overlap_start = max(a.start, b.start)
    overlap_end = min(a.end, b.end)
    intersection = max(0, overlap_end - overlap_start)
    if intersection == 0:
        return 0.0
    union = (a.end - a.start) + (b.end - b.start) - intersection
    return intersection / union if union > 0 else 0.0


def _overlap(a: Span, b: Span) -> int:
    """Character overlap between two spans."""
    return max(0, min(a.end, b.end) - max(a.start, b.start))


def match_strict(
    gold_spans: Sequence[Span],
    pred_spans: Sequence[Span],
    doc_id: str = "",
) -> MatchResult:
    """Strict matching: exact start and end must match."""
    pred_set = {(s.start, s.end) for s in pred_spans}
    gold_set = {(s.start, s.end) for s in gold_spans}

    tp_keys = pred_set & gold_set
    gold_by_pos = {(s.start, s.end): s for s in gold_spans}
    pred_by_pos = {(s.start, s.end): s for s in pred_spans}

    tps = [(gold_by_pos[k], pred_by_pos[k]) for k in tp_keys]
    fns = [s for s in gold_spans if (s.start, s.end) not in tp_keys]
    fps = [s for s in pred_spans if (s.start, s.end) not in tp_keys]

    return MatchResult(
        doc_id=doc_id, true_positives=tps, false_negatives=fns, false_positives=fps
    )


def match_lenient(
    gold_spans: Sequence[Span],
    pred_spans: Sequence[Span],
    doc_id: str = "",
    iou_threshold: float = 0.5,
) -> MatchResult:
    """Lenient matching: IoU >= threshold counts as match.

    Uses maximum-cardinality bipartite matching (Kuhn's augmenting-path
    algorithm) so that the globally optimal set of TP pairs is found.
    Adjacency lists are sorted by descending IoU so that among all
    maximum-cardinality solutions the one with the highest total IoU is
    preferred.
    """
    if not gold_spans or not pred_spans:
        return MatchResult(
            doc_id=doc_id,
            true_positives=[],
            false_negatives=list(gold_spans),
            false_positives=list(pred_spans),
        )

    n_g = len(gold_spans)
    n_p = len(pred_spans)

    # Build adjacency: for each gold index, eligible pred indices sorted by
    # descending IoU so augmenting paths prefer higher-quality matches.
    adj: list[list[int]] = []
    for g in gold_spans:
        eligible = sorted(
            (j for j, p in enumerate(pred_spans) if _iou(g, p) >= iou_threshold),
            key=lambda j: _iou(g, pred_spans[j]),
            reverse=True,
        )
        adj.append(eligible)

    # Kuhn's algorithm: match_p[j] = gold index currently matched to pred j
    match_p: list[int] = [-1] * n_p

    def _augment(g_idx: int, visited: list[bool]) -> bool:
        for p_idx in adj[g_idx]:
            if visited[p_idx]:
                continue
            visited[p_idx] = True
            if match_p[p_idx] == -1 or _augment(match_p[p_idx], visited):
                match_p[p_idx] = g_idx
                return True
        return False

    for g_idx in range(n_g):
        _augment(g_idx, [False] * n_p)

    # Collect results
    matched_gold: set[int] = set()
    tps: list[tuple[Span, Span]] = []
    for p_idx, g_idx in enumerate(match_p):
        if g_idx >= 0:
            tps.append((gold_spans[g_idx], pred_spans[p_idx]))
            matched_gold.add(g_idx)

    fns = [gold_spans[i] for i in range(n_g) if i not in matched_gold]
    matched_pred = {p_idx for p_idx, g_idx in enumerate(match_p) if g_idx >= 0}
    fps = [pred_spans[j] for j in range(n_p) if j not in matched_pred]

    return MatchResult(
        doc_id=doc_id, true_positives=tps, false_negatives=fns, false_positives=fps
    )


def match_token_level(
    text: str,
    gold_spans: Sequence[Span],
    pred_spans: Sequence[Span],
    doc_id: str = "",
) -> MatchResult:
    """Token-level matching: split text by whitespace, classify each token.

    A token is gold-positive if any gold span overlaps it.
    A token is pred-positive if any pred span overlaps it.
    """
    # Tokenize by whitespace
    tokens: list[tuple[int, int]] = []
    i = 0
    while i < len(text):
        # Skip whitespace
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text):
            break
        start = i
        while i < len(text) and not text[i].isspace():
            i += 1
        tokens.append((start, i))

    def _spans_cover_token(spans: Sequence[Span], tok_start: int, tok_end: int) -> bool:
        for s in spans:
            if s.start < tok_end and s.end > tok_start:
                return True
        return False

    tp_tokens: list[tuple[Span, Span]] = []
    fn_tokens: list[Span] = []
    fp_tokens: list[Span] = []

    for tok_start, tok_end in tokens:
        in_gold = _spans_cover_token(gold_spans, tok_start, tok_end)
        in_pred = _spans_cover_token(pred_spans, tok_start, tok_end)
        tok_span = Span(tok_start, tok_end)

        if in_gold and in_pred:
            tp_tokens.append((tok_span, tok_span))
        elif in_gold and not in_pred:
            fn_tokens.append(tok_span)
        elif not in_gold and in_pred:
            fp_tokens.append(tok_span)

    return MatchResult(
        doc_id=doc_id,
        true_positives=tp_tokens,
        false_negatives=fn_tokens,
        false_positives=fp_tokens,
    )


def boundary_quality(match_result: MatchResult) -> dict:
    """Compute boundary quality metrics for lenient matches."""
    if not match_result.true_positives:
        return {
            "mean_iou": 0.0,
            "mean_start_offset": 0.0,
            "mean_end_offset": 0.0,
            "over_extension_rate": 0.0,
            "under_extension_rate": 0.0,
        }

    ious = []
    start_offsets = []
    end_offsets = []
    over_extended = 0
    under_extended = 0

    for gold, pred in match_result.true_positives:
        ious.append(_iou(gold, pred))
        start_offsets.append(abs(pred.start - gold.start))
        end_offsets.append(abs(pred.end - gold.end))
        pred_len = pred.end - pred.start
        gold_len = gold.end - gold.start
        if pred_len > gold_len:
            over_extended += 1
        elif pred_len < gold_len:
            under_extended += 1

    n = len(match_result.true_positives)
    return {
        "mean_iou": sum(ious) / n,
        "mean_start_offset": sum(start_offsets) / n,
        "mean_end_offset": sum(end_offsets) / n,
        "over_extension_rate": over_extended / n,
        "under_extension_rate": under_extended / n,
    }
