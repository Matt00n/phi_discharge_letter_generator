"""Render review_queue.jsonl into a human-readable exclusions report (.md).

One section per excluded document. For each deterministic leak and each
red-team finding we quote only the relevant sentence (or a tight window around
the offending span) rather than the full letter. Negative-integrity issues and
missing planned persons have no in-text location and are listed without a quote.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CORPUS = "sparcling-de"

MAX_SENTENCE = 320  # if the enclosing sentence is longer, fall back to a window
CTX = 140           # chars of context each side in window mode

_BOUND_BEFORE = re.compile(r"[.!?]\s|\n")
_BOUND_AFTER = re.compile(r"[.!?](\s|$)|\n")


def resolve_span(text: str, surface: str, start: int, end: int) -> tuple[int, int] | None:
    """Return trustworthy char offsets for `surface` in `text`.

    Deterministic-leak offsets are already correct; red-team offsets are stale
    (they index an earlier render), so when the stored offsets don't match we
    relocate to the occurrence of `surface` nearest the stored start (handles
    duplicates like repeated dates).
    """
    if surface and 0 <= start < end <= len(text) and text[start:end] == surface:
        return start, end
    if not surface:
        return None
    positions = []
    i = text.find(surface)
    while i != -1:
        positions.append(i)
        i = text.find(surface, i + 1)
    if not positions:
        return None
    best = min(positions, key=lambda p: abs(p - start))
    return best, best + len(surface)


def relevant_quote(text: str, start: int, end: int) -> str:
    """Return the sentence containing [start, end), with the span **bolded**.

    Falls back to a truncated ±CTX window (with ellipses) for very long
    sentences so the report stays readable.
    """
    if not (0 <= start < end <= len(text)):
        return ""
    left = 0
    for m in _BOUND_BEFORE.finditer(text, 0, start):
        left = m.end()
    m = _BOUND_AFTER.search(text, end)
    right = m.start() + 1 if m else len(text)

    lead = trail = ""
    if right - left > MAX_SENTENCE:
        new_left = max(left, start - CTX)
        new_right = min(right, end + CTX)
        lead = "… " if new_left > left else ""
        trail = " …" if new_right < right else ""
        left, right = new_left, new_right

    rel_s, rel_e = start - left, end - left
    seg = text[left:right]
    seg = seg[:rel_s] + "**" + seg[rel_s:rel_e] + "**" + seg[rel_e:]
    seg = " ".join(seg.split())  # collapse newlines/whitespace
    return f"{lead}{seg}{trail}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--corpus",
        default=DEFAULT_CORPUS,
        help="corpus dir under experiments/datasets/ (default: %(default)s)",
    )
    ap.add_argument(
        "--in-path",
        type=Path,
        default=None,
        help="override review_queue.jsonl path (default: datasets/<corpus>/...)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="override output .md path (default: results/exclusions_<corpus>.md)",
    )
    args = ap.parse_args()

    slug = args.corpus.replace("-", "_")
    in_path = args.in_path or (HERE / "datasets" / args.corpus / "review_queue.jsonl")
    out_path = args.out or (HERE / "results" / f"exclusions_{slug}.md")

    with open(in_path, encoding="utf-8") as fh:
        recs = [json.loads(line) for line in fh]
    recs.sort(key=lambda r: r["doc_id"])

    det = sum(1 for r in recs if r.get("leaks") or r.get("negative_findings") or r.get("missing_planned_persons"))
    rt = sum(1 for r in recs if (r.get("red_team") or {}).get("review_items"))

    out: list[str] = []
    out.append(f"# QC exclusions — {len(recs)} docs removed by auto mode\n")
    out.append(
        "_Each was flagged by the deterministic check and/or the red-team pass and "
        "excluded from the dataset. For every located finding the relevant sentence is "
        "quoted (offending span in **bold**); the full letters are intentionally omitted._\n"
    )
    out.append(
        f"_Documents with a deterministic finding: {det}. Documents with a red-team "
        f"finding: {rt}._\n"
    )

    for r in recs:
        text = r["text"]
        out.append("\n---\n")
        out.append(f"## {r['doc_id']} — {r.get('difficulty', '?')} · {r.get('style', '?')}\n")
        reasons = "; ".join(r.get("reasons", [])) or "—"
        out.append(f"**Reasons:** {reasons}\n")

        leaks = r.get("leaks") or []
        if leaks:
            out.append(f"\n**Deterministic PII leaks ({len(leaks)})** — PII surface outside gold spans:\n")
            for lk in leaks:
                span = resolve_span(text, lk["surface"], lk["start"], lk["end"])
                loc = f", @{span[0]}–{span[1]}" if span else ""
                out.append(
                    f"- `{lk['surface']}` (label={lk.get('label', '?')}, "
                    f"entity={lk.get('entity_id', '?')}{loc})"
                )
                if span:
                    q = relevant_quote(text, *span)
                    if q:
                        out.append(f"  > {q}")

        negs = r.get("negative_findings") or []
        if negs:
            out.append(f"\n**Negative-integrity issues ({len(negs)})** — planted lookalikes/eponyms:\n")
            for ng in negs:
                out.append(
                    f"- `{ng['surface']}` (label={ng.get('label', '?')}, "
                    f"entity={ng.get('entity_id', '?')}, reason={ng.get('reason', '?')})"
                )

        mpp = r.get("missing_planned_persons") or []
        if mpp:
            out.append(f"\n**Missing planned persons ({len(mpp)})** — planned entity absent from text:\n")
            for pid in mpp:
                out.append(f"- `{pid}`")

        items = (r.get("red_team") or {}).get("review_items") or []
        if items:
            out.append(f"\n**Red-team findings ({len(items)})** — model flagged as un-annotated PII:\n")
            for it in items:
                cat = it.get("category", "?")
                reasoning = it.get("reasoning", "")
                span = resolve_span(text, it.get("text", ""), it.get("start") or 0, it.get("end") or 0)
                loc = f", @{span[0]}–{span[1]}" if span else ""
                tail = f" — {reasoning}" if reasoning else ""
                out.append(f"- `{it.get('text', '')}` (category={cat}{loc}){tail}")
                if span:
                    q = relevant_quote(text, *span)
                    if q:
                        out.append(f"  > {q}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"Wrote {out_path} ({len(recs)} docs; det={det}, red-team={rt}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
