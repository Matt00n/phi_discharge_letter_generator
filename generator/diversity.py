"""Stage 5b: diversity audit.

Computes holistic diversity metrics across six dimensions:

A. **Lexical** — TTR, inter-document Jaccard (sampled), n-gram novelty, hapax ratio
B. **Structural** — section pattern entropy, structure mode, mean sections/doc
C. **PII surface** — unique surfaces/entities ratio, form entropy per label, density
D. **Negative (trap)** — unique eponym/lookalike/nonpii_id surfaces, density
E. **Clinical content** — ICD/OPS uniqueness, fachabteilung entropy
F. **Aggregate** — single diversity score (harmonic mean of normalized entropies)

All metrics are computable from the standard record dicts produced by the pipeline.
Expensive pairwise operations (Jaccard) are sampled for scalability.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from typing import Any, Iterable


def _entropy(counter: Counter) -> float:
    """Shannon entropy in bits."""
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counter.values() if c)


def _max_entropy(n_categories: int) -> float:
    """Maximum possible entropy for n equally-likely categories."""
    if n_categories <= 1:
        return 0.0
    return math.log2(n_categories)


def _normalized_entropy(counter: Counter) -> float:
    """Entropy normalized to [0, 1] (1 = perfectly uniform)."""
    h = _entropy(counter)
    h_max = _max_entropy(len(counter))
    return h / h_max if h_max > 0 else 0.0


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer for German clinical text."""
    return re.findall(r"\b\w+\b", text.lower())


def _trigrams(tokens: list[str]) -> set[tuple[str, str, str]]:
    """Extract word trigrams from a token list."""
    return {(tokens[i], tokens[i + 1], tokens[i + 2]) for i in range(len(tokens) - 2)}


def _jaccard(set_a: set, set_b: set) -> float:
    """Jaccard distance (1 - similarity). Range [0, 1], higher = more different."""
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return 1.0 - (intersection / union) if union else 0.0


# ---------------------------------------------------------------------------
# A. Lexical diversity
# ---------------------------------------------------------------------------


def _lexical_metrics(
    records: list[dict], *, jaccard_sample_size: int = 500
) -> dict[str, Any]:
    """Compute lexical diversity metrics across all documents."""
    doc_tokens: list[list[str]] = []
    corpus_counter: Counter = Counter()

    for rec in records:
        tokens = _tokenize(rec.get("text", ""))
        doc_tokens.append(tokens)
        corpus_counter.update(tokens)

    n = len(doc_tokens)

    # Type-Token Ratio per document
    ttrs: list[float] = []
    for tokens in doc_tokens:
        if tokens:
            ttrs.append(len(set(tokens)) / len(tokens))
    ttrs.sort()

    # Hapax legomena (words appearing exactly once in entire corpus)
    total_types = len(corpus_counter)
    hapax = sum(1 for c in corpus_counter.values() if c == 1)
    hapax_ratio = hapax / total_types if total_types else 0.0

    # N-gram novelty: fraction of trigrams in each doc NOT seen in any other doc
    doc_trigram_sets = [_trigrams(t) for t in doc_tokens]
    # Build corpus trigram counts
    trigram_doc_count: Counter = Counter()
    for tset in doc_trigram_sets:
        for tri in tset:
            trigram_doc_count[tri] += 1
    # Per-doc: fraction of trigrams unique to this doc
    novelty_scores: list[float] = []
    for tset in doc_trigram_sets:
        if not tset:
            continue
        unique_to_doc = sum(1 for tri in tset if trigram_doc_count[tri] == 1)
        novelty_scores.append(unique_to_doc / len(tset))

    # Inter-document Jaccard distance (sampled pairs for scalability)
    jaccard_distances: list[float] = []
    if n >= 2:
        rng = random.Random(42)
        n_pairs = min(jaccard_sample_size, n * (n - 1) // 2)
        token_sets = [set(t) for t in doc_tokens]
        pairs_seen: set[tuple[int, int]] = set()
        attempts = 0
        while len(jaccard_distances) < n_pairs and attempts < n_pairs * 3:
            i, j = rng.sample(range(n), 2)
            pair = (min(i, j), max(i, j))
            if pair not in pairs_seen:
                pairs_seen.add(pair)
                jaccard_distances.append(_jaccard(token_sets[i], token_sets[j]))
            attempts += 1

    def _pct(vals: list[float], p: float) -> float:
        if not vals:
            return 0.0
        return vals[min(len(vals) - 1, int(p * len(vals)))]

    return {
        "ttr": {
            "median": round(_pct(ttrs, 0.5), 4),
            "p10": round(_pct(ttrs, 0.1), 4),
            "p90": round(_pct(ttrs, 0.9), 4),
        },
        "hapax_ratio": round(hapax_ratio, 4),
        "corpus_vocabulary_size": total_types,
        "trigram_novelty": {
            "mean": round(sum(novelty_scores) / len(novelty_scores), 4)
            if novelty_scores
            else 0.0,
            "median": round(_pct(sorted(novelty_scores), 0.5), 4)
            if novelty_scores
            else 0.0,
        },
        "inter_document_jaccard": {
            "mean": round(sum(jaccard_distances) / len(jaccard_distances), 4)
            if jaccard_distances
            else 0.0,
            "p10": round(_pct(sorted(jaccard_distances), 0.1), 4),
            "p90": round(_pct(sorted(jaccard_distances), 0.9), 4),
            "n_pairs_sampled": len(jaccard_distances),
        },
    }


# ---------------------------------------------------------------------------
# B. Structural diversity
# ---------------------------------------------------------------------------


def _structural_metrics(records: list[dict]) -> dict[str, Any]:
    """Section patterns, structure modes, and document complexity."""
    section_patterns: Counter = Counter()
    structure_modes: Counter = Counter()
    code_styles: Counter = Counter()
    opening_styles: Counter = Counter()
    section_counts: list[int] = []

    for rec in records:
        meta = rec.get("metadata", {})
        sections = meta.get("sections", [])
        section_counts.append(len(sections))
        # Section pattern = ordered tuple of section names
        section_patterns[tuple(sections)] += 1
        # Structure mode from difficulty tags
        tags = meta.get("difficulty_tags", [])
        if "unstructured_text" in tags:
            structure_modes["unstructured_or_mixed"] += 1
        else:
            structure_modes["structured"] += 1
        # Presentation register (code citation + opening formula)
        code_styles[meta.get("code_style", "inline")] += 1
        opening_styles[meta.get("opening_style", "bericht")] += 1

    section_counts.sort()

    return {
        "unique_section_patterns": len(section_patterns),
        "section_pattern_entropy_bits": round(_entropy(section_patterns), 3),
        "structure_mode_distribution": dict(structure_modes),
        "code_style_distribution": dict(code_styles),
        "code_style_entropy_norm": round(_normalized_entropy(code_styles), 3),
        "opening_style_distribution": dict(opening_styles),
        "opening_style_entropy_norm": round(_normalized_entropy(opening_styles), 3),
        "sections_per_doc": {
            "mean": round(sum(section_counts) / len(section_counts), 2)
            if section_counts
            else 0,
            "std": round(
                (
                    sum(
                        (x - sum(section_counts) / len(section_counts)) ** 2
                        for x in section_counts
                    )
                    / len(section_counts)
                )
                ** 0.5,
                2,
            )
            if section_counts
            else 0,
            "min": section_counts[0] if section_counts else 0,
            "max": section_counts[-1] if section_counts else 0,
        },
    }


# ---------------------------------------------------------------------------
# C. PII surface diversity
# ---------------------------------------------------------------------------


def _pii_surface_metrics(records: list[dict]) -> dict[str, Any]:
    """PII surface form diversity and density."""
    # Collect all span surfaces and their labels/forms
    surfaces_per_label: defaultdict[str, Counter] = defaultdict(Counter)
    forms_per_label: defaultdict[str, Counter] = defaultdict(Counter)
    entity_ids: set[str] = set()
    all_surfaces: set[str] = set()
    spans_per_doc: list[int] = []

    for rec in records:
        text = rec.get("text", "")
        doc_spans = rec.get("spans", [])
        spans_per_doc.append(len(doc_spans))
        for sp in doc_spans:
            surface = text[sp["start"] : sp["end"]]
            label = sp["label"]
            surfaces_per_label[label][surface] += 1
            all_surfaces.add(surface)
            entity_ids.add(sp.get("entity_id", ""))
            if sp.get("form"):
                forms_per_label[label][sp["form"]] += 1

    # Form entropy per label (how evenly are forms used?)
    form_entropy_per_label = {}
    for label, form_counter in sorted(forms_per_label.items()):
        form_entropy_per_label[label] = {
            "entropy_bits": round(_entropy(form_counter), 3),
            "n_forms_used": len(form_counter),
            "distribution": dict(form_counter.most_common(8)),
        }

    # Surface/entity ratio
    n_entities = len(entity_ids)
    n_surfaces = len(all_surfaces)

    spans_per_doc.sort()

    return {
        "unique_pii_surfaces": n_surfaces,
        "unique_entity_ids": n_entities,
        "surface_to_entity_ratio": round(n_surfaces / n_entities, 2)
        if n_entities
        else 0,
        "form_entropy_per_label": form_entropy_per_label,
        "spans_per_doc": {
            "mean": round(sum(spans_per_doc) / len(spans_per_doc), 2)
            if spans_per_doc
            else 0,
            "std": round(
                (
                    sum(
                        (x - sum(spans_per_doc) / len(spans_per_doc)) ** 2
                        for x in spans_per_doc
                    )
                    / len(spans_per_doc)
                )
                ** 0.5,
                2,
            )
            if spans_per_doc
            else 0,
            "min": spans_per_doc[0] if spans_per_doc else 0,
            "max": spans_per_doc[-1] if spans_per_doc else 0,
        },
    }


# ---------------------------------------------------------------------------
# D. Negative (trap) diversity
# ---------------------------------------------------------------------------


def _negative_metrics(records: list[dict]) -> dict[str, Any]:
    """Diversity of eponym / lookalike / non-PII ID traps."""
    surfaces_by_label: defaultdict[str, set] = defaultdict(set)
    negatives_per_doc: list[int] = []

    for rec in records:
        text = rec.get("text", "")
        negs = rec.get("negatives", [])
        negatives_per_doc.append(len(negs))
        for neg in negs:
            surface = text[neg["start"] : neg["end"]]
            surfaces_by_label[neg["label"]].add(surface)

    return {
        "unique_eponym_surfaces": len(surfaces_by_label.get("EPONYM", set())),
        "unique_lookalike_surfaces": len(surfaces_by_label.get("LOOKALIKE", set())),
        "unique_nonpii_id_surfaces": len(surfaces_by_label.get("NONPII_ID", set())),
        "total_unique_negative_surfaces": sum(
            len(s) for s in surfaces_by_label.values()
        ),
        "negatives_per_doc": {
            "mean": round(sum(negatives_per_doc) / len(negatives_per_doc), 2)
            if negatives_per_doc
            else 0,
            "max": max(negatives_per_doc) if negatives_per_doc else 0,
            "docs_with_negatives": sum(1 for x in negatives_per_doc if x > 0),
        },
    }


# ---------------------------------------------------------------------------
# E. Clinical content diversity
# ---------------------------------------------------------------------------


def _clinical_metrics(records: list[dict]) -> dict[str, Any]:
    """Medical domain and clinical content diversity."""
    fachabteilung: Counter = Counter()
    diagnoses: set[str] = set()
    procedures: set[str] = set()

    for rec in records:
        meta = rec.get("metadata", {})
        fach = meta.get("fachabteilung", "?")
        fachabteilung[fach] += 1

        # Extract clinical case info if available (from metadata or text)
        # ICD/OPS codes in the text
        text = rec.get("text", "")
        for m in re.finditer(r"\b[A-Z]\d{2}(?:\.\d{1,2})?\b", text):
            diagnoses.add(m.group())
        for m in re.finditer(r"\b\d-\d{2,3}(?:\.\d{1,2})?\b", text):
            procedures.add(m.group())

    return {
        "fachabteilung_entropy_bits": round(_entropy(fachabteilung), 3),
        "fachabteilung_normalized_entropy": round(
            _normalized_entropy(fachabteilung), 4
        ),
        "unique_fachabteilungen": len(fachabteilung),
        "fachabteilung_distribution": dict(fachabteilung.most_common(15)),
        "unique_icd_codes_in_text": len(diagnoses),
        "unique_ops_codes_in_text": len(procedures),
    }


# ---------------------------------------------------------------------------
# F. Aggregate diversity score
# ---------------------------------------------------------------------------


def _aggregate_score(
    origin: Counter,
    surname: Counter,
    difficulty: Counter,
    style: Counter,
    domain: Counter,
    labels: Counter,
    forms: Counter,
    section_patterns: Counter,
) -> dict[str, Any]:
    """Aggregate diversity: harmonic mean of normalized entropies across dimensions.

    Score in [0, 1] where 1 = perfectly uniform across all dimensions.
    Penalizes any single collapsed dimension (harmonic mean property).
    """
    dimensions: dict[str, float] = {}

    for name, counter in [
        ("origin", origin),
        ("surname", surname),
        ("difficulty", difficulty),
        ("style", style),
        ("fachabteilung", domain),
        ("labels", labels),
        ("mention_forms", forms),
        ("section_patterns", section_patterns),
    ]:
        dimensions[name] = _normalized_entropy(counter)

    # Filter out dimensions with 0 (can't contribute to harmonic mean)
    active = {k: v for k, v in dimensions.items() if v > 0}

    if not active:
        score = 0.0
    else:
        # Harmonic mean
        score = len(active) / sum(1.0 / v for v in active.values())

    return {
        "aggregate_diversity_score": round(score, 4),
        "dimension_scores": {k: round(v, 4) for k, v in sorted(dimensions.items())},
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def audit(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Full diversity audit across all dimensions.

    Returns a nested dict suitable for writing to ``diversity.json``.
    """
    records = list(records)
    n = len(records)
    if n == 0:
        return {"n_documents": 0}

    # --- Collect base counters (shared across sections) --------------------
    origin: Counter = Counter()
    surname: Counter = Counter()
    difficulty: Counter = Counter()
    style: Counter = Counter()
    domain: Counter = Counter()
    labels: Counter = Counter()
    tiers: Counter = Counter()
    forms: Counter = Counter()
    tags: Counter = Counter()
    lengths: list[int] = []
    flagged = 0
    section_patterns: Counter = Counter()

    for rec in records:
        meta = rec.get("metadata", {})
        origin[meta.get("patient_origin", "?")] += 1
        surname[meta.get("patient_surname", "?")] += 1
        difficulty[meta.get("difficulty", "?")] += 1
        style[meta.get("style", "?")] += 1
        domain[meta.get("fachabteilung", "?")] += 1
        for t in meta.get("difficulty_tags", []):
            tags[t] += 1
        lengths.append(len(rec.get("text", "")))
        if rec.get("qc", {}).get("flagged"):
            flagged += 1
        for sp in rec.get("spans", []):
            labels[sp["label"]] += 1
            tiers[sp["tier"]] += 1
            if sp.get("form"):
                forms[sp["form"]] += 1
        sections = meta.get("sections", [])
        section_patterns[tuple(sections)] += 1

    lengths.sort()

    def pct(p: float) -> int:
        if not lengths:
            return 0
        return lengths[min(len(lengths) - 1, int(p * len(lengths)))]

    # --- Assemble output ---------------------------------------------------
    result = {
        "n_documents": n,
        "n_flagged": flagged,
        "flagged_fraction": round(flagged / n, 4) if n else 0.0,
        # --- Base distributions (original audit) ---------------------------
        "origin_distribution": dict(origin.most_common()),
        "origin_entropy_bits": round(_entropy(origin), 3),
        "unique_surnames": len(surname),
        "surname_entropy_bits": round(_entropy(surname), 3),
        "difficulty_distribution": dict(difficulty),
        "style_distribution": dict(style),
        "label_distribution": dict(labels.most_common()),
        "tier_distribution": dict(tiers),
        "mention_form_distribution": dict(forms.most_common()),
        "difficulty_tag_counts": dict(tags.most_common()),
        "doc_length_chars": {
            "min": lengths[0] if lengths else 0,
            "p25": pct(0.25),
            "p50": pct(0.5),
            "p75": pct(0.75),
            "p90": pct(0.9),
            "max": lengths[-1] if lengths else 0,
        },
        # --- A. Lexical diversity ------------------------------------------
        "lexical": _lexical_metrics(records),
        # --- B. Structural diversity ---------------------------------------
        "structural": _structural_metrics(records),
        # --- C. PII surface diversity --------------------------------------
        "pii_surface": _pii_surface_metrics(records),
        # --- D. Negative trap diversity ------------------------------------
        "negatives": _negative_metrics(records),
        # --- E. Clinical content diversity ---------------------------------
        "clinical": _clinical_metrics(records),
        # --- F. Aggregate diversity score ----------------------------------
        "aggregate": _aggregate_score(
            origin, surname, difficulty, style, domain, labels, forms, section_patterns
        ),
    }

    return result


def audit_by_splits(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Run ``audit()`` over the full dataset AND per split partition.

    Returns::

        {
            "overall": { ... full audit ... },
            "by_domain": { "train": {...}, "test": {...} },
            "by_difficulty": { "easy": {...}, "medium": {...}, ... },
            "by_surface": { "train": {...}, "test": {...} },
        }

    Records must already have ``metadata.splits`` assigned (via ``assign_splits``).
    Partitions with 0 documents are omitted.
    """
    records = list(records)

    result: dict[str, Any] = {"overall": audit(records)}

    # Group records by each split scheme
    split_schemes = ("by_domain", "by_difficulty", "by_surface")
    for scheme in split_schemes:
        buckets: defaultdict[str, list[dict]] = defaultdict(list)
        for rec in records:
            splits = rec.get("metadata", {}).get("splits", {})
            key = splits.get(scheme)
            if key:
                buckets[key].append(rec)
        if buckets:
            result[scheme] = {k: audit(v) for k, v in sorted(buckets.items())}

    return result
