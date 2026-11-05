"""Corpus loading, sliding-window chunking, and span<->token-label projection.

Everything here works in **character offsets** against the raw document text and
converts to/from wordpiece token labels only inside the tokenizer's chunk
windows. The binary label scheme is IO: every token is PII (1) or O (0);
adjacent PII tokens are merged back into character spans at prediction time.

We deliberately avoid the `datasets` library (not installed) — training examples
are plain dicts consumed by a small torch Dataset in `train.py`.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field

# Label ids for the binary token-classification head.
O_LABEL = 0
PII_LABEL = 1
ID2LABEL = {O_LABEL: "O", PII_LABEL: "PII"}
LABEL2ID = {"O": O_LABEL, "PII": PII_LABEL}
IGNORE_INDEX = -100  # special / padding tokens excluded from the loss


@dataclass
class Doc:
    """A document with character-offset gold PII spans.

    `spans` are (start, end, label) triples; `label` is the original fine-grained
    label kept only for sliced reporting — the model itself is binary.
    """

    doc_id: str
    text: str
    spans: list[tuple[int, int, str]]
    metadata: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Corpus loaders
# --------------------------------------------------------------------------- #
def load_synthetic(path: str, exclude_negatives: bool = True) -> list[Doc]:
    """Load a pipeline dataset.jsonl (SPARCLING-DE).

    Negatives (NONPII decoys: eponyms, lookalikes) are excluded from gold by
    default so the model is not taught to redact them.
    """
    docs: list[Doc] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            spans = [
                (s["start"], s["end"], s.get("label", "PII"))
                for s in rec.get("spans", [])
            ]
            if not exclude_negatives:
                spans += [
                    (s["start"], s["end"], s.get("label", "NEG"))
                    for s in rec.get("negatives", [])
                ]
            docs.append(
                Doc(
                    doc_id=rec["doc_id"],
                    text=rec["text"],
                    spans=spans,
                    metadata=rec.get("metadata", {}),
                )
            )
    return docs


# GRASCCO PHI kind -> coarse family, used only for sliced reporting.
_GRASCCO_FAMILY = {
    "NAME_PATIENT": "NAMES",
    "NAME_DOCTOR": "NAMES",
    "NAME_RELATIVE": "NAMES",
    "NAME_EXT": "NAMES",
    "NAME_USERNAME": "NAMES",
    "NAME_TITLE": "NAMES",
    "DATE": "DATES",
    "ID": "IDS",
    "LOCATION_CITY": "LOCATION",
    "LOCATION_ZIP": "LOCATION",
    "LOCATION_STREET": "LOCATION",
    "LOCATION_HOSPITAL": "LOCATION",
    "LOCATION_ORGANIZATION": "LOCATION",
    "LOCATION_COUNTRY": "LOCATION",
    "CONTACT_PHONE": "CONTACT",
    "CONTACT_FAX": "CONTACT",
    "CONTACT_EMAIL": "CONTACT",
    "AGE": "QUASI",
    "PROFESSION": "QUASI",
}


def load_grascco(directory: str) -> list[Doc]:
    """Load GRASCCO real German clinical letters (WebAnno UIMA CAS JSON).

    Each *_phi.json holds a `uima.cas.Sofa` (text) plus `webanno.custom.PHI`
    feature structures with begin/end/kind. All PHI kinds count as gold PII.
    """
    docs: list[Doc] = []
    for path in sorted(glob.glob(os.path.join(directory, "*_phi.json"))):
        with open(path, encoding="utf-8") as fh:
            cas = json.load(fh)
        fss = cas["%FEATURE_STRUCTURES"]
        sofa = next(x for x in fss if x.get("%TYPE") == "uima.cas.Sofa")
        text = sofa["sofaString"]
        spans: list[tuple[int, int, str]] = []
        for x in fss:
            if x.get("%TYPE") != "webanno.custom.PHI":
                continue
            b, e = x.get("begin"), x.get("end")
            if b is None or e is None or e <= b:
                continue
            kind = x.get("kind") or "PHI"
            spans.append((b, e, _GRASCCO_FAMILY.get(kind, kind)))
        doc_id = os.path.basename(path).replace("_phi.json", "").replace(".txt", "")
        docs.append(Doc(doc_id=doc_id, text=text, spans=spans, metadata={}))
    return docs


# --------------------------------------------------------------------------- #
# Filtering / subsampling helpers (for the ablations)
# --------------------------------------------------------------------------- #
def filter_adversarial(docs: list[Doc], keep_adversarial: bool) -> list[Doc]:
    """Keep or drop the adversarial-difficulty slice."""
    if keep_adversarial:
        return docs
    return [d for d in docs if d.metadata.get("difficulty") != "adversarial"]


def split_train_test(
    docs: list[Doc], test_frac: float = 0.1, seed: int = 42
) -> tuple[list[Doc], list[Doc]]:
    """Deterministic doc-level train/test split, stable across experiments.

    The split is keyed on a hash of (seed, doc_id) so the same held-out
    synthetic test set is used by every training run regardless of filtering or
    subsampling — a doc is never both trained on and evaluated.
    """
    import hashlib

    test, train = [], []
    for d in docs:
        h = hashlib.md5(f"{seed}:{d.doc_id}".encode()).hexdigest()
        frac = int(h[:8], 16) / 0xFFFFFFFF
        (test if frac < test_frac else train).append(d)
    return train, test


def kfold_split(
    docs: list[Doc], k: int, fold: int, seed: int = 42
) -> tuple[list[Doc], list[Doc]]:
    """Deterministic k-fold split; return (train, held-out) for `fold`.

    Fold membership is keyed on a hash of (seed, doc_id), so it is stable and
    order-independent, and the held-out folds are disjoint and cover the corpus.
    Used for GRASCCO cross-validation (only ~63 real docs, too few for a single
    train/test split), so `train.py` and `evaluate.py` must pass the same
    (k, fold, seed) to keep train and eval disjoint.
    """
    import hashlib

    train, held = [], []
    for d in docs:
        h = hashlib.md5(f"{seed}:{d.doc_id}".encode()).hexdigest()
        b = int(h[:8], 16) % k
        (held if b == fold else train).append(d)
    return train, held


def subsample(docs: list[Doc], n: int, seed: int = 0) -> list[Doc]:
    """Deterministic size-scaling subsample of at most `n` docs."""
    if n <= 0 or n >= len(docs):
        return docs
    import random

    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(docs)), n))
    return [docs[i] for i in idx]


# --------------------------------------------------------------------------- #
# Chunking + label projection
# --------------------------------------------------------------------------- #
def _char_pii_mask(length: int, spans: list[tuple[int, int, str]]) -> bytearray:
    mask = bytearray(length)
    for start, end, _ in spans:
        for i in range(max(0, start), min(length, end)):
            mask[i] = 1
    return mask


def _sliding_windows(
    tokenizer, text: str, max_length: int, stride: int
) -> dict[str, list[list]]:
    """Manually window a full tokenization into overlapping fixed-size chunks.

    We deliberately do NOT use the tokenizer's own `return_overflowing_tokens`
    sliding window: its behaviour is not stable across `transformers` versions
    (5.x returns only a *single* overflow segment for a long single sequence,
    silently dropping everything past ~one window — corrupting labels and eval
    on long documents). Instead we tokenize the whole document once and slice
    the token stream ourselves, which is correct and version-independent.

    Returns a dict of parallel per-chunk lists: input_ids, attention_mask,
    offset_mapping, sequence_ids (None on the added special tokens).
    """
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    offs = [tuple(o) for o in enc["offset_mapping"]]
    n = len(ids)

    cls = tokenizer.cls_token_id
    sep = tokenizer.sep_token_id
    if cls is None:
        cls = tokenizer.bos_token_id
    if sep is None:
        sep = tokenizer.eos_token_id
    n_special = int(cls is not None) + int(sep is not None)
    content = max(1, max_length - n_special)
    step = max(1, content - stride)

    chunks: dict[str, list[list]] = {
        "input_ids": [],
        "attention_mask": [],
        "offset_mapping": [],
        "sequence_ids": [],
    }
    start = 0
    while True:
        w_ids = ids[start : start + content]
        w_offs = offs[start : start + content]
        c_ids: list[int] = []
        c_offs: list[tuple[int, int]] = []
        c_seq: list = []
        if cls is not None:
            c_ids.append(cls)
            c_offs.append((0, 0))
            c_seq.append(None)
        c_ids.extend(w_ids)
        c_offs.extend(w_offs)
        c_seq.extend([0] * len(w_ids))
        if sep is not None:
            c_ids.append(sep)
            c_offs.append((0, 0))
            c_seq.append(None)
        chunks["input_ids"].append(c_ids)
        chunks["attention_mask"].append([1] * len(c_ids))
        chunks["offset_mapping"].append(c_offs)
        chunks["sequence_ids"].append(c_seq)
        if start + content >= n:
            break
        start += step
    return chunks


def encode_doc(
    doc: Doc,
    tokenizer,
    max_length: int = 512,
    stride: int = 128,
    with_labels: bool = True,
):
    """Tokenize one doc into overlapping chunks with offset mappings.

    Returns a plain dict with parallel per-chunk lists (input_ids,
    attention_mask, offset_mapping, sequence_ids) plus, when `with_labels`, a
    parallel list of per-token IO label lists (IGNORE_INDEX on special tokens).
    Requires a *fast* tokenizer for offset_mapping.
    """
    enc = _sliding_windows(tokenizer, doc.text, max_length, stride)
    n_chunks = len(enc["input_ids"])
    labels: list[list[int]] = []
    if with_labels:
        mask = _char_pii_mask(len(doc.text), doc.spans)
        for ci in range(n_chunks):
            seq_ids = enc["sequence_ids"][ci]
            offsets = enc["offset_mapping"][ci]
            chunk_labels: list[int] = []
            for tok_i, (a, b) in enumerate(offsets):
                if seq_ids[tok_i] is None or b <= a:
                    chunk_labels.append(IGNORE_INDEX)  # special / empty token
                    continue
                # Token is PII if any character it covers is PII.
                chunk_labels.append(PII_LABEL if any(mask[a:b]) else O_LABEL)
            labels.append(chunk_labels)
    return enc, labels


def reconstruct_spans(
    text: str,
    offsets_per_chunk: list[list[tuple[int, int]]],
    preds_per_chunk: list[list[int]],
    seqids_per_chunk: list[list],
) -> list[tuple[int, int]]:
    """Union chunk-level token predictions into merged character spans.

    Overlapping windows (stride) can cover a character more than once; we take
    the union (a char is PII if any covering chunk predicts PII), then merge
    PII runs and trim whitespace at the edges.

    Inter-word spaces belong to no token, so a naive contiguous merge would
    fragment every multi-word entity (e.g. an institution or full name) at its
    spaces. We therefore bridge PII runs separated only by *intra-line*
    whitespace (spaces/tabs) but never across a newline, which almost always
    separates distinct fields rather than parts of one entity.
    """
    mask = bytearray(len(text))
    for offsets, preds, seq_ids in zip(
        offsets_per_chunk, preds_per_chunk, seqids_per_chunk, strict=False
    ):
        for (a, b), p, sid in zip(offsets, preds, seq_ids, strict=False):
            if sid is None or b <= a:
                continue
            if p == PII_LABEL:
                for i in range(a, min(b, len(text))):
                    mask[i] = 1
    spans: list[tuple[int, int]] = []
    i = 0
    n = len(text)
    while i < n:
        if mask[i]:
            j = i + 1
            while j < n:
                if mask[j]:
                    j += 1
                    continue
                # Bridge a spaces/tabs-only gap to the next PII char.
                k = j
                while k < n and text[k] in " \t":
                    k += 1
                if k < n and mask[k]:
                    j = k + 1
                else:
                    break
            # trim leading/trailing whitespace
            s, e = i, j
            while s < e and text[s].isspace():
                s += 1
            while e > s and text[e - 1].isspace():
                e -= 1
            if e > s:
                spans.append((s, e))
            i = j
        else:
            i += 1
    return spans
