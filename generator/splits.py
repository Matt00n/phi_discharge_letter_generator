"""Stage 5a: split assignment.

Three orthogonal split schemes are attached to every record so the same dataset
supports several evaluation questions:

* ``by_domain``     - hold out whole medical specialties -> out-of-distribution
  generalization across clinical domains.
* ``by_difficulty`` - the difficulty tier itself (easy/medium/hard).
* ``by_surface``    - disjoint patient-surname vocabulary between train and test
  -> a memorization probe (does the model generalize across name distributions,
  or has it memorized a closed vocabulary?).
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterable

DEFAULT_HOLDOUT_DOMAINS = {"Nephrologie", "Dermatologie", "Infektiologie"}


def _bucket(value: str, mod: int) -> int:
    h = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % mod


def assign_splits(
    records: Iterable[dict[str, Any]],
    *,
    holdout_domains: set[str] | None = None,
    surface_test_fraction: float = 0.25,
) -> list[dict[str, Any]]:
    """Attach ``metadata.splits`` to each record in place and return the list."""
    holdout_domains = holdout_domains or DEFAULT_HOLDOUT_DOMAINS
    surface_test_buckets = max(1, round(surface_test_fraction * 20))

    out = []
    for rec in records:
        meta = rec.setdefault("metadata", {})
        domain = meta.get("fachabteilung")
        surname = meta.get("patient_surname", "")

        by_domain = "test" if domain in holdout_domains else "train"
        by_surface = "test" if _bucket(surname, 20) < surface_test_buckets else "train"

        meta["splits"] = {
            "by_domain": by_domain,
            "by_difficulty": meta.get("difficulty"),
            "by_surface": by_surface,
        }
        out.append(rec)
    return out
