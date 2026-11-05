"""A single caveated diversity index over normalized-entropy components.

The index is the harmonic mean of per-dimension normalized entropies, each in
``[0, 1]``. The harmonic mean is used deliberately: it is dominated by the least
diverse dimension, so a corpus cannot mask a collapsed dimension (for example a
single dominant specialty) behind other varied ones.
"""

from __future__ import annotations


def aggregate_diversity_index(
    components: dict[str, float | None],
) -> dict[str, object]:
    """Combine normalized-entropy components into one harmonic-mean index.

    ``components`` maps a dimension name to its normalized entropy in ``[0, 1]``
    or ``None`` when that dimension was not computed for this corpus. Identity
    fields (surnames, entity identifiers, random seeds) are intentionally never
    passed in: high entropy there is trivial and would inflate the index.
    """

    available = {name: value for name, value in components.items() if value is not None}
    omitted = sorted(name for name, value in components.items() if value is None)
    if not available:
        return {
            "status": "not_computed",
            "reason": "no diversity components were available for this corpus",
            "dimensions_omitted": omitted,
        }
    zero_dimensions = sorted(name for name, value in available.items() if value <= 0.0)
    if zero_dimensions:
        harmonic_mean = 0.0
    else:
        harmonic_mean = len(available) / sum(
            1.0 / value for value in available.values()
        )
    return {
        "status": "computed",
        "definition": (
            "harmonic mean of per-dimension normalized Shannon entropies, each in "
            "[0, 1]; dominated by the least diverse dimension"
        ),
        "harmonic_mean_normalized_entropy": harmonic_mean,
        "dimensions_included": sorted(available),
        "dimensions_omitted": omitted,
        "zero_entropy_dimensions": zero_dimensions,
        "component_values": {name: available[name] for name in sorted(available)},
        "interpretation": (
            "Identity fields (surnames, entity identifiers, seeds) are excluded by "
            "design; their entropy is trivially high and would not reflect "
            "controllable content diversity."
        ),
    }
