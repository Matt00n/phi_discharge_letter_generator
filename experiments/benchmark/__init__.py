"""Anonymizer benchmark harness.

Evaluates de-identification methods against a pipeline-generated gold dataset.

The harness is scoped to the *text-replacement* evaluation path: an anonymizer
returns the full document with every PII span replaced by a marker, and the
bundled :mod:`experiments.benchmark.scoring` package (``--mode text``) aligns that output back to
character offsets before scoring. Gemini is the first supported model; add more
by registering an :class:`~experiments.benchmark.anonymizers.base.Anonymizer`
in :mod:`experiments.benchmark.anonymizers`.
"""
