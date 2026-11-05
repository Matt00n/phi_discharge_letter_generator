"""Ablation experiments for the SPARCLING-DE pipeline.

Three ablations, each stripping out one load-bearing pipeline stage to show what
it buys us. All arms reuse the *identical* Stage-4 QC and Stage-5 diversity code
as the main pipeline, so the comparison is apples-to-apples.

* ``no_qc``         — analysis only: what would ship if Stage 4 were disabled.
* ``no_templating`` — LLM writes free text + self-annotates (no placeholders);
                      show QC fires heavily.
* ``no_sampling``   — LLM invents a template AND fills it (no Stage-1 sampler);
                      show diversity collapses.
"""
