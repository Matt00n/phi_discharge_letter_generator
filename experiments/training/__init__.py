"""Fine-tuning harness: train a small German encoder as a binary PII detector.

Trains medBERT.de (or any HF encoder) for token-level binary PII / not-PII
detection on the synthetic SPARCLING-DE corpus and evaluates it on a target
corpus (held-out synthetic and/or real GRASCCO). Long letters are handled with
sliding-window chunking; predictions are reconstructed to character spans and
scored with the existing benchmark scorer so results are directly comparable to
the de-identification benchmark.
"""
