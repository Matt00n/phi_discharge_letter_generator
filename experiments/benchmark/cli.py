"""Command-line entry point for the anonymizer benchmark."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from experiments.benchmark.config import load_config
from experiments.benchmark.runner import run_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m experiments.benchmark",
        description=(
            "Benchmark a de-identification method on a pipeline-generated gold "
            "dataset. The anonymizer returns full PHI-replaced text, which is "
            "aligned to character offsets and scored against the gold spans."
        ),
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="Override output_dir from the YAML configuration.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process only the first N documents (overrides config).",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_config(args.config)
    run_benchmark(
        config,
        output_dir=args.output.resolve() if args.output else None,
        limit=args.limit,
    )
    return 0
