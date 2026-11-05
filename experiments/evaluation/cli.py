"""Command-line entry point for corpus evaluation."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from experiments.evaluation.config import load_config
from experiments.evaluation.runner import run_evaluation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute reproducible diversity, controllability, clinical/textual "
            "realism, and cross-corpus statistics from a validated YAML specification."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="Override output_path from the YAML configuration.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing report atomically.",
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
    run_evaluation(
        config,
        output_path=args.output.resolve() if args.output else None,
        overwrite=True if args.overwrite else None,
    )
    return 0
