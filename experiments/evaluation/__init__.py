"""Paper-facing corpus diversity, controllability, and realism evaluation."""

from experiments.evaluation.config import EvaluationConfig, load_config
from experiments.evaluation.runner import run_evaluation

__all__ = ["EvaluationConfig", "load_config", "run_evaluation"]
