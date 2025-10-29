"""Evaluation utilities for clinical NER."""

from typing import Any, Dict


def evaluate_model(config: Dict[str, Any]) -> Dict[str, float]:
    """Run seqeval metrics and return the results."""
    raise NotImplementedError("Evaluation not implemented yet.")


if __name__ == "__main__":
    raise SystemExit("Use scripts/run_eval.sh with a config file to launch evaluation.")
