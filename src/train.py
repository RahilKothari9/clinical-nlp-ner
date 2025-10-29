"""Training entry point for the clinical NER pipeline."""

from typing import Any, Dict


def run_training(config: Dict[str, Any]) -> None:
    """Train BioBERT on BC5CDR using configuration values."""
    raise NotImplementedError("Training loop not implemented yet.")


if __name__ == "__main__":
    raise SystemExit("Use scripts/run_train.sh with a config file to launch training.")
