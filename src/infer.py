"""Command line inference for clinical NER."""

from typing import List


def tag_text(text: str) -> List[str]:
    """Return entity tags for the supplied clinical text."""
    raise NotImplementedError("Inference not implemented yet.")


if __name__ == "__main__":
    raise SystemExit("Use python -m src.infer 'your text' once implemented.")
