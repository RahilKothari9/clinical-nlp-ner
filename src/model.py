"""Model and tokenizer factory for clinical NER."""

from typing import Tuple

from transformers import AutoModelForTokenClassification, AutoTokenizer


DEFAULT_MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"


def load_model_and_tokenizer(model_name: str = DEFAULT_MODEL_NAME) -> Tuple[AutoModelForTokenClassification, AutoTokenizer]:
    """Load a BioBERT token classification model and matching tokenizer."""
    raise NotImplementedError("Model initialization not implemented yet.")
