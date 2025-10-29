"""Model and tokenizer factory for clinical NER."""

from __future__ import annotations

from typing import Dict, Optional, Union

import torch
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

DEFAULT_MODEL_NAME = "dmis-lab/biobert-v1.1"
_FALLBACK_MODEL_NAME = "bert-base-cased"
TorchDTypeLike = Optional[Union[str, torch.dtype]]


class ModelLoadingError(RuntimeError):
    """Raised when neither the primary nor fallback models can be loaded."""


def _resolve_dtype(dtype_name: TorchDTypeLike):
    """Normalize dtype aliases to arguments accepted by Transformers."""
    if dtype_name is None:
        return {}

    if isinstance(dtype_name, torch.dtype):
        return {"torch_dtype": dtype_name}

    normalized = dtype_name.lower()
    mapping = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "f32": torch.float32,
        "auto": "auto",
    }
    resolved = mapping.get(normalized)
    if resolved is None:
        return {}
    return {"torch_dtype": resolved}


def get_tokenizer(model_name: str = DEFAULT_MODEL_NAME) -> AutoTokenizer:
    """Load a tokenizer suited for biomedical NER with fallback."""
    last_error: Optional[Exception] = None
    for candidate in (model_name, _FALLBACK_MODEL_NAME):
        try:
            return AutoTokenizer.from_pretrained(candidate, use_fast=True)
        except Exception as exc:  # pragma: no cover - network/cache failures
            last_error = exc
            if candidate == _FALLBACK_MODEL_NAME:
                raise ModelLoadingError(
                    f"Failed to load tokenizer for {model_name} or fallback {_FALLBACK_MODEL_NAME}."
                ) from exc
    raise ModelLoadingError("Tokenizer loading failed unexpectedly") from last_error


def get_model(
    model_name: str = DEFAULT_MODEL_NAME,
    num_labels: int = 5,
    id2label: Optional[Dict[int, str]] = None,
    label2id: Optional[Dict[str, int]] = None,
    *,
    gradient_checkpointing: bool = False,
    torch_dtype: TorchDTypeLike = None,
) -> AutoModelForTokenClassification:
    """Instantiate a token classification model configured for clinical NER."""

    if id2label is None and label2id is None:
        id2label = {idx: str(idx) for idx in range(num_labels)}
        label2id = {label: idx for idx, label in id2label.items()}
    elif id2label is None:
        id2label = {idx: label for label, idx in label2id.items()}
    elif label2id is None:
        label2id = {label: idx for idx, label in id2label.items()}

    dtype_kwargs = _resolve_dtype(torch_dtype)
    last_error: Optional[Exception] = None

    for candidate in (model_name, _FALLBACK_MODEL_NAME):
        try:
            config = AutoConfig.from_pretrained(
                candidate,
                num_labels=num_labels,
                id2label=id2label,
                label2id=label2id,
            )
            model = AutoModelForTokenClassification.from_pretrained(
                candidate,
                config=config,
                **dtype_kwargs,
            )
            if gradient_checkpointing:
                model.gradient_checkpointing_enable()
            return model
        except Exception as exc:  # pragma: no cover - network/cache failures
            last_error = exc
            if candidate == _FALLBACK_MODEL_NAME:
                raise ModelLoadingError(
                    f"Failed to load model {model_name} or fallback {_FALLBACK_MODEL_NAME}."
                ) from exc

    raise ModelLoadingError("Model loading failed unexpectedly") from last_error
