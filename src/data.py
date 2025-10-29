"""Utilities for downloading and preprocessing the BC5CDR dataset."""

from typing import Dict, Tuple

from datasets import DatasetDict


def load_bc5cdr(split: str = "all") -> DatasetDict:
    """Download the BC5CDR dataset via Hugging Face Datasets.

    Parameters
    ----------
    split: Which split to load ("all", "train", "validation", "test").

    Returns
    -------
    DatasetDict containing token-level annotations.
    """
    raise NotImplementedError("Dataset loading not implemented yet.")


def preprocess_samples(dataset: DatasetDict) -> Tuple[DatasetDict, Dict[str, int]]:
    """Apply paper-inspired preprocessing to the BC5CDR dataset."""
    raise NotImplementedError("Preprocessing logic not implemented yet.")
