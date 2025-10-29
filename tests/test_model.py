"""Smoke tests for model utilities."""

import pytest

from src import model


def test_load_model_and_tokenizer_not_implemented():
    with pytest.raises(NotImplementedError):
        model.load_model_and_tokenizer()
