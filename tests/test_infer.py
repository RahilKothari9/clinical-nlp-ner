"""Smoke tests for inference utilities."""

import pytest

from src import infer


def test_tag_text_not_implemented():
    with pytest.raises(NotImplementedError):
        infer.tag_text("fever and cough")
