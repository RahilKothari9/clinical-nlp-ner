"""Smoke tests for data utilities."""

import pytest

from src import data


def test_load_bc5cdr_not_implemented():
    with pytest.raises(NotImplementedError):
        data.load_bc5cdr()
