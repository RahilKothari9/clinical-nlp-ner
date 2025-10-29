"""Tests for BC5CDR data utilities."""

import pytest

from src import data


def test_load_bc5cdr_unsupported_scheme():
    with pytest.raises(NotImplementedError):
        data.load_bc5cdr(label_scheme="BIOES")
