"""Tests for inference utilities with mocked transformers."""

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from src import infer


class DummyTokenizer:
    def __call__(self, text, return_tensors="pt", return_offsets_mapping=True, truncation=True):
        assert text == "Chest pain"
        return {
            "input_ids": torch.tensor([[101, 999, 888, 102]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
            "offset_mapping": torch.tensor([[[0, 0], [0, 5], [6, 10], [0, 0]]]),
        }


class DummyModel:
    def __init__(self):
        self.config = SimpleNamespace(id2label={0: "O", 1: "B-Disease", 2: "I-Disease"})

    def to(self, device):  # pylint: disable=unused-argument
        return self

    def eval(self):
        return None

    def __call__(self, **kwargs):  # pylint: disable=unused-argument
        logits = torch.zeros((1, 4, 3), dtype=torch.float32)
        logits[0, 1, 1] = 5.0
        logits[0, 2, 2] = 5.0
        return SimpleNamespace(logits=logits)


@mock.patch("src.infer.AutoModelForTokenClassification.from_pretrained", return_value=DummyModel())
@mock.patch("src.infer.AutoTokenizer.from_pretrained", return_value=DummyTokenizer())
def test_predict_entities_merges_bio_spans(mock_tokenizer, mock_model):  # pylint: disable=unused-argument
    entities = infer.predict_entities("Chest pain", model_dir=Path("unused"))
    assert entities == [
        {"label": "Disease", "start": 0, "end": 10, "text": "Chest pain"}
    ]


def test_predict_entities_handles_empty_text():
    assert infer.predict_entities("", model_dir=Path("unused")) == []
