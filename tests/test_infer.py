"""Tests for inference utilities."""

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from src import infer


class DummyTokenizer:
    def __call__(self, text, return_tensors="pt", return_offsets_mapping=True, truncation=True):
        assert text == "Chest pain"
        return {
            "input_ids": torch.tensor([[101, 200, 201, 102]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
            "offset_mapping": torch.tensor([[[0, 0], [0, 5], [6, 10], [0, 0]]]),
        }


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(id2label={0: "O", 1: "B-Disease", 2: "I-Disease"})

    def to(self, device):  # pylint: disable=unused-argument
        return self

    def eval(self):
        return None

    def forward(self, **kwargs):  # type: ignore[override]
        logits = torch.zeros((1, 4, 3))
        logits[0, 1, 1] = 5.0
        logits[0, 2, 2] = 5.0
        return SimpleNamespace(logits=logits)


@mock.patch("src.infer._load_artifacts", return_value=(DummyTokenizer(), DummyModel(), {0: "O", 1: "B-Disease", 2: "I-Disease"}, {"O": 0, "B-Disease": 1, "I-Disease": 2}))
def test_predict_entities_returns_well_formed_spans(mock_loader):  # pylint: disable=unused-argument
    entities = infer.predict_entities("Chest pain", model_dir=Path("unused"))
    assert len(entities) == 1
    entity = entities[0]
    assert set(entity.keys()) == {"label", "start", "end", "text"}
    assert entity["label"] == "Disease"
    assert isinstance(entity["start"], int)
    assert isinstance(entity["end"], int)
    assert entity["start"] < entity["end"]
    assert entity["text"] == "Chest pain"


def test_predict_entities_handles_empty_input():
    assert infer.predict_entities("", model_dir=Path("unused")) == []
