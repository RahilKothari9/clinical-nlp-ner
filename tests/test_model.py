"""Tests for model utilities leveraging lightweight stubs."""

from types import SimpleNamespace
from unittest import mock

import torch

from src import model


class DummyConfig:
    def __init__(self, *, num_labels, id2label, label2id):
        self.num_labels = num_labels
        self.id2label = id2label
        self.label2id = label2id


class DummyClassifier(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.linear = torch.nn.Linear(8, config.num_labels)

    def gradient_checkpointing_enable(self):  # pragma: no cover - simple stub
        return None

    def forward(self, input_ids=None, attention_mask=None):  # type: ignore[override]
        batch_size, seq_len = input_ids.shape
        hidden = torch.ones(batch_size, seq_len, 8)
        logits = self.linear(hidden)
        return SimpleNamespace(logits=logits)


@mock.patch("src.model.AutoModelForTokenClassification.from_pretrained")
@mock.patch("src.model.AutoConfig.from_pretrained")
def test_get_model_forward_pass(mock_config, mock_model_from_pretrained):
    mock_config.side_effect = lambda *args, **kwargs: DummyConfig(**kwargs)
    mock_model_from_pretrained.side_effect = lambda *args, **kwargs: DummyClassifier(kwargs["config"])

    label_list = ["O", "B-Chemical", "B-Disease"]
    id2label = {idx: label for idx, label in enumerate(label_list)}
    label2id = {label: idx for idx, label in id2label.items()}

    classifier = model.get_model(
        model_name=model.DEFAULT_MODEL_NAME,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
        gradient_checkpointing=True,
    )

    dummy_inputs = torch.ones((2, 4), dtype=torch.long)
    outputs = classifier(input_ids=dummy_inputs, attention_mask=dummy_inputs)

    assert outputs.logits.shape == (2, 4, len(label_list))
