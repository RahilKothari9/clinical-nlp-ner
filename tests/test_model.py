"""Tests for model utilities without hitting external downloads."""

from unittest import mock

from src import model


@mock.patch("src.model.AutoTokenizer.from_pretrained")
def test_get_tokenizer_fallback(mock_from_pretrained):
    def side_effect(name, use_fast=True):  # pylint: disable=unused-argument
        if name == model.DEFAULT_MODEL_NAME:
            raise OSError("missing cache")
        return "tokenizer"

    mock_from_pretrained.side_effect = side_effect
    tokenizer = model.get_tokenizer()
    assert tokenizer == "tokenizer"


@mock.patch("src.model.AutoModelForTokenClassification.from_pretrained")
@mock.patch("src.model.AutoConfig.from_pretrained")
def test_get_model_uses_label_mappings(mock_config, mock_from_pretrained):
    mock_model = mock.Mock()
    mock_model.gradient_checkpointing_enable = mock.Mock()
    mock_from_pretrained.return_value = mock_model

    label_list = ["O", "B-Chemical"]
    id2label = {idx: label for idx, label in enumerate(label_list)}
    label2id = {label: idx for idx, label in id2label.items()}

    result_model = model.get_model(
        model_name=model.DEFAULT_MODEL_NAME,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
        gradient_checkpointing=True,
        torch_dtype="fp16",
    )

    assert result_model is mock_model
    mock_config.assert_called_once()
    mock_from_pretrained.assert_called_once()
    mock_model.gradient_checkpointing_enable.assert_called_once()
