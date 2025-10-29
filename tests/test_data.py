"""Integration tests for BC5CDR data utilities."""

from typing import List

from src import data


def _has_non_empty_labels(example: dict) -> bool:
    labels: List[int] = example["ner_tags"]
    return any(label != 0 for label in labels)


def test_load_bc5cdr_returns_expected_splits():
    dataset, label_list, label_to_id, id_to_label = data.load_bc5cdr()

    assert {"train", "validation", "test"}.issubset(dataset.keys())
    assert label_list
    assert label_to_id
    assert id_to_label

    train_example = dataset["train"][0]
    assert len(train_example["tokens"]) == len(train_example["ner_tags"])
    assert _has_non_empty_labels(train_example)

    validation_example = dataset["validation"][0]
    assert len(validation_example["tokens"]) == len(validation_example["ner_tags"])

    test_example = dataset["test"][0]
    assert len(test_example["tokens"]) == len(test_example["ner_tags"])
