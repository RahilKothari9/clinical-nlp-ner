"""Evaluation utilities for clinical NER."""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import yaml
from datasets import DatasetDict
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from transformers import AutoModelForTokenClassification, AutoTokenizer, Trainer, TrainingArguments

from . import data

LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG_PATH = Path("configs/ner_biobert.yaml")


def _load_config(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload or {}


def _prepare_dataset(dataset: DatasetDict, tokenizer: AutoTokenizer) -> DatasetDict:
    def tokenize_and_align_labels(examples: Dict[str, Any]) -> Dict[str, Any]:
        tokenized_inputs = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
        )

        labels: List[List[int]] = []
        for index in range(len(examples["tokens"])):
            word_ids = tokenized_inputs.word_ids(batch_index=index)
            sample_labels = examples["ner_tags"][index]
            label_ids: List[int] = []
            for word_id in word_ids:
                if word_id is None:
                    label_ids.append(-100)
                else:
                    label_ids.append(sample_labels[word_id])
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    remove_cols = dataset["train"].column_names
    return dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=remove_cols,
    )


def _collect_predictions(
    trainer: Trainer,
    tokenized_dataset: DatasetDict,
    label_list: List[str],
) -> Tuple[List[List[str]], List[List[str]], Dict[str, float]]:
    prediction_output = trainer.predict(tokenized_dataset["test"])

    predictions = np.argmax(prediction_output.predictions, axis=-1)
    label_ids = prediction_output.label_ids

    true_sequences: List[List[str]] = []
    pred_sequences: List[List[str]] = []

    for preds, labels in zip(predictions, label_ids):
        current_true: List[str] = []
        current_pred: List[str] = []
        for pred_id, label_id in zip(preds, labels):
            if label_id == -100:
                continue
            current_true.append(label_list[int(label_id)])
            current_pred.append(label_list[int(pred_id)])
        if current_true:
            true_sequences.append(current_true)
            pred_sequences.append(current_pred)

    overall_metrics = {
        "micro_precision": precision_score(true_sequences, pred_sequences, average="micro"),
        "micro_recall": recall_score(true_sequences, pred_sequences, average="micro"),
        "micro_f1": f1_score(true_sequences, pred_sequences, average="micro"),
        "macro_precision": precision_score(true_sequences, pred_sequences, average="macro"),
        "macro_recall": recall_score(true_sequences, pred_sequences, average="macro"),
        "macro_f1": f1_score(true_sequences, pred_sequences, average="macro"),
    }

    return true_sequences, pred_sequences, overall_metrics


def _compute_per_label_metrics(
    true_sequences: List[List[str]],
    pred_sequences: List[List[str]],
    label_list: List[str],
) -> Dict[str, Dict[str, float]]:
    precision, recall, f1, support = precision_recall_fscore_support(
        true_sequences,
        pred_sequences,
        average=None,
        labels=label_list,
        zero_division=0,
    )

    per_label = {}
    for label, p, r, f, s in zip(label_list, precision, recall, f1, support):
        per_label[label] = {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "support": int(s),
        }
    return per_label


def _write_classification_report(
    report_dir: Path,
    true_sequences: List[List[str]],
    pred_sequences: List[List[str]],
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "test_report.txt"
    report_text = classification_report(true_sequences, pred_sequences, digits=4)
    report_path.write_text(report_text, encoding="utf-8")
    LOGGER.info("Classification report written to %s", report_path)


def _write_confusion_matrix(
    report_dir: Path,
    true_sequences: List[List[str]],
    pred_sequences: List[List[str]],
    label_list: List[str],
    label_to_id: Dict[str, int],
) -> None:
    matrix = [[0 for _ in label_list] for _ in label_list]

    for true_seq, pred_seq in zip(true_sequences, pred_sequences):
        for true_label, pred_label in zip(true_seq, pred_seq):
            matrix[label_to_id[true_label]][label_to_id[pred_label]] += 1

    report_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = report_dir / "cm.csv"
    with matrix_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["label"] + label_list)
        for label, row in zip(label_list, matrix):
            writer.writerow([label] + row)
    LOGGER.info("Confusion matrix written to %s", matrix_path)


def evaluate_model(config: Dict[str, Any]) -> Dict[str, Any]:
    logging.basicConfig(level=getattr(logging, config.get("log_level", "INFO")))

    dataset_args = {
        "label_scheme": config.get("label_scheme", "BIO"),
        "cache_dir": config.get("cache_dir"),
        "local_files_only": config.get("local_files_only", False),
    }
    dataset, label_list, label_to_id, id_to_label = data.load_bc5cdr(**dataset_args)

    model_dir = Path(config.get("model_dir", config.get("output_dir", "models/biobert")))
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory '{model_dir}' does not exist.")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    ner_model = AutoModelForTokenClassification.from_pretrained(model_dir)

    tokenized_dataset = _prepare_dataset(dataset, tokenizer)

    eval_batch_size = int(config.get("per_device_eval_batch_size", 16))
    training_args = TrainingArguments(
        output_dir=str(model_dir),
        per_device_eval_batch_size=eval_batch_size,
        dataloader_num_workers=int(config.get("dataloader_num_workers", 0)),
        report_to=config.get("report_to", []),
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=ner_model,
        args=training_args,
        tokenizer=tokenizer,
    )

    true_sequences, pred_sequences, overall_metrics = _collect_predictions(
        trainer,
        tokenized_dataset,
        label_list,
    )

    per_label_metrics = _compute_per_label_metrics(true_sequences, pred_sequences, label_list)

    reports_dir = Path(config.get("reports_dir", "reports"))
    _write_classification_report(reports_dir, true_sequences, pred_sequences)
    _write_confusion_matrix(reports_dir, true_sequences, pred_sequences, label_list, label_to_id)

    metrics = {
        "overall": overall_metrics,
        "per_label": per_label_metrics,
    }
    LOGGER.info("Evaluation metrics: %s", metrics)
    return metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained NER model on BC5CDR.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to YAML configuration file used for evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = _load_config(args.config)
    evaluate_model(config)


if __name__ == "__main__":
    main()
