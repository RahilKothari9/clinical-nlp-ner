"""Training entry point for the clinical NER pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml
from datasets import DatasetDict
from seqeval.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from . import data, model

LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG_PATH = Path("configs/ner_biobert.yaml")


def _load_config(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload or {}


def _prepare_datasets(
    dataset: DatasetDict,
    tokenizer: AutoTokenizer,
) -> DatasetDict:
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


def _build_training_args(config: Dict[str, Any]) -> TrainingArguments:
    output_dir = Path(config.get("output_dir", "models/biobert"))
    output_dir.mkdir(parents=True, exist_ok=True)

    per_device_train_batch_size = config.get("per_device_train_batch_size", 16)
    per_device_eval_batch_size = config.get("per_device_eval_batch_size", 16)

    return TrainingArguments(
        output_dir=str(output_dir),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=float(config.get("learning_rate", 3e-5)),
        per_device_train_batch_size=int(per_device_train_batch_size),
        per_device_eval_batch_size=int(per_device_eval_batch_size),
        num_train_epochs=float(config.get("num_train_epochs", 5)),
        weight_decay=float(config.get("weight_decay", 0.01)),
        warmup_ratio=float(config.get("warmup_ratio", 0.1)),
        logging_steps=int(config.get("logging_steps", 50)),
        seed=int(config.get("seed", 42)),
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model="eval_micro_f1",
        greater_is_better=True,
        save_total_limit=int(config.get("save_total_limit", 2)),
        report_to=config.get("report_to", []),
    )


def _compute_metrics_builder(label_list: List[str]):
    def compute_metrics(eval_preds):
        predictions, labels = eval_preds
        predictions = predictions.argmax(axis=-1)

        true_labels: List[List[str]] = []
        true_predictions: List[List[str]] = []

        for prediction, label in zip(predictions, labels):
            current_true: List[str] = []
            current_pred: List[str] = []
            for pred_id, label_id in zip(prediction, label):
                if label_id == -100:
                    continue
                current_true.append(label_list[int(label_id)])
                current_pred.append(label_list[int(pred_id)])
            true_labels.append(current_true)
            true_predictions.append(current_pred)

        metrics = {
            "micro_f1": f1_score(true_labels, true_predictions, average="micro"),
            "macro_f1": f1_score(true_labels, true_predictions, average="macro"),
            "micro_precision": precision_score(true_labels, true_predictions, average="micro"),
            "macro_precision": precision_score(true_labels, true_predictions, average="macro"),
            "micro_recall": recall_score(true_labels, true_predictions, average="micro"),
            "macro_recall": recall_score(true_labels, true_predictions, average="macro"),
        }
        return metrics

    return compute_metrics


def run_training(config: Dict[str, Any]) -> None:
    logging.basicConfig(level=getattr(logging, config.get("log_level", "INFO")))

    dataset_args = {
        "label_scheme": config.get("label_scheme", "BIO"),
        "cache_dir": config.get("cache_dir"),
        "local_files_only": config.get("local_files_only", False),
    }
    dataset, label_list, label_to_id, id_to_label = data.load_bc5cdr(**dataset_args)

    model_name = config.get("model_name", model.DEFAULT_MODEL_NAME)
    tokenizer = model.get_tokenizer(model_name=model_name)
    tokenized_dataset = _prepare_datasets(dataset, tokenizer)

    ner_model = model.get_model(
        model_name=model_name,
        num_labels=len(label_list),
        id2label=id_to_label,
        label2id=label_to_id,
        gradient_checkpointing=config.get("gradient_checkpointing", False),
        torch_dtype=config.get("torch_dtype"),
    )

    training_args = _build_training_args(config)
    data_collator = DataCollatorForTokenClassification(tokenizer)
    compute_metrics = _compute_metrics_builder(label_list)

    trainer = Trainer(
        model=ner_model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)

    test_metrics = trainer.evaluate(tokenized_dataset["test"])
    LOGGER.info("Test metrics: %s", test_metrics)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BioBERT on BC5CDR for NER.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to YAML configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = _load_config(args.config)
    run_training(config)


if __name__ == "__main__":
    main()
