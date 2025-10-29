"""Command line inference for clinical NER."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL_DIR = Path("models/biobert")


def _normalize_id2label(id2label) -> Dict[int, str]:
    if isinstance(id2label, dict):
        return {int(idx): str(label) for idx, label in id2label.items()}
    if isinstance(id2label, list):
        return {idx: str(label) for idx, label in enumerate(id2label)}
    raise ValueError("Unsupported id2label structure in model config.")


def _load_artifacts(model_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    id2label = _normalize_id2label(model.config.id2label)
    label2id = {label: idx for idx, label in id2label.items()}
    return tokenizer, model, id2label, label2id


def _aggregate_entities(
    offsets: List[List[int]],
    labels: List[str],
    text: str,
) -> List[Dict[str, object]]:
    entities: List[Dict[str, object]] = []
    current: Optional[Dict[str, object]] = None

    for (start, end), tag in zip(offsets, labels):
        if start == end:
            continue

        if tag == "O" or not tag:
            if current is not None:
                current["text"] = text[current["start"] : current["end"]]
                entities.append(current)
                current = None
            continue

        if "-" in tag:
            prefix, entity_label = tag.split("-", 1)
        else:
            prefix, entity_label = "B", tag

        if prefix == "B" or current is None or current["label"] != entity_label:
            if current is not None:
                current["text"] = text[current["start"] : current["end"]]
                entities.append(current)
            current = {"label": entity_label, "start": start, "end": end}
        elif prefix == "I" and current["label"] == entity_label:
            current["end"] = end
        else:
            if current is not None:
                current["text"] = text[current["start"] : current["end"]]
                entities.append(current)
            current = {"label": entity_label, "start": start, "end": end}

    if current is not None:
        current["text"] = text[current["start"] : current["end"]]
        entities.append(current)

    return entities


def predict_entities(
    text: str,
    model_dir: Path = DEFAULT_MODEL_DIR,
) -> List[Dict[str, object]]:
    """Predict clinical entities for free-text input."""
    if not text:
        return []

    tokenizer, ner_model, id2label, _ = _load_artifacts(model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ner_model.to(device)
    ner_model.eval()

    encoded = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
    )

    offset_mapping = encoded.pop("offset_mapping")[0].tolist()
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        outputs = ner_model(**encoded)
        predictions = outputs.logits.argmax(dim=-1).squeeze(0).tolist()

    labels = [id2label.get(int(pred), "O") for pred in predictions]
    entities = _aggregate_entities(offset_mapping, labels, text)
    return entities


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run clinical NER inference.")
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Clinical sentence or document to analyze.",
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Directory containing the fine-tuned model checkpoint.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="WARNING",
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.WARNING))
    entities = predict_entities(args.text, model_dir=args.model_dir)
    print(json.dumps({"text": args.text, "entities": entities}, indent=2))


if __name__ == "__main__":
    main()
