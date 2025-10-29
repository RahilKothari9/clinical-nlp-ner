"""Gradio demo for clinical named entity recognition."""

from __future__ import annotations

import argparse
import logging
from html import escape
from pathlib import Path
from typing import Dict, List, Tuple

import gradio as gr
import torch

from . import infer

LOGGER = logging.getLogger(__name__)
DEFAULT_BASE_DIR = infer.DEFAULT_MODEL_DIR
BEST_CHECKPOINT_DIRNAME = "checkpoint-best"
LABEL_CHOICES = {
    "both": "Both (Chemical + Disease)",
    "disease": "Diseases only",
    "chemical": "Chemicals only",
}


def _resolve_model_dir(base_dir: Path) -> Path:
    best_dir = base_dir / BEST_CHECKPOINT_DIRNAME
    if best_dir.exists():
        LOGGER.info("Using best checkpoint at %s", best_dir)
        return best_dir
    LOGGER.info("Using base model directory %s", base_dir)
    return base_dir


def _load_inference_stack(model_dir: Path):
    tokenizer, model, id2label, _ = infer._load_artifacts(model_dir)  # pylint: disable=protected-access
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, id2label, device


def _predict(text: str, tokenizer, model, id2label, device):
    if not text.strip():
        return []

    encoded = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
    )
    offsets = encoded.pop("offset_mapping")[0].tolist()
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        logits = model(**encoded).logits
        predictions = logits.argmax(dim=-1).squeeze(0).tolist()

    labels = [id2label.get(int(pred), "O") for pred in predictions]
    entities = infer._aggregate_entities(offsets, labels, text)  # pylint: disable=protected-access
    return entities


def _filter_entities(entities: List[Dict[str, object]], choice: str) -> List[Dict[str, object]]:
    if choice == "both":
        return entities

    target = "disease" if choice == "disease" else "chemical"
    return [entity for entity in entities if str(entity["label"]).lower() == target]


def _render_html(text: str, entities: List[Dict[str, object]]) -> str:
    if not text:
        return "<p><em>No text provided.</em></p>"

    pieces: List[str] = []
    cursor = 0

    for entity in sorted(entities, key=lambda item: item["start"]):
        start = int(entity["start"])
        end = int(entity["end"])
        label = str(entity["label"])
        if start > cursor:
            pieces.append(escape(text[cursor:start]))
        highlighted = escape(text[start:end])
        pieces.append(
            f"<mark class='ner-entity'><span class='ner-text'>{highlighted}</span>"
            f"<span class='ner-label'>{escape(label)}</span></mark>"
        )
        cursor = end

    if cursor < len(text):
        pieces.append(escape(text[cursor:]))

    if not pieces:
        return f"<p>{escape(text)}</p>"

    style = (
        "<style>"
        ".ner-entity { background-color: #ffef99; padding: 0 0.2em; margin: 0 0.1em; border-radius: 4px; display: inline-flex; flex-direction: column; align-items: center; }"
        ".ner-entity .ner-text { font-weight: 600; }"
        ".ner-entity .ner-label { font-size: 0.7em; text-transform: uppercase; color: #444; }"
        "</style>"
    )
    return style + "<p>" + "".join(pieces) + "</p>"


def build_demo(model_dir: Path) -> gr.Blocks:
    tokenizer, model, id2label, device = _load_inference_stack(model_dir)

    def _predict_for_ui(text: str, label_choice: str) -> Tuple[str, List[Dict[str, object]]]:
        entities = _predict(text, tokenizer, model, id2label, device)
        filtered = _filter_entities(entities, label_choice)
        html = _render_html(text, filtered)
        return html, filtered

    with gr.Blocks(title="Clinical NER Demo") as demo:
        gr.Markdown("## Clinical Named Entity Recognition\nPaste biomedical text and highlight detected entities.")
        text_input = gr.Textbox(
            label="Clinical text",
            lines=6,
            placeholder="Patient has elevated troponin...",
        )
        label_dropdown = gr.Dropdown(
            choices=[LABEL_CHOICES[key] for key in LABEL_CHOICES],
            value=LABEL_CHOICES["both"],
            label="Entity display",
        )
        highlight_output = gr.HTML(label="Highlighted text")
        entities_output = gr.JSON(label="Entities")
        submit_button = gr.Button("Run NER")

        def _map_choice(choice_display: str) -> str:
            for key, display in LABEL_CHOICES.items():
                if display == choice_display:
                    return key
            return "both"

        def _adapter(text: str, choice_display: str) -> Tuple[str, List[Dict[str, object]]]:
            choice_key = _map_choice(choice_display)
            return _predict_for_ui(text, choice_key)

        submit_button.click(
            fn=_adapter,
            inputs=[text_input, label_dropdown],
            outputs=[highlight_output, entities_output],
        )
    return demo


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the clinical NER Gradio app.")
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help="Directory containing the trained model (defaults to models/biobert).",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Enable Gradio sharing (public link).",
    )
    parser.add_argument(
        "--server_port",
        type=int,
        default=None,
        help="Optional custom port for the Gradio server.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Python logging level (e.g. INFO, DEBUG).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    resolved_dir = _resolve_model_dir(args.model_dir)
    if not resolved_dir.exists():
        raise FileNotFoundError(f"Model directory '{resolved_dir}' not found.")

    demo = build_demo(resolved_dir)
    demo.launch(share=args.share, server_port=args.server_port)


if __name__ == "__main__":
    main()
