"""BC5CDR dataset utilities for clinical named entity recognition."""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from datasets import ClassLabel, DatasetDict, Sequence, Value, load_dataset

LOGGER = logging.getLogger(__name__)

_DATASET_ID = "tner/bc5cdr"
_DATASET_CONFIG = "bc5cdr"
_LABEL_LIST = ["O", "B-Chemical", "B-Disease", "I-Disease", "I-Chemical"]
_LABEL_TO_ID = {label: idx for idx, label in enumerate(_LABEL_LIST)}
_ID_TO_LABEL = {idx: label for label, idx in _LABEL_TO_ID.items()}

PathLike = Union[str, Path]


def load_bc5cdr(
    label_scheme: str = "BIO",
    cache_dir: Optional[PathLike] = None,
    local_files_only: bool = False,
) -> Tuple[DatasetDict, List[str], Dict[str, int], Dict[int, str]]:
    """Load the BC5CDR chemical/disease NER dataset with token-level BIO tags.

    Parameters
    ----------
    label_scheme:
        Tagging scheme to apply. Currently only "BIO" tags are supported.
    cache_dir:
        Optional cache directory forwarded to ``datasets.load_dataset``.
    local_files_only:
        If ``True`` the loader only looks for already-downloaded data.

    Returns
    -------
    ``(dataset, label_list, label_to_id, id_to_label)`` where ``dataset`` is a
    :class:`datasets.DatasetDict` with ``train``, ``validation``, and ``test``
    splits containing ``tokens`` and ``ner_tags`` columns that align with
    Hugging Face Transformers token classification expectations.
    """
    normalized_scheme = label_scheme.upper()
    if normalized_scheme != "BIO":
        raise NotImplementedError(
            f"Unsupported label scheme '{label_scheme}'. Only BIO is implemented."
        )

    cache_path: Optional[Path] = Path(cache_dir) if cache_dir else None

    LOGGER.info(
        "Loading BC5CDR dataset from '%s' (config=%s)...",
        _DATASET_ID,
        _DATASET_CONFIG,
    )
    dataset = load_dataset(
        _DATASET_ID,
        name=_DATASET_CONFIG,
        cache_dir=str(cache_path) if cache_path else None,
        local_files_only=local_files_only,
    )

    # Align column names with Transformers token-classification conventions.
    if "ner_tags" not in dataset["train"].column_names:
        dataset = dataset.rename_column("tags", "ner_tags")

    label_feature = ClassLabel(names=_LABEL_LIST)
    sequence_string = Sequence(Value("string"))
    sequence_labels = Sequence(label_feature)

    dataset = dataset.cast_column("tokens", sequence_string)
    dataset = dataset.cast_column("ner_tags", sequence_labels)

    return dataset, list(_LABEL_LIST), dict(_LABEL_TO_ID), dict(_ID_TO_LABEL)


def _save_dataset(
    dataset: DatasetDict,
    out_dir: PathLike,
    labels: List[str],
    label_to_id: Dict[str, int],
    id_to_label: Dict[int, str],
) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Saving tokenized BC5CDR splits to %s", out_path)
    dataset.save_to_disk(str(out_path))

    labels_path = out_path / "labels.json"
    labels_payload = {
        "labels": labels,
        "label_to_id": label_to_id,
        "id_to_label": {str(idx): label for idx, label in id_to_label.items()},
    }
    labels_path.write_text(json.dumps(labels_payload, indent=2), encoding="utf-8")
    LOGGER.info("Label mapping written to %s", labels_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare BC5CDR for NER tasks.")
    parser.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="Directory where the processed dataset will be saved.",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=None,
        help="Optional path to reuse/download the raw dataset cache.",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Only use already cached files (offline mode).",
    )
    parser.add_argument(
        "--label_scheme",
        type=str,
        default="BIO",
        help="Tagging scheme to generate (currently only BIO).",
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

    dataset, labels, label_to_id, id_to_label = load_bc5cdr(
        label_scheme=args.label_scheme,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    _save_dataset(dataset, args.out_dir, labels, label_to_id, id_to_label)


if __name__ == "__main__":
    main()
