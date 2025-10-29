"""BC5CDR dataset utilities for clinical named entity recognition."""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value
from huggingface_hub import hf_hub_download

LOGGER = logging.getLogger(__name__)

_DATASET_ID = "tner/bc5cdr"
_LABELS_FILENAME = "dataset/label.json"
_SPLIT_FILES = {
    "train": "dataset/train.json",
    "validation": "dataset/valid.json",
    "test": "dataset/test.json",
}

PathLike = Union[str, Path]


def _download_file(
    filename: str,
    cache_dir: Optional[PathLike] = None,
    local_files_only: bool = False,
) -> Path:
    download_kwargs = {
        "repo_id": _DATASET_ID,
        "filename": filename,
        "repo_type": "dataset",
        "local_files_only": local_files_only,
    }
    if cache_dir is not None:
        download_kwargs["cache_dir"] = str(cache_dir)

    path = hf_hub_download(**download_kwargs)
    return Path(path)


def _load_label_mappings(
    cache_dir: Optional[PathLike] = None,
    local_files_only: bool = False,
) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    labels_path = _download_file(
        _LABELS_FILENAME,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    mapping = json.loads(labels_path.read_text(encoding="utf-8"))
    label_to_id = {label: int(idx) for label, idx in mapping.items()}

    label_list = [None] * (max(label_to_id.values()) + 1)
    for label, idx in label_to_id.items():
        label_list[idx] = label
    if any(item is None for item in label_list):
        raise ValueError("Label mapping contains gaps; unable to construct label list.")

    id_to_label = {idx: label for idx, label in enumerate(label_list)}
    return label_list, label_to_id, id_to_label


def _load_split(
    split_name: str,
    file_path: Path,
    features: Features,
) -> Dataset:
    records: List[Dict[str, List[Union[str, int]]]] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            sample = json.loads(stripped)
            tokens = sample.get("tokens", [])
            labels = sample.get("tags") or sample.get("ner_tags")
            if labels is None:
                raise ValueError(f"Sample in {split_name} is missing label sequence.")
            if len(tokens) != len(labels):
                raise ValueError(
                    f"Token/label length mismatch in {split_name}: {len(tokens)} vs {len(labels)}"
                )
            records.append({"tokens": tokens, "ner_tags": labels})

    LOGGER.info("Loaded %s samples for split '%s'", len(records), split_name)
    return Dataset.from_list(records, features=features)


def load_bc5cdr(
    label_scheme: str = "BIO",
    cache_dir: Optional[PathLike] = None,
    local_files_only: bool = False,
) -> Tuple[DatasetDict, List[str], Dict[str, int], Dict[int, str]]:
    """Load the BC5CDR chemical/disease NER dataset with token-level BIO tags."""

    normalized_scheme = label_scheme.upper()
    if normalized_scheme != "BIO":
        raise NotImplementedError(
            f"Unsupported label scheme '{label_scheme}'. Only BIO is implemented."
        )

    cache_path: Optional[Path] = Path(cache_dir) if cache_dir else None

    label_list, label_to_id, id_to_label = _load_label_mappings(
        cache_dir=cache_path,
        local_files_only=local_files_only,
    )

    label_feature = ClassLabel(names=list(label_list))
    features = Features(
        {
            "tokens": Sequence(Value("string")),
            "ner_tags": Sequence(label_feature),
        }
    )

    split_datasets = {}
    for split_name, filename in _SPLIT_FILES.items():
        split_path = _download_file(
            filename,
            cache_dir=cache_path,
            local_files_only=local_files_only,
        )
        split_datasets[split_name] = _load_split(split_name, split_path, features)

    dataset_dict = DatasetDict(split_datasets)
    return dataset_dict, list(label_list), dict(label_to_id), dict(id_to_label)


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
