#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH=${1:-configs/ner_biobert.yaml}

echo "Training entry point not implemented. Expected to call python -m src.train ${CONFIG_PATH}."
