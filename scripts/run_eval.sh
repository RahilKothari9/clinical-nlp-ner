#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH=${1:-configs/ner_biobert.yaml}

echo "Evaluation entry point not implemented. Expected to call python -m src.eval ${CONFIG_PATH}."
