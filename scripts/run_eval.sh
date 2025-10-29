#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH=${1:-configs/ner_biobert.yaml}

python -m src.eval --config "${CONFIG_PATH}"
