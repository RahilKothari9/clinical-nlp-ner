#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH=${1:-configs/ner_biobert.yaml}

python -m src.train --config "${CONFIG_PATH}"
