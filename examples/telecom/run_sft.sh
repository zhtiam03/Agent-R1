#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 -m recipes.telecom.sft.train_lora --config "${TELECOM_CONFIG:-recipes/telecom/config.toml}" "$@"
