#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 -m recipes.telecom.launch_grpo --config "${TELECOM_CONFIG:-recipes/telecom/config.toml}" --reward-mode state_aware "$@"
