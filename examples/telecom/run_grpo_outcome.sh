#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# Default prints overrides only; pass --run explicitly on a CUDA server.
python3 -m recipes.telecom.launch_grpo --config "${TELECOM_CONFIG:-recipes/telecom/config.toml}" --reward-mode outcome_only "$@"
