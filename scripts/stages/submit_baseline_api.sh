#!/usr/bin/env bash
# Submit baseline (openai_compat) on Slurm; key defaults to <repo>/key.txt.
# Usage: from repo root,  bash scripts/stages/submit_baseline_api.sh
# Or: export EXPERIMENT_DIR=...  then bash scripts/stages/submit_baseline_api.sh

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

export MODEL_TYPE=openai_compat
export OPENAI_COMPAT_KEY_FILE="${OPENAI_COMPAT_KEY_FILE:-${REPO_ROOT}/key.txt}"
export EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-config/baseline_config.yaml}"

exec sbatch "${REPO_ROOT}/scripts/stages/baseline_orchestrator_api.slurm"
