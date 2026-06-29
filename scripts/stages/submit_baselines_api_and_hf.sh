#!/usr/bin/env bash
# Submit two baseline orchestrators: OpenAI-compatible API vs local HuggingFace/vLLM.
# Run from repo root:  bash scripts/stages/submit_baselines_api_and_hf.sh
#
# Required for API run: OPENAI_COMPAT_KEY_FILE (e.g. export OPENAI_COMPAT_KEY_FILE=$PWD/key.txt)
# Optional: EXPERIMENT_CONFIG (defaults to config/baseline_config.yaml if present via load_config)
#
# Uses task_env so the driver does not need vLLM for testcase grids (testcase+API would
# otherwise require USE_EXISTING_GRID_SPECS=1 — see src/baseline.py).

set -euo pipefail

cd /home/avani/projects/aip-lelis/avani/DSL_Generator

TS="$(date +%Y%m%d_%H%M%S)"

# Shared knobs (override before calling this script)
export BASELINE_VARIANT="${BASELINE_VARIANT:-task_env}"
export SPEC_FILE="${SPEC_FILE:-prompt_specifications/specification_with_updated_nld_baseline.txt}"
export NLD_PATH="${NLD_PATH:-prompt_specifications/nld_crafter.txt}"
export BASELINE_UNIQUE_DIR="${BASELINE_UNIQUE_DIR:-true}"
unset EXPERIMENT_DIR  # ignore any pre-set dir; each submit_baseline picks a fresh unique path

# API path
export MODEL_TYPE="openai_compat"
export JOB_PREFIX="${JOB_PREFIX_API:-baseline_api_${TS}}"
if [ -z "${OPENAI_COMPAT_KEY_FILE:-}" ]; then
  echo "ERROR: set OPENAI_COMPAT_KEY_FILE to your key file path (first line = API key)." >&2
  exit 1
fi
unset EXPERIMENT_DIR

echo "=== Submitting API baseline: JOB_PREFIX=$JOB_PREFIX ==="
bash scripts/stages/submit_baseline.sh

# HF path (new experiment dir from submit_baseline unique naming)
export MODEL_TYPE="huggingface"
export JOB_PREFIX="${JOB_PREFIX_HF:-baseline_hf_${TS}}"
unset OPENAI_COMPAT_KEY_FILE
unset EXPERIMENT_DIR

echo "=== Submitting HuggingFace baseline: JOB_PREFIX=$JOB_PREFIX ==="
bash scripts/stages/submit_baseline.sh

echo "Done. Check scripts/log/<experiment_basename>/ for stage_baseline_*.out"
