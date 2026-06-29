#!/bin/bash
#SBATCH --job-name=baseline_api
#SBATCH --account=aip-lelis
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=scripts/log/baseline_orchestrator_%j.out
#SBATCH --error=scripts/log/baseline_orchestrator_%j.err
#
# CPU-only API baseline; key defaults to <repo>/key.txt.
# Submit from repo root:
#   cd /path/to/DSL_Generator && sbatch scripts/stages/baseline_orchestrator_api.slurm
#
# Optional exports before sbatch: EXPERIMENT_DIR, SPEC_FILE, NLD_PATH, CODEBASE_PATH,
# BASELINE_VARIANT, FUNC_EVOLUTION_ROUND, TOTAL_SAMPLES, NUM_EXPLICIT_FEEDBACK_ITERATIONS,
# EXPERIMENT_CONFIG, GRID_PROMPT, JOB_PREFIX, OPENAI_COMPAT_KEY_FILE

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR}"
cd "${ROOT}"
mkdir -p scripts/log

export MODEL_TYPE=openai_compat
export EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-config/baseline_config.yaml}"
export OPENAI_COMPAT_KEY_FILE="${OPENAI_COMPAT_KEY_FILE:-${ROOT}/key.txt}"

. new_dsl_env/bin/activate

SPEC_FILE="${SPEC_FILE:-prompt_specifications/specification_with_updated_nld_baseline.txt}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-experiments/baseline_testcases}"
NLD_PATH="${NLD_PATH:-prompt_specifications/nld.txt}"
CODEBASE_PATH="${CODEBASE_PATH:-prompt_specifications/codebase.txt}"
BASELINE_VARIANT="${BASELINE_VARIANT:-task_env}"
FUNC_EVOLUTION_ROUND="${FUNC_EVOLUTION_ROUND:-0}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-1000}"
NUM_EXPLICIT_FEEDBACK_ITERATIONS="${NUM_EXPLICIT_FEEDBACK_ITERATIONS:-30}"
GRID_PROMPT="${GRID_PROMPT:-prompt_specifications/grid_prompt_withcoordinateinfo.txt}"

JOB_PREFIX_ARG=()
if [[ -n "${JOB_PREFIX:-}" ]]; then
  JOB_PREFIX_ARG=(--job-prefix "${JOB_PREFIX}")
fi

exec python src/baseline.py \
  --spec_file "${SPEC_FILE}" \
  --experiment_dir "${EXPERIMENT_DIR}" \
  --model_type openai_compat \
  --openai_compat_key_file "${OPENAI_COMPAT_KEY_FILE}" \
  --nld_path "${NLD_PATH}" \
  --codebase_path "${CODEBASE_PATH}" \
  --func_evolution_round "${FUNC_EVOLUTION_ROUND}" \
  --total_samples "${TOTAL_SAMPLES}" \
  --num_explicit_feedback_iterations "${NUM_EXPLICIT_FEEDBACK_ITERATIONS}" \
  --baseline_variant "${BASELINE_VARIANT}" \
  --grid_prompt "${GRID_PROMPT}" \
  "${JOB_PREFIX_ARG[@]}"
