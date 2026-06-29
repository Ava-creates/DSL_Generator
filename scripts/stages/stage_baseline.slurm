#!/usr/bin/env bash
# Submit from repo root via scripts/stages/submit_baseline.sh

#SBATCH --job-name=baseline_orchestrator
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --account=aip-lelis

set -euo pipefail

ROOT="${DSL_GENERATOR_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$ROOT" ]]; then
  _here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  ROOT="$(cd "${_here}/../.." && pwd)"
fi
cd "$ROOT"

source new_dsl_env/bin/activate

export EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-config/baseline_config.yaml}"
export MODEL_TYPE="${MODEL_TYPE:-openai_compat}"
export OPENAI_COMPAT_KEY_FILE="${OPENAI_COMPAT_KEY_FILE:-${ROOT}/key.txt}"

if [[ "${MODEL_TYPE}" == "openai_compat" ]]; then
  if [[ ! -f "${OPENAI_COMPAT_KEY_FILE}" ]] && [[ -z "${OPENAI_COMPAT_API_KEY:-}" ]]; then
    echo "Missing API key: set OPENAI_COMPAT_API_KEY or create ${OPENAI_COMPAT_KEY_FILE}" >&2
    exit 1
  fi
fi

SPEC_FILE="${SPEC_FILE:-prompt_specifications/specification_with_updated_nld_baseline.txt}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-experiments/baseline_testcases}"
NLD_PATH="${NLD_PATH:-prompt_specifications/nld.txt}"
CODEBASE_PATH="${CODEBASE_PATH:-prompt_specifications/codebase.txt}"
BASELINE_VARIANT="${BASELINE_VARIANT:-task_env}"
FUNC_EVOLUTION_ROUND="${FUNC_EVOLUTION_ROUND:-0}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-1000}"
NUM_EXPLICIT_FEEDBACK_ITERATIONS="${NUM_EXPLICIT_FEEDBACK_ITERATIONS:-30}"
GRID_PROMPT="${GRID_PROMPT:-prompt_specifications/grid_prompt_withcoordinateinfo.txt}"
RECIPES_PATH="${RECIPES_PATH:-craft/resources/recipes.yaml}"
HINTS_PATH="${HINTS_PATH:-craft/resources/hints.yaml}"

JOB_PREFIX_ARG=()
if [[ -n "${JOB_PREFIX:-}" ]]; then
  JOB_PREFIX_ARG=(--job_prefix "${JOB_PREFIX}")
fi

exec python src/baseline.py \
  --spec_file "${SPEC_FILE}" \
  --experiment_dir "${EXPERIMENT_DIR}" \
  --model_type "${MODEL_TYPE}" \
  --openai_compat_key_file "${OPENAI_COMPAT_KEY_FILE}" \
  --nld_path "${NLD_PATH}" \
  --codebase_path "${CODEBASE_PATH}" \
  --func_evolution_round "${FUNC_EVOLUTION_ROUND}" \
  --total_samples "${TOTAL_SAMPLES}" \
  --num_explicit_feedback_iterations "${NUM_EXPLICIT_FEEDBACK_ITERATIONS}" \
  --baseline_variant "${BASELINE_VARIANT}" \
  --grid_prompt "${GRID_PROMPT}" \
  --recipes_path "${RECIPES_PATH}" \
  --hints_path "${HINTS_PATH}" \
  "${JOB_PREFIX_ARG[@]}"
