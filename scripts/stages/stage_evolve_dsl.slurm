#!/bin/bash
# Slurm: run DSL / CFG evolution (stage_evolve_dsl.py).
#
# Default: MODEL_TYPE=huggingface + GPU + local vLLM (pipeline_hf-style).
# API-only:
#   sbatch --export=ALL,MODEL_TYPE=openai_compat scripts/stages/stage_evolve_dsl.slurm
#
# Example:
#   export EXPERIMENT_DIR=experiments/pipeline_hf_20260423_181855_1747661
#   sbatch scripts/stages/stage_evolve_dsl.slurm

#SBATCH --job-name=evolve_dsl
#SBATCH --account=aip-lelis
#SBATCH --chdir=/home/avani/projects/aip-lelis/avani/DSL_Generator
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=scripts/log/%x_%j.out
#SBATCH --error=scripts/log/%x_%j.err

set -euo pipefail

# Slurm runs a *copy* of this script from spool; BASH_SOURCE then breaks ../../ resolution.
# Match #SBATCH --chdir (override with DSL_GENERATOR_ROOT for other clones).
REPO_ROOT="${DSL_GENERATOR_ROOT:-/home/avani/projects/aip-lelis/avani/DSL_Generator}"
cd "${REPO_ROOT}"

mkdir -p scripts/log

EXPERIMENT_DIR="${EXPERIMENT_DIR:-experiments/pipeline_hf_20260423_181855_1747661}"
DSL_VERSION="${DSL_VERSION:-0}"
MAX_RETRIES="${MAX_RETRIES:-10}"
RECIPES_PATH="${RECIPES_PATH:-craft/resources/recipes.yaml}"
FAILING_TASKS="${FAILING_TASKS:-get[gem] get[wood] get[gold]}"
MAX_FAILED_PROGRAMS="${MAX_FAILED_PROGRAMS:-30}"
OPENAI_COMPAT_KEY_FILE="${OPENAI_COMPAT_KEY_FILE:-}"

if [[ -f "new_dsl_env/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source new_dsl_env/bin/activate
else
  echo "Expected ${REPO_ROOT}/new_dsl_env — edit script to use your venv." >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
export MODEL_TYPE="${MODEL_TYPE:-huggingface}"

source scripts/stages/log_chain_context.sh
log_chain_context "stage_evolve_dsl"

# shellcheck disable=SC2086
EXTRA_ARGS=()
if [[ -n "${OPENAI_COMPAT_KEY_FILE}" ]]; then
  EXTRA_ARGS+=(--openai_compat_key_file "${OPENAI_COMPAT_KEY_FILE}")
fi

python src/pipeline/stages/stage_evolve_dsl.py \
  --experiment_dir "${EXPERIMENT_DIR}" \
  --dsl_version "${DSL_VERSION}" \
  --recipes_path "${RECIPES_PATH}" \
  --max_retries "${MAX_RETRIES}" \
  --max_failed_programs "${MAX_FAILED_PROGRAMS}" \
  --failing_tasks ${FAILING_TASKS} \
  "${EXTRA_ARGS[@]}"

# Chain to next stage based on state (backup if in-process chaining did not submit).
if [ $? -eq 0 ]; then
  source scripts/stages/chain_next_stage.sh
  chain_based_on_state
fi
