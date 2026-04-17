#!/bin/bash
#SBATCH --job-name=stage_baseline
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=avani4@ualberta.ca
#SBATCH --output=scripts/log/stage_baseline_%j.out
#SBATCH --error=scripts/log/stage_baseline_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --account=aip-lelis

set -euo pipefail

source new_dsl_env/bin/activate
cd /home/avani/projects/aip-lelis/avani/DSL_Generator

mkdir -p scripts/log

# Optional config loader (supports EXPERIMENT_CONFIG and env overrides)
if [ -n "${EXPERIMENT_CONFIG:-}" ] || [ -f "config/baseline_config.yaml" ]; then
  source scripts/load_config.sh
fi

# Defaults
SPEC_FILE="${SPEC_FILE:-prompt_specifications/specification_with_updated_nld_baseline.txt}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-experiments/baseline_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}}"
MODEL_TYPE="${MODEL_TYPE:-huggingface}"
DSL_ROUND="${DSL_ROUND:-0}"
FUNC_EVOLUTION_ROUND="${FUNC_EVOLUTION_ROUND:-0}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-1000}"
NUM_EXPLICIT_FEEDBACK_ITERATIONS="${NUM_EXPLICIT_FEEDBACK_ITERATIONS:-30}"
GRID_PROMPT_PATH="${GRID_PROMPT_PATH:-prompt_specifications/grid_prompt.txt}"
REQUIRE_TEST_TYPE="${REQUIRE_TEST_TYPE:-true}"
SKIP_POSITIVE_GRIDS="${SKIP_POSITIVE_GRIDS:-false}"
CFG_TEXT="${CFG_TEXT:-}"
NLD_PATH="${NLD_PATH:-}"
CODEBASE_PATH="${CODEBASE_PATH:-}"
JOB_PREFIX="${JOB_PREFIX:-baseline}"
BASELINE_MODE="${BASELINE_MODE:-1}"
BASELINE_VARIANT="${BASELINE_VARIANT:-testcase}"
PHASE2_ONLY="${PHASE2_ONLY:-false}"
PHASE2_SEED_ROUND="${PHASE2_SEED_ROUND:-}"
TASK_ENV_ROUNDS="${TASK_ENV_ROUNDS:-1}"
OPENAI_COMPAT_KEY_FILE="${OPENAI_COMPAT_KEY_FILE:-}"

CMD=(
  python src/baseline.py
  --spec_file "$SPEC_FILE"
  --experiment_dir "$EXPERIMENT_DIR"
  --model_type "$MODEL_TYPE"
  --dsl_round "$DSL_ROUND"
  --func_evolution_round "$FUNC_EVOLUTION_ROUND"
  --total_samples "$TOTAL_SAMPLES"
  --num_explicit_feedback_iterations "$NUM_EXPLICIT_FEEDBACK_ITERATIONS"
  --job_prefix "$JOB_PREFIX"
  --grid_prompt "$GRID_PROMPT_PATH"
  --require_test_type "$REQUIRE_TEST_TYPE"
  --skip_positive_grids "$SKIP_POSITIVE_GRIDS"
  --baseline_variant "$BASELINE_VARIANT"
  --task_env_rounds "$TASK_ENV_ROUNDS"
)

if [ -n "$CFG_TEXT" ]; then
  CMD+=(--cfg_text "$CFG_TEXT")
fi

if [ -n "$NLD_PATH" ]; then
  CMD+=(--nld_path "$NLD_PATH")
fi
if [ -n "$CODEBASE_PATH" ]; then
  CMD+=(--codebase_path "$CODEBASE_PATH")
fi
if [ -n "$OPENAI_COMPAT_KEY_FILE" ]; then
  CMD+=(--openai_compat_key_file "$OPENAI_COMPAT_KEY_FILE")
fi
if [ "$PHASE2_ONLY" = "true" ] || [ "$PHASE2_ONLY" = "1" ]; then
  CMD+=(--phase2_only)
fi
if [ -n "$PHASE2_SEED_ROUND" ]; then
  CMD+=(--phase2_seed_round "$PHASE2_SEED_ROUND")
fi

echo "Running baseline with:"
echo "  SPEC_FILE=$SPEC_FILE"
echo "  EXPERIMENT_DIR=$EXPERIMENT_DIR"
echo "  MODEL_TYPE=$MODEL_TYPE"
"${CMD[@]}"
