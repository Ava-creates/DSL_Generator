#!/bin/bash
#SBATCH --job-name=stage_test_tasks
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=avani4@ualberta.ca
#SBATCH --output=scripts/log/stage_test_tasks_%j.out
#SBATCH --error=scripts/log/stage_test_tasks_%j.err
#SBATCH --time=15:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --account=aip-lelis

# Activate virtual environment
source new_dsl_env/bin/activate

# Set working directory to project root
cd /home/avani/projects/aip-lelis/avani/DSL_Generator

source scripts/stages/log_chain_context.sh
log_chain_context "stage_test_tasks"

# Enable detailed LLM call tracing only for this stage.
export TEST_TASKS_LLM_DEBUG="${TEST_TASKS_LLM_DEBUG:-1}"

# Convert tasks array to space-separated string if needed
if [ -n "$TASKS" ]; then
    TASKS_ARGS="$TASKS"
else
    TASKS_ARGS="${TASKS_JSON:-}"
fi

# Run the stage script
if [ -n "${FUNC_EVOLUTION_ROUND:-}" ]; then
    FUNC_ROUND_ARG="--func_evolution_round $FUNC_EVOLUTION_ROUND"
else
    FUNC_ROUND_ARG=""
fi

if [ -n "${SYNTHESIS_PROMPT:-}" ]; then
    SYNTH_PROMPT_ARG="--synthesis_prompt $SYNTHESIS_PROMPT"
else
    SYNTH_PROMPT_ARG=""
fi

if [ "${SINGLE_TASK_JOB:-0}" = "1" ]; then
    SINGLE_TASK_ARG="--single_task_job"
else
    SINGLE_TASK_ARG=""
fi

# Must match DEFAULT_TEST_SEEDS in src/pipeline/stages/stage_test_tasks.py (0,5,...,45).
export TEST_SEEDS="0 5 10 15 20 25 30 35 40 45"
TEST_SEEDS_ARG="--test_seeds ${TEST_SEEDS}"

python src/pipeline/stages/stage_test_tasks.py \
    --experiment_dir "$EXPERIMENT_DIR" \
    --tasks $TASKS_ARGS \
    --recipes_path "${RECIPES_PATH:-craft/resources/recipes.yaml}" \
    --hints_path "${HINTS_PATH:-craft/resources/hints.yaml}" \
    --max_attempts "${MAX_ATTEMPTS:-30}" \
    $TEST_SEEDS_ARG \
    --dsl_round "${DSL_ROUND:-0}" \
    $FUNC_ROUND_ARG \
    $SINGLE_TASK_ARG \
    $SYNTH_PROMPT_ARG

# Chain to next stage based on state (run in background so job exits immediately)
# sbatch commands are non-blocking and submit jobs immediately, so chaining should be fast
if [ $? -eq 0 ] || [ $? -eq 1 ]; then
    (
        source scripts/stages/chain_next_stage.sh
        chain_based_on_state
    ) &
    # Give chaining a brief moment to submit jobs (sbatch is fast but we ensure jobs are queued)
    sleep 30
    # Job exits immediately - background chaining will continue and jobs are already submitted
fi

