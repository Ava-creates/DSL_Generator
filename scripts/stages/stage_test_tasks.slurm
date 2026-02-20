#!/bin/bash
#SBATCH --job-name=stage_test_tasks
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=avani4@ualberta.ca
#SBATCH --output=scripts/log/stage_test_tasks_%j.out
#SBATCH --error=scripts/log/stage_test_tasks_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --account=aip-lelis

# Activate virtual environment
source new_dsl_env/bin/activate

# Set working directory to project root
cd /home/avani/projects/aip-lelis/avani/DSL_Generator

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

python src/pipeline/stages/stage_test_tasks.py \
    --experiment_dir "$EXPERIMENT_DIR" \
    --tasks $TASKS_ARGS \
    --recipes_path "${RECIPES_PATH:-craft/resources/recipes.yaml}" \
    --hints_path "${HINTS_PATH:-craft/resources/hints.yaml}" \
    --max_attempts "${MAX_ATTEMPTS:-30}" \
    --dsl_round "${DSL_ROUND:-0}" \
    $FUNC_ROUND_ARG

# Chain to next stage based on state (run in background so job exits immediately)
# sbatch commands are non-blocking and submit jobs immediately, so chaining should be fast
if [ $? -eq 0 ] || [ $? -eq 1 ]; then
    (
        source scripts/stages/chain_next_stage.sh
        chain_based_on_state
    ) &
    # Give chaining a brief moment to submit jobs (sbatch is fast but we ensure jobs are queued)
    sleep 1
    # Job exits immediately - background chaining will continue and jobs are already submitted
fi

