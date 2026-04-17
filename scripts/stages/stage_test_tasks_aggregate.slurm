#!/bin/bash
#SBATCH --job-name=stage_test_tasks_aggregate
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=avani4@ualberta.ca
#SBATCH --output=scripts/log/stage_test_tasks_aggregate_%j.out
#SBATCH --error=scripts/log/stage_test_tasks_aggregate_%j.err
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --account=aip-lelis

source new_dsl_env/bin/activate
cd /home/avani/projects/aip-lelis/avani/DSL_Generator

source scripts/stages/log_chain_context.sh
log_chain_context "stage_test_tasks_aggregate"

if [ -n "$TASKS" ]; then
    TASKS_ARGS="$TASKS"
else
    TASKS_ARGS="${TASKS_JSON:-}"
fi

python src/pipeline/stages/stage_test_tasks_aggregate.py \
    --experiment_dir "$EXPERIMENT_DIR" \
    --tasks $TASKS_ARGS \
    --dsl_round "${DSL_ROUND:-0}" \
    --func_evolution_round "${FUNC_EVOLUTION_ROUND:-0}"

if [ $? -eq 0 ]; then
    (
        source scripts/stages/chain_next_stage.sh
        chain_based_on_state
    ) &
fi
