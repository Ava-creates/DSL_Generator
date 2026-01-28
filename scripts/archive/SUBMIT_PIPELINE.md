# How to Submit the Self-Chaining Pipeline

The pipeline is now fully self-chaining! You only need to submit the first job (`stage_get_cfg.slurm`) and it will automatically chain through all subsequent stages.

**All job submissions happen in SLURM scripts** - Python stages only update the state file, and the SLURM scripts check state and submit the next stage.

## Quick Start

### Basic Submission

```bash
# Set required environment variables
export EXPERIMENT_DIR="experiment_001"  # Or omit to auto-generate
export TASKS='["make[stick]", "get[gem]"]'  # JSON array format
export SPEC_FILE="prompt_specifications/specification_with_updated_nld.txt"

# Optional: Set other parameters
export MAX_DSL_EVOLUTIONS=3
export MAX_FUNCTION_EVOLUTIONS=3
export MODEL_TYPE="huggingface"
export TOTAL_SAMPLES=1000
export NUM_EXPLICIT_FEEDBACK_ITERATIONS=1
export MAX_ATTEMPTS=1
export RECIPES_PATH="craft/resources/recipes.yaml"
export HINTS_PATH="craft/resources/hints.yaml"

# Submit the first job - pipeline will chain automatically
sbatch --export=ALL scripts/stages/stage_get_cfg.slurm
```

**Note**: If `EXPERIMENT_DIR` is not set, the pipeline will auto-generate a unique directory name with a timestamp (e.g., `experiment_20251223_090205`). However, for parallel experiments, it's recommended to use the helper script (`run_multiple_experiments.sh`) which generates unique names upfront.

### Alternative: Space-Separated Tasks

If you prefer space-separated format:

```bash
export EXPERIMENT_DIR="experiment_001"
export TASKS="make[stick] get[gem]"  # Space-separated
export SPEC_FILE="prompt_specifications/specification_with_updated_nld.txt"

sbatch --export=ALL scripts/stages/stage_get_cfg.slurm
```

## Environment Variables

### Required Variables

- **`EXPERIMENT_DIR`**: Directory name for this experiment (e.g., `experiment_001`)
- **`TASKS`**: List of tasks to solve. Can be:
  - JSON array: `'["make[stick]", "get[gem]"]'`
  - Space-separated: `"make[stick] get[gem]"`
- **`SPEC_FILE`**: Path to specification file (default: `prompt_specifications/specification_with_updated_nld.txt`)

### Optional Variables

- **`MAX_DSL_EVOLUTIONS`**: Maximum DSL evolution rounds (default: `3`). Set this to control how many times the DSL can evolve before the pipeline stops.
- **`MAX_FUNCTION_EVOLUTIONS`**: Maximum function evolution rounds (default: `1`). Set this to control how many times functions can evolve per DSL round before moving to DSL evolution.
- **`MODEL_TYPE`**: Model type - `huggingface`, `ollama`, or `gemini` (default: `huggingface`)
- **`TOTAL_SAMPLES`**: Total samples for FunSearch per function (default: `1000`)
- **`NUM_EXPLICIT_FEEDBACK_ITERATIONS`**: Explicit feedback iterations (default: `1`)
- **`MAX_ATTEMPTS`**: Maximum attempts per task (default: `1`)
- **`RECIPES_PATH`**: Path to recipes YAML (default: `craft/resources/recipes.yaml`)
- **`HINTS_PATH`**: Path to hints YAML (default: `craft/resources/hints.yaml`)
- **`SKIP_CFG_GENERATION`**: Set to `"true"` to skip CFG generation (default: `false`)
- **`CFG_OUTPUT_FILE`**: Path to existing CFG file to load (if skipping generation)
- **`MAX_CFG_RETRIES`**: Maximum CFG generation retries (default: `10`)

## Running Multiple Experiments in Parallel

You can run multiple experiments simultaneously. Each experiment will have:
- Its own experiment directory and state file
- Unique job names (prefixed with experiment directory name)
- Independent pipeline execution

### Using the Helper Script

```bash
# Run multiple experiments from command line (with specific directory names)
./scripts/run_multiple_experiments.sh experiment_001 experiment_002 experiment_003

# Auto-generate multiple experiment directories
./scripts/run_multiple_experiments.sh --auto-generate 3

# Auto-generate with custom prefix
./scripts/run_multiple_experiments.sh --auto-generate 5 --prefix my_exp

# Or use a config file (one experiment directory per line, or "AUTO" for auto-generation)
cat > experiments.txt << EOF
experiment_001
AUTO
experiment_002
AUTO
EOF

./scripts/run_multiple_experiments.sh experiments.txt
```

**Note**: When using `--auto-generate` or `AUTO` in config files, unique experiment directory names are generated upfront (with timestamp + random component) and passed to the jobs. This ensures each experiment has a unique name even when submitted simultaneously.

### Manual Submission

```bash
# Submit multiple experiments manually
for exp_dir in experiment_001 experiment_002 experiment_003; do
    export EXPERIMENT_DIR="$exp_dir"
    export TASKS='["make[stick]"]'
    sbatch --job-name="${exp_dir}_get_cfg" --export=ALL scripts/stages/stage_get_cfg.slurm
done
```

### Monitoring Multiple Experiments

```bash
# View all jobs for a specific experiment
squeue -u $USER --name=<experiment_name>_*

# Cancel all jobs for a specific experiment
scancel --job-name=<experiment_name>_*

# View state of multiple experiments
for exp_dir in experiment_*/; do
    echo "=== $exp_dir ==="
    cat "$exp_dir/pipeline_state.txt" 2>/dev/null || echo "No state file"
    echo ""
done
```

## Examples

### Example 1: Simple Run

```bash
export EXPERIMENT_DIR="my_experiment"
export TASKS='["make[stick]"]'
export SPEC_FILE="prompt_specifications/specification_with_updated_nld.txt"

sbatch --export=ALL scripts/stages/stage_get_cfg.slurm
```

### Example 2: Multiple Tasks with Custom Settings

```bash
export EXPERIMENT_DIR="experiment_multi_task"
export TASKS='["make[stick]", "get[gem]", "make[plank]"]'
export SPEC_FILE="prompt_specifications/specification_with_updated_nld.txt"
export MAX_DSL_EVOLUTIONS=5
export MAX_FUNCTION_EVOLUTIONS=5
export TOTAL_SAMPLES=2000

sbatch --export=ALL scripts/stages/stage_get_cfg.slurm
```

### Example 3: Using Existing CFG

```bash
export EXPERIMENT_DIR="experiment_continue"
export TASKS='["make[stick]", "get[gem]"]'
export SPEC_FILE="prompt_specifications/specification_with_updated_nld.txt"
export SKIP_CFG_GENERATION="true"
export CFG_OUTPUT_FILE="previous_experiment/cfg/cfg_output.json"

sbatch --export=ALL scripts/stages/stage_get_cfg.slurm
```

## Monitoring

### Check Job Status

```bash
# View all your jobs
squeue -u $USER

# View specific job
squeue -j <job_id>
```

### View Logs

```bash
# View CFG generation log
tail -f scripts/log/stage_get_cfg_<job_id>.out

# View file generation log
tail -f scripts/log/stage_file_generation_<job_id>.out

# View FunSearch logs
tail -f scripts/log/stage_funsearch_*_<job_id>.out

# View explicit feedback logs
tail -f scripts/log/stage_explicit_feedback_*_<job_id>.out

# View test task logs
tail -f scripts/log/stage_test_task_*_<job_id>.out
```

### Check Pipeline State

```bash
# View current pipeline state
cat <experiment_dir>/pipeline_state.txt

# Example output:
# terminal_functions_remaining=2
# terminal_functions_total=5
# explicit_feedback_remaining=0
# test_tasks_remaining=1
# phase=initial
# dsl_round=0
# func_evolution_round=0
```

### Check Stage Status Files

```bash
# Check CFG status
cat <experiment_dir>/stage_get_cfg_status.json

# Check file generation status
cat <experiment_dir>/stage_file_generation_status.json

# Check individual function statuses
cat <experiment_dir>/stage_funsearch_<function_name>_status.json
cat <experiment_dir>/stage_explicit_feedback_<function_name>_status.json
cat <experiment_dir>/stage_test_task_<task>_status.json
```

## Pipeline Flow

Once submitted, the pipeline automatically chains through:

1. **Get CFG** → Generates/loads CFG, initializes state file
2. **File Generation** → Generates function files
3. **FunSearch** → Runs FunSearch for each terminal function (parallel)
4. **Explicit Feedback** → Runs explicit feedback for each function (parallel)
5. **Test Tasks** → Tests each task (parallel)
6. **Function Evolution** (if tasks fail) → Evolves functions, then chains back to FunSearch
7. **DSL Evolution** (if still failing) → Evolves DSL, then chains back to File Generation

## Troubleshooting

### Job Not Submitting

- Check that you're in the project root directory
- Verify environment variables are set correctly
- Check SLURM account: `sacctmgr show assoc user=$USER`

### Pipeline Stuck

- Check the state file: `cat <experiment_dir>/pipeline_state.txt`
- Check recent logs for errors
- Verify all required files exist

### Cancel All Jobs

```bash
# Cancel all your jobs
scancel -u $USER

# Cancel specific job and its dependencies
scancel <job_id>
```

## Notes

- The pipeline uses a state file (`pipeline_state.txt`) to track progress
- All stages chain automatically - no manual intervention needed
- Each stage has its own time limit in the SLURM script
- Jobs run in parallel where possible (FunSearch, explicit feedback, test tasks)
- The pipeline handles failures gracefully and continues when possible
- **Multiple experiments can run in parallel** - each has unique job names and independent state files
- Job names are prefixed with the experiment directory name (first 20 characters) for easy identification

