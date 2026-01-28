# Pipeline Orchestrator

This directory contains a modular pipeline system where each stage runs as a separate SLURM job, orchestrated by a lightweight Python script that runs on the login node.

## Overview

The pipeline is broken down into separate stages:

1. **Stage 1: Get CFG** - Generate or load context-free grammar
2. **Stage 2: File Generation** - Generate function-specific prompts and func_init files
3. **Stage 3: FunSearch** - Run FunSearch for each terminal function
4. **Stage 4: Explicit Feedback** - Run explicit feedback generation
5. **Stage 5: Test Tasks** - Test CFG on tasks
6. **Stage 6: Evolve Functions** - Evolve functions with failing tasks (optional)
7. **Stage 7: Evolve DSL** - Evolve DSL when tasks still fail (optional)

Each stage runs as a separate SLURM job with proper dependencies, allowing for better resource management and easier debugging.

## Quick Start

### Basic Usage

The simplest way to run the pipeline:

```bash
bash scripts/run_orchestrator.sh \
    --experiment_dir experiment_001 \
    --tasks "make[stick]" "get[gem]"
```

This will:
- Use the default spec file: `prompt_specifications/specification_with_updated_nld.txt`
- Run in a tmux session (so you can detach and reattach)
- Submit separate jobs for each function and task

### Run with tmux (Recommended for Long Pipelines)

The orchestrator runs in a tmux session on the login node by default, allowing you to detach and reattach:

```bash
# Uses default spec file
bash scripts/run_orchestrator.sh \
    --experiment_dir experiment_001 \
    --tasks "make[stick]" "get[gem]"

# Attach to tmux session to monitor
tmux attach -t pipeline_orchestrator

# Or view logs
tail -f scripts/log/orchestrator_*.log
```

### Run without tmux

```bash
bash scripts/run_orchestrator.sh \
    --no-tmux \
    --experiment_dir experiment_001 \
    --tasks "make[stick]" "get[gem]"
```

### Using tasks from JSON file

```bash
bash scripts/run_orchestrator.sh \
    --experiment_dir experiment_001 \
    --tasks config/task_config.json
```

### Using a custom spec file

```bash
bash scripts/run_orchestrator.sh \
    --experiment_dir experiment_001 \
    --spec_file prompt_specifications/custom_spec.txt \
    --tasks "make[stick]" "get[gem]"
```

### Full Example with All Options

```bash
bash scripts/run_orchestrator.sh \
    --experiment_dir experiment_unified_$(date +%Y%m%d_%H%M%S) \
    --spec_file prompt_specifications/specification_with_updated_nld.txt \
    --tasks "make[stick]" "get[gem]" "make[axe]" \
    --max_dsl_evolutions 3 \
    --max_function_evolutions 3 \
    --max_attempts 1 \
    --model_type huggingface
```

## Architecture

### Stage Scripts

Each stage is implemented as a standalone Python script in `src/pipeline/stages/`:

**Batch Stages (run once):**
- `stage_get_cfg.py` - Get CFG
- `stage_file_generation.py` - Generate function files
- `stage_evolve_dsl.py` - Evolve DSL

**Per-Function Stages (run once per function, in parallel):**
- `stage_funsearch_single.py` - Run FunSearch for a single function
- `stage_explicit_feedback_single.py` - Run explicit feedback for a single function
- `stage_evolve_function_single.py` - Evolve a single function

**Per-Task Stages (run once per task, in parallel):**
- `stage_test_task_single.py` - Test a single task

Each stage script:
- Reads input from previous stages via status JSON files
- Writes output to status JSON files for next stages
- Can be run independently for debugging
- Per-function/task stages create individual status files (e.g., `stage_funsearch_MOVE_status.json`)

### SLURM Scripts

Each stage has a corresponding SLURM script in `scripts/stages/`:

- `stage_get_cfg.slurm`
- `stage_file_generation.slurm`
- `stage_funsearch.slurm`
- `stage_explicit_feedback.slurm`
- `stage_test_tasks.slurm`
- `stage_evolve_functions.slurm`
- `stage_evolve_dsl.slurm`

### Orchestrator

The orchestrator (`scripts/orchestrator.py`) is a lightweight Python script that:

1. Submits jobs in the correct order with SLURM dependencies
2. Waits for jobs to complete
3. Checks exit codes and status files
4. Decides whether to continue with function evolution or DSL evolution
5. Handles the loop for DSL evolutions and function evolutions

The orchestrator runs on the login node and only submits jobs and waits - it doesn't do heavy computation, making it compliant with login node policies.

## Workflow

### Initial Run

1. **Get CFG** - Generate initial CFG (1 job)
2. **File Generation** - Generate function prompts and init files (1 job)
3. **FunSearch** - Run FunSearch for each function **in parallel** (N jobs, one per function)
4. **Explicit Feedback** - Run explicit feedback for each function **in parallel** (N jobs, one per function)
5. **Test Tasks** - Test each task **in parallel** (M jobs, one per task)

### If Tasks Fail

6. **Evolve Functions** (up to `max_function_evolutions` times):
   - Evolve each function with failing tasks **in parallel** (N jobs, one per function)
   - Re-run FunSearch for evolved functions **in parallel** (N jobs)
   - Re-run explicit feedback for evolved functions **in parallel** (N jobs)
   - Re-test failing tasks **in parallel** (M jobs, one per task)
   - If all solved, exit successfully

### If Still Failing

7. **Evolve DSL**:
   - Evolve the DSL
   - Return to step 2 (File Generation) with new DSL
   - Repeat up to `max_dsl_evolutions` times

## Monitoring

### View orchestrator logs

If running in tmux:
```bash
tmux attach -t pipeline_orchestrator
```

Or view the log file:
```bash
tail -f scripts/log/orchestrator_*.log
```

### Monitor individual stage jobs

```bash
# View all running jobs
squeue -u $USER

# View logs for a specific stage type
tail -f scripts/log/stage_funsearch_*.out
tail -f scripts/log/stage_explicit_feedback_*.out
tail -f scripts/log/stage_test_task_*.out

# View logs for a specific job ID
tail -f scripts/log/stage_funsearch_<job_id>.out
```

### Check stage status

Each stage writes a status JSON file to the experiment directory:

```bash
# Check if CFG stage completed
cat experiment_001/stage_get_cfg_status.json

# Check FunSearch status for a specific function
cat experiment_001/stage_funsearch_MOVE_status.json

# Check explicit feedback status for a specific function
cat experiment_001/stage_explicit_feedback_MOVE_status.json

# Check test status for a specific task
cat experiment_001/stage_test_task_make_stick__status.json
```

### Monitor job progress

```bash
# See all jobs in queue
squeue -u $USER

# See detailed info about a job
scontrol show job <job_id>

# See job efficiency
seff <job_id>

# Cancel a specific job
scancel <job_id>

# Cancel all your jobs
scancel -u $USER
```

## Configuration

### Environment Variables

The orchestrator and stage scripts use environment variables for configuration:

- `EXPERIMENT_DIR` - Experiment directory
- `SPEC_FILE` - Specification file path
- `TASKS` - Space-separated list of tasks
- `DSL_ROUND` - Current DSL evolution round (0-indexed)
- `FUNC_EVOLUTION_ROUND` - Current function evolution round (0-indexed)
- `MODEL_TYPE` - Model type (huggingface, ollama, gemini)
- `MAX_ATTEMPTS` - Maximum attempts per task
- `RECIPES_PATH` - Path to recipes YAML
- `HINTS_PATH` - Path to hints YAML

### Command Line Options

See `bash scripts/run_orchestrator.sh --help` for all options.

## Benefits

1. **Modularity** - Each stage is independent and can be debugged separately
2. **Resource Management** - Each stage can have different resource requirements
3. **Fault Tolerance** - Failed stages can be re-run independently
4. **Login Node Compliance** - Orchestrator is lightweight and only submits jobs
5. **Long-Running Pipelines** - Can run in tmux session for days/weeks
6. **Easy Monitoring** - Each stage has its own logs and status files

## Troubleshooting

### Job fails

1. Check the stage log: `tail -f scripts/log/stage_*_<job_id>.err`
2. Check the status file: `cat <experiment_dir>/stage_<stage_name>_status.json`
3. Re-run the stage manually if needed

### Orchestrator stops

1. Check orchestrator log: `tail -f scripts/log/orchestrator_*.log`
2. Check if tmux session is still running: `tmux ls`
3. Re-attach to tmux: `tmux attach -t pipeline_orchestrator`

### Jobs stuck in queue

1. Check queue: `squeue -u $USER`
2. Check job dependencies: `squeue -j <job_id> -o "%D"`
3. Cancel and restart if needed: `scancel <job_id>`

## Example: Running a Full Pipeline

```bash
# Start orchestrator in tmux (uses default spec file)
bash scripts/run_orchestrator.sh \
    --experiment_dir experiment_unified_$(date +%Y%m%d_%H%M%S) \
    --tasks config/task_config.json \
    --max_dsl_evolutions 3 \
    --max_function_evolutions 3 \
    --max_attempts 1

# Monitor progress
tmux attach -t pipeline_orchestrator

# Or check logs
tail -f scripts/log/orchestrator_*.log
```

