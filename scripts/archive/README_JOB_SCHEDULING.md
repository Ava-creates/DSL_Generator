# Job Scheduling Workflow

This directory contains SLURM scripts for running the unified pipeline with separate jobs for each CFG version.

## Overview

The workflow consists of:
1. **First Job**: Runs for 2 days or until first DSL iteration completes
2. **Subsequent Jobs**: One job per CFG version, automatically submitted when DSL evolves

## Quick Start

### Step 1: Submit First Job

```bash
# Option 1: Use the helper script
bash scripts/submit_first_job.sh

# Option 2: Submit directly
sbatch scripts/run_first_iteration.slurm
```

The first job will:
- Run for up to 2 days
- Get initial CFG
- Implement CFG, test on tasks, evolve functions if needed
- Evolve DSL (if tasks still fail)
- Exit with code **100** when DSL evolves

### Step 2: Submit Jobs for Each CFG Version

After the first job completes with exit code 100, submit the next job:

```bash
# Set experiment directory (from first job output)
export EXPERIMENT_DIR=<experiment_dir>

# Submit job for CFG version 1
sbatch scripts/run_cfg_version.slurm
```

**Note**: The `run_cfg_version_wrapper.sh` script will automatically submit the next job when DSL evolves (exit code 100).

## Scripts

### `run_first_iteration.slurm`
- **Time Limit**: 2 days
- **Purpose**: Run first DSL iteration
- **Exit Codes**:
  - `0`: All tasks solved
  - `100`: DSL evolved (submit next job)
  - Other: Error or time limit reached

### `run_cfg_version.slurm`
- **Time Limit**: 2 days per CFG version
- **Purpose**: Process one CFG version
- **Requires**: `EXPERIMENT_DIR` environment variable
- **Exit Codes**:
  - `0`: All tasks solved
  - `100`: DSL evolved (auto-submits next job)
  - Other: Error or time limit reached

### `submit_first_job.sh`
- Helper script to submit the first job
- Optional: Pass experiment directory as argument

## Workflow Details

### First Job Flow

1. Job starts, gets initial CFG
2. Implements CFG (funsearch, explicit feedback)
3. Tests CFG on tasks
4. If tasks fail, evolves functions (up to 3 rounds)
5. If still failing, evolves DSL
6. **Exits with code 100** when DSL evolves
7. Checkpoint saved with CFG version 1

### Subsequent Jobs Flow

1. Job loads checkpoint from previous job
2. Processes the CFG version specified in checkpoint
3. Implements CFG, tests, evolves functions if needed
4. If tasks still fail, evolves DSL
5. **Exits with code 100** when DSL evolves
6. **Automatically submits next job** for new CFG version
7. Checkpoint updated with new CFG version

### Manual Submission

If auto-submission fails, manually submit next job:

```bash
export EXPERIMENT_DIR=<experiment_dir>
sbatch scripts/run_cfg_version.slurm
```

## Environment Variables

You can set these before submitting jobs:

```bash
export EXPERIMENT_DIR="my_experiment"  # Required for CFG version jobs
export SPEC_FILE="prompt_specifications/specification_for_cfg.txt"
export TASKS="config/task_config.json"
export MAX_FUNCTION_EVOLUTIONS=3
export MAX_DSL_EVOLUTIONS=3
export RECIPES_PATH="craft/resources/recipes.yaml"
export HINTS_PATH="craft/resources/hints.yaml"
export MODEL_TYPE="huggingface"
```

## Monitoring

### Check Job Status

```bash
# Check all your jobs
squeue -u $USER

# Check specific job
squeue -j <job_id>
```

### View Logs

```bash
# First iteration job
tail -f scripts/log/first_iteration_<job_id>.out

# CFG version jobs
tail -f scripts/log/cfg_version_<job_id>.out
```

### Check Checkpoint

```bash
# View checkpoint (shows CFG version, DSL round, etc.)
cat <experiment_dir>/checkpoint.json | python3 -m json.tool
```

## Exit Codes

- **0**: All tasks solved - pipeline complete!
- **100**: DSL evolved - next job should be submitted
- **Other**: Error or time limit reached - check logs

## Benefits

1. **Better Scheduling**: Short jobs (2 days) get scheduled faster
2. **Fault Tolerance**: If a job fails, only that CFG version needs rerun
3. **Resource Management**: Each CFG version processed independently
4. **Progress Tracking**: Easy to see which CFG versions completed

## Example

```bash
# 1. Submit first job
bash scripts/submit_first_job.sh

# Wait for job to complete (check exit code in log)

# 2. If exit code is 100, submit next job
export EXPERIMENT_DIR=experiment_unified_20250115_120000_job12345
sbatch scripts/run_cfg_version.slurm

# 3. Subsequent jobs will auto-submit when DSL evolves
# (or submit manually if needed)
```

## Notes

- Each job has a 2-day time limit
- Checkpoints are saved in `<experiment_dir>/checkpoint.json`
- The unified pipeline automatically handles checkpoint resumption
- Jobs will continue until all tasks are solved or max DSL evolutions reached

