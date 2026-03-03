#!/bin/bash

log_chain_context() {
    local stage_name="$1"
    local ts
    ts="$(date '+%Y-%m-%d %H:%M:%S')"

    echo "[CHAIN-CONTEXT][$stage_name] timestamp=${ts}"
    echo "[CHAIN-CONTEXT][$stage_name] slurm_job_id=${SLURM_JOB_ID:-<unset>} slurm_job_name=${SLURM_JOB_NAME:-<unset>}"
    echo "[CHAIN-CONTEXT][$stage_name] node=${SLURMD_NODENAME:-${HOSTNAME:-<unknown>}}"
    echo "[CHAIN-CONTEXT][$stage_name] experiment_dir=${EXPERIMENT_DIR:-<unset>}"
    echo "[CHAIN-CONTEXT][$stage_name] experiment_config=${EXPERIMENT_CONFIG:-<unset>}"
    echo "[CHAIN-CONTEXT][$stage_name] cwd=$(pwd)"
}
