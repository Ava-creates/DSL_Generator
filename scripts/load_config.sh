#!/bin/bash
# Helper script to load configuration from YAML file and export to environment
# This script is sourced by SLURM scripts and submit_with_config.sh.

# Repo root (this file lives at scripts/load_config.sh)
_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"

# Prefer project venv so PyYAML and deps match the pipeline
_CONFIG_PYTHON="${_REPO_ROOT}/new_dsl_env/bin/python"
if [ ! -x "${_CONFIG_PYTHON}" ]; then
    _CONFIG_PYTHON="python3"
fi

CONFIG_FILE="${EXPERIMENT_CONFIG:-config/experiment_config.yaml}"
if [[ "${CONFIG_FILE}" != /* ]]; then
    CONFIG_FILE="${_REPO_ROOT}/${CONFIG_FILE}"
fi

if [ -f "$CONFIG_FILE" ]; then
    _exports="$(
        CONFIG_FILE="$CONFIG_FILE" \
        REPO_ROOT="$_REPO_ROOT" \
        "${_CONFIG_PYTHON}" << 'PY'
import os
import sys

repo = os.environ["REPO_ROOT"]
cfg_path = os.environ["CONFIG_FILE"]
sys.path.insert(0, repo)
os.chdir(repo)

from src.utils.config_loader import load_config

config = load_config(cfg_path)

for key, value in config.items():
    if key == "tasks" and isinstance(value, list):
        tasks_str = " ".join(str(t) for t in value)
        print(f'export TASKS="{tasks_str}"')
    elif key == "skip_cfg_generation":
        if not os.environ.get("SKIP_CFG_GENERATION"):
            print(f'export SKIP_CFG_GENERATION="{"true" if value else "false"}"')
    elif value is not None:
        value_str = str(value).replace('"', '\\"')
        env_mapping = {
            "experiment_dir": "EXPERIMENT_DIR",
            "spec_file": "SPEC_FILE",
            "nld_path": "NLD_PATH",
            "codebase_path": "CODEBASE_PATH",
            "failure_analysis_prompt": "FAILURE_ANALYSIS_PROMPT",
            "cfg_evolution_prompt": "CFG_EVOLUTION_PROMPT",
            "synthesis_prompt": "SYNTHESIS_PROMPT",
            "max_dsl_evolutions": "MAX_DSL_EVOLUTIONS",
            "max_function_evolutions": "MAX_FUNCTION_EVOLUTIONS",
            "total_samples": "TOTAL_SAMPLES",
            "num_explicit_feedback_iterations": "NUM_EXPLICIT_FEEDBACK_ITERATIONS",
            "max_attempts": "MAX_ATTEMPTS",
            "model_type": "MODEL_TYPE",
            "recipes_path": "RECIPES_PATH",
            "hints_path": "HINTS_PATH",
            "cfg_output_file": "CFG_OUTPUT_FILE",
            "max_cfg_retries": "MAX_CFG_RETRIES",
            "grid_regeneration_attempts": "GRID_REGENERATION_ATTEMPTS",
            "positive_grids": "POSITIVE_GRIDS",
            "negative_grids": "NEGATIVE_GRIDS",
            "edge_grids": "EDGE_GRIDS",
            "grid_spec_llm_attempts": "GRID_SPEC_LLM_ATTEMPTS",
            "use_existing_grid_specs": "USE_EXISTING_GRID_SPECS",
            "grid_spec_dir": "GRID_SPEC_DIR",
            "grid_prompt_path": "GRID_PROMPT_PATH",
            "require_test_type": "REQUIRE_TEST_TYPE",
            "skip_positive_grids": "SKIP_POSITIVE_GRIDS",
            "cfg_generator_prompt_path": "CFG_GENERATOR_PROMPT_PATH",
            "domain_context_template_path": "DOMAIN_CONTEXT_TEMPLATE_PATH",
            "cfg_text": "CFG_TEXT",
            "job_prefix": "JOB_PREFIX",
            "implement_cfg_single_time": "IMPLEMENT_CFG_SINGLE_TIME",
            "baseline_variant": "BASELINE_VARIANT",
            "phase2_only": "PHASE2_ONLY",
            "phase2_seed_round": "PHASE2_SEED_ROUND",
            "task_env_rounds": "TASK_ENV_ROUNDS",
        }
        if key == "use_existing_grid_specs":
            if not os.environ.get("USE_EXISTING_GRID_SPECS"):
                print(f'export USE_EXISTING_GRID_SPECS="{"true" if value else "false"}"')
        elif key == "require_test_type":
            if not os.environ.get("REQUIRE_TEST_TYPE"):
                print(f'export REQUIRE_TEST_TYPE="{"true" if value else "false"}"')
        elif key == "skip_positive_grids":
            if not os.environ.get("SKIP_POSITIVE_GRIDS"):
                print(f'export SKIP_POSITIVE_GRIDS="{"true" if value else "false"}"')
        elif key == "phase2_only":
            if not os.environ.get("PHASE2_ONLY"):
                print(f'export PHASE2_ONLY="{"true" if value else "false"}"')
        elif key in env_mapping:
            env_var = env_mapping[key]
            if not os.environ.get(env_var):
                print(f'export {env_var}="{value_str}"')
PY
)"
    _status=$?
    if [ "${_status}" -ne 0 ]; then
        echo "Error: failed to load YAML config via ${_CONFIG_PYTHON} (exit ${_status})." >&2
        echo "  Config path: ${CONFIG_FILE}" >&2
        echo "  Use repo venv: ${_REPO_ROOT}/new_dsl_env/bin/python" >&2
        return 2 2>/dev/null || exit 2
    fi
    eval "${_exports}"
    echo "Loaded configuration from: $CONFIG_FILE" >&2
else
    echo "Config file not found: $CONFIG_FILE (using environment variables or defaults)"
fi
