#!/bin/bash
# Helper script to load configuration from YAML file and export to environment
# This is sourced by SLURM scripts to set environment variables

# Get the config file path from environment or use default
CONFIG_FILE="${EXPERIMENT_CONFIG:-config/experiment_config.yaml}"

# If config file exists, load it and export to environment
if [ -f "$CONFIG_FILE" ]; then
    # Use Python to load config and export to environment
    # This script is sourced, so exports will be available to the calling script
    eval "$(python3 << EOF
import sys
import os
sys.path.insert(0, '/home/avani/projects/aip-lelis/avani/DSL_Generator')
from src.utils.config_loader import load_config

config = load_config('$CONFIG_FILE')

# Print export statements for bash to evaluate
for key, value in config.items():
    if key == 'tasks' and isinstance(value, list):
        # Export tasks as space-separated string
        tasks_str = ' '.join(str(t) for t in value)
        print(f"export TASKS=\"{tasks_str}\"")
    elif key == 'skip_cfg_generation':
        if not os.environ.get('SKIP_CFG_GENERATION'):
            print(f"export SKIP_CFG_GENERATION=\"{'true' if value else 'false'}\"")
    elif value is not None:
        # Escape quotes in values
        value_str = str(value).replace('"', '\\"')
        env_var = key.upper()
        # Map config keys to environment variable names
        env_mapping = {
            'experiment_dir': 'EXPERIMENT_DIR',
            'spec_file': 'SPEC_FILE',
            'nld_path': 'NLD_PATH',
            'codebase_path': 'CODEBASE_PATH',
            'failure_analysis_prompt': 'FAILURE_ANALYSIS_PROMPT',
            'cfg_evolution_prompt': 'CFG_EVOLUTION_PROMPT',
            'synthesis_prompt': 'SYNTHESIS_PROMPT',
            'max_dsl_evolutions': 'MAX_DSL_EVOLUTIONS',
            'max_function_evolutions': 'MAX_FUNCTION_EVOLUTIONS',
            'total_samples': 'TOTAL_SAMPLES',
            'num_explicit_feedback_iterations': 'NUM_EXPLICIT_FEEDBACK_ITERATIONS',
            'max_attempts': 'MAX_ATTEMPTS',
            'model_type': 'MODEL_TYPE',
            'recipes_path': 'RECIPES_PATH',
            'hints_path': 'HINTS_PATH',
            'cfg_output_file': 'CFG_OUTPUT_FILE',
            'max_cfg_retries': 'MAX_CFG_RETRIES',
            'grid_regeneration_attempts': 'GRID_REGENERATION_ATTEMPTS',
            'positive_grids': 'POSITIVE_GRIDS',
            'negative_grids': 'NEGATIVE_GRIDS',
            'edge_grids': 'EDGE_GRIDS',
            'grid_spec_llm_attempts': 'GRID_SPEC_LLM_ATTEMPTS',
            'use_existing_grid_specs': 'USE_EXISTING_GRID_SPECS',
            'grid_spec_dir': 'GRID_SPEC_DIR',
            'grid_prompt_path': 'GRID_PROMPT_PATH',
            'require_test_type': 'REQUIRE_TEST_TYPE',
            'skip_positive_grids': 'SKIP_POSITIVE_GRIDS',
            'cfg_generator_prompt_path': 'CFG_GENERATOR_PROMPT_PATH',
            'domain_context_template_path': 'DOMAIN_CONTEXT_TEMPLATE_PATH',
            'cfg_text': 'CFG_TEXT',
            'job_prefix': 'JOB_PREFIX',
            'baseline_variant': 'BASELINE_VARIANT',
            'phase2_only': 'PHASE2_ONLY',
            'phase2_seed_round': 'PHASE2_SEED_ROUND',
            'task_env_rounds': 'TASK_ENV_ROUNDS',
        }
        if key == 'use_existing_grid_specs':
            if not os.environ.get('USE_EXISTING_GRID_SPECS'):
                print(f"export USE_EXISTING_GRID_SPECS=\"{'true' if value else 'false'}\"")
        elif key == 'require_test_type':
            if not os.environ.get('REQUIRE_TEST_TYPE'):
                print(f"export REQUIRE_TEST_TYPE=\"{'true' if value else 'false'}\"")
        elif key == 'skip_positive_grids':
            if not os.environ.get('SKIP_POSITIVE_GRIDS'):
                print(f"export SKIP_POSITIVE_GRIDS=\"{'true' if value else 'false'}\"")
        elif key == 'phase2_only':
            if not os.environ.get('PHASE2_ONLY'):
                print(f"export PHASE2_ONLY=\"{'true' if value else 'false'}\"")
        elif key in env_mapping:
            env_var = env_mapping[key]
            if not os.environ.get(env_var):
                print(f"export {env_var}=\"{value_str}\"")
EOF
)" 2>/dev/null
    echo "Loaded configuration from: $CONFIG_FILE" >&2
else
    echo "Config file not found: $CONFIG_FILE (using environment variables or defaults)"
fi

