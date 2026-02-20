#!/usr/bin/env python3
"""
Configuration file loader for experiment parameters.
Supports YAML config files with environment variable overrides.
"""

import os
import sys
import yaml
import json
from typing import Dict, Any, Optional, List
from pathlib import Path


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load experiment configuration from YAML file or environment variables.
    
    Priority order:
    1. Environment variables (highest priority)
    2. Config file values
    3. Default values (lowest priority)
    
    Args:
        config_path: Path to YAML config file. If None, looks for:
                    - EXPERIMENT_CONFIG environment variable
                    - config/experiment_config.yaml in project root
                    - config/experiment_config.yaml.example as fallback
    
    Returns:
        Dictionary of configuration values
    """
    config = {}
    
    # Try to load from config file
    if config_path is None:
        # Check environment variable first
        config_path = os.environ.get("EXPERIMENT_CONFIG")
        
        # If not set, try default locations
        if config_path is None:
            project_root = Path(__file__).parent.parent.parent
            default_path = project_root / "config" / "experiment_config.yaml"
            example_path = project_root / "config" / "experiment_config.yaml.example"
            
            if default_path.exists():
                config_path = str(default_path)
            elif example_path.exists():
                print(f"⚠ Warning: Using example config file. Copy to experiment_config.yaml to customize.", file=sys.stderr)
                config_path = str(example_path)
    
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
            # Print to stderr to avoid interfering with bash eval
            print(f"✓ Loaded config from: {config_path}", file=sys.stderr)
        except Exception as e:
            print(f"⚠ Warning: Failed to load config file {config_path}: {e}", file=sys.stderr)
            config = {}
    elif config_path:
        print(f"⚠ Warning: Config file not found: {config_path}", file=sys.stderr)
    
    # Environment variable overrides (highest priority)
    # Map environment variable names to config keys
    env_mappings = {
        "EXPERIMENT_DIR": "experiment_dir",
        "SPEC_FILE": "spec_file",
        "TASKS": "tasks",  # Can be JSON array or space-separated
        "MAX_DSL_EVOLUTIONS": "max_dsl_evolutions",
        "MAX_FUNCTION_EVOLUTIONS": "max_function_evolutions",
        "TOTAL_SAMPLES": "total_samples",
        "NUM_EXPLICIT_FEEDBACK_ITERATIONS": "num_explicit_feedback_iterations",
        "MAX_ATTEMPTS": "max_attempts",
        "MODEL_TYPE": "model_type",
        "RECIPES_PATH": "recipes_path",
        "HINTS_PATH": "hints_path",
        "SKIP_CFG_GENERATION": "skip_cfg_generation",
        "CFG_OUTPUT_FILE": "cfg_output_file",
        "MAX_CFG_RETRIES": "max_cfg_retries",
        "GRID_REGENERATION_ATTEMPTS": "grid_regeneration_attempts",
        "USE_EXISTING_GRID_SPECS": "use_existing_grid_specs",
        "GRID_SPEC_DIR": "grid_spec_dir",
    }
    
    for env_var, config_key in env_mappings.items():
        env_value = os.environ.get(env_var)
        if env_value is not None:
            # Convert string values to appropriate types
            if config_key in ["max_dsl_evolutions", "max_function_evolutions", 
                             "total_samples", "num_explicit_feedback_iterations",
                             "max_attempts", "max_cfg_retries", "grid_regeneration_attempts"]:
                try:
                    config[config_key] = int(env_value)
                except ValueError:
                    pass
            elif config_key == "skip_cfg_generation":
                config[config_key] = env_value.lower() in ("true", "1", "yes")
            elif config_key == "use_existing_grid_specs":
                config[config_key] = env_value.lower() in ("true", "1", "yes")
            elif config_key == "grid_spec_dir":
                config[config_key] = env_value
            elif config_key == "tasks":
                # Handle tasks - can be JSON array or space-separated
                try:
                    # Try parsing as JSON first
                    config[config_key] = json.loads(env_value)
                except (json.JSONDecodeError, TypeError):
                    # If not JSON, treat as space-separated
                    config[config_key] = env_value.split()
            else:
                config[config_key] = env_value
    
    # Set defaults for missing values
    defaults = {
        "experiment_dir": None,
        "spec_file": "prompt_specifications/specification_with_updated_nld.txt",
        "tasks": [],
        "max_dsl_evolutions": 2,
        "max_function_evolutions": 1,
        "total_samples": 1000,
        "num_explicit_feedback_iterations": 30,
        "max_attempts": 30,
        "model_type": "huggingface",
        "recipes_path": "craft/resources/recipes.yaml",
        "hints_path": "craft/resources/hints.yaml",
        "skip_cfg_generation": False,
        "cfg_output_file": None,
        "max_cfg_retries": 10,
        "grid_regeneration_attempts": 5,
        "use_existing_grid_specs": False,
        "grid_spec_dir": None,
    }
    
    for key, default_value in defaults.items():
        if key not in config:
            config[key] = default_value
    
    return config


def export_config_to_env(config: Dict[str, Any]) -> None:
    """
    Export configuration values to environment variables for SLURM scripts.
    
    Args:
        config: Configuration dictionary
    """
    env_mappings = {
        "experiment_dir": "EXPERIMENT_DIR",
        "spec_file": "SPEC_FILE",
        "tasks": "TASKS",  # Will be converted to space-separated string
        "max_dsl_evolutions": "MAX_DSL_EVOLUTIONS",
        "max_function_evolutions": "MAX_FUNCTION_EVOLUTIONS",
        "total_samples": "TOTAL_SAMPLES",
        "num_explicit_feedback_iterations": "NUM_EXPLICIT_FEEDBACK_ITERATIONS",
        "max_attempts": "MAX_ATTEMPTS",
        "model_type": "MODEL_TYPE",
        "recipes_path": "RECIPES_PATH",
        "hints_path": "HINTS_PATH",
        "skip_cfg_generation": "SKIP_CFG_GENERATION",
        "cfg_output_file": "CFG_OUTPUT_FILE",
        "max_cfg_retries": "MAX_CFG_RETRIES",
        "grid_regeneration_attempts": "GRID_REGENERATION_ATTEMPTS",
        "use_existing_grid_specs": "USE_EXISTING_GRID_SPECS",
        "grid_spec_dir": "GRID_SPEC_DIR",
    }
    
    for config_key, env_var in env_mappings.items():
        value = config.get(config_key)
        if value is not None:
            if config_key == "tasks" and isinstance(value, list):
                # Convert list to space-separated string for SLURM
                os.environ[env_var] = " ".join(str(t) for t in value)
            elif config_key in ("skip_cfg_generation", "use_existing_grid_specs"):
                os.environ[env_var] = "true" if value else "false"
            elif config_key == "grid_spec_dir" and value is not None:
                os.environ[env_var] = str(value)
            else:
                os.environ[env_var] = str(value)


def get_config_value(config: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    Get a configuration value with optional default.
    
    Args:
        config: Configuration dictionary
        key: Configuration key
        default: Default value if key not found
    
    Returns:
        Configuration value or default
    """
    return config.get(key, default)


if __name__ == "__main__":
    # Test the config loader
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    config = load_config(config_path)
    print("\nLoaded configuration:")
    print(json.dumps(config, indent=2, default=str))

