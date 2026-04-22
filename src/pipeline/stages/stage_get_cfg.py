#!/usr/bin/env python3
"""
Stage 1: Get CFG
This stage generates or loads a CFG from the experiment directory.
"""

import os
import sys
import json
import argparse

# Add project root to path
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _project_root)

from src.pipeline.cfg_to_funsearch_pipeline import get_cfg
from src.utils.pipeline_state import update_state

# Import vLLM for shared instance
try:
    from vllm import LLM as vLLM
except ImportError:
    vLLM = None


def main():
    parser = argparse.ArgumentParser(description="Stage 1: Get CFG")
    parser.add_argument('--experiment_dir', type=str, default=None, help='Experiment directory (auto-generated if not provided)')
    parser.add_argument('--skip_cfg_generation', action='store_true', help='Skip CFG generation and load from file')
    parser.add_argument('--cfg_output_file', type=str, default=None, help='File to load CFG from')
    parser.add_argument('--max_cfg_retries', type=int, default=int(os.environ.get("MAX_CFG_RETRIES", "10")), help='Maximum retries for CFG generation')
    parser.add_argument('--nld_path', type=str, default=os.environ.get("NLD_PATH", "prompt_specifications/nld.txt"), help='Path to natural language domain description')
    parser.add_argument('--recipes_path', type=str, default=os.environ.get("RECIPES_PATH"), help='Optional path to recipes/domain file')
    parser.add_argument('--cfg_generator_prompt_path', type=str, default=os.environ.get("CFG_GENERATOR_PROMPT_PATH", "prompt_specifications/cfg_generator.txt"), help='Path to CFG generator prompt template')
    parser.add_argument('--domain_context_template_path', type=str, default=os.environ.get("DOMAIN_CONTEXT_TEMPLATE_PATH"), help='Path to domain context template')
    
    args = parser.parse_args()
    # Auto-generate experiment directory if not provided or empty
    if not args.experiment_dir or args.experiment_dir.strip() == "":
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_root = "experiments"
        os.makedirs(base_root, exist_ok=True)
        args.experiment_dir = os.path.join(base_root, f"experiment_{timestamp}")
        print(f"\n{'='*80}", flush=True)
        print(f"Experiment directory not provided, auto-generating: {args.experiment_dir}", flush=True)
        print(f"{'='*80}\n", flush=True)
        sys.stdout.flush()
    
    # Create shared LLM instance if needed.
    # openai_compat uses HTTP API wrapper; other modes keep existing vLLM behavior.
    shared_vllm = None
    if not args.skip_cfg_generation:
        model_type = os.environ.get("MODEL_TYPE", "huggingface").strip().lower()
        if model_type == "openai_compat":
            from src.pipeline.explicit_feedback_generation import OpenAICompatLLMWrapper
            key_file = os.environ.get("OPENAI_COMPAT_KEY_FILE", "").strip() or None
            print("\n[Setup] Using OpenAI-compatible API for CFG generation...")
            shared_vllm = OpenAICompatLLMWrapper(key_file)
            print(" OpenAI-compatible API wrapper created")
        elif vLLM is not None:
            try:
                print("\n[Setup] Initializing shared vLLM instance...")
                shared_vllm = vLLM(model="/scratch/avani/gpt", tensor_parallel_size=4)
                print(" Shared vLLM instance created")
            except Exception as e:
                print(f" Warning: Could not create shared vLLM instance: {e}")
                shared_vllm = None
    
    # Ensure experiment directory exists
    os.makedirs(args.experiment_dir, exist_ok=True)
    os.makedirs(os.path.join(args.experiment_dir, "cfg"), exist_ok=True)
    
    # Get CFG
    cfg, terminals, example, success = get_cfg(
        experiment_dir=args.experiment_dir,
        skip_cfg_generation=args.skip_cfg_generation,
        cfg_output_file=args.cfg_output_file,
        max_cfg_retries=args.max_cfg_retries,
        nld_path=args.nld_path,
        recipes_path=args.recipes_path,
        cfg_generator_prompt_path=args.cfg_generator_prompt_path,
        domain_context_template_path=args.domain_context_template_path,
        shared_vllm=shared_vllm
    )
    
    if not success or not cfg or not terminals:
        print(" Failed to get valid CFG", file=sys.stderr)
        return 1
    
    print(f"\n Got CFG with {len(terminals)} terminal functions")
    
    # Get tasks from environment variable, with default from config file
    tasks_env = os.environ.get("TASKS", "")
    if tasks_env:
        # Try to parse as JSON first, then as space-separated
        try:
            tasks = json.loads(tasks_env)
        except:
            tasks = tasks_env.split()
    else:
        # Use default tasks from config file if TASKS environment variable not set
        task_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "config", "task_config.json")
        if os.path.exists(task_config_path):
            try:
                with open(task_config_path, 'r') as f:
                    task_config = json.load(f)
                tasks = task_config.get("tasks", ["make[stick]"])  # Fallback to single task if config missing
            except:
                tasks = ["make[stick]"]  # Fallback if config file can't be read
        else:
            tasks = ["make[stick]"]  # Fallback if config file doesn't exist
    
    # Get evolution limits from environment
    max_dsl_evolutions = int(os.environ.get("MAX_DSL_EVOLUTIONS", "2"))
    max_function_evolutions = int(os.environ.get("MAX_FUNCTION_EVOLUTIONS", "1"))
    
    # Update state file with total number of terminal functions and tasks
    update_state(
        args.experiment_dir,
        function_implementation_total=len(terminals),
        function_implementation_remaining=len(terminals),
        test_tasks_total=len(tasks),  # Informational only (test_tasks runs in single job)
        tasks=json.dumps(tasks) if tasks else "[]",
        phase="initial",
        dsl_round=0,
        max_dsl_evolutions=max_dsl_evolutions,
        dsl_evolutions_remaining=max_dsl_evolutions,
        func_evolution_round=0,
        max_function_evolutions=max_function_evolutions
    )
    print(f"  Updated state file: {len(terminals)} terminal functions, {len(tasks)} tasks")
    
    # Save stage completion marker
    stage_status = {
        "stage": "get_cfg",
        "status": "completed",
        "cfg_path": os.path.join(args.experiment_dir, "cfg", "cfg_output.json"),
        "num_terminals": len(terminals)
    }
    status_file = os.path.join(args.experiment_dir, "stage_get_cfg_status.json")
    with open(status_file, 'w') as f:
        json.dump(stage_status, f, indent=2)
    
    # Chaining will be handled by the SLURM script after this Python script completes
    print("\n[Chaining] State file updated. SLURM script will handle chaining to next stage.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


