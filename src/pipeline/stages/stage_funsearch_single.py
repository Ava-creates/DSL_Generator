#!/usr/bin/env python3
"""
Stage 3: FunSearch (Single Function)
This stage runs FunSearch for a single terminal function.
"""

import os
import sys
import json
import argparse
from typing import Dict, Optional

# Add project root to path
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _project_root)

from src.pipeline.cfg_to_funsearch_pipeline import determine_inputs
from funsearch.implementation.funsearch import FunSearch
from funsearch.implementation import config as config_lib
from src.utils.results_tracker import ResultsTracker
from src.utils.pipeline_state import decrement_function_implementation, read_state

# Import vLLM for shared instance
try:
    from vllm import LLM as vLLM
except ImportError:
    vLLM = None


def main():
    parser = argparse.ArgumentParser(description="Stage 3: FunSearch (Single Function)")
    parser.add_argument('--experiment_dir', type=str, required=True, help='Experiment directory')
    parser.add_argument('--spec_file', type=str, required=True, help='Path to specification file')
    parser.add_argument('--function_name', type=str, required=True, help='Name of the function to run FunSearch for')
    parser.add_argument('--model_type', type=str, default='huggingface', choices=['huggingface', 'ollama', 'gemini'])
    parser.add_argument('--dsl_round', type=int, default=0, help='DSL evolution round number')
    parser.add_argument('--func_evolution_round', type=int, default=None, help='Function evolution round number')
    parser.add_argument('--total_samples', type=int, default=1000, help='Total number of samples for FunSearch (default: 1000)')
    parser.add_argument(
        '--grid_regeneration_attempts',
        type=int,
        default=int(os.environ.get("GRID_REGENERATION_ATTEMPTS", 5)),
        help='Attempts to regenerate grids when initial pass_check fails'
    )
    
    args = parser.parse_args()
    
    # Load CFG
    cfg_path = os.path.join(args.experiment_dir, "cfg", "cfg_output.json")
    if not os.path.exists(cfg_path):
        print(f" CFG file not found: {cfg_path}", file=sys.stderr)
        return 1
    
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg_data = json.load(f)
    cfg = cfg_data.get("cfg", "")
    terminals = cfg_data.get("terminals", {})
    
    if not cfg or not terminals:
        print(" Invalid CFG data", file=sys.stderr)
        return 1
    
    if args.function_name not in terminals:
        print(f" Function {args.function_name} not found in terminals", file=sys.stderr)
        return 1
    
    description = terminals[args.function_name]
    
    # Load file generation status to get func_files and func_init_files
    file_gen_status_path = os.path.join(args.experiment_dir, "stage_file_generation_status.json")
    if not os.path.exists(file_gen_status_path):
        print(f" File generation status not found: {file_gen_status_path}", file=sys.stderr)
        return 1
    
    with open(file_gen_status_path, 'r') as f:
        file_gen_status = json.load(f)
    
    func_files = file_gen_status.get("func_files", {})
    func_init_files = file_gen_status.get("func_init_files", {})
    
    if args.function_name not in func_files or args.function_name not in func_init_files:
        print(f" Function files not found for {args.function_name}", file=sys.stderr)
        return 1
    
    func_file = func_files[args.function_name]
    func_init_file = func_init_files[args.function_name]
    
    # Load specification
    if not os.path.exists(args.spec_file):
        print(f" Specification file not found: {args.spec_file}", file=sys.stderr)
        return 1
    
    with open(args.spec_file, 'r', encoding='utf-8') as f:
        specification = f.read()
    
    # Create shared vLLM instance
    shared_vllm = None
    if args.model_type == "huggingface" and vLLM is not None:
        try:
            print("\n[Setup] Initializing shared vLLM instance...")
            # Use gpu_memory_utilization=0.85 to leave headroom and avoid OOM
            # This helps prevent memory fragmentation issues
            shared_vllm = vLLM(
                model="/scratch/avani/gpt", 
                tensor_parallel_size=4,
                gpu_memory_utilization=0.85  # Use 85% of GPU memory to leave headroom
            )
            print(" Shared vLLM instance created")
        except Exception as e:
            print(f" Warning: Could not create shared vLLM instance: {e}")
            shared_vllm = None
    
    # Configure FunSearch - match the original configuration pattern
    # Match evaluators to samples_per_prompt for clean parallelization
    config = config_lib.Config(
        num_samplers=1,  # Single sampler - generates samples_per_prompt samples per iteration
        num_evaluators=2,  # Match samples_per_prompt - each evaluator handles one sample
        samples_per_prompt=2,  # 2 samples per prompt
        total_samples=args.total_samples,  # Total samples across all iterations
        programs_database=config_lib.ProgramsDatabaseConfig(),
        grid_regeneration_attempts=args.grid_regeneration_attempts,
    )
    
    # Calculate iterations from config (for tracking)
    import math
    if config.total_samples is not None:
        num_iterations = math.ceil(config.total_samples / (config.num_samplers * config.samples_per_prompt))
        total_samples_expected = config.num_samplers * num_iterations * config.samples_per_prompt
    else:
        num_iterations = config.num_iterations
        total_samples_expected = config.num_samplers * num_iterations * config.samples_per_prompt
    
    # Results directory
    results_dir = os.path.join(args.experiment_dir, "results", "funsearch")
    os.makedirs(results_dir, exist_ok=True)
    
    # Create results tracker to track environment steps
    results_tracker = ResultsTracker(args.experiment_dir)
    
    # Get initial funsearch steps (before this run)
    initial_funsearch_steps = results_tracker.interactions.get("funsearch", 0)
    
    # Run FunSearch for this function
    print(f"\n[{args.function_name}] Starting FunSearch...")
    print(f"  Target: {args.total_samples} total samples")
    print(f"  Config: {config.num_samplers} sampler(s), {config.samples_per_prompt} samples per prompt")
    print(f"  Will run: {num_iterations} iterations")
    print(f"  Expected total samples: {total_samples_expected}")
    
    # Prepare config info for status file (before try block so it's available in exception handler)
    config_info = {
        "total_samples": args.total_samples,
        "num_samplers": config.num_samplers,
        "num_evaluators": config.num_evaluators,
        "samples_per_prompt": config.samples_per_prompt,
        "num_iterations": num_iterations,
        "expected_total_samples": total_samples_expected
    }
    
    try:
        funsearch = FunSearch(model_type=args.model_type, shared_vllm=shared_vllm)
        # Set results tracker on FunSearch instance so evaluators can use it
        funsearch.results_tracker = results_tracker
        
        inputs = determine_inputs(args.function_name, description, cfg)
        
        funsearch.run(
            specification=specification,
            inputs=inputs,
            config=config,
            function_to_implement=func_file,
            function_init=func_init_file,
            spec_file=args.spec_file,
            experiment_dir=results_dir
        )
        print(f"[{args.function_name}]  Completed FunSearch")
        
        # Get total funsearch steps after completion
        final_funsearch_steps = results_tracker.interactions.get("funsearch", 0)
        steps_taken = final_funsearch_steps - initial_funsearch_steps
        
        print(f"[{args.function_name}] Environment steps taken: {steps_taken}")
        
        # Save stage completion marker for this function with tracking info
        status_file = os.path.join(args.experiment_dir, f"stage_funsearch_{args.function_name}_status.json")
        status_dir_file = os.path.join(
            args.experiment_dir, "status", "funsearch", f"{args.function_name}.json"
        )
        os.makedirs(os.path.dirname(status_dir_file), exist_ok=True)
        stage_status = {
            "stage": "funsearch",
            "function_name": args.function_name,
            "status": "completed",
            "dsl_round": args.dsl_round,
            "func_evolution_round": args.func_evolution_round,
            "config": config_info,
            "env_steps": steps_taken
        }
        for path in (status_file, status_dir_file):
            with open(path, 'w') as f:
                json.dump(stage_status, f, indent=2)
        
        # Decrement terminal function count and check if we should trigger explicit feedback
        print(f"\n[Chaining] Decrementing terminal function count...")
        remaining = decrement_function_implementation(args.experiment_dir)
        print(f"  Remaining terminal functions: {remaining}")
        
        # If this was the last funsearch job (remaining is now 0), let the SLURM chaining
        # script submit explicit feedback jobs. We avoid setting submission flags here.
        if remaining == 0:
            print(f"\n[Chaining] All FunSearch jobs completed. Chaining script will submit explicit feedback jobs.")
        
        return 0
    except Exception as e:
        error_msg = str(e)
        print(f"[{args.function_name}]  Error: {error_msg}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        
        # Get funsearch steps even on failure (if any were taken)
        final_funsearch_steps = results_tracker.interactions.get("funsearch", 0)
        steps_taken = final_funsearch_steps - initial_funsearch_steps
        
        # Save failure status with config info
        status_file = os.path.join(args.experiment_dir, f"stage_funsearch_{args.function_name}_status.json")
        status_dir_file = os.path.join(
            args.experiment_dir, "status", "funsearch", f"{args.function_name}.json"
        )
        os.makedirs(os.path.dirname(status_dir_file), exist_ok=True)
        stage_status = {
            "stage": "funsearch",
            "function_name": args.function_name,
            "status": "failed",
            "error": error_msg,
            "dsl_round": args.dsl_round,
            "func_evolution_round": args.func_evolution_round,
            "config": config_info,
            "env_steps": steps_taken
        }
        for path in (status_file, status_dir_file):
            with open(path, 'w') as f:
                json.dump(stage_status, f, indent=2)
        
        # Still decrement terminal function count even on failure
        # This ensures the pipeline can continue even if some jobs fail
        print(f"\n[Chaining] Decrementing terminal function count (after failure)...")
        remaining = decrement_function_implementation(args.experiment_dir)
        print(f"  Remaining terminal functions: {remaining}")
        
        # If this was the last funsearch job, still try to submit explicit feedback
        # Use atomic flag to prevent duplicate submissions
        if remaining == 0:
            if not mark_explicit_feedback_submitted(args.experiment_dir):
                print(f"  Explicit feedback jobs already submitted by another process")
            else:
                print(f"\n[Chaining] All FunSearch jobs completed (some may have failed). Submitting explicit feedback jobs...")
                
                # Load terminals to get all function names
                cfg_path = os.path.join(args.experiment_dir, "cfg", "cfg_output.json")
                with open(cfg_path, 'r', encoding='utf-8') as f:
                    cfg_data = json.load(f)
                terminals = cfg_data.get("terminals", {})
                
                # Chaining will be handled by the SLURM script after this Python script completes
                print(f"\n[Chaining] State file updated. SLURM script will handle chaining to explicit feedback jobs.")
        
        return 1


if __name__ == "__main__":
    sys.exit(main())

