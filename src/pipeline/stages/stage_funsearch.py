#!/usr/bin/env python3
"""
Stage 3: FunSearch
This stage runs FunSearch for each terminal function.
"""

import os
import sys
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _project_root)

from src.pipeline.cfg_to_funsearch_pipeline import determine_inputs
from funsearch.implementation.funsearch import FunSearch
from funsearch.implementation import config as config_lib
from src.utils.status_manager import read_status, write_status
from src.utils.pipeline_state import read_state

# Import vLLM for shared instance
try:
    from vllm import LLM as vLLM
except ImportError:
    vLLM = None


def main():
    parser = argparse.ArgumentParser(description="Stage 3: FunSearch")
    parser.add_argument('--experiment_dir', type=str, required=True, help='Experiment directory')
    parser.add_argument('--spec_file', type=str, required=True, help='Path to specification file')
    parser.add_argument('--model_type', type=str, default='huggingface', choices=['huggingface', 'ollama', 'gemini'])
    parser.add_argument('--dsl_round', type=int, default=0, help='DSL evolution round number')
    parser.add_argument('--func_evolution_round', type=int, default=None, help='Function evolution round number')
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
    
    # Get current DSL round from state (fallback to args if provided)
    state = read_state(args.experiment_dir)
    state_dsl_round = state.get("dsl_round", 0)
    if args.dsl_round != state_dsl_round:
        print(f"[FunSearch] Using dsl_round={state_dsl_round} from state file (cmd: {args.dsl_round})")
        args.dsl_round = state_dsl_round
    
    # Load file generation status to get func_files and func_init_files
    # Use backward-compatible read that tries both new and legacy locations
    file_gen_status = read_status(args.experiment_dir, args.dsl_round, "file_generation")
    if file_gen_status is None:
        print(f" File generation status not found for dsl_round={args.dsl_round}", file=sys.stderr)
        return 1
    
    func_files = file_gen_status.get("func_files", {})
    func_init_files = file_gen_status.get("func_init_files", {})
    
    if not func_files or not func_init_files:
        print(" Function files not found in file generation status", file=sys.stderr)
        return 1
    
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
            shared_vllm = vLLM(model="/scratch/avani/gpt", tensor_parallel_size=4)
            print(" Shared vLLM instance created")
        except Exception as e:
            print(f" Warning: Could not create shared vLLM instance: {e}")
            shared_vllm = None
    
    # Configure FunSearch
    config = config_lib.Config(
        num_samplers=1,
        num_evaluators=2,
        samples_per_prompt=2,
        total_samples=100,
        programs_database=config_lib.ProgramsDatabaseConfig(),
        grid_regeneration_attempts=args.grid_regeneration_attempts,
    )
    
    # Results directory
    results_dir = os.path.join(args.experiment_dir, "results", "funsearch")
    os.makedirs(results_dir, exist_ok=True)
    
    # Helper function to run FunSearch for a single function
    def run_funsearch_for_function(func_name, func_file, func_init_file, description):
        """Run FunSearch for a single function."""
        try:
            print(f"[{func_name}] Starting FunSearch...")
            funsearch = FunSearch(model_type=args.model_type, shared_vllm=shared_vllm)
            inputs = determine_inputs(func_name, description, cfg)
            
            funsearch.run(
                specification=specification,
                inputs=inputs,
                config=config,
                function_to_implement=func_file,
                function_init=func_init_file,
                spec_file=args.spec_file,
                experiment_dir=results_dir
            )
            print(f"[{func_name}]  Completed FunSearch")
            return func_name, "success", None
        except Exception as e:
            error_msg = str(e)
            print(f"[{func_name}]  Error: {error_msg}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return func_name, "error", error_msg
    
    # Run FunSearch in parallel
    print(f"\n[Step 4] Running FunSearch in parallel for {len(terminals)} functions...")
    max_workers = min(len(terminals), 16)
    print(f"  Using {max_workers} parallel workers")
    
    results = {}
    errors = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_func = {
            executor.submit(
                run_funsearch_for_function,
                func_name,
                func_files[func_name],
                func_init_files[func_name],
                description
            ): func_name
            for func_name, description in terminals.items()
        }
        
        for future in as_completed(future_to_func):
            func_name, status, error = future.result()
            results[func_name] = status
            if error:
                errors[func_name] = error
    
    # Check for errors
    if errors:
        print(f"\n FunSearch failed for {len(errors)} function(s):")
        for func_name, error in errors.items():
            print(f"  - {func_name}: {error}")
        print(" Stage failed", file=sys.stderr)
        return 1
    
    # Save stage completion marker
    stage_status = {
        "stage": "funsearch",
        "status": "completed",
        "dsl_round": args.dsl_round,
        "func_evolution_round": args.func_evolution_round,
        "results": results
    }
    write_status(args.experiment_dir, args.dsl_round, "funsearch", stage_status)
    
    print(f"\n All {len(terminals)} functions completed FunSearch successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())


