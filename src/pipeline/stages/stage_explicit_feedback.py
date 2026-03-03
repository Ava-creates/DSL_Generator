#!/usr/bin/env python3
"""
Stage 4: Explicit Feedback Generation
This stage runs explicit feedback generation for each function.
"""

import os
import sys
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional

# Add project root to path
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _project_root)

from src.pipeline.cfg_to_funsearch_pipeline import run_explicit_feedback_generation

# Import vLLM for shared instance
try:
    from vllm import LLM as vLLM
except ImportError:
    vLLM = None


def main():
    parser = argparse.ArgumentParser(description="Stage 4: Explicit Feedback Generation")
    parser.add_argument('--experiment_dir', type=str, required=True, help='Experiment directory')
    parser.add_argument('--spec_file', type=str, required=True, help='Path to specification file')
    parser.add_argument('--model_type', type=str, default='huggingface', choices=['huggingface', 'ollama', 'gemini'])
    parser.add_argument('--dsl_round', type=int, default=0, help='DSL evolution round number')
    parser.add_argument('--func_evolution_round', type=int, default=None, help='Function evolution round number')
    parser.add_argument('--num_iterations', type=int, default=30, help='Number of explicit feedback iterations')
    
    args = parser.parse_args()
    
    # Load CFG
    cfg_path = os.path.join(args.experiment_dir, "cfg", "cfg_output.json")
    if not os.path.exists(cfg_path):
        print(f" CFG file not found: {cfg_path}", file=sys.stderr)
        return 1
    
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg_data = json.load(f)
    terminals = cfg_data.get("terminals", {})
    
    # Load file generation status
    file_gen_status_path = os.path.join(args.experiment_dir, "stage_file_generation_status.json")
    if not os.path.exists(file_gen_status_path):
        print(f" File generation status not found: {file_gen_status_path}", file=sys.stderr)
        return 1
    
    with open(file_gen_status_path, 'r') as f:
        file_gen_status = json.load(f)
    
    func_files = file_gen_status.get("func_files", {})
    func_signatures = file_gen_status.get("func_signatures", {})
    
    # Load funsearch status
    funsearch_status_path = os.path.join(args.experiment_dir, "stage_funsearch_status.json")
    if not os.path.exists(funsearch_status_path):
        print(f" FunSearch status not found: {funsearch_status_path}", file=sys.stderr)
        return 1
    
    with open(funsearch_status_path, 'r') as f:
        funsearch_status = json.load(f)
    
    funsearch_results = funsearch_status.get("results", {})
    successful_funcs = [func_name for func_name in terminals.keys() 
                       if funsearch_results.get(func_name) == "success"]
    
    if not successful_funcs:
        print(" No successful FunSearch results", file=sys.stderr)
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
    
    # Results directory
    results_dir = os.path.join(args.experiment_dir, "results", "funsearch")
    explicit_feedback_dir = os.path.join(args.experiment_dir, "explicit_feedback")
    os.makedirs(explicit_feedback_dir, exist_ok=True)
    
    # Helper function to run explicit feedback for a single function
    def run_explicit_feedback_for_function(func_name, func_file):
        """Run explicit feedback for a single function."""
        try:
            print(f"[{func_name}] Starting explicit feedback generation...")
            current_func_file = func_file
            
            # Run multiple iterations
            final_func = None
            current_func_code = None  # Keep function code in memory instead of saving intermediate files
            
            # Read initial function code
            with open(func_file, 'r', encoding='utf-8') as f:
                current_func_code = f.read()
            
            import tempfile
            for iteration in range(args.num_iterations):
                # Use temporary file for this iteration (will be cleaned up automatically)
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp_file:
                    tmp_file.write(current_func_code)
                    tmp_file_path = tmp_file.name
                
                try:
                    iter_func = run_explicit_feedback_generation(
                        func_name, results_dir, func_file, args.experiment_dir, explicit_feedback_dir,
                        specification, k=5, shared_vllm=shared_vllm, 
                        func_signature=func_signatures.get(func_name, ""),
                        results_tracker=None,
                        dsl_round=args.dsl_round, func_evolution_round=args.func_evolution_round
                    )
                    
                    if iter_func:
                        final_func = iter_func
                        current_func_code = iter_func  # Update for next iteration
                finally:
                    # Clean up temporary file immediately
                    try:
                        os.remove(tmp_file_path)
                    except OSError:
                        pass
            
            if final_func:
                print(f"[{func_name}]  Completed explicit feedback ({args.num_iterations} iterations)")
                return func_name, final_func, None
            else:
                print(f"[{func_name}]  No final function extracted")
                return func_name, None, None
        except Exception as e:
            error_msg = str(e)
            print(f"[{func_name}]  Error: {error_msg}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return func_name, None, error_msg
    
    # Run explicit feedback in parallel
    print(f"\n[Step 5] Running explicit feedback generation for {len(successful_funcs)} functions...")
    max_workers_ef = min(len(successful_funcs), 8)
    print(f"  Using {max_workers_ef} parallel workers")
    
    final_functions = {}
    errors = {}
    
    with ThreadPoolExecutor(max_workers=max_workers_ef) as executor:
        future_to_func = {
            executor.submit(
                run_explicit_feedback_for_function,
                func_name,
                func_files[func_name]
            ): func_name
            for func_name in successful_funcs
        }
        
        for future in as_completed(future_to_func):
            func_name, final_func, error = future.result()
            if final_func:
                final_functions[func_name] = final_func
            elif error:
                errors[func_name] = error
    
    if errors:
        print(f"\n Explicit feedback failed for {len(errors)} function(s):")
        for func_name, error in errors.items():
            print(f"  - {func_name}: {error}")
    
    # Save final functions
    from src.pipeline.cfg_to_funsearch_pipeline import sanitize_function_name
    final_functions_dir = os.path.join(args.experiment_dir, "final_functions")
    os.makedirs(final_functions_dir, exist_ok=True)
    
    for func_name, func_code in final_functions.items():
        safe_name = sanitize_function_name(func_name)
        if args.dsl_round is not None:
            if args.func_evolution_round is not None:
                func_file = os.path.join(final_functions_dir, f"{safe_name}_dsl{args.dsl_round}_func{args.func_evolution_round}.py")
            else:
                # Initial functions should be func0
                func_file = os.path.join(final_functions_dir, f"{safe_name}_dsl{args.dsl_round}_func0.py")
        else:
            func_file = os.path.join(final_functions_dir, f"{safe_name}.py")
        
        with open(func_file, 'w', encoding='utf-8') as f:
            f.write(func_code)
        print(f"  Saved {func_name} to {os.path.basename(func_file)}")
    
    # Save stage completion marker
    stage_status = {
        "stage": "explicit_feedback",
        "status": "completed",
        "dsl_round": args.dsl_round,
        "func_evolution_round": args.func_evolution_round,
        "final_functions": list(final_functions.keys()),
        "errors": list(errors.keys()) if errors else []
    }
    status_file = os.path.join(args.experiment_dir, "stage_explicit_feedback_status.json")
    with open(status_file, 'w') as f:
        json.dump(stage_status, f, indent=2)
    
    print(f"\n Explicit feedback completed for {len(final_functions)} functions")
    return 0 if final_functions else 1


if __name__ == "__main__":
    sys.exit(main())


