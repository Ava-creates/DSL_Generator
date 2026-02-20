#!/usr/bin/env python3
"""
Stage 4: Explicit Feedback Generation (Single Function)
This stage runs explicit feedback generation for a single function.
"""

import os
import glob
import sys
import json
import argparse
from typing import Dict, Optional

# Add project root to path
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _project_root)

from src.pipeline.cfg_to_funsearch_pipeline import run_explicit_feedback_generation, sanitize_function_name
from src.utils.pipeline_state import decrement_function_implementation, read_state, mark_test_tasks_submitted

# Import vLLM for shared instance
try:
    from vllm import LLM as vLLM
except ImportError:
    vLLM = None


def main():
    parser = argparse.ArgumentParser(description="Stage 4: Explicit Feedback Generation (Single Function)")
    parser.add_argument('--experiment_dir', type=str, required=True, help='Experiment directory')
    parser.add_argument('--spec_file', type=str, required=True, help='Path to specification file')
    parser.add_argument('--function_name', type=str, required=True, help='Name of the function to run explicit feedback for')
    parser.add_argument('--model_type', type=str, default='huggingface', choices=['huggingface', 'ollama', 'gemini'])
    parser.add_argument('--dsl_round', type=int, default=0, help='DSL evolution round number')
    parser.add_argument('--func_evolution_round', type=int, default=None, help='Function evolution round number')
    parser.add_argument('--num_iterations', type=int, default=30, help='Number of explicit feedback iterations')
    
    args = parser.parse_args()
    
    # Debug: Print DSL round to verify it's being passed correctly
    print(f"[DEBUG] Explicit feedback stage called with dsl_round={args.dsl_round} (from --dsl_round argument)")
    print(f"[DEBUG] DSL_ROUND environment variable: {os.environ.get('DSL_ROUND', 'NOT SET')}")
    
    # Load CFG
    cfg_path = os.path.join(args.experiment_dir, "cfg", "cfg_output.json")
    if not os.path.exists(cfg_path):
        print(f"✗ CFG file not found: {cfg_path}", file=sys.stderr)
        return 1
    
    # Load file generation status
    file_gen_status_path = os.path.join(args.experiment_dir, "stage_file_generation_status.json")
    if not os.path.exists(file_gen_status_path):
        print(f"✗ File generation status not found: {file_gen_status_path}", file=sys.stderr)
        return 1
    
    with open(file_gen_status_path, 'r') as f:
        file_gen_status = json.load(f)
    
    func_files = file_gen_status.get("func_files", {})
    func_signatures = file_gen_status.get("func_signatures", {})
    
    if args.function_name not in func_files:
        print(f"✗ Function file not found for {args.function_name}", file=sys.stderr)
        return 1
    
    func_file = func_files[args.function_name]
    
    # Check if FunSearch completed for this function (prefer grouped status, fallback to legacy)
    funsearch_status_file = os.path.join(args.experiment_dir, f"stage_funsearch_{args.function_name}_status.json")
    funsearch_status_grouped = os.path.join(
        args.experiment_dir, "status", "funsearch", f"{args.function_name}.json"
    )
    status_path = funsearch_status_grouped if os.path.exists(funsearch_status_grouped) else funsearch_status_file
    if not os.path.exists(status_path):
        print(f"✗ FunSearch status not found for {args.function_name}", file=sys.stderr)
        return 1
    
    with open(status_path, 'r') as f:
        funsearch_status = json.load(f)
    
    if funsearch_status.get("status") != "completed":
        print(f"✗ FunSearch did not complete successfully for {args.function_name}", file=sys.stderr)
        return 1
    
    # Load specification
    if not os.path.exists(args.spec_file):
        print(f"✗ Specification file not found: {args.spec_file}", file=sys.stderr)
        return 1
    
    with open(args.spec_file, 'r', encoding='utf-8') as f:
        specification = f.read()
    
    # Create shared vLLM instance
    shared_vllm = None
    if args.model_type == "huggingface" and vLLM is not None:
        try:
            print("\n[Setup] Initializing shared vLLM instance...")
            shared_vllm = vLLM(model="/scratch/avani/gpt", tensor_parallel_size=4)
            print("✓ Shared vLLM instance created")
        except Exception as e:
            print(f"⚠ Warning: Could not create shared vLLM instance: {e}")
            shared_vllm = None
    
    # Results directory
    results_dir = os.path.join(args.experiment_dir, "results", "funsearch")
    explicit_feedback_dir = os.path.join(args.experiment_dir, "explicit_feedback")
    os.makedirs(explicit_feedback_dir, exist_ok=True)
    
    # Run explicit feedback for this function
    print(f"\n[{args.function_name}] Starting explicit feedback generation...")
    try:
        current_func_code = None  # Keep function code in memory instead of saving intermediate files
        
        # Read initial function code
        with open(func_file, 'r', encoding='utf-8') as f:
            current_func_code = f.read()
        
        # Run multiple iterations
        final_func = None
        import tempfile
        for iteration in range(max(args.num_iterations, 1)):
            # Use temporary file for this iteration (will be cleaned up automatically)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp_file:
                tmp_file.write(current_func_code)
                tmp_file_path = tmp_file.name
            
            try:
                iter_func = run_explicit_feedback_generation(
                    args.function_name, results_dir, func_file, args.experiment_dir, explicit_feedback_dir,
                    specification, k=5, shared_vllm=shared_vllm, 
                    func_signature=func_signatures.get(args.function_name, ""),
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
            # Save final function
            final_functions_dir = os.path.join(args.experiment_dir, "final_functions")
            os.makedirs(final_functions_dir, exist_ok=True)
            
            safe_name = sanitize_function_name(args.function_name)
            if args.dsl_round is not None:
                if args.func_evolution_round is not None:
                    func_file_path = os.path.join(final_functions_dir, f"{safe_name}_dsl{args.dsl_round}_func{args.func_evolution_round}.py")
                else:
                    # Initial functions should be func0
                    func_file_path = os.path.join(final_functions_dir, f"{safe_name}_dsl{args.dsl_round}_func0.py")
            else:
                func_file_path = os.path.join(final_functions_dir, f"{safe_name}.py")
            
            with open(func_file_path, 'w', encoding='utf-8') as f:
                f.write(final_func)
            print(f"  Saved {args.function_name} to {os.path.basename(func_file_path)}")
            
            # Clean up intermediate explicit feedback artifacts (iter files and feedback JSONs) for this function
            safe_name = sanitize_function_name(args.function_name)
            patterns = [
                os.path.join(explicit_feedback_dir, f"{safe_name}_dsl{args.dsl_round}_iter_*.py"),
                os.path.join(explicit_feedback_dir, f"feedback_{safe_name}_*.json"),
                os.path.join(explicit_feedback_dir, f"{safe_name}_iter_*.py"),
                # Clean up old unversioned files (for backward compatibility)
                os.path.join(explicit_feedback_dir, f"eval_{safe_name}.py"),
                os.path.join(explicit_feedback_dir, f"feedback_{safe_name}.json"),
            ]
            for pattern in patterns:
                for path in glob.glob(pattern):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            
            # Save stage completion marker (legacy flat path + grouped folder path)
            status_file = os.path.join(args.experiment_dir, f"stage_explicit_feedback_{args.function_name}_status.json")
            status_dir_file = os.path.join(
                args.experiment_dir, "status", "explicit_feedback", f"{args.function_name}.json"
            )
            os.makedirs(os.path.dirname(status_dir_file), exist_ok=True)
            stage_status = {
                "stage": "explicit_feedback",
                "function_name": args.function_name,
                "status": "completed",
                "dsl_round": args.dsl_round,
                "func_evolution_round": args.func_evolution_round
            }
            for path in (status_file, status_dir_file):
                with open(path, 'w') as f:
                    json.dump(stage_status, f, indent=2)
            
            print(f"[{args.function_name}] ✓ Completed explicit feedback ({args.num_iterations} iterations)")
            
            # Decrement explicit feedback counter and check if we should trigger test tasks
            print(f"\n[Chaining] Decrementing explicit feedback count...")
            remaining = decrement_function_implementation(args.experiment_dir)
            print(f"  Remaining explicit feedback jobs: {remaining}")
            
            # If this was the last explicit feedback job, let chaining script handle test task submission
            if remaining == 0:
                # Load tasks from state file to check if any exist
                state = read_state(args.experiment_dir)
                tasks_str = state.get("tasks", "[]")
                try:
                    tasks = json.loads(tasks_str) if isinstance(tasks_str, str) else tasks_str
                except:
                    # Fallback: try to get from environment or use empty list
                    tasks = os.environ.get("TASKS", "").split() if os.environ.get("TASKS") else []
                
                if not tasks:
                    print(f"\n[Chaining] All explicit feedback jobs completed, but no tasks found in state file.")
                    print(f"  ⚠ Warning: Test tasks cannot be submitted without a tasks list.")
                    print(f"  Chaining script will check for tasks and handle submission.")
                else:
                    print(f"\n[Chaining] All explicit feedback jobs completed. Chaining script will submit test task jobs.")
            
            return 0
        else:
            print(f"[{args.function_name}] ⚠ No final function extracted")
            return 1
    except Exception as e:
        error_msg = str(e)
        print(f"[{args.function_name}] ✗ Error: {error_msg}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        
        # Save failure status (legacy + grouped folder path)
        status_file = os.path.join(args.experiment_dir, f"stage_explicit_feedback_{args.function_name}_status.json")
        status_dir_file = os.path.join(
            args.experiment_dir, "status", "explicit_feedback", f"{args.function_name}.json"
        )
        os.makedirs(os.path.dirname(status_dir_file), exist_ok=True)
        stage_status = {
            "stage": "explicit_feedback",
            "function_name": args.function_name,
            "status": "failed",
            "error": error_msg,
            "dsl_round": args.dsl_round,
            "func_evolution_round": args.func_evolution_round
        }
        for path in (status_file, status_dir_file):
            with open(path, 'w') as f:
                json.dump(stage_status, f, indent=2)
        
        # Still decrement even on failure
        print(f"\n[Chaining] Decrementing explicit feedback count (after failure)...")
        remaining = decrement_explicit_feedback(args.experiment_dir)
        print(f"  Remaining explicit feedback jobs: {remaining}")
        
        # If this was the last explicit feedback job, let chaining script handle test task submission
        if remaining == 0:
            state = read_state(args.experiment_dir)
            tasks_str = state.get("tasks", "[]")
            try:
                tasks = json.loads(tasks_str) if isinstance(tasks_str, str) else tasks_str
            except:
                tasks = os.environ.get("TASKS", "").split() if os.environ.get("TASKS") else []
            
            if not tasks:
                print(f"\n[Chaining] All explicit feedback jobs completed (some may have failed), but no tasks found.")
                print(f"  ⚠ Warning: Test tasks cannot be submitted without a tasks list.")
                print(f"  Chaining script will check for tasks and handle submission.")
            else:
                print(f"\n[Chaining] All explicit feedback jobs completed (some may have failed). Chaining script will submit test task jobs.")
        
        return 1


if __name__ == "__main__":
    sys.exit(main())

