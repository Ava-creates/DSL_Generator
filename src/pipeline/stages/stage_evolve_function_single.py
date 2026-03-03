#!/usr/bin/env python3
"""
Stage 6: Evolve Function (Single Function)
This stage evolves a single function with failing tasks.
"""

import os
import sys
import json
import argparse
from typing import Dict, List

# Add project root to path
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _project_root)

from src.pipeline.integrated_pipeline import evolve_functions_with_failing_tasks
from src.utils.pipeline_state import decrement_function_evolution, read_state, update_state

# Import vLLM for shared instance
try:
    from vllm import LLM as vLLM
except ImportError:
    vLLM = None


def main():
    parser = argparse.ArgumentParser(description="Stage 6: Evolve Function (Single Function)")
    parser.add_argument('--experiment_dir', type=str, required=True, help='Experiment directory')
    parser.add_argument('--spec_file', type=str, required=True, help='Path to specification file')
    parser.add_argument('--function_name', type=str, required=True, help='Name of the function to evolve')
    parser.add_argument('--failing_tasks', type=str, nargs='*', default=[], help='List of failing tasks (optional, kept for API compat)')
    parser.add_argument('--model_type', type=str, default='huggingface', choices=['huggingface', 'ollama', 'gemini'])
    parser.add_argument('--dsl_round', type=int, default=0, help='DSL evolution round number')
    parser.add_argument('--func_evolution_round', type=int, default=0, help='Function evolution round number')
    parser.add_argument('--total_samples', type=int, default=1000, help='Total samples to generate in FunSearch (default: 1000)')
    
    args = parser.parse_args()
    
    # Initialize shared_vllm early so it's always defined in exception handlers
    shared_vllm = None
    
    # Load CFG
    cfg_path = os.path.join(args.experiment_dir, "cfg", "cfg_output.json")
    if not os.path.exists(cfg_path):
        print(f" CFG file not found: {cfg_path}", file=sys.stderr)
        return 1
    
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg_data = json.load(f)
    cfg = cfg_data.get("cfg", "")
    terminals = cfg_data.get("terminals", {})
    
    if not cfg:
        print(" Invalid CFG data: missing CFG string", file=sys.stderr)
        return 1
    
    # Ensure terminals match CFG (re-extract if missing or incomplete)
    # Note: shared_vllm is initialized later, so we pass None here
    # LLM-based description generation will fall back to pattern-based if no vLLM is available
    from src.pipeline.integrated_pipeline import ensure_terminals_match_cfg
    old_terminals = terminals.copy()  # Preserve existing descriptions
    terminals = ensure_terminals_match_cfg(cfg, terminals, old_terminals=old_terminals, shared_vllm=None)
    
    if not terminals:
        print(" Invalid CFG data: no terminals found after extraction", file=sys.stderr)
        return 1
    
    if args.function_name not in terminals:
        print(f" Function {args.function_name} not found in terminals", file=sys.stderr)
        print(f"  Available terminals: {list(terminals.keys())}", file=sys.stderr)
        return 1
    
    # Load specification
    if not os.path.exists(args.spec_file):
        print(f" Specification file not found: {args.spec_file}", file=sys.stderr)
        return 1
    
    with open(args.spec_file, 'r', encoding='utf-8') as f:
        specification = f.read()
    
    # Create shared vLLM instance (used by both FunSearch and explicit feedback in function evolution)
    # This ensures we only create ONE instance instead of separate ones for each stage
    # If creation fails, fail the stage to prevent OOM from multiple instances
    # Clean up GPU memory before creating new instance (in case previous stage left memory allocated)
    if args.model_type == "huggingface" and vLLM is not None:
        try:
            # Aggressive GPU memory cleanup before creating new instance
            import gc
            import torch
            print("\n[Setup] Performing aggressive GPU memory cleanup before creating vLLM instance...")
            
            # Multiple rounds of cleanup to handle fragmentation
            for cleanup_round in range(3):
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    # Check memory after cleanup
                    if cleanup_round == 0:
                        for gpu_id in range(torch.cuda.device_count()):
                            allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
                            reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
                            total = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
                            print(f"  GPU {gpu_id}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.2f}GB total")
            
            # Additional cleanup: try to reset CUDA context if possible
            if torch.cuda.is_available():
                # Force synchronization and clear any pending operations
                torch.cuda.synchronize()
                # One more round of cleanup after sync
                gc.collect()
                torch.cuda.empty_cache()
            
            print("[Setup] Initializing shared vLLM instance (for FunSearch and explicit feedback)...")
            print("  This may take a few minutes and requires significant GPU memory...")
            shared_vllm = vLLM(
                model="/scratch/avani/gpt", 
                tensor_parallel_size=4,
                gpu_memory_utilization=0.75  # Reduced from 0.85 to leave more headroom for parallel jobs
            )
            print(" Shared vLLM instance created - will be reused for both FunSearch and explicit feedback")
        except RuntimeError as e:
            error_msg = str(e)
            print(f" ERROR: Could not create shared vLLM instance: {error_msg}", file=sys.stderr)
            
            # Provide diagnostic information
            try:
                import torch
                if torch.cuda.is_available():
                    print("\n[Diagnostics] Current GPU memory state:", file=sys.stderr)
                    for gpu_id in range(torch.cuda.device_count()):
                        allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
                        reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
                        total = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
                        free = total - reserved
                        print(f"  GPU {gpu_id}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, "
                              f"{free:.2f}GB free, {total:.2f}GB total", file=sys.stderr)
            except ImportError:
                pass  # torch not available for diagnostics
            
            print(f"\n  Possible causes:", file=sys.stderr)
            print(f"    1. GPU memory fragmentation (previous operations left fragmented memory)", file=sys.stderr)
            print(f"    2. Insufficient free GPU memory (need ~{0.75 * 4 * 80:.0f}GB for 4 GPUs at 75% utilization)", file=sys.stderr)
            print(f"    3. CUDA context issues from previous operations", file=sys.stderr)
            print(f"    4. Another process/job using the same GPUs", file=sys.stderr)
            print(f"\n  Failing stage to prevent multiple instances from being created (which would cause OOM)", file=sys.stderr)
            print(f"  Shared instance is required to avoid GPU memory issues when running FunSearch and explicit feedback sequentially", file=sys.stderr)
            return 1
        except Exception as e:
            print(f" ERROR: Unexpected error creating shared vLLM instance: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            return 1
    
    # Evolve this single function
    # Create a terminals dict with only this function
    single_function_terminals = {args.function_name: terminals[args.function_name]}
    
    print(f"\n[{args.function_name}] Evolving function with {len(args.failing_tasks)} failing tasks...")
    try:
        evolved = evolve_functions_with_failing_tasks(
            experiment_dir=args.experiment_dir,
            failing_tasks=args.failing_tasks,
            terminals=single_function_terminals,
            specification=specification,
            spec_file=args.spec_file,
            dsl_round=args.dsl_round,
            func_evolution_round=args.func_evolution_round,
            cfg=cfg,
            shared_vllm=shared_vllm,
            total_samples=args.total_samples
        )
        
        # Save stage completion marker (legacy + grouped folder path)
        status_file = os.path.join(args.experiment_dir, f"stage_evolve_function_{args.function_name}_status.json")
        status_dir_file = os.path.join(
            args.experiment_dir, "status", "evolve_function", f"{args.function_name}.json"
        )
        os.makedirs(os.path.dirname(status_dir_file), exist_ok=True)
        stage_status = {
            "stage": "evolve_function",
            "function_name": args.function_name,
            "status": "completed" if evolved else "failed",
            "evolved": evolved,
            "failing_tasks": args.failing_tasks,
            "dsl_round": args.dsl_round,
            "func_evolution_round": args.func_evolution_round
        }
        for path in (status_file, status_dir_file):
            with open(path, 'w') as f:
                json.dump(stage_status, f, indent=2)
        
        if evolved:
            print(f"[{args.function_name}]  Function evolution completed")
        else:
            print(f"[{args.function_name}]  Function evolution failed or produced no results")
        
        # Clean up vLLM instance to free GPU memory before chaining
        if shared_vllm is not None:
            try:
                print("\n[Cleanup] Freeing shared vLLM instance and GPU memory...")
                del shared_vllm
                import gc
                import torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                print(" GPU memory freed")
            except Exception as cleanup_error:
                print(f"   Warning: Error during cleanup: {cleanup_error}")
        
        # Decrement function evolution counter and check if we should trigger funsearch
        print(f"\n[Chaining] Decrementing function evolution count...")
        remaining = decrement_function_evolution(args.experiment_dir)
        print(f"  Remaining function evolution jobs: {remaining}")
        
        # If this was the last function evolution job, submit funsearch jobs again
        if remaining == 0:
            print(f"\n[Chaining] All function evolution jobs completed. Submitting FunSearch jobs for evolved functions...")
            
            # Load terminals
            cfg_path = os.path.join(args.experiment_dir, "cfg", "cfg_output.json")
            with open(cfg_path, 'r', encoding='utf-8') as f:
                cfg_data = json.load(f)
            terminals = cfg_data.get("terminals", {})
            
            # Update state for funsearch phase
            state = read_state(args.experiment_dir)
            current_func_round = state.get("func_evolution_round", 0)
            # Increment func_evolution_round since we just completed a round
            new_func_round = current_func_round + 1
            # After all function evolution is done: set test_tasks_submitted=0 and function_impl_remaining=0
            # This will trigger test_tasks to run again
            update_state(
                args.experiment_dir,
                phase="initial",  # Back to initial phase but with func_evolution_round
                func_evolution_round=new_func_round,  # Increment the round
                function_implementation_total=len(terminals),
                function_implementation_remaining=0,  # Set to 0 to trigger test_tasks
                test_tasks_submitted=0  # Set to 0 to trigger test_tasks
            )
            print(f"  Updated func_evolution_round: {current_func_round} -> {new_func_round}")
            
            scripts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "scripts", "stages")
            spec_file = os.environ.get("SPEC_FILE", "prompt_specifications/specification_with_updated_nld.txt")
            model_type = os.environ.get("MODEL_TYPE", "huggingface")
            total_samples = os.environ.get("TOTAL_SAMPLES", "1000")  # Default: 1000 (matches other stages)
            
            # Chaining will be handled by the SLURM script (chain_next_stage.sh)
            # It will submit FunSearch jobs when all function evolution jobs complete
            print(f"\n[Chaining] State updated. Chaining script will submit FunSearch jobs when all function evolution jobs complete.")
        
        return 0 if evolved else 1
    except Exception as e:
        error_msg = str(e)
        print(f"[{args.function_name}]  Error: {error_msg}", file=sys.stderr)
        
        # Clean up vLLM instance to free GPU memory even on error
        if shared_vllm is not None:
            try:
                print("\n[Cleanup] Freeing vLLM instance and GPU memory...")
                del shared_vllm
                import gc
                import torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                print(" GPU memory freed")
            except Exception as cleanup_error:
                print(f"   Warning: Error during cleanup: {cleanup_error}")
        import traceback
        traceback.print_exc()
        
        # Save failure status (legacy + grouped folder path)
        status_file = os.path.join(args.experiment_dir, f"stage_evolve_function_{args.function_name}_status.json")
        status_dir_file = os.path.join(
            args.experiment_dir, "status", "evolve_function", f"{args.function_name}.json"
        )
        os.makedirs(os.path.dirname(status_dir_file), exist_ok=True)
        stage_status = {
            "stage": "evolve_function",
            "function_name": args.function_name,
            "status": "failed",
            "error": error_msg,
            "failing_tasks": args.failing_tasks,
            "dsl_round": args.dsl_round,
            "func_evolution_round": args.func_evolution_round
        }
        for path in (status_file, status_dir_file):
            with open(path, 'w') as f:
                json.dump(stage_status, f, indent=2)
        
        # Still decrement even on failure
        print(f"\n[Chaining] Decrementing function evolution count (after failure)...")
        remaining = decrement_function_evolution(args.experiment_dir)
        print(f"  Remaining function evolution jobs: {remaining}")
        
        # If this was the last function evolution job, update state for chaining
        # The chaining script (chain_next_stage.sh) will handle FunSearch submission
        if remaining == 0:
            print(f"\n[Chaining] All function evolution jobs completed (some may have failed).")
            print(f"  Chaining script will submit FunSearch jobs when this job completes.")
            
            cfg_path = os.path.join(args.experiment_dir, "cfg", "cfg_output.json")
            with open(cfg_path, 'r', encoding='utf-8') as f:
                cfg_data = json.load(f)
            terminals = cfg_data.get("terminals", {})
            
            state = read_state(args.experiment_dir)
            # After all function evolution is done: set test_tasks_submitted=0 and function_impl_remaining=0
            update_state(
                args.experiment_dir,
                phase="initial",
                function_implementation_total=len(terminals),
                function_implementation_remaining=0,  # Set to 0 to trigger test_tasks
                test_tasks_submitted=0  # Set to 0 to trigger test_tasks
            )
        
        return 1


if __name__ == "__main__":
    sys.exit(main())

