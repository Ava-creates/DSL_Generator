#!/usr/bin/env python3
"""
Stage 7: Evolve DSL
This stage evolves the DSL when tasks still fail after function evolution.
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Tuple

# Add project root to path
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _project_root)

from src.pipeline.integrated_pipeline import evolve_dsl
from src.utils.pipeline_state import read_state, update_state
from src.utils.file_utils import version_file

# Import vLLM for shared instance
try:
    from vllm import LLM as vLLM
except ImportError:
    vLLM = None


def main():
    parser = argparse.ArgumentParser(description="Stage 7: Evolve DSL")
    parser.add_argument('--experiment_dir', type=str, required=True, help='Experiment directory')
    parser.add_argument('--failing_tasks', type=str, nargs='+', required=True, help='List of failing tasks')
    parser.add_argument('--recipes_path', type=str, default="craft/resources/recipes.yaml", help='Path to recipes YAML')
    parser.add_argument('--max_retries', type=int, default=10, help='Maximum retries for DSL evolution')
    
    args = parser.parse_args()
    
    # Load CFG
    cfg_path = os.path.join(args.experiment_dir, "cfg", "cfg_output.json")
    if not os.path.exists(cfg_path):
        print(f"✗ CFG file not found: {cfg_path}", file=sys.stderr)
        return 1
    
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg_data = json.load(f)
    cfg = cfg_data.get("cfg", "")
    terminals = cfg_data.get("terminals", {})
    
    if not cfg or not terminals:
        print("✗ Invalid CFG data", file=sys.stderr)
        return 1
    
    # Load recipes
    if not os.path.exists(args.recipes_path):
        print(f"✗ Recipes file not found: {args.recipes_path}", file=sys.stderr)
        return 1
    
    with open(args.recipes_path, 'r') as f:
        recipes = f.read()
    
    # Create shared vLLM instance
    shared_vllm = None
    if vLLM is not None:
        try:
            print("\n[Setup] Initializing shared vLLM instance...")
            shared_vllm = vLLM(model="/scratch/avani/gpt", tensor_parallel_size=4)
            print("✓ Shared vLLM instance created")
        except Exception as e:
            print(f"⚠ Warning: Could not create shared vLLM instance: {e}")
            shared_vllm = None
    
    # Evolve DSL with retries
    print(f"\n[Step 7] Evolving DSL with {len(args.failing_tasks)} failing tasks...")
    dsl_success = False
    new_cfg = cfg
    new_terminals = terminals
    
    for dsl_attempt in range(1, args.max_retries + 1):
        if dsl_attempt > 1:
            print(f"\n[DSL Evolution Retry] Attempt {dsl_attempt}/{args.max_retries}")
        
        new_cfg, new_terminals, attempt_success = evolve_dsl(
            experiment_dir=args.experiment_dir,
            failing_tasks=args.failing_tasks,
            cfg=cfg,
            recipes=recipes,
            terminals=terminals,
            shared_vllm=shared_vllm
        )
        
        # Check if evolution was successful and CFG is different
        if attempt_success and new_cfg != cfg:
            dsl_success = True
            print(f"\n✓ DSL evolved successfully on attempt {dsl_attempt}")
            break
        else:
            if attempt_success:
                print(f"  ⚠ Attempt {dsl_attempt}: Evolved CFG is same as original, retrying...")
            else:
                print(f"  ⚠ Attempt {dsl_attempt}: DSL evolution failed, retrying...")
    
    if dsl_success and new_cfg != cfg:
        # Note: evolve_dsl() in integrated_pipeline.py already versions and saves the CFG file
        # So we don't need to do it again here - just verify it was saved
        if os.path.exists(cfg_path):
            print(f"✓ Evolved CFG already saved by evolve_dsl() function")
        else:
            print(f"⚠ Warning: CFG file not found after evolution, saving manually...")
            # Version existing file before saving new one (if it exists)
            if os.path.exists(cfg_path):
                try:
                    version_file(cfg_path, keep_original=False)
                    print(f"  ✓ Versioned previous CFG file")
                except Exception as e:
                    print(f"  ⚠ Warning: Failed to version CFG file: {e}")
            
            cfg_data = {
                "cfg": new_cfg,
                "terminals": new_terminals,
                "example": cfg_data.get("example", None)
            }
            with open(cfg_path, 'w', encoding='utf-8') as f:
                json.dump(cfg_data, f, indent=2, ensure_ascii=False)
            print(f"✓ Saved evolved CFG to {cfg_path}")
        
        # Save stage completion marker (legacy + grouped folder path)
        stage_status = {
            "stage": "evolve_dsl",
            "status": "completed",
            "failing_tasks": args.failing_tasks,
            "evolved": True,
            "attempt": dsl_attempt
        }
        status_file = os.path.join(args.experiment_dir, "stage_evolve_dsl_status.json")
        status_dir_file = os.path.join(args.experiment_dir, "status", "evolve_dsl", "status.json")
        os.makedirs(os.path.dirname(status_dir_file), exist_ok=True)
        for path in (status_file, status_dir_file):
            with open(path, 'w') as f:
                json.dump(stage_status, f, indent=2)
        
        # Update state and chain back to file generation
        state = read_state(args.experiment_dir)
        new_dsl_round = state.get("dsl_round", 0) + 1
        dsl_evolutions_remaining = state.get("dsl_evolutions_remaining", 3) - 1
        
        # Update state for new DSL round with new terminal function counts
        # The new CFG has new terminals, so we need to update the counts
        # File generation will update these properly when it runs, but we set them here as a placeholder
        # Note: test_tasks_total and max_function_evolutions are preserved (not reset)
        # test_tasks runs in single job, so test_tasks_remaining is not needed
        state = read_state(args.experiment_dir)
        num_new_terminals = len(new_terminals)
        test_tasks_total = state.get("test_tasks_total", 0)  # Preserve existing task count (informational)
        max_function_evolutions = state.get("max_function_evolutions", 1)  # Preserve max function evolutions setting
        
        update_state(
            args.experiment_dir,
            phase="initial",
            dsl_round=new_dsl_round,
            dsl_evolutions_remaining=dsl_evolutions_remaining,
            func_evolution_round=0,  # Reset function evolution round (start fresh with new DSL)
            max_function_evolutions=max_function_evolutions,  # Preserve the max setting
            function_implementation_total=num_new_terminals,  # Update with new terminal count
            function_implementation_remaining=num_new_terminals,  # Will be updated by file generation
            # implement_cfg_submitted removed - use status files as source of truth
            test_tasks_submitted=0,  # Reset so test tasks can be submitted again
            function_evolution_submitted=0,
            file_generation_submitted=0,  # Reset so file generation can be submitted again
            dsl_evolution_submitted=0  # Reset so DSL evolution can be submitted again in next round
            # Note: test_tasks_total is preserved (informational only, test_tasks runs in single job)
        )
        
        print(f"  Updated state: {num_new_terminals} terminal functions in new DSL (was {len(terminals)})")
        print(f"  Preserved: {test_tasks_total} test tasks, max_function_evolutions={max_function_evolutions} (unchanged)")
        
        print(f"\n[Chaining] DSL evolved to round {new_dsl_round}. Submitting file generation job...")
        
        # Submit file generation job
        scripts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "scripts", "stages")
        spec_file = os.environ.get("SPEC_FILE", "prompt_specifications/specification_with_updated_nld.txt")
        
        env_vars = {
            "EXPERIMENT_DIR": args.experiment_dir,
            "SPEC_FILE": spec_file,
            "DSL_ROUND": str(new_dsl_round)
        }
        
        try:
            file_gen_script = os.path.join(scripts_dir, "stage_file_generation.slurm")
            if os.path.exists(file_gen_script):
                # Chaining will be handled by the SLURM script
                print(f"  ✓ Will submit file generation job")
            else:
                print(f"  ⚠ Warning: File generation script not found: {file_gen_script}")
        except Exception as e:
            print(f"  ⚠ Warning: Failed to submit file generation job: {e}")
        
        return 0
    else:
        print(f"\n✗ DSL evolution failed after {args.max_retries} attempts")
        
        # Save stage completion marker
        stage_status = {
            "stage": "evolve_dsl",
            "status": "failed",
            "failing_tasks": args.failing_tasks,
            "evolved": False,
            "attempts": args.max_retries
        }
        status_file = os.path.join(args.experiment_dir, "stage_evolve_dsl_status.json")
        status_dir_file = os.path.join(args.experiment_dir, "status", "evolve_dsl", "status.json")
        os.makedirs(os.path.dirname(status_dir_file), exist_ok=True)
        for path in (status_file, status_dir_file):
            with open(path, 'w') as f:
                json.dump(stage_status, f, indent=2)
        
        return 1


if __name__ == "__main__":
    sys.exit(main())


