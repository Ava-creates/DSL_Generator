#!/usr/bin/env python3
"""
Stage 7: Evolve DSL
This stage evolves the DSL when tasks still fail after function evolution.
"""

import os
import sys
import json
import argparse
from typing import Dict, List

# Add project root to path
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _project_root)

from src.pipeline.integrated_pipeline import evolve_dsl
from src.utils.pipeline_state import read_state, update_state
from src.utils.file_utils import version_file
from src.utils.status_manager import write_status

# Import vLLM for shared instance
try:
    from vllm import LLM as vLLM
except ImportError:
    vLLM = None


def extract_failed_programs_from_synthesis_results(
    experiment_dir: str,
    failing_tasks: List[str],
    dsl_version: int = 0,
    max_programs_per_task: int = 30,
) -> Dict[str, List[str]]:
    """Extract failed programs for failing tasks from synthesis_results.json.

    Uses only results from:
    - the current DSL version being evolved (cfg_version == dsl_version)
    - the latest function evolution round observed for that DSL version
    Then caps programs per task to max_programs_per_task (most recent entries).
    """
    synthesis_results_path = os.path.join(experiment_dir, "results_tracking", "synthesis_results.json")
    
    if not os.path.exists(synthesis_results_path):
        print(f"Warning: synthesis_results.json not found at {synthesis_results_path}")
        return {}
    
    failed_programs_by_task = {}
    
    try:
        with open(synthesis_results_path, 'r') as f:
            synthesis_results = json.load(f)
    except Exception as e:
        print(f"Warning: Could not read synthesis_results.json: {e}")
        return {}
    
    source_cfg_version = dsl_version

    relevant_results = [
        r for r in synthesis_results
        if r.get("cfg_version", 0) == source_cfg_version and r.get("task") in failing_tasks
    ]

    if not relevant_results:
        print(
            "Warning: No synthesis results found for "
            f"source cfg_version={source_cfg_version} (requested dsl_version={dsl_version}) "
            "and requested failing tasks"
        )
        return {}

    latest_func_round = max(r.get("func_evolution_round", 0) for r in relevant_results)
    print(
        f"Using source cfg_version={source_cfg_version} "
        f"(requested dsl_version={dsl_version}), "
        f"latest func_evolution_round={latest_func_round}"
    )

    latest_round_results = [
        r for r in relevant_results
        if r.get("func_evolution_round", 0) == latest_func_round
    ]

    for task in failing_tasks:
        failed_programs = []
        seen_program_keys = set()
        duplicate_skips = 0

        task_failed_results = [
            r for r in latest_round_results
            if r.get("task") == task and not r.get("success", False)
        ]

        # Keep most recent unique programs first (reverse assumes synthesis_results append order)
        for result in reversed(task_failed_results):
            program = result.get("program", "")
            program_key = " ".join(str(program).split()).strip().lower()
            if program_key in seen_program_keys:
                duplicate_skips += 1
                continue
            seen_program_keys.add(program_key)

            failure_reason = result.get("failure_reason", "Unknown")

            # Try to extract inventory information from different possible fields
            inventory_before = result.get("inventory_before", {})
            inventory_after = result.get("inventory_after", {})
            inventory_trace = result.get("inventory_trace", [])

            # Format program information with available inventory data
            lines = [f"Program:\n{program}"]

            # Add inventory information if available
            if inventory_trace:
                lines.append("Inventory changes during the program (whole inventory after the function where the change happened):")
                for entry in inventory_trace:
                    token = entry.get("token", "?")
                    inv = entry.get("inventory", [])
                    inv_str = ", ".join(inv) if inv else "<empty>"
                    lines.append(f"  {token} -> {inv_str}")
            elif inventory_before or inventory_after:
                lines.append(f"Inventory before: {inventory_before}")
                lines.append(f"Inventory after: {inventory_after}")

            if failure_reason and str(failure_reason).strip().lower() != "unknown":
                lines.append(f"Failure: {failure_reason}")
            failed_programs.append("\n".join(lines))

            if max_programs_per_task > 0 and len(failed_programs) >= max_programs_per_task:
                break

        # restore chronological order in prompt context
        failed_programs.reverse()

        if failed_programs:
            failed_programs_by_task[task] = failed_programs
            print(
                f"Found {len(failed_programs)} failed programs for task: {task} "
                f"(capped at {max_programs_per_task}, latest func round only, duplicates skipped={duplicate_skips})"
            )
    
    return failed_programs_by_task


def main():
    parser = argparse.ArgumentParser(description="Stage 7: Evolve DSL")
    parser.add_argument('--experiment_dir', type=str, required=True, help='Experiment directory')
    parser.add_argument('--failing_tasks', type=str, nargs='+', required=True, help='List of failing tasks')
    parser.add_argument('--recipes_path', type=str, default="craft/resources/recipes.yaml", help='Path to recipes YAML')
    parser.add_argument('--max_retries', type=int, default=10, help='Maximum retries for DSL evolution')
    parser.add_argument('--dsl_version', type=int, default=0, help='DSL version to load (e.g., 0 for cfg_output_0.json)')
    parser.add_argument('--max_failed_programs', type=int, default=30, help='Maximum failed programs per task for failure-analysis context')
    
    args = parser.parse_args()
    
    # Load CFG — convention: cfg_output_N.json = round N, cfg_output.json = fallback for round 0
    cfg_path = os.path.join(args.experiment_dir, "cfg", f"cfg_output_{args.dsl_version}.json")
    if not os.path.exists(cfg_path):
        if args.dsl_version == 0:
            # Backward compat: round-0 CFG was saved as cfg_output.json in older experiments
            fallback = os.path.join(args.experiment_dir, "cfg", "cfg_output.json")
            if os.path.exists(fallback):
                import shutil
                shutil.copy2(fallback, cfg_path)
                print(f" Created {cfg_path} from cfg_output.json (backward compat)")
            else:
                print(f" CFG file not found: {cfg_path}", file=sys.stderr)
                return 1
        else:
            print(f" CFG file not found: {cfg_path}", file=sys.stderr)
            return 1
    
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg_data = json.load(f)
    cfg = cfg_data.get("cfg", "")
    terminals = cfg_data.get("terminals", {})
    
    if not cfg or not terminals:
        print(" Invalid CFG data", file=sys.stderr)
        return 1
    
    # Load recipes
    if not os.path.exists(args.recipes_path):
        print(f" Recipes file not found: {args.recipes_path}", file=sys.stderr)
        return 1
    
    with open(args.recipes_path, 'r') as f:
        recipes = f.read()
    
    # Create shared vLLM instance
    shared_vllm = None
    if vLLM is not None:
        try:
            print("\n[Setup] Initializing shared vLLM instance...")
            shared_vllm = vLLM(model="/scratch/avani/gpt", tensor_parallel_size=4)
            print(" Shared vLLM instance created")
        except Exception as e:
            print(f" Warning: Could not create shared vLLM instance: {e}")
            shared_vllm = None
    
    # Evolve DSL with retries
    print(f"\n[Step 7] Evolving DSL with {len(args.failing_tasks)} failing tasks...")
    dsl_success = False
    new_cfg = cfg
    new_terminals = terminals
    
    # Extract failed programs from synthesis results for context
    print("\n[Step 1] Extracting failed programs from synthesis results...")
    failed_programs_by_task = extract_failed_programs_from_synthesis_results(args.experiment_dir, args.failing_tasks, args.dsl_version)
    
    for dsl_attempt in range(1, args.max_retries + 1):
        if dsl_attempt > 1:
            print(f"\n[DSL Evolution Retry] Attempt {dsl_attempt}/{args.max_retries}")
        
        new_cfg, new_terminals, attempt_success = evolve_dsl(
            experiment_dir=args.experiment_dir,
            failing_tasks=args.failing_tasks,
            cfg=cfg,
            recipes=recipes,
            terminals=terminals,
            shared_vllm=shared_vllm,
            failed_programs_by_task=failed_programs_by_task,
            new_dsl_round=args.dsl_version + 1,
        )
        
        # Check if evolution was successful and CFG is different
        if attempt_success and new_cfg != cfg:
            dsl_success = True
            print(f"\n DSL evolved successfully on attempt {dsl_attempt}")
            break
        else:
            if attempt_success:
                print(f"   Attempt {dsl_attempt}: Evolved CFG is same as original, retrying...")
            else:
                print(f"   Attempt {dsl_attempt}: DSL evolution failed, retrying...")
    
    if dsl_success and new_cfg != cfg:
        # Note: evolve_dsl() in integrated_pipeline.py already versions and saves the CFG file
        # So we don't need to do it again here - just verify it was saved
        if os.path.exists(cfg_path):
            print(" Evolved CFG already saved by evolve_dsl() function")
        else:
            print(" Warning: CFG file not found after evolution, saving manually...")
            # Version existing file before saving new one (if it exists)
            if os.path.exists(cfg_path):
                try:
                    version_file(cfg_path, keep_original=False)
                    print("   Versioned previous CFG file")
                except Exception as e:
                    print(f"   Warning: Failed to version CFG file: {e}")
            
            # Save evolved CFG to next version
            next_version = args.dsl_version + 1
            output_cfg_path = os.path.join(args.experiment_dir, "cfg", f"cfg_output_{next_version}.json")
            
            cfg_data = {
                "cfg": new_cfg,
                "terminals": new_terminals,
                "example": cfg_data.get("example", None)
            }
            with open(output_cfg_path, 'w', encoding='utf-8') as f:
                json.dump(cfg_data, f, indent=2, ensure_ascii=False)
            print(f" Saved evolved CFG to {output_cfg_path}")
        
        # Save stage completion marker with DSL versioning
        stage_status = {
            "stage": "evolve_dsl",
            "status": "completed",
            "failing_tasks": args.failing_tasks,
            "evolved": True,
            "attempt": dsl_attempt,
            "dsl_round": args.dsl_version + 1
        }
        # Write to versioned location: status/evolve_dsl/dsl{N}/status.json
        write_status(
            args.experiment_dir, 
            args.dsl_version,  # Save status for the current DSL round being evolved
            "evolve_dsl", 
            stage_status
        )
        
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
        
        try:
            file_gen_script = os.path.join(scripts_dir, "stage_file_generation.slurm")
            if os.path.exists(file_gen_script):
                # Chaining will be handled by the SLURM script
                print("   Will submit file generation job")
            else:
                print(f"   Warning: File generation script not found: {file_gen_script}")
        except Exception as e:
            print(f"   Warning: Failed to submit file generation job: {e}")
        
        return 0
    else:
        print(f"\n DSL evolution failed after {args.max_retries} attempts")
        
        # Save stage completion marker with DSL versioning
        stage_status = {
            "stage": "evolve_dsl",
            "status": "failed",
            "failing_tasks": args.failing_tasks,
            "evolved": False,
            "attempts": args.max_retries,
            "dsl_round": args.dsl_version
        }
        # Write to versioned location: status/evolve_dsl/dsl{N}/status.json
        write_status(
            args.experiment_dir, 
            args.dsl_version,
            "evolve_dsl", 
            stage_status
        )
        
        return 1


if __name__ == "__main__":
    sys.exit(main())


