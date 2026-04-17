#!/usr/bin/env python3
"""
Unified pipeline that orchestrates:
1. Get CFG
2. Loop up to 3 times (DSL evolution):
   a. Implement CFG (generate prompts, funsearch, explicit feedback)
   b. Test CFG on tasks
   c. If tasks fail, loop up to 3 times (function evolution):
      - Evolve functions with failing tasks
      - Test CFG on tasks
   d. If still failing, evolve DSL and continue
"""

import os
import sys
import json
import argparse
from typing import List, Optional

# Exit codes for job resubmission
EXIT_CODE_DSL_EVOLVED = 100  # DSL evolved, need to resubmit job
EXIT_CODE_SUCCESS = 0
EXIT_CODE_FAILURE = 1

# Add project root to path (go up to project root)
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _project_root)

from src.pipeline.cfg_to_funsearch_pipeline import (
    get_cfg, implement_cfg
)
from src.pipeline.integrated_pipeline import (
    test_cfg_on_tasks,
    evolve_functions_with_failing_tasks,
    evolve_dsl,
    run_failure_analysis_for_dsl_evolution,
)
from src.utils.results_tracker import ResultsTracker

# Import vLLM for shared instance
try:
    from vllm import LLM as vLLM
except ImportError:
    vLLM = None


def run_unified_pipeline(
    experiment_dir: str,
    spec_file: str,
    tasks: List[str],
    max_dsl_evolutions: int = 2,
    max_function_evolutions: int = 3,
    skip_cfg_generation: bool = False,
    cfg_output_file: Optional[str] = None,
    max_cfg_retries: int = 10,
    nld_path: str = "prompt_specifications/nld.txt",
    codebase_path: Optional[str] = None,
    recipes_path: str = "craft/resources/recipes.yaml",
    hints_path: str = "craft/resources/hints.yaml",
    max_attempts: int = 1,
    model_type: str = "huggingface",
    shared_vllm=None,
    resume_from_checkpoint: bool = False,
    local: bool = False,
) -> int:
    """Run the unified pipeline with the structured flow.
    
    Flow:
    1. Get CFG
    2. Loop up to max_dsl_evolutions times:
       a. Implement CFG
       b. Test CFG on tasks
       c. If tasks fail, loop up to max_function_evolutions times:
          - Evolve functions with failing tasks
          - Test CFG on tasks
       d. If still failing, evolve DSL and continue
    
    Args:
        resume_from_checkpoint: If True, start from the DSL round in checkpoint
    
    Returns:
        0 on success, 1 on failure, EXIT_CODE_DSL_EVOLVED (100) when DSL evolves
    """
    print(f"\n{'='*80}")
    print("UNIFIED PIPELINE")
    print(f"{'='*80}")
    print(f"Experiment directory: {experiment_dir}")
    print(f"Tasks to solve: {tasks}")
    print(f"Max DSL evolutions: {max_dsl_evolutions}")
    print(f"Max function evolutions: {max_function_evolutions}")
    
    # Load checkpoint if resuming
    start_dsl_round = 0
    checkpoint_type = None
    resume_func_round = 0
    resume_failing_tasks = []
    if resume_from_checkpoint:
        checkpoint_path = os.path.join(experiment_dir, "checkpoint.json")
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
            start_dsl_round = checkpoint.get("dsl_round", 0)
            checkpoint_type = checkpoint.get("checkpoint_type", "dsl_evolution")  # Default to DSL evolution
            # Restore parameters from checkpoint
            max_dsl_evolutions = checkpoint.get("max_dsl_evolutions", max_dsl_evolutions)
            max_function_evolutions = checkpoint.get("max_function_evolutions", max_function_evolutions)
            spec_file = checkpoint.get("spec_file", spec_file)
            tasks = checkpoint.get("tasks", tasks)
            recipes_path = checkpoint.get("recipes_path", recipes_path)
            hints_path = checkpoint.get("hints_path", hints_path)
            model_type = checkpoint.get("model_type", model_type)
            skip_cfg_generation = checkpoint.get("skip_cfg_generation", skip_cfg_generation)
            cfg_output_file = checkpoint.get("cfg_output_file", cfg_output_file)
            print(f"Resuming from checkpoint: DSL round {start_dsl_round + 1}/{max_dsl_evolutions}")
            if checkpoint_type == "function_evolution":
                resume_func_round = checkpoint.get("func_round", 0)
                resume_failing_tasks = checkpoint.get("failing_tasks", [])
                print(f"  Checkpoint type: Function Evolution (round {resume_func_round + 1})")
                print(f"  Failing tasks: {resume_failing_tasks}")
            else:
                print("  Checkpoint type: DSL Evolution")
            print("  Restored parameters from checkpoint")
        else:
            print("   Checkpoint file not found, starting from beginning")
    
    # Create shared vLLM instance if not provided
    if shared_vllm is None:
        if vLLM is not None:
            try:
                print("\n[Setup] Initializing shared vLLM instance...")
                shared_vllm = vLLM(model="/scratch/avani/gpt", tensor_parallel_size=4)
                print(" Shared vLLM instance created")
            except Exception as e:
                print(f" Warning: Could not create shared vLLM instance: {e}")
                print("  Will create individual instances as needed")
                shared_vllm = None
        else:
            print("\n[Setup] vLLM not available, will use regular LLM instances")
    else:
        print("\n[Setup] Using provided shared vLLM instance")
    
    # Ensure experiment directory structure exists
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "function_specific_prompts"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "functions_generated"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "results", "funsearch"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "cfg"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "final_functions"), exist_ok=True)
    
    # Initialize results tracker
    results_tracker = ResultsTracker(experiment_dir)
    print("\n[Setup] Initialized results tracker for interaction and reward tracking")
    
    # Step 1: Get CFG
    cfg, terminals, example, success = get_cfg(
        experiment_dir=experiment_dir,
        skip_cfg_generation=skip_cfg_generation,
        cfg_output_file=cfg_output_file,
        max_cfg_retries=max_cfg_retries,
        nld_path=nld_path,
        recipes_path=recipes_path,
        shared_vllm=shared_vllm
    )
    
    if not success or not cfg or not terminals:
        print(" Failed to get valid CFG. Cannot proceed.")
        return 1
    
    print(f"\n Got CFG with {len(terminals)} terminal functions")
    
    # Load recipes for DSL evolution
    with open(recipes_path, 'r') as f:
        recipes = f.read()
    
    # Main loop: DSL evolution (up to max_dsl_evolutions times)
    for dsl_round in range(start_dsl_round, max_dsl_evolutions):
        print(f"\n{'='*80}")
        print(f"DSL Evolution Round {dsl_round + 1}/{max_dsl_evolutions}")
        print(f"{'='*80}")
        
        # Save checkpoint at the start of each DSL round (for timeout recovery)
        checkpoint_path = os.path.join(experiment_dir, "checkpoint.json")
        checkpoint_data = {
            "dsl_round": dsl_round,  # Current round (will resume from here if timeout)
            "func_round": 0,  # Reset function round at start of DSL round
            "max_dsl_evolutions": max_dsl_evolutions,
            "max_function_evolutions": max_function_evolutions,
            "spec_file": spec_file,
            "tasks": tasks,
            "recipes_path": recipes_path,
            "hints_path": hints_path,
            "model_type": model_type,
            "skip_cfg_generation": True,  # We already have CFG
            "cfg_output_file": os.path.join(experiment_dir, "cfg", "cfg_output.json"),
            "failing_tasks": []  # Will be updated after testing
        }
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        print(f"   Saved checkpoint at start of DSL round {dsl_round + 1}")
        
        # Check if we're resuming from a function evolution checkpoint
        # If so, skip CFG implementation and testing, go straight to function evolution
        skip_implementation = False
        if resume_from_checkpoint and checkpoint_type == "function_evolution" and resume_failing_tasks and dsl_round == start_dsl_round:
            print("\n[Resuming] Skipping CFG implementation - resuming from function evolution")
            failing_tasks = resume_failing_tasks
            all_solved = False
            skip_implementation = True
            # Clear resume flag after first use
            resume_from_checkpoint = False
        
        if not skip_implementation:
            # Reset evolution interactions for new DSL round
            if results_tracker is not None:
                results_tracker.current_evolution_interactions = {
                    "funsearch": 0,
                    "explicit_feedback": 0,
                    "program_synthesis": 0
                }
            
            # Step 2a: Implement CFG
            print("\n[Step 2a] Implementing CFG...")
            implementation_success, final_functions = implement_cfg(
                cfg=cfg,
                terminals=terminals,
                example=example,
                spec_file=spec_file,
                experiment_dir=experiment_dir,
                model_type=model_type,
                shared_vllm=shared_vllm,
                results_tracker=results_tracker,
                dsl_round=dsl_round,
                func_evolution_round=None,  # Initial implementation, not a function evolution
                nld_path=nld_path,
                codebase_path=codebase_path,
            )
                
            if not implementation_success:
                print(" CFG implementation failed. Stopping pipeline.")
                return 1
            
            # Step 2b: Test CFG on tasks
            print("\n[Step 2b] Testing CFG on tasks...")
            task_results = test_cfg_on_tasks(
                experiment_dir=experiment_dir,
                tasks=tasks,
                cfg=cfg,
                terminals=terminals,
                recipes_path=recipes_path,
                hints_path=hints_path,
                max_attempts=max_attempts,
                shared_vllm=shared_vllm,
                results_tracker=results_tracker,
                cfg_version=dsl_round,
                func_evolution_round=None,  # Initial testing, not function evolution
                model_type=model_type,
            )
            
            all_solved = all(task_results.values())
            failing_tasks = [task for task, success in task_results.items() if not success]
            
            # Save evolution metrics for initial testing
            if results_tracker is not None:
                # Get rewards from results in this evolution (func_evolution_round=None)
                rewards_per_task = {}
                for task in tasks:
                    task_results = results_tracker.get_task_results(task)
                    # Filter for results from this evolution (None means initial testing)
                    evolution_results = [r for r in task_results 
                                        if r.get("func_evolution_round") is None and r.get("cfg_version") == dsl_round]
                    if evolution_results:
                        rewards_per_task[task] = max([r["reward"] for r in evolution_results])
                    else:
                        rewards_per_task[task] = 0.0
                
                results_tracker.save_evolution_metrics(
                    dsl_round=dsl_round,
                    func_evolution_round=None,
                    steps_in_evolution=results_tracker.current_evolution_interactions.copy(),
                    rewards_per_task=rewards_per_task
                )
            
            # Plot after testing all tasks (to show progress)
            print("\n[Generating Plots] Creating reward vs interactions plots...")
            results_tracker.plot_reward_vs_interactions(dsl_round=dsl_round, func_evolution_round=None)
            results_tracker.plot_all_tasks_combined(dsl_round=dsl_round, func_evolution_round=None)
            # Also generate separate plots per task from evolution metrics
            results_tracker.plot_tasks_separately_from_metrics(dsl_round=dsl_round, func_evolution_round=None)
            
            if all_solved:
                print(f"\n{'='*80}")
                print(" ALL TASKS SOLVED!")
                print(f"{'='*80}")
                # Remove checkpoint on success
                checkpoint_path = os.path.join(experiment_dir, "checkpoint.json")
                if os.path.exists(checkpoint_path):
                    os.remove(checkpoint_path)
                
                # Print summary
                summary = results_tracker.get_summary()
                print("\n[Results Summary]")
                print(f"  Total results: {summary['total_results']}")
                print(f"  Total interactions: {summary['total_interactions']['total']}")
                print(f"    - FunSearch: {summary['total_interactions']['funsearch']}")
                print(f"    - Explicit Feedback: {summary['total_interactions']['explicit_feedback']}")
                print(f"    - Program Synthesis: {summary['total_interactions']['program_synthesis']}")
                print(f"  Tasks: {len(summary['tasks'])}")
                print(f"  CFG Versions: {summary['cfg_versions']}")
                
                return 0
            
            print(f"\n   {len(failing_tasks)}/{len(tasks)} tasks failed: {failing_tasks}")
        
        # Update checkpoint with failing tasks before function evolution
        checkpoint_path = os.path.join(experiment_dir, "checkpoint.json")
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            checkpoint_data["failing_tasks"] = failing_tasks
            checkpoint_data["func_round"] = 0  # Start function evolution
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
        
        # Step 2c: Function evolution loop (up to max_function_evolutions times)
        # Check if resuming from checkpoint
        start_func_round = 0
        if resume_from_checkpoint:
            checkpoint_path = os.path.join(experiment_dir, "checkpoint.json")
            if os.path.exists(checkpoint_path):
                with open(checkpoint_path, 'r') as f:
                    checkpoint = json.load(f)
                start_func_round = checkpoint.get("func_round", 0)
                if start_func_round > 0:
                    print(f"  Resuming from function evolution round {start_func_round + 1}")
        
        for func_round in range(start_func_round, max_function_evolutions):
            print(f"\n  {'-'*60}")
            print(f"  Function Evolution Round {func_round + 1}/{max_function_evolutions}")
            print(f"  {'-'*60}")
            
            # Reset evolution interactions for new function evolution round
            if results_tracker is not None:
                results_tracker.current_evolution_interactions = {
                    "funsearch": 0,
                    "explicit_feedback": 0,
                    "program_synthesis": 0
                }
            
            # Save checkpoint at start of each function evolution round
            checkpoint_path = os.path.join(experiment_dir, "checkpoint.json")
            if os.path.exists(checkpoint_path):
                with open(checkpoint_path, 'r') as f:
                    checkpoint_data = json.load(f)
                checkpoint_data["func_round"] = func_round
                checkpoint_data["checkpoint_type"] = "function_evolution"  # Mark as function evolution checkpoint
                with open(checkpoint_path, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2)
                print(f"   Saved checkpoint at start of function evolution round {func_round + 1}")
            
            # Load specification
            if os.path.exists(spec_file):
                with open(spec_file, 'r') as f:
                    specification = f.read()
            else:
                print("   Specification file not found")
                specification = ""
            
            # Evolve functions with failing tasks
            print("\n  [Evolving Functions] Evolving functions with failing tasks...")
            evolved = evolve_functions_with_failing_tasks(
                experiment_dir=experiment_dir,
                failing_tasks=failing_tasks,
                terminals=terminals,
                specification=specification,
                spec_file=spec_file,
                dsl_round=dsl_round,
                func_evolution_round=func_round,
                cfg=cfg,
                max_evolutions=1,
                shared_vllm=shared_vllm
            )
            
            if not evolved:
                print("   Function evolution failed or produced no results")
                break
            
            # Re-test tasks
            print("\n  [Re-testing Tasks] Re-testing tasks after function evolution...")
            task_results = test_cfg_on_tasks(
                experiment_dir=experiment_dir,
                tasks=failing_tasks,
                cfg=cfg,
                terminals=terminals,
                recipes_path=recipes_path,
                hints_path=hints_path,
                max_attempts=max_attempts,
                shared_vllm=shared_vllm,
                results_tracker=results_tracker,
                cfg_version=dsl_round,
                func_evolution_round=func_round,
                model_type=model_type,
            )
            
            all_solved = all(task_results.values())
            failing_tasks = [task for task, success in task_results.items() if not success]
            
            print(f"\n  Task Results after function evolution round {func_round + 1}:")
            for task, success in task_results.items():
                status = "" if success else ""
                print(f"    {status} {task}")
            
            # Save evolution metrics for this function evolution round
            if results_tracker is not None:
                # Get rewards from results in this specific evolution round
                rewards_per_task = {}
                for task in failing_tasks:
                    task_results = results_tracker.get_task_results(task)
                    # Filter for results from this specific evolution round
                    evolution_results = [r for r in task_results 
                                        if r.get("func_evolution_round") == func_round and r.get("cfg_version") == dsl_round]
                    if evolution_results:
                        rewards_per_task[task] = max([r["reward"] for r in evolution_results])
                    else:
                        rewards_per_task[task] = 0.0
                
                results_tracker.save_evolution_metrics(
                    dsl_round=dsl_round,
                    func_evolution_round=func_round,
                    steps_in_evolution=results_tracker.current_evolution_interactions.copy(),
                    rewards_per_task=rewards_per_task
                )
                
                # Reset evolution interactions for next function evolution round
                results_tracker.current_evolution_interactions = {
                    "funsearch": 0,
                    "explicit_feedback": 0,
                    "program_synthesis": 0
                }
            
            # Plot after re-testing tasks (to show progress)
            print("\n[Generating Plots] Creating reward vs interactions plots...")
            results_tracker.plot_reward_vs_interactions(dsl_round=dsl_round, func_evolution_round=func_round)
            results_tracker.plot_all_tasks_combined(dsl_round=dsl_round, func_evolution_round=func_round)
            # Also generate separate plots per task from evolution metrics
            results_tracker.plot_tasks_separately_from_metrics(dsl_round=dsl_round, func_evolution_round=func_round)
            
            if all_solved:
                print(f"\n  All tasks solved after function evolution round {func_round + 1}!")
                print(f"\n{'='*80}")
                print(" ALL TASKS SOLVED!")
                print(f"{'='*80}")
                
                # Print summary (plots already generated above)
                summary = results_tracker.get_summary()
                print("\n[Results Summary]")
                print(f"  Total results: {summary['total_results']}")
                print(f"  Total interactions: {summary['total_interactions']['total']}")
                print(f"    - FunSearch: {summary['total_interactions']['funsearch']}")
                print(f"    - Explicit Feedback: {summary['total_interactions']['explicit_feedback']}")
                print(f"    - Program Synthesis: {summary['total_interactions']['program_synthesis']}")
                print(f"  Tasks: {len(summary['tasks'])}")
                print(f"  CFG Versions: {summary['cfg_versions']}")
                
                return 0
        
        # Step 2d: If still failing, evolve DSL
        if failing_tasks:
            print(f"\n  {'-'*60}")
            print("  DSL Evolution (tasks still failing)")
            print(f"  {'-'*60}")
            
            # Retry DSL evolution up to 10 times if CFG is rejected or same as original
            max_dsl_retries = 10
            dsl_success = False
            new_cfg = cfg
            new_terminals = terminals

            failure_analysis_cached = run_failure_analysis_for_dsl_evolution(
                experiment_dir=experiment_dir,
                failing_tasks=failing_tasks,
                cfg=cfg,
                terminals=terminals,
                failed_programs_by_task=None,
                shared_vllm=shared_vllm,
            )

            for dsl_attempt in range(1, max_dsl_retries + 1):
                if dsl_attempt > 1:
                    print(
                        f"\n  [DSL Evolution Retry] Attempt {dsl_attempt}/{max_dsl_retries} "
                        f"(CFG evolution only)"
                    )

                new_cfg, new_terminals, attempt_success = evolve_dsl(
                    experiment_dir=experiment_dir,
                    failing_tasks=failing_tasks,
                    cfg=cfg,
                    recipes=recipes,
                    terminals=terminals,
                    failure_analysis=failure_analysis_cached,
                    shared_vllm=shared_vllm,
                )
                
                # Check if evolution was successful and CFG is different
                if attempt_success and new_cfg != cfg:
                    dsl_success = True
                    print(f"\n   DSL evolved successfully on attempt {dsl_attempt}")
                    break
                else:
                    if attempt_success:
                        print(f"   Attempt {dsl_attempt}: Evolved CFG is same as original, retrying...")
                    else:
                        print(f"   Attempt {dsl_attempt}: DSL evolution failed, retrying...")
            
            if dsl_success and new_cfg != cfg:
                print("\n   DSL evolved successfully")
                cfg = new_cfg
                terminals = new_terminals
                
                # Update example if available
                cfg_path = os.path.join(experiment_dir, "cfg", "cfg_output.json")
                if os.path.exists(cfg_path):
                    with open(cfg_path, 'r') as f:
                        cfg_data = json.load(f)
                        example = cfg_data.get("example", None)
                
                # Save checkpoint for resubmission
                checkpoint_path = os.path.join(experiment_dir, "checkpoint.json")
                checkpoint_data = {
                    "dsl_round": dsl_round + 1,  # Next round to run
                    "func_round": 0,  # Reset function round for new DSL round
                    "max_dsl_evolutions": max_dsl_evolutions,
                    "max_function_evolutions": max_function_evolutions,
                    "spec_file": spec_file,
                    "tasks": tasks,
                    "recipes_path": recipes_path,
                    "hints_path": hints_path,
                    "model_type": model_type,
                    "skip_cfg_generation": True,  # We already have CFG
                    "cfg_output_file": cfg_path,
                    "checkpoint_type": "dsl_evolution",
                    "failing_tasks": []
                }
                with open(checkpoint_path, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2)
                print(f"   Saved checkpoint to {checkpoint_path}")
                
                # Generate plots before exiting (to show progress so far)
                print("\n[Generating Plots] Creating reward vs interactions plots...")
                results_tracker.plot_reward_vs_interactions(dsl_round=dsl_round, func_evolution_round=None)
                results_tracker.plot_all_tasks_combined(dsl_round=dsl_round, func_evolution_round=None)
                # Also generate separate plots per task from evolution metrics
                results_tracker.plot_tasks_separately_from_metrics(dsl_round=dsl_round, func_evolution_round=None)
                
                if dsl_round < max_dsl_evolutions - 1:
                    if local:
                        print(f"\n  {'='*80}")
                        print("  DSL EVOLVED - CONTINUING LOCALLY")
                        print(f"  Next DSL round: {dsl_round + 2}/{max_dsl_evolutions}")
                        print(f"  {'='*80}")
                        continue
                    print(f"\n  {'='*80}")
                    print("  DSL EVOLVED - EXITING TO RESUBMIT JOB")
                    print(f"  Next DSL round will be: {dsl_round + 2}/{max_dsl_evolutions}")
                    print(f"  Current dsl_round: {dsl_round}, max_dsl_evolutions: {max_dsl_evolutions}")
                    print(f"  Exiting with code: {EXIT_CODE_DSL_EVOLVED}")
                    print(f"  {'='*80}")
                    return EXIT_CODE_DSL_EVOLVED
                else:
                    print("  Reached maximum DSL evolution rounds")
                    
                    # Generate plots even if max evolutions reached
                    print("\n[Generating Plots] Creating reward vs interactions plots...")
                    results_tracker.plot_reward_vs_interactions(dsl_round=dsl_round, func_evolution_round=None)
                    results_tracker.plot_all_tasks_combined(dsl_round=dsl_round, func_evolution_round=None)
                    # Also generate separate plots per task from evolution metrics
                    results_tracker.plot_tasks_separately_from_metrics(dsl_round=dsl_round, func_evolution_round=None)
            else:
                print(f"\n   DSL evolution failed after {max_dsl_retries} attempts")
                print(f"  Could not generate a valid, different CFG after {max_dsl_retries} retries")
                
                # Generate plots even if DSL evolution failed
                print("\n[Generating Plots] Creating reward vs interactions plots...")
                results_tracker.plot_reward_vs_interactions(dsl_round=dsl_round, func_evolution_round=None)
                results_tracker.plot_all_tasks_combined(dsl_round=dsl_round, func_evolution_round=None)
                # Also generate separate plots per task from evolution metrics
                results_tracker.plot_tasks_separately_from_metrics(dsl_round=dsl_round, func_evolution_round=None)
                if dsl_round < max_dsl_evolutions - 1:
                    print("  Continuing to next DSL evolution round...")
                    continue
                else:
                    print("  Reached maximum DSL evolution rounds")
        else:
            # This shouldn't happen, but just in case
            print("\n   All tasks solved (should have returned earlier)")
            return 0
    
    # If we get here, we've exhausted all evolution rounds
    print(f"\n{'='*80}")
    print(" PIPELINE COMPLETED WITHOUT SOLVING ALL TASKS")
    print(f"{'='*80}")
    print(f"Remaining failing tasks: {failing_tasks}")
    return 1


def main():
    parser = argparse.ArgumentParser(
        description="Unified pipeline: Get CFG, implement, test, evolve functions, evolve DSL"
    )
    parser.add_argument(
        '--experiment_dir',
        type=str,
        required=True,
        help='Path to experiment directory'
    )
    parser.add_argument(
        '--spec_file',
        type=str,
        required=True,
        help='Path to specification file for funsearch'
    )
    parser.add_argument(
        '--tasks',
        type=str,
        nargs='+',
        required=True,
        help='List of tasks to solve (e.g., "make[stick]" "get[gem]")'
    )
    parser.add_argument(
        '--max_dsl_evolutions',
        type=int,
        default=3,
        help='Maximum number of DSL evolution rounds (default: 3)'
    )
    parser.add_argument(
        '--max_function_evolutions',
        type=int,
        default=3,
        help='Maximum number of function evolution rounds per DSL round (default: 3)'
    )
    parser.add_argument(
        '--skip_cfg_generation',
        action='store_true',
        help='Skip CFG generation and load from file'
    )
    parser.add_argument(
        '--cfg_output_file',
        type=str,
        default=None,
        help='File to load CFG from (if skip_cfg_generation is True)'
    )
    parser.add_argument(
        '--max_cfg_retries',
        type=int,
        default=10,
        help='Maximum number of attempts to generate a valid CFG (default: 10)'
    )
    parser.add_argument(
        '--nld_path',
        type=str,
        default="prompt_specifications/nld.txt",
        help='Path to natural language domain description file'
    )
    parser.add_argument(
        '--codebase_path',
        type=str,
        default=None,
        help='Path to codebase description for <<CODEBASE>> in spec (default: experiment config)'
    )
    parser.add_argument(
        '--recipes_path',
        type=str,
        default="craft/resources/recipes.yaml",
        help='Path to recipes YAML file'
    )
    parser.add_argument(
        '--hints_path',
        type=str,
        default="craft/resources/hints.yaml",
        help='Path to hints YAML file'
    )
    parser.add_argument(
        '--max_attempts',
        type=int,
        default=1,
        help='Maximum number of attempts to synthesize a program for each task (default: 1)'
    )
    parser.add_argument(
        '--model_type',
        type=str,
        choices=['huggingface', 'ollama', 'gemini', 'openai_compat'],
        default='huggingface',
        help='Model type for funsearch'
    )
    parser.add_argument(
        '--resume_from_checkpoint',
        action='store_true',
        help='Resume from checkpoint if available'
    )
    parser.add_argument(
        '--local',
        action='store_true',
        help='Run fully locally in one process (do not exit with resubmission code after DSL evolution)'
    )
    
    args = parser.parse_args()
    
    # Handle case where tasks argument is a JSON file path
    tasks = args.tasks
    if len(tasks) == 1 and tasks[0].endswith('.json'):
        # Load tasks from JSON file
        tasks_file = tasks[0]
        if os.path.exists(tasks_file):
            with open(tasks_file, 'r') as f:
                config = json.load(f)
                tasks = config.get("tasks", [])
                print(f"Loaded {len(tasks)} tasks from {tasks_file}")
        else:
            print(f" Error: Tasks file not found: {tasks_file}")
            return 1
    
    return run_unified_pipeline(
        experiment_dir=args.experiment_dir,
        spec_file=args.spec_file,
        tasks=tasks,
        max_dsl_evolutions=args.max_dsl_evolutions,
        max_function_evolutions=args.max_function_evolutions,
        skip_cfg_generation=args.skip_cfg_generation,
        cfg_output_file=args.cfg_output_file,
        max_cfg_retries=args.max_cfg_retries,
        nld_path=args.nld_path,
        codebase_path=args.codebase_path,
        recipes_path=args.recipes_path,
        hints_path=args.hints_path,
        max_attempts=args.max_attempts,
        model_type=args.model_type,
        resume_from_checkpoint=args.resume_from_checkpoint,
        local=args.local,
    )


if __name__ == "__main__":
    sys.exit(main())

