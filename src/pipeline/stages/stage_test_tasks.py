#!/usr/bin/env python3
"""
Stage 5: Test Tasks
This stage tests the CFG on tasks by synthesizing programs.
"""

import os
import sys
import json
import argparse
import re
import shutil

# Add project root to path
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _project_root)

from src.pipeline.integrated_pipeline import test_cfg_on_tasks, ensure_terminals_match_cfg
from src.pipeline.cfg_to_funsearch_pipeline import sanitize_function_name
from src.utils.results_tracker import ResultsTracker
from src.utils.pipeline_state import update_state, read_state, resolve_model_type_for_chained_jobs
from src.utils.status_manager import write_status
from src.utils.per_task_test_paths import program_synthesis_task_shard_dir

# Import vLLM for shared instance
try:
    from vllm import LLM as vLLM
except ImportError:
    vLLM = None


DEFAULT_TEST_SEEDS = list(range(0, 50, 5))


def _safe_task_token(task: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(task)).strip("_")
    return token or "task"


def _task_status_filename(task: str) -> str:
    return f"{_safe_task_token(task)}.json"


def main():
    parser = argparse.ArgumentParser(description="Stage 5: Test Tasks")
    parser.add_argument('--experiment_dir', type=str, required=True, help='Experiment directory')
    parser.add_argument('--tasks', type=str, nargs='+', required=True, help='List of tasks to solve')
    parser.add_argument('--recipes_path', type=str, default="craft/resources/recipes.yaml", help='Path to recipes YAML')
    parser.add_argument('--hints_path', type=str, default="craft/resources/hints.yaml", help='Path to hints YAML')
    parser.add_argument('--max_attempts', type=int, default=30, help='Maximum attempts per task')
    parser.add_argument(
        '--test_seeds',
        type=int,
        nargs='+',
        default=DEFAULT_TEST_SEEDS,
        help='Seeds used for test-task synthesis (default: 0 5 10 15 20 25 30 35 40 45)',
    )
    parser.add_argument('--synthesis_prompt', type=str, default=None, help='Path to synthesis prompt template file (default: prompt_specifications/prompt_synth_with_grid_and_failures.txt)')
    parser.add_argument(
        '--with_final_functions_in_prompt',
        action='store_true',
        help='Include final function Python source in the synthesis prompt (default: off; default prompt unchanged)',
    )
    parser.add_argument(
        '--openai_compat_key_file',
        type=str,
        default=None,
        help='File with OpenAI-compatible API key (first non-empty line). Default: <repo>/key.txt if OPENAI_COMPAT_API_KEY unset.',
    )
    parser.add_argument(
        '--model_type',
        type=str,
        default=os.environ.get("MODEL_TYPE", "huggingface"),
        choices=['huggingface', 'ollama', 'gemini', 'openai_compat'],
        help='Model type for program synthesis (default: MODEL_TYPE env or huggingface)',
    )
    parser.add_argument('--dsl_round', type=int, default=int(os.environ.get("DSL_ROUND", "0")), help='DSL evolution round number')
    parser.add_argument(
        '--func_evolution_round',
        type=int,
        default=int(os.environ.get("FUNC_EVOLUTION_ROUND", "0")),
        help='Function evolution round number (0 = initial terminal functions)',
    )
    parser.add_argument(
        '--single_task_job',
        action='store_true',
        help='Run in single-task job mode and write per-task status/results for later aggregation',
    )
    parser.add_argument(
        '--tasks_subdir',
        type=str,
        default=os.environ.get("PROG_SYNTH_TASKS_SUBDIR", "tasks"),
        help='Subdir under results_tracking/dsl{n}/func{m}/ for shard output (e.g. tasks_api)',
    )

    args = parser.parse_args()

    args.model_type = resolve_model_type_for_chained_jobs(args.experiment_dir, args.model_type)
    print(f"[Config] Resolved model_type={args.model_type} for program synthesis")

    if args.openai_compat_key_file and not os.environ.get("OPENAI_COMPAT_API_KEY", "").strip():
        from src.utils.openai_compat_key import resolve_openai_compat_api_key
        os.environ["OPENAI_COMPAT_API_KEY"] = resolve_openai_compat_api_key(args.openai_compat_key_file)
    
    # Handle tasks - can be passed as space-separated string or JSON file
    tasks = args.tasks
    
    # If only one argument and it's a JSON file, load from file
    if len(tasks) == 1 and tasks[0].endswith('.json'):
        tasks_file = tasks[0]
        if os.path.exists(tasks_file):
            with open(tasks_file, 'r') as f:
                config = json.load(f)
                tasks = config.get("tasks", [])
        else:
            print(f" Tasks file not found: {tasks_file}", file=sys.stderr)
            return 1
    # If only one argument and it's a JSON string, parse it
    elif len(tasks) == 1 and tasks[0].startswith('['):
        try:
            tasks = json.loads(tasks[0])
        except:
            pass  # Keep as is if not valid JSON
    # If tasks come from environment as space-separated, they should already be split by argparse
    # But if they're passed as a single string with spaces, we need to handle it
    elif len(tasks) == 1 and ' ' in tasks[0]:
        # Split by space if it's a single string with spaces
        tasks = tasks[0].split()
    
    # Deduplicate tasks
    tasks = list(dict.fromkeys(tasks))  # Preserves order while removing duplicates
    if args.single_task_job and len(tasks) > 1:
        print(f"[Config] single_task_job enabled with multiple tasks; using only first task: {tasks[0]}")
        tasks = [tasks[0]]

    test_seeds = list(dict.fromkeys(int(s) for s in args.test_seeds))
    print(f"[Config] Test seeds: {test_seeds}")
    print(f"[Config] model_type={args.model_type} (synthesis: {'OpenAI-compatible API' if args.model_type == 'openai_compat' else 'local vLLM'})")
    
    # Load CFG - use dsl_round to select which cfg version
    from src.utils.file_utils import resolve_cfg_path
    cfg_path = resolve_cfg_path(args.experiment_dir, args.dsl_round)
    if not os.path.exists(cfg_path):
        print(f" CFG file not found: {cfg_path}", file=sys.stderr)
        return 1
    
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg_data = json.load(f)
    cfg = cfg_data.get("cfg", "")
    terminals = cfg_data.get("terminals", {})
    
    if not cfg:
        print(" Invalid CFG data", file=sys.stderr)
        return 1
    
    # Ensure terminals match CFG (add missing functions from CFG)
    terminals = ensure_terminals_match_cfg(cfg, terminals, shared_vllm=None)
    
    if not terminals:
        print(" No terminal functions found in CFG", file=sys.stderr)
        return 1
    
    # Guard: Check that final functions exist for this exact round
    # If func_evolution_round > 0, we must have func{N} files — don't silently fall back to func0
    if args.func_evolution_round is not None and args.func_evolution_round > 0:
        final_functions_dir = os.path.join(args.experiment_dir, "final_functions")
        missing_for_round = []
        for func_name in terminals:
            safe_name = sanitize_function_name(func_name)
            expected_file = os.path.join(
                final_functions_dir,
                f"{safe_name}_dsl{args.dsl_round}_func{args.func_evolution_round}.py"
            )
            if not os.path.exists(expected_file):
                missing_for_round.append(f"{safe_name}_dsl{args.dsl_round}_func{args.func_evolution_round}.py")
        
        if missing_for_round:
            print(f"\n ERROR: test_tasks called for func_evolution_round={args.func_evolution_round} "
                  f"but the following final function files are missing:", file=sys.stderr)
            for m in missing_for_round:
                print(f"  - {m}", file=sys.stderr)
            print("\n  The evolve stage must produce these files before test_tasks can run.", file=sys.stderr)
            print("  Skipping test_tasks for this round.", file=sys.stderr)
            return 1
    
    # Create shared vLLM instance (not used when model_type is openai_compat — skip to avoid GPU init noise)
    shared_vllm = None
    if args.model_type != "openai_compat" and vLLM is not None:
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
            
            print("[Setup] Initializing shared vLLM instance...")
            print("  This may take a few minutes and requires significant GPU memory...")
            shared_vllm = vLLM(
                model="/scratch/avani/gpt",
                tensor_parallel_size=4,
                gpu_memory_utilization=0.75  # Use 75% of GPU memory instead of default 90%
            )
            print(" Shared vLLM instance created")
        except Exception as e:
            print(f" ERROR: Failed to create shared vLLM instance: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            # Clean up GPU memory after failure
            import gc
            import torch
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            shared_vllm = None
    
    # Create ResultsTracker. In parallel single-task mode we isolate writes per task
    # under results_tracking/dsl{n}/func{m}/prog_synthoutput/<task>/ (no nested results_tracking).
    single_task_shard_dir = None
    if args.single_task_job and tasks:
        func_round = args.func_evolution_round if args.func_evolution_round is not None else 0
        single_task_shard_dir = program_synthesis_task_shard_dir(
            args.experiment_dir,
            dsl_round=args.dsl_round,
            func_evolution_round=func_round,
            task_token=_safe_task_token(tasks[0]),
            tasks_subdir=args.tasks_subdir,
        )
        if os.path.isdir(single_task_shard_dir):
            shutil.rmtree(single_task_shard_dir)
        os.makedirs(single_task_shard_dir, exist_ok=True)
        results_tracker = ResultsTracker(
            args.experiment_dir,
            results_dir=single_task_shard_dir,
        )
        print(f"[Config] Single-task program synthesis shard: {single_task_shard_dir}")
    else:
        results_tracker = ResultsTracker(args.experiment_dir)
    
    # Test CFG on tasks
    print(f"\n[Step 5] Testing CFG on {len(tasks)} tasks...")
    task_results = test_cfg_on_tasks(
        experiment_dir=args.experiment_dir,
        tasks=tasks,
        cfg=cfg,
        terminals=terminals,
        recipes_path=args.recipes_path,
        hints_path=args.hints_path,
        max_attempts=args.max_attempts,
        test_seeds=test_seeds,
        shared_vllm=shared_vllm,
        results_tracker=results_tracker,
        cfg_version=args.dsl_round,
        func_evolution_round=args.func_evolution_round,
        synthesis_prompt_path=args.synthesis_prompt,
        model_type=args.model_type,
        include_final_functions_in_prompt=args.with_final_functions_in_prompt,
        openai_compat_key_file=args.openai_compat_key_file,
        seed_outcome_log_path=(
            os.path.join(single_task_shard_dir, "program_synthesis_seed_outcomes.jsonl")
            if single_task_shard_dir
            else None
        ),
    )
    
    all_solved = all(task_results.values())
    failing_tasks = [task for task, success in task_results.items() if not success]

    if args.single_task_job:
        task_name = tasks[0] if tasks else "unknown_task"
        task_status = {
            "stage": "test_tasks",
            "mode": "single_task_job",
            "status": "completed",
            "task": task_name,
            "dsl_round": args.dsl_round,
            "func_evolution_round": args.func_evolution_round,
            "success": bool(task_results.get(task_name, False)),
            "task_results": task_results,
            "all_solved": all_solved,
            "failing_tasks": failing_tasks,
        }
        write_status(
            args.experiment_dir,
            args.dsl_round,
            "test_tasks_tasks",
            task_status,
            filename=_task_status_filename(task_name),
        )
        print(f"[Single Task Job] Wrote status for task '{task_name}'")

        # Per-task jobs intentionally skip consolidated plotting/state updates.
        # Aggregation is handled once all task jobs complete.
        if shared_vllm is not None:
            try:
                print("\n[Cleanup] Cleaning up vLLM instance and GPU memory...")
                del shared_vllm
                import gc
                import torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                print(" Cleanup complete")
            except Exception as e:
                print(f" Warning: Error during cleanup: {e}")

        return 0
    
    # Save consolidated stage completion marker (single file for all tasks)
    stage_status = {
        "stage": "test_tasks",
        "status": "completed",
        "dsl_round": args.dsl_round,
        "func_evolution_round": args.func_evolution_round,
        "task_results": task_results,
        "all_solved": all_solved,
        "failing_tasks": failing_tasks
    }
    write_status(args.experiment_dir, args.dsl_round, "test_tasks", stage_status)
    
    print(f"\n{'='*80}")
    print("Task Test Results")
    print(f"{'='*80}")
    for task, success in task_results.items():
        status = "" if success else ""
        print(f"  {status} {task}")
    
    # Generate plots from results tracking
    print("\n[Generating Plots] Creating plots from results tracking...")
    try:
        if results_tracker.results:
            results_tracker.plot_reward_vs_interactions(
                dsl_round=args.dsl_round,
                func_evolution_round=args.func_evolution_round
            )
            results_tracker.plot_all_tasks_combined(
                dsl_round=args.dsl_round,
                func_evolution_round=args.func_evolution_round
            )
            print(" Plots generated successfully")
        else:
            print(" No results found for plotting")
    except Exception as e:
        print(f" Warning: Could not generate plots: {e}")
        import traceback
        traceback.print_exc()
    
    # After test_tasks completes: set function_impl_remaining to max and test_tasks_submitted to 1
    # This prepares for function evolution (which will reset both to 0 when done)
    print("\n[Chaining] All test tasks completed. Updating state...")
    state = read_state(args.experiment_dir)
    function_impl_total = state.get("function_implementation_total", 0)
    
    if all_solved:
        # If all solved, set both to 0 (pipeline complete)
        update_state(args.experiment_dir, 
                     test_tasks_submitted=0,
                     function_implementation_remaining=0)
        print("  All tasks solved! Pipeline complete.")
    else:
        # If not all solved, set function_impl_remaining to max and test_tasks_submitted to 1
        # This prepares for function evolution
        update_state(args.experiment_dir,
                     test_tasks_submitted=1,
                     function_implementation_remaining=function_impl_total)
        print(f"  Set function_implementation_remaining={function_impl_total}, test_tasks_submitted=1")
        print("  Chaining script will handle function evolution submission.")
    
    if all_solved:
        print("\n ALL TASKS SOLVED! Pipeline complete.")
        update_state(args.experiment_dir, phase="complete")
    else:
        print(f"\n {len(failing_tasks)}/{len(tasks)} tasks failed: {failing_tasks}")
        print("  Chaining script will check results and submit evolution jobs if needed.")
    
    # Clean up vLLM instance and GPU memory before exiting
    if shared_vllm is not None:
        try:
            print("\n[Cleanup] Cleaning up vLLM instance and GPU memory...")
            del shared_vllm
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            print(" Cleanup complete")
        except Exception as e:
            print(f" Warning: Error during cleanup: {e}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


