#!/usr/bin/env python3
"""
Stage 7: Evolve DSL
This stage evolves the DSL when tasks still fail after function evolution.
"""

import os
import sys
import json
import argparse
import subprocess

# Add project root to path
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _project_root)

from src.pipeline.integrated_pipeline import evolve_dsl, run_failure_analysis_for_dsl_evolution
from src.utils.pipeline_state import read_state, resolve_model_type_for_chained_jobs, update_state
from src.utils.synthesis_failed_programs import extract_failed_programs_from_synthesis_results
from src.utils.file_utils import version_file
from src.utils.status_manager import write_status

# Import vLLM for shared instance
try:
    from vllm import LLM as vLLM
except ImportError:
    vLLM = None


def _chain_next_stage(experiment_dir: str) -> None:
    """Submit the next pipeline stage (e.g. file_generation after DSL evolution)."""
    env = os.environ.copy()
    env["EXPERIMENT_DIR"] = experiment_dir
    print("\n[Chaining] Invoking chain_next_stage...")
    subprocess.run(
        ["bash", "-c", "source scripts/stages/chain_next_stage.sh && chain_based_on_state"],
        cwd=_project_root,
        env=env,
        check=False,
    )


def main():
    parser = argparse.ArgumentParser(description="Stage 7: Evolve DSL")
    parser.add_argument('--experiment_dir', type=str, required=True, help='Experiment directory')
    parser.add_argument('--failing_tasks', type=str, nargs='+', required=True, help='List of failing tasks')
    parser.add_argument('--recipes_path', type=str, default="craft/resources/recipes.yaml", help='Path to recipes YAML')
    parser.add_argument('--max_retries', type=int, default=10, help='Maximum retries for DSL evolution')
    parser.add_argument('--dsl_version', type=int, default=0, help='DSL version to load (e.g., 0 for cfg_output_0.json)')
    parser.add_argument('--max_failed_programs', type=int, default=30, help='Maximum failed programs per task for failure-analysis context')
    parser.add_argument(
        '--model_type',
        type=str,
        default=None,
        choices=['huggingface', 'ollama', 'gemini', 'openai_compat'],
        help='LLM backend (default: resolve from pipeline_state / MODEL_TYPE; huggingface uses local vLLM)',
    )
    parser.add_argument(
        '--openai_compat_key_file',
        type=str,
        default=None,
        help='Key file for OpenAI-compatible API when model_type is openai_compat',
    )

    args = parser.parse_args()

    cli_model_type = args.model_type
    if cli_model_type is None:
        cli_model_type = os.environ.get("MODEL_TYPE", "").strip() or None

    resolved_model_type = resolve_model_type_for_chained_jobs(args.experiment_dir, cli_model_type)

    if args.openai_compat_key_file and not os.environ.get("OPENAI_COMPAT_API_KEY", "").strip():
        from src.utils.openai_compat_key import resolve_openai_compat_api_key
        os.environ["OPENAI_COMPAT_API_KEY"] = resolve_openai_compat_api_key(args.openai_compat_key_file)
    
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

    print(f"\n[Setup] DSL evolution LLM backend: {resolved_model_type}")

    # Shared vLLM only when not using HTTP API (matches stage_get_cfg / stage_evolve_functions pattern).
    shared_vllm = None
    if resolved_model_type == "openai_compat":
        print("[Setup] Using OpenAI-compatible API for failure analysis + CFG evolution (no local vLLM)")
    elif vLLM is not None:
        try:
            print("\n[Setup] Initializing shared vLLM instance...")
            shared_vllm = vLLM(model="/scratch/avani/gpt", tensor_parallel_size=4)
            print(" Shared vLLM instance created")
        except Exception as e:
            print(f" Warning: Could not create shared vLLM instance: {e}")
            shared_vllm = None

    if resolved_model_type != "openai_compat" and shared_vllm is None:
        print(
            "\nERROR: DSL evolution needs local vLLM for model_type=huggingface, "
            "but vLLM did not start (no GPU / CUDA, wrong node, or bad model path).\n"
            "Fix one of:\n"
            "  • Submit this stage on a GPU partition with CUDA visible.\n"
            "  • Or use the HTTP API: export MODEL_TYPE=openai_compat and set "
            "OPENAI_COMPAT_API_KEY or pass --openai_compat_key_file.",
            file=sys.stderr,
        )
        return 1

    # Evolve DSL with retries
    print(f"\n[Step 7] Evolving DSL with {len(args.failing_tasks)} failing tasks...")
    dsl_success = False
    new_cfg = cfg
    new_terminals = terminals
    
    # Extract failed programs from synthesis results for context
    print("\n[Step 1] Extracting failed programs from synthesis results...")
    failed_programs_by_task = extract_failed_programs_from_synthesis_results(
        args.experiment_dir,
        args.failing_tasks,
        args.dsl_version,
        max_programs_per_task=int(args.max_failed_programs),
    )

    # Failure analysis LLM runs once; retries only re-run CFG evolution with the same analysis.
    print("\n[Step 2] Running failure analysis (once per session)...")
    failure_analysis_cached = run_failure_analysis_for_dsl_evolution(
        experiment_dir=args.experiment_dir,
        failing_tasks=args.failing_tasks,
        cfg=cfg,
        terminals=terminals,
        failed_programs_by_task=failed_programs_by_task,
        shared_vllm=shared_vllm,
        model_type=resolved_model_type,
        openai_compat_key_file=args.openai_compat_key_file,
    )

    for dsl_attempt in range(1, args.max_retries + 1):
        if dsl_attempt > 1:
            print(f"\n[DSL Evolution Retry] Attempt {dsl_attempt}/{args.max_retries} (CFG evolution only)")

        new_cfg, new_terminals, attempt_success = evolve_dsl(
            experiment_dir=args.experiment_dir,
            failing_tasks=args.failing_tasks,
            cfg=cfg,
            recipes=recipes,
            terminals=terminals,
            failure_analysis=failure_analysis_cached,
            shared_vllm=shared_vllm,
            new_dsl_round=args.dsl_version + 1,
            model_type=resolved_model_type,
            openai_compat_key_file=args.openai_compat_key_file,
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
                    version_file(cfg_path)
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
        new_dsl_round = args.dsl_version + 1
        # Status lives under the *new* DSL round (evolve on dsl0 produces dsl1).
        write_status(
            args.experiment_dir,
            new_dsl_round,
            "evolve_dsl",
            stage_status,
        )

        # Update state and chain back to file generation
        state = read_state(args.experiment_dir)
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
        
        print(f"\n[Chaining] DSL evolved to round {new_dsl_round}.")
        _chain_next_stage(args.experiment_dir)
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


