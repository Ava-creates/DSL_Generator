#!/usr/bin/env python3
"""
Stage: Implement CFG Single Function (FunSearch + Explicit Feedback Package)
This stage runs FunSearch and Explicit Feedback together for a single function.
"""

import os
import sys
import json
import argparse
import glob
import re

# Add project root to path
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _project_root)

from src.utils.saved_function import (
    normalize_saved_function,
    resolve_func_signature,
    prepare_function_module_source,
    best_function_from_funsearch_log,
)
from src.pipeline.cfg_to_funsearch_pipeline import (
    determine_inputs,
    run_explicit_feedback_generation,
    sanitize_function_name,
    find_funsearch_log_file,
    apply_specification_template_placeholders,
)
from funsearch.implementation.funsearch import FunSearch
from funsearch.implementation import config as config_lib
from src.utils.results_tracker import (
    ResultsTracker,
    generate_reward_plots_for_function,
)
from src.utils.pipeline_state import (
    decrement_function_implementation,
    read_state,
    update_state,
)
from src.utils.status_manager import read_status, write_function_status
from src.utils.config_loader import funsearch_grid_regen_kwargs_from_config, load_config
from src.utils.file_utils import resolve_cfg_path, resolve_final_function_path
from src.pipeline.llm_terminal_function_generation import (
    run_llm_best_of_n,
    run_llm_chained,
    find_llm_log_file,
)

# Import vLLM for shared instance
try:
    from vllm import LLM as vLLM
except ImportError:
    vLLM = None


def main():
    _exp_cfg = load_config()
    _default_grid_regen = int(_exp_cfg.get("grid_regeneration_attempts", 5))
    parser = argparse.ArgumentParser(description="Stage: Implement CFG Single Function (FunSearch + Explicit Feedback Package)")
    parser.add_argument('--experiment_dir', type=str, required=True, help='Experiment directory')
    parser.add_argument('--spec_file', type=str, required=True, help='Path to specification file')
    parser.add_argument('--function_name', type=str, required=True, help='Name of the function to process')
    parser.add_argument('--model_type', type=str, default='huggingface', choices=['huggingface', 'ollama', 'gemini', 'openai_compat'])
    parser.add_argument('--dsl_round', type=int, default=0, help='DSL evolution round number')
    parser.add_argument(
        '--func_evolution_round',
        type=int,
        default=int(os.environ.get('FUNC_EVOLUTION_ROUND', '0')),
        help='Function evolution round (passed by chain/slurm)',
    )
    parser.add_argument('--total_samples', type=int, default=1000, help='Total number of samples for FunSearch (default: 1000)')
    parser.add_argument('--num_iterations', type=int, default=30, help='Number of explicit feedback iterations')
    parser.add_argument(
        '--grid_regeneration_attempts',
        type=int,
        default=_default_grid_regen,
        help='Attempts to regenerate grids when initial pass_check fails (default: experiment config / GRID_REGENERATION_ATTEMPTS)',
    )
    parser.add_argument('--nld_path', type=str, default=None, help='NLD file for <<NLD>> (default: experiment config / NLD_PATH)')
    parser.add_argument('--codebase_path', type=str, default=None, help='Codebase file for <<CODEBASE>> (default: experiment config / CODEBASE_PATH)')
    parser.add_argument('--baseline_mode', action='store_true', help='Allow baseline runs with empty cfg text')
    parser.add_argument('--openai_compat_key_file', type=str, default=None, help='File with OpenAI-compatible API key (first non-empty line). Default: <repo>/key.txt if OPENAI_COMPAT_API_KEY unset.')
    parser.add_argument(
        '--skip_funsearch',
        action='store_true',
        help='Skip FunSearch and run explicit feedback only, using the existing function artifact (func_file from file generation status).',
    )
    parser.add_argument(
        '--terminal_function_mode',
        type=str,
        default=None,
        choices=['funsearch', 'llm_best_of_n', 'llm_chained'],
        help='How to generate terminal functions: funsearch (default), llm_best_of_n, or llm_chained.',
    )
    parser.add_argument(
        '--skip_explicit_feedback',
        action='store_true',
        help='Skip explicit feedback (normally false for all terminal_function_mode values).',
    )
    parser.add_argument(
        '--funsearch_vector_clustering',
        action='store_true',
        help='Cluster FunSearch programs by per-test ans vector (grid/seed pass-fail) instead of scalar reward.',
    )

    args = parser.parse_args()

    _exp_cfg = load_config()
    if args.terminal_function_mode is None:
        args.terminal_function_mode = _exp_cfg.get('terminal_function_mode', 'funsearch')
    if not args.skip_explicit_feedback:
        env_skip_ef = os.environ.get('SKIP_EXPLICIT_FEEDBACK', '').strip().lower()
        if env_skip_ef in {'1', 'true', 'yes'}:
            args.skip_explicit_feedback = True
        elif _exp_cfg.get('skip_explicit_feedback'):
            args.skip_explicit_feedback = bool(_exp_cfg.get('skip_explicit_feedback'))

    if os.environ.get("SKIP_FUNSEARCH", "").strip().lower() in {"1", "true", "yes"}:
        args.skip_funsearch = True
    # CLI --terminal_function_mode wins over TERMINAL_FUNCTION_MODE env (ablation passes mode explicitly).
    if args.terminal_function_mode is None and os.environ.get("TERMINAL_FUNCTION_MODE", "").strip():
        args.terminal_function_mode = os.environ.get("TERMINAL_FUNCTION_MODE", "").strip()
    if os.environ.get("FUNSEARCH_VECTOR_CLUSTERING", "").strip().lower() in {"1", "true", "yes"}:
        args.funsearch_vector_clustering = True
    if args.funsearch_vector_clustering:
        os.environ["FUNSEARCH_VECTOR_CLUSTERING"] = "1"
        print("[Config] FunSearch vector clustering enabled (signature = ans pass/fail vector)")

    # If a key file is given, resolve and export it so deep callers (sampler, evaluator) pick it up.
    if args.openai_compat_key_file and not os.environ.get("OPENAI_COMPAT_API_KEY", "").strip():
        from src.utils.openai_compat_key import resolve_openai_compat_api_key
        os.environ["OPENAI_COMPAT_API_KEY"] = resolve_openai_compat_api_key(args.openai_compat_key_file)
    
    # Read state file to get current DSL and function evolution rounds
    # This ensures consistency after DSL evolution (which resets func_evolution_round to 0)
    state = read_state(args.experiment_dir)
    state_dsl_round = state.get("dsl_round", 0)
    
    force_dsl_round = os.environ.get("FORCE_DSL_ROUND", "").strip().lower() in {"1", "true", "yes"}
    if force_dsl_round:
        print(f"[Implement CFG] FORCE_DSL_ROUND set; keeping dsl_round={args.dsl_round}")
    elif args.dsl_round != state_dsl_round:
        print("[Implement CFG]  Warning: dsl_round mismatch!")
        print(f"  Command line: {args.dsl_round}, State file: {state_dsl_round}")
        print(f"  Using state file value: {state_dsl_round}")
        args.dsl_round = state_dsl_round
    
    print(f"[Implement CFG] Using dsl_round={args.dsl_round}")

    # Persist so chained jobs (test_tasks) pick the same backend when MODEL_TYPE is missing from env.
    update_state(args.experiment_dir, pipeline_model_type=args.model_type)
    
    # Load CFG for this DSL round only (no cross-round fallback).
    cfg_path = resolve_cfg_path(args.experiment_dir, args.dsl_round)
    if not os.path.exists(cfg_path):
        print(f" CFG file not found for dsl_round={args.dsl_round}: {cfg_path}", file=sys.stderr)
        return 1
    
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg_data = json.load(f)
    cfg = cfg_data.get("cfg", "")
    terminals = cfg_data.get("terminals", {})
    
    if not terminals:
        print(" Invalid CFG data", file=sys.stderr)
        return 1
    if not cfg and not args.baseline_mode:
        print(" Invalid CFG data", file=sys.stderr)
        return 1
    if not cfg and args.baseline_mode:
        print(" Warning: Empty CFG text allowed in baseline mode", file=sys.stderr)
    
    if args.function_name not in terminals:
        print(f" Function {args.function_name} not found in terminals", file=sys.stderr)
        return 1
    
    description = terminals[args.function_name]
    
    # Load file generation status to get func_files and func_init_files
    file_gen_status = read_status(args.experiment_dir, args.dsl_round, "file_generation")
    if file_gen_status is None:
        print(" File generation status not found at dsl<round>/file_generation/status", file=sys.stderr)
        return 1
    
    func_files = file_gen_status.get("func_files", {})
    func_init_files = file_gen_status.get("func_init_files", {})
    func_signatures = file_gen_status.get("func_signatures", {})
    
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
    
    specification = apply_specification_template_placeholders(
        specification,
        cfg=cfg if cfg else None,
        nld_path=args.nld_path,
        codebase_path=args.codebase_path,
    )
    
    # Create shared LLM instance used by both FunSearch and explicit feedback.
    # For openai_compat, use a lightweight HTTP wrapper (no GPU needed).
    shared_vllm = None
    if args.model_type == "openai_compat":
        from src.pipeline.explicit_feedback_generation import OpenAICompatLLMWrapper
        shared_vllm = OpenAICompatLLMWrapper(args.openai_compat_key_file)
        print("[Setup] Using OpenAI-compatible API for LLM inference (no GPU required)")
    elif args.model_type == "huggingface" and vLLM is not None:
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
            
            print("\n  Possible causes:", file=sys.stderr)
            print("    1. GPU memory fragmentation (previous operations left fragmented memory)", file=sys.stderr)
            print(f"    2. Insufficient free GPU memory (need ~{0.75 * 4 * 80:.0f}GB for 4 GPUs at 75% utilization)", file=sys.stderr)
            print("    3. CUDA context issues from previous operations", file=sys.stderr)
            print("    4. Another process/job using the same GPUs", file=sys.stderr)
            print("\n  Failing stage to prevent multiple instances from being created (which would cause OOM)", file=sys.stderr)
            print("  Shared instance is required to avoid GPU memory issues when running FunSearch and explicit feedback sequentially", file=sys.stderr)
            return 1
        except Exception as e:
            print(f" ERROR: Unexpected error creating shared vLLM instance: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            return 1
    
    # Create results tracker
    results_tracker = ResultsTracker(args.experiment_dir)
    
    # Results directories
    results_dir = os.path.join(args.experiment_dir, "results", "funsearch")
    os.makedirs(results_dir, exist_ok=True)
    dsl_folder = f"dsl{args.dsl_round}" if args.dsl_round is not None else "dsl_unknown"
    explicit_feedback_dir = os.path.join(args.experiment_dir, "explicit_feedback", dsl_folder)
    os.makedirs(explicit_feedback_dir, exist_ok=True)
    
    func_signature = resolve_func_signature(
        args.function_name,
        func_signatures,
    )

    def _prepare_final_function(code: str) -> str:
        # EF already returns a full function; only strip FunSearch harness artifacts if present.
        prepared = prepare_function_module_source(code, func_signature, allow_trivial=False)
        if not prepared:
            raise ValueError(
                f"No usable explicit-feedback implementation for {args.function_name}"
            )
        return prepared
    
    print(f"\n{'='*80}")
    print(f"Implementing CFG for function: {args.function_name}")
    print(f"  DSL Round: {args.dsl_round}")
    print(f"  Terminal function mode: {args.terminal_function_mode}")
    print(f"{'='*80}")

    inputs = determine_inputs(args.function_name, description, cfg)
    candidate_log_file: str | None = None

    # Step 1: Generate terminal function candidates
    if args.skip_funsearch and args.terminal_function_mode == 'funsearch':
        print(f"\n[Step 1] Skipping FunSearch; using existing artifact: {func_file}")
        if not os.path.isfile(func_file):
            print(f" Missing FunSearch artifact: {func_file}", file=sys.stderr)
            return 1
        with open(func_file, "r", encoding="utf-8") as _sf:
            _body = _sf.read()
        if not _body.strip():
            print(f" FunSearch artifact is empty: {func_file}", file=sys.stderr)
            return 1
        funsearch_status = {
            "stage": "funsearch",
            "function_name": args.function_name,
            "status": "completed",
            "dsl_round": args.dsl_round,
            "env_steps": 0,
            "skipped_funsearch": True,
        }
        write_function_status(args.experiment_dir, args.dsl_round, "funsearch", args.function_name, funsearch_status)
        candidate_log_file = find_funsearch_log_file(
            args.function_name,
            results_dir,
            dsl_round=args.dsl_round,
        )
        if not candidate_log_file:
            print(
                f"[{args.function_name}] No FunSearch log for dsl_round={args.dsl_round}",
                file=sys.stderr,
            )
            return 1
    elif args.terminal_function_mode == 'llm_best_of_n':
        print(f"\n[Step 1] Running LLM best-of-n for {args.function_name} ({args.total_samples} samples)...")
        candidate_log_file = run_llm_best_of_n(
            specification=specification,
            inputs=inputs,
            func_file=func_file,
            func_init_file=func_init_file,
            spec_file=args.spec_file,
            experiment_dir=args.experiment_dir,
            model_type=args.model_type,
            shared_vllm=shared_vllm,
            results_tracker=results_tracker,
            num_samples=args.total_samples,
            grid_regeneration_attempts=args.grid_regeneration_attempts,
            grid_lookup_experiment_dir=args.experiment_dir,
        )
        write_function_status(args.experiment_dir, args.dsl_round, "funsearch", args.function_name, {
            "stage": "llm_best_of_n",
            "function_name": args.function_name,
            "status": "completed",
            "dsl_round": args.dsl_round,
            "log_file": candidate_log_file,
            "num_samples": args.total_samples,
        })
    elif args.terminal_function_mode == 'llm_chained':
        print(f"\n[Step 1] Running LLM chained feedback for {args.function_name} ({args.total_samples} iterations)...")
        candidate_log_file = run_llm_chained(
            specification=specification,
            inputs=inputs,
            func_file=func_file,
            func_init_file=func_init_file,
            spec_file=args.spec_file,
            experiment_dir=args.experiment_dir,
            model_type=args.model_type,
            shared_vllm=shared_vllm,
            results_tracker=results_tracker,
            num_iterations=args.total_samples,
            grid_regeneration_attempts=args.grid_regeneration_attempts,
            grid_lookup_experiment_dir=args.experiment_dir,
        )
        write_function_status(args.experiment_dir, args.dsl_round, "funsearch", args.function_name, {
            "stage": "llm_chained",
            "function_name": args.function_name,
            "status": "completed",
            "dsl_round": args.dsl_round,
            "log_file": candidate_log_file,
            "num_iterations": args.total_samples,
        })
    else:
        print(f"\n[Step 1] Running FunSearch for {args.function_name}...")
        
        config = config_lib.Config(
            **funsearch_grid_regen_kwargs_from_config(),
            num_samplers=1,
            num_evaluators=1 if args.model_type == "openai_compat" else 2,
            samples_per_prompt=2,
            total_samples=args.total_samples,
            programs_database=config_lib.ProgramsDatabaseConfig(),
            grid_regeneration_attempts=args.grid_regeneration_attempts,
        )
        
        initial_funsearch_steps = results_tracker.interactions.get("funsearch", 0)
        
        funsearch = FunSearch(model_type=args.model_type, shared_vllm=shared_vllm)
        funsearch.results_tracker = results_tracker
        
        funsearch.run(
            specification=specification,
            inputs=inputs,
            config=config,
            function_to_implement=func_file,
            function_init=func_init_file,
            spec_file=args.spec_file,
            experiment_dir=results_dir,
            grid_lookup_experiment_dir=args.experiment_dir,
        )
        
        final_funsearch_steps = results_tracker.interactions.get("funsearch", 0)
        steps_taken = final_funsearch_steps - initial_funsearch_steps
        
        print(f"[{args.function_name}]  Completed FunSearch (env steps: {steps_taken})")
        
        funsearch_status = {
            "stage": "funsearch",
            "function_name": args.function_name,
            "status": "completed",
            "dsl_round": args.dsl_round,
            "env_steps": steps_taken
        }
        write_function_status(args.experiment_dir, args.dsl_round, "funsearch", args.function_name, funsearch_status)
        candidate_log_file = find_funsearch_log_file(
            args.function_name,
            results_dir,
            dsl_round=args.dsl_round,
        )
    
    # Step 2: final_functions come from explicit feedback (FunSearch log is input to EF only).
    final_func = None

    if args.skip_explicit_feedback:
        print(f"\n[Step 2] Skipping explicit feedback (terminal_function_mode={args.terminal_function_mode})")
        if not candidate_log_file or not os.path.isfile(candidate_log_file):
            print(f"[{args.function_name}] Missing candidate log for skip_explicit_feedback mode", file=sys.stderr)
            return 1
        final_func = best_function_from_funsearch_log(candidate_log_file, func_signature)
    else:
        print(f"\n[Step 2] Running explicit feedback generation for {args.function_name}...")
        ef_results_dir = results_dir
        if args.terminal_function_mode in ('llm_best_of_n', 'llm_chained') and candidate_log_file:
            ef_results_dir = os.path.dirname(candidate_log_file)

        ef_func = run_explicit_feedback_generation(
            args.function_name,
            ef_results_dir,
            func_file,
            args.experiment_dir,
            explicit_feedback_dir,
            specification,
            k=5,
            shared_vllm=shared_vllm,
            func_signature=func_signatures.get(args.function_name, ""),
            results_tracker=results_tracker,
            dsl_round=args.dsl_round,
            num_iterations=max(args.num_iterations, 1),
            log_file=candidate_log_file,
        )
        if not ef_func:
            print(
                f"[{args.function_name}] Explicit feedback produced no function",
                file=sys.stderr,
            )
            return 1
        final_func = prepare_function_module_source(ef_func, func_signature, allow_trivial=False)
        if not final_func:
            print(
                f"[{args.function_name}] Explicit-feedback result was empty or trivial after normalization",
                file=sys.stderr,
            )
            return 1
        print(f"[{args.function_name}] Using explicit-feedback result")

    if final_func:
        final_functions_dir = os.path.join(args.experiment_dir, "final_functions")
        os.makedirs(final_functions_dir, exist_ok=True)

        safe_name = sanitize_function_name(args.function_name)
        func_file_path = resolve_final_function_path(
            args.experiment_dir, args.function_name, args.dsl_round
        )

        with open(func_file_path, 'w', encoding='utf-8') as f:
            f.write(_prepare_final_function(final_func))
        print(f"  Saved {args.function_name} to {os.path.basename(func_file_path)}")

        patterns = [
            os.path.join(explicit_feedback_dir, f"{safe_name}_dsl{args.dsl_round}_iter_*.py"),
            os.path.join(explicit_feedback_dir, f"{safe_name}_iter_*.py"),
        ]
        for pattern in patterns:
            for path in glob.glob(pattern):
                os.remove(path)

        explicit_fb_status = {
            "stage": "explicit_feedback",
            "function_name": args.function_name,
            "status": "completed",
            "dsl_round": args.dsl_round,
            "skipped": args.skip_explicit_feedback,
        }
        write_function_status(args.experiment_dir, args.dsl_round, "explicit_feedback", args.function_name, explicit_fb_status)

        if not args.skip_explicit_feedback:
            print(f"[{args.function_name}]  Completed explicit feedback ({args.num_iterations} iterations)")

        generated = generate_reward_plots_for_function(
            experiment_dir=args.experiment_dir,
            function_name=args.function_name,
            dsl_round=args.dsl_round,
        )
        print(
            f"   Reward plots for {args.function_name}: "
            f"funsearch={generated['funsearch']} "
            f"explicit={generated['explicit']} "
            f"baseline={generated['baseline']}"
        )

        print("\n[Chaining] Decrementing function implementation counter...")
        implementation_remaining = decrement_function_implementation(args.experiment_dir)
        print(f"  Function implementations remaining: {implementation_remaining}")

        if shared_vllm is not None:
            print("\n[Cleanup] Freeing shared vLLM instance and GPU memory...")
            del shared_vllm
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            print(" GPU memory freed")

        return 0

    print(f"[{args.function_name}] No final function extracted from explicit feedback or FunSearch log")
    if not candidate_log_file or not os.path.isfile(candidate_log_file):
        print(f"[{args.function_name}] Missing FunSearch log file", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())

