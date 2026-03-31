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

# Add project root to path
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _project_root)

from src.pipeline.cfg_to_funsearch_pipeline import (
    determine_inputs, 
    run_explicit_feedback_generation,
    sanitize_function_name,
    find_funsearch_log_file,
)
from funsearch.implementation.funsearch import FunSearch
from funsearch.implementation import config as config_lib
from src.utils.results_tracker import (
    ResultsTracker,
    plot_funsearch_reward_vs_interactions,
    plot_explicit_feedback_reward_vs_interactions,
    plot_baseline_reward_vs_interactions,
)
from src.utils.pipeline_state import (
    decrement_function_implementation
)
from src.utils.status_manager import read_status, write_function_status

# Import vLLM for shared instance
try:
    from vllm import LLM as vLLM
except ImportError:
    vLLM = None


def main():
    parser = argparse.ArgumentParser(description="Stage: Implement CFG Single Function (FunSearch + Explicit Feedback Package)")
    parser.add_argument('--experiment_dir', type=str, required=True, help='Experiment directory')
    parser.add_argument('--spec_file', type=str, required=True, help='Path to specification file')
    parser.add_argument('--function_name', type=str, required=True, help='Name of the function to process')
    parser.add_argument('--model_type', type=str, default='huggingface', choices=['huggingface', 'ollama', 'gemini'])
    parser.add_argument('--dsl_round', type=int, default=0, help='DSL evolution round number')
    parser.add_argument('--func_evolution_round', type=int, default=None, help='Function evolution round number')
    parser.add_argument('--total_samples', type=int, default=1000, help='Total number of samples for FunSearch (default: 1000)')
    parser.add_argument('--num_iterations', type=int, default=30, help='Number of explicit feedback iterations')
    parser.add_argument(
        '--grid_regeneration_attempts',
        type=int,
        default=int(os.environ.get("GRID_REGENERATION_ATTEMPTS", 5)),
        help='Attempts to regenerate grids when initial pass_check fails'
    )
    
    args = parser.parse_args()
    
    # Read state file to get current DSL and function evolution rounds
    # This ensures consistency after DSL evolution (which resets func_evolution_round to 0)
    from src.utils.pipeline_state import read_state
    state = read_state(args.experiment_dir)
    state_dsl_round = state.get("dsl_round", 0)
    state_func_round = state.get("func_evolution_round", 0)
    
    # Use dsl_round from state file if not provided or if it doesn't match
    if args.dsl_round != state_dsl_round:
        print("[Implement CFG]  Warning: dsl_round mismatch!")
        print(f"  Command line: {args.dsl_round}, State file: {state_dsl_round}")
        print(f"  Using state file value: {state_dsl_round}")
        args.dsl_round = state_dsl_round
    
    # If func_evolution_round was not provided or doesn't match state, use state value
    # This ensures consistency after DSL evolution (which resets func_evolution_round to 0)
    if args.func_evolution_round is None:
        args.func_evolution_round = state_func_round
        print(f"[Implement CFG] Using func_evolution_round={args.func_evolution_round} from state file")
    elif args.func_evolution_round != state_func_round:
        print("[Implement CFG]  Warning: func_evolution_round mismatch!")
        print(f"  Command line: {args.func_evolution_round}, State file: {state_func_round}")
        print(f"  Using state file value: {state_func_round}")
        args.func_evolution_round = state_func_round
    
    print(f"[Implement CFG] Using dsl_round={args.dsl_round}, func_evolution_round={args.func_evolution_round}")
    
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
    
    # Replace DSL section in specification with current CFG
    import re
    if cfg:
        dsl_pattern = r'(## DSL[^\n]*\n.*?"""\n)(.*?)(\n"""\n)'
        dsl_match = re.search(dsl_pattern, specification, re.DOTALL)
        if dsl_match:
            header = dsl_match.group(1)
            footer = dsl_match.group(3)
            cfg_section = header + cfg + footer
            specification = re.sub(dsl_pattern, cfg_section, specification, flags=re.DOTALL)
        else:
            dsl_pattern_simple = r'(## DSL[^\n]*\n"""\n)(.*?)(\n"""\n)'
            dsl_match_simple = re.search(dsl_pattern_simple, specification, re.DOTALL)
            if dsl_match_simple:
                header = dsl_match_simple.group(1)
                footer = dsl_match_simple.group(3)
                cfg_section = header + cfg + footer
                specification = re.sub(dsl_pattern_simple, cfg_section, specification, flags=re.DOTALL)
    
    # Create shared vLLM instance (used by both FunSearch and explicit feedback)
    # This ensures we only create ONE instance instead of separate ones for each stage
    # If creation fails, fail the stage to prevent OOM from multiple instances
    # Clean up GPU memory before creating new instance (in case previous stage left memory allocated)
    shared_vllm = None
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
    
    func_evolution_round = args.func_evolution_round if args.func_evolution_round is not None else 0
    
    print(f"\n{'='*80}")
    print(f"Implementing CFG for function: {args.function_name}")
    print(f"  DSL Round: {args.dsl_round}")
    print(f"  Function Evolution Round: {func_evolution_round}")
    print(f"{'='*80}")
    
    # Step 1: Run FunSearch
    print(f"\n[Step 1] Running FunSearch for {args.function_name}...")
    
    config = config_lib.Config(
        num_samplers=1,
        num_evaluators=2,
        samples_per_prompt=2,
        total_samples=args.total_samples,
        programs_database=config_lib.ProgramsDatabaseConfig(),
        grid_regeneration_attempts=args.grid_regeneration_attempts,
    )
    
    initial_funsearch_steps = results_tracker.interactions.get("funsearch", 0)
    
    try:
        funsearch = FunSearch(model_type=args.model_type, shared_vllm=shared_vllm)
        funsearch.results_tracker = results_tracker
        
        inputs = determine_inputs(args.function_name, description, cfg)
        
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
        
        # Save FunSearch status
        funsearch_status = {
            "stage": "funsearch",
            "function_name": args.function_name,
            "status": "completed",
            "dsl_round": args.dsl_round,
            "func_evolution_round": func_evolution_round,
            "env_steps": steps_taken
        }
        write_function_status(args.experiment_dir, args.dsl_round, "funsearch", args.function_name, funsearch_status)

        # Plot FunSearch reward vs interactions for this function
        try:
            log_file = find_funsearch_log_file(args.function_name, results_dir)
            if log_file:
                funsearch_plot_dir = os.path.join(
                    args.experiment_dir, "results_tracking", "funsearch"
                )
                plot_funsearch_reward_vs_interactions(
                    log_file=log_file,
                    output_dir=funsearch_plot_dir,
                    function_name=args.function_name,
                )
            else:
                print(f"   No FunSearch log found for plotting: {args.function_name}")
        except Exception as plot_error:
            print(f"   Failed to plot FunSearch metrics: {plot_error}")
        
    except Exception as e:
        error_msg = str(e)
        print(f"[{args.function_name}]  FunSearch failed: {error_msg}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        
        # Save failure status
        funsearch_status = {
            "stage": "funsearch",
            "function_name": args.function_name,
            "status": "failed",
            "error": error_msg,
            "dsl_round": args.dsl_round,
            "func_evolution_round": func_evolution_round
        }
        write_function_status(args.experiment_dir, args.dsl_round, "funsearch", args.function_name, funsearch_status)
        
        return 1
    
    # Step 2: Run Explicit Feedback
    print(f"\n[Step 2] Running explicit feedback generation for {args.function_name}...")
    
    # Read initial function code from FunSearch result (this will be our fallback)
    funsearch_result_code = None
    try:
        with open(func_file, 'r', encoding='utf-8') as f:
            funsearch_result_code = f.read()
    except Exception as e:
        print(f"   Warning: Could not read FunSearch result file: {e}", file=sys.stderr)
    
    try:
        current_func_code = funsearch_result_code  # Start with FunSearch result
        
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
                    results_tracker=results_tracker,
                    dsl_round=args.dsl_round, func_evolution_round=args.func_evolution_round
                )
                
                if iter_func:
                    final_func = iter_func
                    current_func_code = iter_func  # Update for next iteration
                else:
                    # Explicit feedback didn't return improved function, keep current
                    final_func = current_func_code
            finally:
                # Clean up temporary file immediately
                try:
                    os.remove(tmp_file_path)
                except OSError:
                    pass
        
        # If no explicit feedback iterations ran or all failed, use FunSearch result
        if final_func is None and current_func_code:
            final_func = current_func_code
        
        # Final fallback: if everything failed, use FunSearch result
        if final_func is None and funsearch_result_code:
            final_func = funsearch_result_code
            print("   Using FunSearch result as final function (explicit feedback failed)")
        
        if final_func:
            # Save final function
            final_functions_dir = os.path.join(args.experiment_dir, "final_functions")
            os.makedirs(final_functions_dir, exist_ok=True)
            
            safe_name = sanitize_function_name(args.function_name)
            if args.dsl_round is not None:
                if args.func_evolution_round is not None:
                    func_file_path = os.path.join(final_functions_dir, f"{safe_name}_dsl{args.dsl_round}_func{args.func_evolution_round}.py")
                else:
                    func_file_path = os.path.join(final_functions_dir, f"{safe_name}_dsl{args.dsl_round}_func0.py")
            else:
                func_file_path = os.path.join(final_functions_dir, f"{safe_name}.py")
            
            with open(func_file_path, 'w', encoding='utf-8') as f:
                f.write(final_func)
            print(f"  Saved {args.function_name} to {os.path.basename(func_file_path)}")
            
            # Clean up only intermediate iteration files (not the final versioned explicit feedback files)
            # The final versioned files (eval_{name}_dsl{num}_func{num}.py and feedback_{name}_dsl{num}_func{num}.json)
            # should be preserved as they are the actual explicit feedback outputs
            patterns = [
                # Only clean up intermediate iteration files
                os.path.join(explicit_feedback_dir, f"{safe_name}_dsl{args.dsl_round}_iter_*.py"),
                os.path.join(explicit_feedback_dir, f"{safe_name}_iter_*.py"),
                # Clean up old unversioned files (for backward compatibility) - but NOT versioned ones
                # Only remove if they don't have versioning in the name
            ]
            for pattern in patterns:
                for path in glob.glob(pattern):
                    try:
                        # Double-check: don't delete versioned files (they contain _dsl{num}_func{num})
                        basename = os.path.basename(path)
                        # Skip if it's a versioned file (contains _dsl followed by digits and _func)
                        if re.search(r'_dsl\d+_func\d+', basename):
                            continue
                        os.remove(path)
                    except OSError:
                        pass
            
            # Save explicit feedback status
            explicit_fb_status = {
                "stage": "explicit_feedback",
                "function_name": args.function_name,
                "status": "completed",
                "dsl_round": args.dsl_round,
                "func_evolution_round": func_evolution_round
            }
            write_function_status(args.experiment_dir, args.dsl_round, "explicit_feedback", args.function_name, explicit_fb_status)
            
            print(f"[{args.function_name}]  Completed explicit feedback ({args.num_iterations} iterations)")
            
            # Plot explicit feedback reward vs interactions and baseline combined plot
            try:
                safe_name = sanitize_function_name(args.function_name)
                if args.dsl_round is not None:
                    if args.func_evolution_round is not None:
                        feedback_filename = f"feedback_{safe_name}_dsl{args.dsl_round}_func{args.func_evolution_round}.json"
                    else:
                        feedback_filename = f"feedback_{safe_name}_dsl{args.dsl_round}_func0.json"
                else:
                    feedback_filename = f"feedback_{safe_name}.json"
                feedback_file = os.path.join(explicit_feedback_dir, feedback_filename)
                
                explicit_plot_dir = os.path.join(
                    args.experiment_dir, "results_tracking", "explicit_feedback", dsl_folder
                )
                if os.path.exists(feedback_file):
                    plot_explicit_feedback_reward_vs_interactions(
                        feedback_file=feedback_file,
                        output_dir=explicit_plot_dir,
                        function_name=args.function_name,
                    )
                else:
                    print(f"   No explicit feedback file found for plotting: {feedback_file}")
                
                # Combined baseline plot (FunSearch + Explicit Feedback)
                log_file = find_funsearch_log_file(args.function_name, results_dir)
                if log_file and os.path.exists(feedback_file):
                    baseline_plot_dir = os.path.join(
                        args.experiment_dir, "results_tracking", "baseline", dsl_folder
                    )
                    plot_baseline_reward_vs_interactions(
                        funsearch_log_file=log_file,
                        explicit_feedback_file=feedback_file,
                        output_dir=baseline_plot_dir,
                        function_name=args.function_name,
                    )
            except Exception as plot_error:
                print(f"   Failed to plot explicit feedback/baseline metrics: {plot_error}")
            
            # Decrement counter (both FunSearch and Explicit Feedback are part of implement_cfg)
            print("\n[Chaining] Decrementing function implementation counter...")
            implementation_remaining = decrement_function_implementation(args.experiment_dir)
            print(f"  Function implementations remaining: {implementation_remaining}")
            
            # Clean up vLLM instance to free GPU memory
            # If we had a shared instance, clean it up. If not, explicit feedback may have created its own.
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
            
            return 0
        else:
            print(f"[{args.function_name}]  No final function extracted")
            return 1
            
    except Exception as e:
        error_msg = str(e)
        print(f"[{args.function_name}]  Explicit feedback failed: {error_msg}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        
        # Extract best function from FunSearch log file as fallback
        print("   Extracting best function from FunSearch log as fallback...")
        best_func_from_log = None
        
        try:
            from src.pipeline.cfg_to_funsearch_pipeline import find_funsearch_log_file
            from src.pipeline.explicit_feedback_generation import parse_log_file
            import re
            
            # Find the log file
            log_file = find_funsearch_log_file(args.function_name, results_dir)
            if log_file:
                # Parse top function from log (highest score)
                funcs = parse_log_file(log_file, k=1)
                if funcs:
                    best_log_score, best_log_func = funcs[0]
                    
                    # Extract function signature from func_signatures or construct it
                    safe_name = sanitize_function_name(args.function_name)
                    func_signature = func_signatures.get(args.function_name, "")
                    if not func_signature:
                        # Construct signature from function name
                        func_signature = f"def {safe_name}(env, turndir)" if "TURN" in args.function_name else f"def {safe_name}(env)"
                    
                    # Extract function name from signature
                    func_name_match = re.search(r'def\s+(\w+)', func_signature)
                    target_func_name = func_name_match.group(1) if func_name_match else safe_name
                    
                    # The best_log_func from parse_log_file should be just the function body
                    # But it might contain wrapper functions (solve, evaluate, etc.)
                    # Extract only the target function definition
                    func_body = best_log_func.strip()
                    
                    # Check if the function body contains the target function
                    # Look for the function definition with the target name
                    target_func_pattern = rf'def\s+{re.escape(target_func_name)}\s*\([^)]*\):'
                    if re.search(target_func_pattern, func_body):
                        # Extract just the target function (stop at next def or @ decorator)
                        func_match = re.search(
                            rf'(def\s+{re.escape(target_func_name)}\s*\([^)]*\):.*?)(?=\n\ndef\s|\n@|\Z)',
                            func_body, re.DOTALL
                        )
                        if func_match:
                            func_body = func_match.group(1).strip()
                    else:
                        # Function body doesn't contain the target function name
                        # It might be just the function body without signature, or it's a different function
                        # Use the signature and append the body
                        sig_clean = func_signature.strip()
                        if not sig_clean.endswith(':'):
                            sig_clean += ':'
                        # Check if func_body already starts with 'def'
                        if not func_body.startswith('def'):
                            func_body = f"{sig_clean}\n  {func_body}"
                        else:
                            # It already has a def, use as is
                            pass
                    
                    # Extract imports from FunSearch result or specification
                    imports = []
                    if funsearch_result_code:
                        # Extract import statements from FunSearch result
                        import_pattern = r'^(import\s+\S+|from\s+\S+\s+import\s+.*?)$'
                        seen_imports = set()
                        for line in funsearch_result_code.split('\n'):
                            line_stripped = line.strip()
                            if re.match(import_pattern, line_stripped) and line_stripped not in seen_imports:
                                imports.append(line_stripped)
                                seen_imports.add(line_stripped)
                    
                    # Combine imports and function
                    if imports:
                        best_func_from_log = '\n'.join(imports) + '\n\n' + func_body
                    else:
                        best_func_from_log = func_body
                    
                    print(f"   Extracted best function from log (score: {best_log_score:.4f})")
                else:
                    print("   No functions found in log file")
            else:
                print(f"   Could not find log file for {args.function_name}")
        except Exception as extract_error:
            print(f"   Failed to extract function from log: {extract_error}", file=sys.stderr)
            import traceback
            traceback.print_exc()
        
        # Save the best function from log, or fall back to FunSearch result file if log extraction failed
        # Note: log file (.log) contains clean function bodies, result file (.py) has wrapper code
        final_func_to_save = best_func_from_log if best_func_from_log else funsearch_result_code
        
        if final_func_to_save:
            print("  Saving best function as final function (explicit feedback failed)...")
            final_functions_dir = os.path.join(args.experiment_dir, "final_functions")
            os.makedirs(final_functions_dir, exist_ok=True)
            
            safe_name = sanitize_function_name(args.function_name)
            if args.dsl_round is not None:
                if args.func_evolution_round is not None:
                    func_file_path = os.path.join(final_functions_dir, f"{safe_name}_dsl{args.dsl_round}_func{args.func_evolution_round}.py")
                else:
                    func_file_path = os.path.join(final_functions_dir, f"{safe_name}_dsl{args.dsl_round}_func0.py")
            else:
                func_file_path = os.path.join(final_functions_dir, f"{safe_name}.py")
            
            try:
                with open(func_file_path, 'w', encoding='utf-8') as f:
                    f.write(final_func_to_save)
                source = "FunSearch log" if best_func_from_log else "FunSearch result"
                print(f"  Saved {args.function_name} to {os.path.basename(func_file_path)} ({source})")
            except Exception as save_error:
                print(f"   Failed to save final function: {save_error}", file=sys.stderr)
        
        # Save failure status
        explicit_fb_status = {
            "stage": "explicit_feedback",
            "function_name": args.function_name,
            "status": "failed",
            "error": error_msg,
            "dsl_round": args.dsl_round,
            "func_evolution_round": func_evolution_round
        }
        write_function_status(args.experiment_dir, args.dsl_round, "explicit_feedback", args.function_name, explicit_fb_status)
        
        # Still decrement counters
        implementation_remaining = decrement_function_implementation(args.experiment_dir)
        
        # Clean up vLLM instance to free GPU memory
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
            except Exception as e:
                print(f"   Warning: Error during cleanup: {e}")
        
        return 1


if __name__ == "__main__":
    sys.exit(main())

