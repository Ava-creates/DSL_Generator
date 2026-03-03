#!/usr/bin/env python3
"""
Stage: Implement CFG (FunSearch + Explicit Feedback Package)
This stage runs FunSearch and Explicit Feedback together as a package for the initial phase.
"""

import os
import sys
import json
import argparse

# Add project root to path
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _project_root)

from src.pipeline.cfg_to_funsearch_pipeline import implement_cfg
from src.utils.results_tracker import ResultsTracker
from src.utils.pipeline_state import read_state, update_state

# Import vLLM for shared instance
try:
    from vllm import LLM as vLLM
except ImportError:
    vLLM = None


def main():
    parser = argparse.ArgumentParser(description="Stage: Implement CFG (FunSearch + Explicit Feedback Package)")
    parser.add_argument('--experiment_dir', type=str, required=True, help='Experiment directory')
    parser.add_argument('--spec_file', type=str, required=True, help='Path to specification file')
    parser.add_argument('--model_type', type=str, default="huggingface", help='Model type')
    parser.add_argument('--dsl_round', type=int, default=0, help='DSL evolution round number')
    parser.add_argument('--func_evolution_round', type=int, default=None, help='Function evolution round number')
    
    args = parser.parse_args()
    
    # Load CFG
    cfg_path = os.path.join(args.experiment_dir, "cfg", "cfg_output.json")
    if not os.path.exists(cfg_path):
        print(f" CFG file not found: {cfg_path}", file=sys.stderr)
        return 1
    
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg_data = json.load(f)
    cfg = cfg_data.get("cfg", "")
    terminals = cfg_data.get("terminals", {})
    example = cfg_data.get("example", None)
    
    if not cfg or not terminals:
        print(" Invalid CFG data", file=sys.stderr)
        return 1
    
    # Load specification
    if not os.path.exists(args.spec_file):
        print(f" Specification file not found: {args.spec_file}", file=sys.stderr)
        return 1
    
    with open(args.spec_file, 'r', encoding='utf-8') as f:
        specification = f.read()
    
    # Create shared vLLM instance
    shared_vllm = None
    if vLLM is not None and args.model_type == "huggingface":
        try:
            print("\n[Setup] Initializing shared vLLM instance...")
            shared_vllm = vLLM(model="/scratch/avani/gpt", tensor_parallel_size=4)
            print(" Shared vLLM instance created")
        except Exception as e:
            print(f" Warning: Could not create shared vLLM instance: {e}")
            shared_vllm = None
    
    # Create ResultsTracker
    results_tracker = ResultsTracker(args.experiment_dir)
    
    # Run implement_cfg (FunSearch + Explicit Feedback package)
    print(f"\n[Implement CFG] Running FunSearch and Explicit Feedback package for {len(terminals)} functions...")
    print(f"  DSL Round: {args.dsl_round}")
    if args.func_evolution_round is not None:
        print(f"  Function Evolution Round: {args.func_evolution_round}")
    else:
        print(f"  Function Evolution Round: 0 (initial)")
    
    try:
        success, final_functions = implement_cfg(
            cfg=cfg,
            terminals=terminals,
            example=example,
            spec_file=args.spec_file,
            experiment_dir=args.experiment_dir,
            model_type=args.model_type,
            shared_vllm=shared_vllm,
            results_tracker=results_tracker,
            dsl_round=args.dsl_round,
            func_evolution_round=args.func_evolution_round if args.func_evolution_round is not None else 0
        )
        
        if not success:
            print(" Failed to implement CFG", file=sys.stderr)
            return 1
        
        print(f"\n Successfully implemented CFG for {len(final_functions)} functions")
        
        # Mark implement_cfg as complete in state (includes both FunSearch and Explicit Feedback)
        # Status files are the source of truth, but we update counter for consistency
        update_state(
            args.experiment_dir,
            function_implementation_remaining=0
        )
        
        # Create explicit feedback status files for each function (required by chaining logic)
        explicit_fb_status_dir = os.path.join(args.experiment_dir, "status", "explicit_feedback")
        os.makedirs(explicit_fb_status_dir, exist_ok=True)
        
        func_evolution_round = args.func_evolution_round if args.func_evolution_round is not None else 0
        
        for func_name in final_functions.keys():
            # Create status file in grouped location
            status_file = os.path.join(explicit_fb_status_dir, f"{func_name}.json")
            legacy_status_file = os.path.join(args.experiment_dir, f"stage_explicit_feedback_{func_name}_status.json")
            
            status = {
                "stage": "explicit_feedback",
                "function_name": func_name,
                "status": "completed",
                "dsl_round": args.dsl_round,
                "func_evolution_round": func_evolution_round
            }
            
            # Write to both locations for compatibility
            with open(status_file, 'w') as f:
                json.dump(status, f, indent=2)
            with open(legacy_status_file, 'w') as f:
                json.dump(status, f, indent=2)
        
        print(f" Created explicit feedback status files for {len(final_functions)} functions")
        print(" Updated pipeline state: FunSearch and Explicit Feedback marked as complete")
        
        return 0
        
    except Exception as e:
        print(f" Error implementing CFG: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

