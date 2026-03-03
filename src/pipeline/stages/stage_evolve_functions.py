#!/usr/bin/env python3
"""
Stage 6: Evolve Functions
This stage evolves functions with failing tasks.
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

# Import vLLM for shared instance
try:
    from vllm import LLM as vLLM
except ImportError:
    vLLM = None


def main():
    parser = argparse.ArgumentParser(description="Stage 6: Evolve Functions")
    parser.add_argument('--experiment_dir', type=str, required=True, help='Experiment directory')
    parser.add_argument('--spec_file', type=str, required=True, help='Path to specification file')
    parser.add_argument('--failing_tasks', type=str, nargs='*', default=[], help='List of failing tasks (optional, kept for API compat)')
    parser.add_argument('--model_type', type=str, default='huggingface', choices=['huggingface', 'ollama', 'gemini'])
    parser.add_argument('--dsl_round', type=int, default=0, help='DSL evolution round number')
    parser.add_argument('--func_evolution_round', type=int, default=0, help='Function evolution round number')
    
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
    if args.model_type == "huggingface" and vLLM is not None:
        try:
            print("\n[Setup] Initializing shared vLLM instance...")
            shared_vllm = vLLM(model="/scratch/avani/gpt", tensor_parallel_size=4)
            print(" Shared vLLM instance created")
        except Exception as e:
            print(f" Warning: Could not create shared vLLM instance: {e}")
            shared_vllm = None
    
    # Evolve functions
    print(f"\n[Step 6] Evolving functions with {len(args.failing_tasks)} failing tasks...")
    evolved = evolve_functions_with_failing_tasks(
        experiment_dir=args.experiment_dir,
        failing_tasks=args.failing_tasks,
        terminals=terminals,
        specification=specification,
        spec_file=args.spec_file,
        dsl_round=args.dsl_round,
        func_evolution_round=args.func_evolution_round,
        cfg=cfg,
        max_evolutions=1,
        shared_vllm=shared_vllm
    )
    
    # Save stage completion marker
    stage_status = {
        "stage": "evolve_functions",
        "status": "completed" if evolved else "failed",
        "dsl_round": args.dsl_round,
        "func_evolution_round": args.func_evolution_round,
        "failing_tasks": args.failing_tasks,
        "evolved": evolved
    }
    status_file = os.path.join(args.experiment_dir, "stage_evolve_functions_status.json")
    with open(status_file, 'w') as f:
        json.dump(stage_status, f, indent=2)
    
    if evolved:
        print(f"\n Function evolution completed")
        return 0
    else:
        print(f"\n Function evolution failed or produced no results")
        return 1


if __name__ == "__main__":
    sys.exit(main())


