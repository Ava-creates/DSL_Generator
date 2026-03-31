#!/usr/bin/env python3
"""
Stage 2: File Generation
This stage generates function-specific prompts and func_init files.
"""

import os
import sys
import json
import argparse
import re

# Add project root to path
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _project_root)

from src.pipeline.cfg_to_funsearch_pipeline import (
    generate_function_prompt, generate_func_init, sanitize_function_name
)
from src.utils.pipeline_state import read_state, update_state
from src.utils.status_manager import write_status
from src.utils.config_loader import load_config

# Import vLLM for LLM-based evaluation generation
try:
    from vllm import LLM as vLLM
except ImportError:
    vLLM = None


def main():
    parser = argparse.ArgumentParser(description="Stage 2: File Generation")
    parser.add_argument('--experiment_dir', type=str, required=True, help='Experiment directory')
    parser.add_argument('--spec_file', type=str, required=True, help='Path to specification file')
    parser.add_argument('--dsl_round', type=int, default=0, help='DSL evolution round number')
    parser.add_argument('--func_evolution_round', type=int, default=None, help='Function evolution round number')
    parser.add_argument('--grid_prompt', type=str, default='prompt_specifications/grid_prompt.txt', help='Path to grid generation prompt template')
    parser.add_argument('--require_test_type', type=lambda x: x.lower() != 'false', default=True, help='Whether to require test_type in grid specs (set false for old prompts; defaults test_type to positive)')
    parser.add_argument('--skip_positive_grids', type=lambda x: x.lower() == 'true', default=False, help='When true, discard LLM-generated positive grids (only save negative/edge); useful for supplementing existing positives with new negative cases')
    
    args = parser.parse_args()
    config = load_config()
    positive_grids = int(config.get("positive_girds", config.get("positive_grids", 10)))
    negative_grids = int(config.get("negative_grids", 4))
    edge_grids = int(config.get("edge_grids", 1))
    
    # Read state file to get current DSL and function evolution rounds
    # This ensures consistency after DSL evolution
    state = read_state(args.experiment_dir)
    state_dsl_round = state.get("dsl_round", 0)
    state_func_round = state.get("func_evolution_round", 0)
    
    # Use dsl_round from state file if not provided or if it doesn't match
    # This is critical after DSL evolution - the state file has the correct round
    if args.dsl_round != state_dsl_round:
        print("[File Generation]  Warning: dsl_round mismatch!")
        print(f"  Command line: {args.dsl_round}, State file: {state_dsl_round}")
        print(f"  Using state file value: {state_dsl_round}")
        args.dsl_round = state_dsl_round
    
    # If func_evolution_round was not provided or doesn't match state, use state value
    # This ensures consistency after DSL evolution (which resets func_evolution_round to 0)
    if args.func_evolution_round is None:
        args.func_evolution_round = state_func_round
        print(f"[File Generation] Using func_evolution_round={args.func_evolution_round} from state file")
    elif args.func_evolution_round != state_func_round:
        print("[File Generation]  Warning: func_evolution_round mismatch!")
        print(f"  Command line: {args.func_evolution_round}, State file: {state_func_round}")
        print(f"  Using state file value: {state_func_round}")
        args.func_evolution_round = state_func_round
    
    print(f"[File Generation] Using dsl_round={args.dsl_round}, func_evolution_round={args.func_evolution_round}")

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
    
    # Re-extract terminals from CFG for use during file generation
    # This ensures we use complete terminals even if the file has incomplete ones
    # But we don't modify the file - it stays as-is
    from src.pipeline.integrated_pipeline import ensure_terminals_match_cfg
    terminals = ensure_terminals_match_cfg(cfg, terminals, shared_vllm=shared_vllm)
    
    if not terminals:
        print(" Invalid CFG data: no terminals found after extraction", file=sys.stderr)
        return 1
    
    # After DSL evolution, func_evolution_round should be 0
    # If it's not, this might indicate a state inconsistency
    if args.dsl_round > 0 and args.func_evolution_round != 0:
        # Check if this is the first file generation for this DSL round
        # by checking if func0 files already exist
        func0_exists = False
        for func_name in terminals.keys():
            safe_name = sanitize_function_name(func_name)
            func0_file = os.path.join(args.experiment_dir, "function_specific_prompts", 
                                     f"{safe_name}_dsl{args.dsl_round}_func0.txt")
            if os.path.exists(func0_file):
                func0_exists = True
                break
        
        if not func0_exists:
            print(f"[File Generation]  No func0 files found for DSL round {args.dsl_round}")
            print("  This should be the initial file generation after DSL evolution.")
            print("  Forcing func_evolution_round=0 to create func0 files")
            args.func_evolution_round = 0
    
    # Load specification
    if not os.path.exists(args.spec_file):
        print(f" Specification file not found: {args.spec_file}", file=sys.stderr)
        return 1
    
    with open(args.spec_file, 'r', encoding='utf-8') as f:
        specification = f.read()
    
    # Replace DSL section in specification with current CFG
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
    
    # Ensure directories exist
    os.makedirs(os.path.join(args.experiment_dir, "function_specific_prompts"), exist_ok=True)
    os.makedirs(os.path.join(args.experiment_dir, "functions_generated"), exist_ok=True)
    
    # Generate function-specific prompts
    print("\n[Step 2] Generating function-specific prompts...")
    print(f"  Generating function-specific prompts for all {len(terminals)} functions")
    
    func_files = {}
    func_signatures = {}
    
    for idx, (func_name, description) in enumerate(terminals.items(), 1):
        print(f"\n  [{idx}/{len(terminals)}] Processing {func_name}...")
        func_file, func_signature = generate_function_prompt(
            func_name, description, cfg, specification,
            experiment_dir=args.experiment_dir,
            dsl_round=args.dsl_round,
            func_evolution_round=args.func_evolution_round,
            shared_vllm=shared_vllm,
            grid_prompt_path=args.grid_prompt,
            require_test_type=args.require_test_type,
            skip_positive_grids=args.skip_positive_grids,
            positive_grids=positive_grids,
            negative_grids=negative_grids,
            edge_grids=edge_grids,
        )
        func_files[func_name] = func_file
        func_signatures[func_name] = func_signature
    
    # Generate func_init files
    print("\n[Step 3] Generating func_init files...")
    func_init_files = {}
    for func_name, description in terminals.items():
        func_init_file = generate_func_init(
            func_name, description, cfg,
            experiment_dir=args.experiment_dir,
            dsl_round=args.dsl_round,
            func_evolution_round=args.func_evolution_round
        )
        func_init_files[func_name] = func_init_file
    
    # Save stage completion marker with DSL versioning
    stage_status = {
        "stage": "file_generation",
        "status": "completed",
        "dsl_round": args.dsl_round,
        "func_evolution_round": args.func_evolution_round,
        "func_files": func_files,
        "func_init_files": func_init_files,
        "func_signatures": func_signatures
    }
    # Write to versioned location: status/file_generation/dsl{N}/status.json
    write_status(
        args.experiment_dir,
        args.dsl_round,
        "file_generation",
        stage_status
    )
    
    print(f"\n Generated files for {len(terminals)} functions")
    
    # Read state file to get terminal function count
    state = read_state(args.experiment_dir)
    num_terminals = len(terminals)
    
    # Update state with current terminal count and reset counters
    update_state(
        args.experiment_dir,
        function_implementation_total=num_terminals,
        function_implementation_remaining=num_terminals,
        # implement_cfg_submitted removed - use status files as source of truth
        # test_tasks runs in single job, no counter needed
    )
    
    print(f"\n[Chaining] State file shows {num_terminals} terminal functions to process")
    
    # Chaining will be handled by the SLURM script after this Python script completes
    print(f"\n[Chaining] State file updated with {num_terminals} terminal functions.")
    print("  SLURM script will handle chaining to FunSearch jobs.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


