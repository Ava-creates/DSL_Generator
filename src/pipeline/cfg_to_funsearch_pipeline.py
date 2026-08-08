#!/usr/bin/env python3
"""
Pipeline script that:
1. Generates CFG from domain description
2. Creates function-specific prompts for each terminal function
3. Creates func_init files with stub implementations
4. Runs funsearch for each terminal function
"""

import os
import sys
import re
import json
import argparse
from datetime import datetime
from typing import Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import glob
from funsearch.implementation.funsearch import FunSearch
from funsearch.implementation import config as config_lib
from src.utils.file_utils import version_file
from src.utils.results_tracker import ResultsTracker, plot_funsearch_reward_vs_interactions
from src.utils.config_loader import funsearch_grid_regen_kwargs_from_config, load_config
from src.pipeline.grid_generation import ensure_function_grid_spec
from src.pipeline.domain_templates import (
    craft_solve_template_basic,
    craft_solve_template_task_env_basic,
    craft_evaluate_template,
    craft_env_setup,
    craft_baseline_evaluate_template,
)

# Import vLLM for shared instance
try:
    from vllm import LLM as vLLM
except ImportError:
    vLLM = None


def parse_function_name_and_args(func_name: str) -> tuple[str, list[str]]:
    """Parse function name that may include arguments in parentheses.
    
    Examples:
        "MOVE()" -> ("MOVE", [])
        "TURN(DIR)" -> ("TURN", ["DIR"])
        "USE(TOOL, OBSTACLE)" -> ("USE", ["TOOL", "OBSTACLE"])
        "MOVE" -> ("MOVE", [])
    
    Returns:
        Tuple of (base_function_name, list_of_argument_names)
    """
    # Check if function name has parentheses with arguments
    match = re.match(r'^(\w+)\(([^)]*)\)$', func_name)
    if match:
        base_name = match.group(1)
        args_str = match.group(2).strip()
        if args_str:
            # Split by comma and clean up
            args = [arg.strip() for arg in args_str.split(',') if arg.strip()]
        else:
            args = []
        return base_name, args
    
    # No parentheses, just the function name
    return func_name, []


def sanitize_function_name(func_name: str) -> str:
    """Convert terminal function name to valid Python identifier."""
    # First parse to get base name
    base_name, _ = parse_function_name_and_args(func_name)
    # Convert to lowercase and replace non-alphanumeric with underscore
    func_name = base_name.lower()
    func_name = re.sub(r'\W|^(?=\d)', '_', func_name)
    return func_name


def _extract_function_body(func_code: str, func_name: str) -> str:
    """Extract the body of a function from Python source."""
    pattern = rf'^def\s+{re.escape(func_name)}\s*\([^)]*\)\s*:'
    lines = func_code.splitlines()
    start_idx = None
    for idx, line in enumerate(lines):
        if re.match(pattern, line.strip()):
            start_idx = idx + 1
            break
    if start_idx is None:
        return ""
    body_lines = lines[start_idx:]
    # Drop leading blank lines
    while body_lines and not body_lines[0].strip():
        body_lines = body_lines[1:]
    return "\n".join(body_lines).rstrip()


def _versioned_name(safe_name: str, dsl_round: Optional[int], func_evolution_round: Optional[int] = None) -> str:
    """Return the versioned base filename prefix, e.g. 'move_dsl0' or 'move'."""
    del func_evolution_round  # function evolution versioning removed; kept for call-site compat
    if dsl_round is not None:
        return f"{safe_name}_dsl{dsl_round}"
    return safe_name


def _load_seed_body(
    experiment_dir: Optional[str],
    safe_name: str,
    dsl_round: Optional[int],
    func_evolution_round: Optional[int] = None,
) -> str:
    """Load previous DSL round function body to seed evolve prompt."""
    del func_evolution_round
    if not experiment_dir:
        return ""
    prev_path = None
    if dsl_round is not None and dsl_round > 0:
        prev_path = os.path.join(experiment_dir, "final_functions", f"{_versioned_name(safe_name, dsl_round - 1)}.py")
    elif dsl_round is None:
        prev_path = os.path.join(experiment_dir, "final_functions", f"{safe_name}.py")
    if prev_path and os.path.exists(prev_path):
        try:
            with open(prev_path, "r", encoding="utf-8") as f:
                code = f.read()
            return _extract_function_body(code, safe_name)
        except Exception:
            return ""
    return ""


def extract_raw_cfg_arg_symbols(func_name: str, cfg: str) -> list[str]:
    """Extract raw CFG argument symbols in order (e.g. ['INTEGER'] or ['ITEM', 'OBSTACLE'])."""
    base_name, args_from_name = parse_function_name_and_args(func_name)

    if args_from_name:
        return [a.strip().upper() for a in args_from_name if a.strip()]

    if not cfg:
        print(f"No CFG found for {func_name}")
        sys.exit(1)

    func_name_for_search = base_name

    pattern_with_lpar = rf"{re.escape(func_name_for_search)}\s+LPAR\s+(.*?)\s+RPAR"
    match = re.search(pattern_with_lpar, cfg, re.IGNORECASE | re.MULTILINE | re.DOTALL)
    if match:
        args_content = match.group(1).strip()
        args_list: list[str] = []
        parts = re.split(r'\s+COMMA\s+', args_content, flags=re.IGNORECASE)
        for part in parts:
            part = part.strip()
            if part and part.upper() not in ['LPAR', 'RPAR', 'COMMA', 'SEMI', 'SEMICOLON']:
                arg_match = re.match(r'^(\w+)', part)
                if arg_match:
                    arg = arg_match.group(1).strip().upper()
                    if arg and arg not in ['LPAR', 'RPAR', 'COMMA', 'SEMI', 'LPAREN', 'RPAREN']:
                        args_list.append(arg)
        if args_list:
            return args_list

    single_arg_pattern = rf"{re.escape(func_name_for_search)}\s+LPAR\s+(\w+)\s+RPAR"
    match = re.search(single_arg_pattern, cfg, re.IGNORECASE | re.MULTILINE)
    if match:
        arg = match.group(1).strip().upper()
        if arg and arg not in ['LPAR', 'RPAR', 'COMMA', 'SEMI', 'LPAREN', 'RPAREN', '(', ')']:
            return [arg]

    pattern_literal = rf"{re.escape(func_name_for_search)}\s*\(\s*([^)]+)\s*\)"
    match = re.search(pattern_literal, cfg, re.IGNORECASE | re.MULTILINE)
    if match:
        args_content = match.group(1).strip()
        args_list = []
        for arg in args_content.split(','):
            arg = arg.strip().upper()
            if arg and arg not in ['LPAR', 'RPAR', 'COMMA', 'SEMI']:
                args_list.append(arg)
        if args_list:
            return args_list

    single_arg_pattern2 = rf"{re.escape(func_name_for_search)}\s*\(\s*(\w+)\s*\)"
    match = re.search(single_arg_pattern2, cfg, re.IGNORECASE | re.MULTILINE)
    if match:
        arg = match.group(1).strip().upper()
        if arg and arg not in ['LPAR', 'RPAR', 'COMMA', 'SEMI']:
            return [arg]

    from src.pipeline.cfg_parser import CFGParser
    parser = CFGParser(cfg)
    for fname, fargs in parser.get_terminal_functions():
        if fname.strip().upper() == func_name_for_search.strip().upper():
            if fargs:
                return [a.strip().upper() for a in fargs if a.strip()]
            break

    return []


def extract_function_args(func_name: str, cfg: str) -> str:
    """Python parameter names derived from CFG argument symbols (lowercased as-is)."""
    symbols = extract_raw_cfg_arg_symbols(func_name, cfg)
    if not symbols:
        return "arg"
    return ", ".join(s.lower() for s in symbols)


def infer_return_type(description: str) -> tuple[str, str]:
    """Infer return type and default return value from function description."""

    # Default to list of actions
    return "list[int]", "[]"


def infer_argument_type(arg_name: str, cfg: str, description: str = "") -> str:
    """Infer argument type from the CFG rule for this symbol (domain-agnostic)."""
    for lookup_name in (arg_name, arg_name.upper()):
        inferred = _infer_argument_type_from_cfg_rule(lookup_name, cfg)
        if inferred is not None:
            return inferred
    return "str"


def _infer_argument_type_from_cfg_rule(arg_name: str, cfg: str) -> str | None:
    """Return int/float/str from a CFG ::= rule, or None if rule not found."""
    if cfg:
        rule_head_pattern = rf"^\s*{re.escape(arg_name)}\s*::=\s*(.*)$"
        head_match = re.search(rule_head_pattern, cfg, re.IGNORECASE | re.MULTILINE)
        if head_match:
            rhs_parts = [head_match.group(1).strip()]
            remaining_lines = cfg[head_match.end():].splitlines()
            for line in remaining_lines:
                if re.match(r"^\s*\|", line):
                    rhs_parts.append(re.sub(r"^\s*\|\s*", "", line))
                else:
                    break

            raw_values = []
            for rhs in rhs_parts:
                raw_values.extend(v.strip() for v in rhs.split("|") if v.strip())

            values = [v.strip().strip('"').strip("'") for v in raw_values if v.strip()]
            if values:
                value_types = set()
                for v in values:
                    if re.fullmatch(r"-?\d+", v):
                        value_types.add("int")
                    elif re.fullmatch(r"-?(?:\d+\.\d*|\d*\.\d+)", v):
                        value_types.add("float")
                    else:
                        # Grammar symbols/tokens/literals are treated as strings.
                        value_types.add("str")

                if value_types == {"int"}:
                    return "int"
                if value_types == {"float"}:
                    return "float"
                if value_types <= {"int", "float"}:
                    return "float"
                return "str"

    return None


def create_experiment_directory(base_name: str = "experiment") -> str:
    """Create a dated/numbered experiment directory.
    
    Args:
        base_name: Base name for the experiment directory
        
    Returns:
        Path to the created experiment directory
    """
    base_root = "experiments"
    os.makedirs(base_root, exist_ok=True)
    # Get current date and time
    current_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    # Try to find the next available number
    experiment_num = 1
    while True:
        experiment_dir = os.path.join(base_root, f"{base_name}_{current_date}_{experiment_num:03d}")
        if not os.path.exists(experiment_dir):
            break
        experiment_num += 1
    
    # Create the directory structure
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "function_specific_prompts"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "functions_generated"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "results", "funsearch"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "cfg"), exist_ok=True)
    
    return experiment_dir


def generate_function_prompt(func_name: str, description: str, cfg: str, 
                             specification: str = "",
                             func_dir: str = "function_specific_prompts",
                             experiment_dir: Optional[str] = None,
                             dsl_round: Optional[int] = None,
                             func_evolution_round: Optional[int] = None,
                             shared_vllm=None,
                             forced_task_name: Optional[str] = None,
                             use_task_env: bool = False,
                             grid_prompt_path: str = "prompt_specifications/grid_prompt.txt",
                             require_test_type: bool = True,
                             skip_positive_grids: bool = False,
                             positive_grids: int = 10,
                             negative_grids: int = 4,
                             edge_grids: int = 1) -> tuple[str, str]:
    """Generate a function-specific prompt file for funsearch.
    
    Args:
        func_name: Name of the function
        description: Description of the function
        cfg: CFG string
        specification: Specification text
        func_dir: Directory for function prompts
        experiment_dir: Experiment directory
        dsl_round: DSL evolution round number (0-indexed)
        func_evolution_round: Function evolution round number (0-indexed, None for initial)
    """
    # Use experiment directory if provided, otherwise use default
    if experiment_dir:
        func_dir = os.path.join(experiment_dir, "function_specific_prompts")
    
    os.makedirs(func_dir, exist_ok=True)
    
    # Parse function name to get base name and arguments
    base_name, args_list = parse_function_name_and_args(func_name)
    
    # Get sanitized name for file naming
    safe_name = sanitize_function_name(func_name)
    
    # Extract argument name(s) - prefer from function name, fallback to CFG when available.
    if args_list:
        args = ", ".join([a.strip().lower() for a in args_list if a.strip()])
    elif cfg and str(cfg).strip():
        args = extract_function_args(func_name, cfg)
    else:
        args = ""

    if (args == "arg" or not args) and cfg and str(cfg).strip():
            from src.pipeline.cfg_parser import CFGParser
            parser = CFGParser(cfg)
            for fname, fargs in parser.get_terminal_functions():
                if fname.strip().lower() == base_name.strip().lower():
                    if fargs:
                        args_list = [a.strip().lower() for a in fargs if a.strip()]
                        args = ", ".join(args_list)
                    break

    # Check if function has arguments (check args_list first, then args)
    # args is now a comma-separated string like "tool, item" or just "tool"
    has_args = bool(args and args != "arg" and args.strip())
    
    # For display purposes, use the base function name
    display_name = base_name
    
    func_file = os.path.join(func_dir, f"{_versioned_name(safe_name, dsl_round, func_evolution_round)}.txt")

    # Infer return type from description
    _, default_return = infer_return_type(description)
    
    # Infer argument types if function has arguments
    arg_type = "str"  # default
    typed_args = args if args else ""
    arg_list = []
    if has_args:
        arg_list = [a.strip() for a in args.split(',') if a.strip()]
        inferred_types = []
        for a in arg_list:
            inferred_types.append(infer_argument_type(a, cfg, description))
        if inferred_types:
            typed_args = ", ".join([f"{n}:{t}" for n, t in zip(arg_list, inferred_types)])
        # fallback single type for docstring
        arg_type = inferred_types[0] if inferred_types else arg_type
    
    # Build function signature parts
    if has_args:
        func_params = f"env, {args}"
        func_call_args = f"env, {args}" 
        args_docstring = f"      {args} ({arg_type}): Function-specific argument(s).\n  "
    else:
        func_params = "env"
        func_call_args = "env"
        args_docstring = ""

    default_task_name = forced_task_name
    recipes_path = "craft/resources/recipes.yaml"
    hints_path = "craft/resources/hints.yaml"
    # grid_prompt_path is passed as a parameter (default: prompt_specifications/grid_prompt.txt)
    # Load env description (absolute path to avoid cwd issues)
    env_description = ""
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    nld_path = os.path.join(_repo_root, "..", "prompt_specifications", "nld.txt")
    try:
        with open(nld_path, "r", encoding="utf-8") as f:
            env_description = f.read().strip()
    except Exception as e:
        print(f"  Warning: could not read env description ({nld_path}): {e}")
        env_description = ""
    recipes_text = ""
    try:
        with open(recipes_path, "r", encoding="utf-8") as f:
            recipes_text = f.read().strip()
    except Exception:
        recipes_text = ""
    codebase_text = ""
    codebase_path = os.path.join(_repo_root, "..", "prompt_specifications", "codebase.txt")
    try:
        with open(codebase_path, "r", encoding="utf-8") as f:
            codebase_text = f.read().strip()
    except Exception as e:
        print(f"  Warning: could not read codebase description ({codebase_path}): {e}")
        codebase_text = ""
    grid_spec_paths = []
    grid_spec = None
    if use_task_env:
        if not default_task_name:
            raise ValueError(f"Missing forced_task_name for task-env mode: {func_name}")
        task_name_for_env = default_task_name
        grid_spec_path = None
    else:
        grid_dir_override = os.environ.get("GRID_SPEC_DIR")
        grid_dir = grid_dir_override if grid_dir_override else (os.path.join(experiment_dir, "grids") if experiment_dir else "grids")
        os.makedirs(grid_dir, exist_ok=True)
        num_grid_tests = (negative_grids + edge_grids) if skip_positive_grids else (positive_grids + negative_grids + edge_grids)
        total_grid_generation_attempts = 200

        # Optional: reuse pre-generated grid specs via env vars.
        use_existing_grids = str(os.environ.get("USE_EXISTING_GRID_SPECS", "")).lower() in {"1", "true", "yes"}
        if use_existing_grids:
            # Only scan grid_dir; prefer exact function/dsl matches, then fall back to any JSONs.
            if os.path.isdir(grid_dir):
                def _match(fname: str) -> bool:
                    if not fname.lower().endswith(".json"):
                        return False
                    if dsl_round is not None:
                        return f"{safe_name}_dsl{dsl_round}_" in fname
                    return f"{safe_name}_" in fname

                for fname in sorted(os.listdir(grid_dir)):
                    if _match(fname):
                        grid_spec_paths.append(os.path.join(grid_dir, fname))
                if not grid_spec_paths and dsl_round is None:
                    for fname in sorted(os.listdir(grid_dir)):
                        if fname.lower().endswith(".json") and f"{safe_name}_" in fname:
                            grid_spec_paths.append(os.path.join(grid_dir, fname))
                if not grid_spec_paths and dsl_round is None:
                    for fname in sorted(os.listdir(grid_dir)):
                        if fname.lower().endswith(".json"):
                            grid_spec_paths.append(os.path.join(grid_dir, fname))
            if grid_spec_paths:
                try:
                    with open(grid_spec_paths[0], "r", encoding="utf-8") as f:
                        grid_spec = json.load(f)
                except Exception as e:
                    print(f"  Warning: could not load first grid spec {grid_spec_paths[0]}: {e}")

    if not grid_spec_paths and shared_vllm is not None:
        generated_cases = []  # all cases including positives, used for LLM context
        saved_count = 0  # only saved (non-positive when skip_positive_grids=True) cases
        attempts_for_case = max(1, total_grid_generation_attempts // max(1, num_grid_tests))
        max_total_iters = num_grid_tests * 8 if skip_positive_grids else num_grid_tests * 2
        total_iters = 0
        # Count existing files so new cases don't overwrite them
        if dsl_round is not None:
            _prefix = f"{safe_name}_dsl{dsl_round}_case"
        else:
            _prefix = f"{safe_name}_case"
        _existing_count = len([f for f in os.listdir(grid_dir)
                                if f.startswith(_prefix) and f.endswith(".json")]) if os.path.isdir(grid_dir) else 0
        # Preload existing files as LLM context
        def _case_num(fname):
            """Extract numeric case index for natural sort (case2 < case10)."""
            import re as _re
            m = _re.search(r'_case(\d+)\.json$', fname)
            return int(m.group(1)) if m else -1

        if _existing_count > 0:
            for _ef in sorted(os.listdir(grid_dir), key=_case_num):
                if _ef.startswith(_prefix) and _ef.endswith(".json"):
                    try:
                        with open(os.path.join(grid_dir, _ef), "r", encoding="utf-8") as _fh:
                            _loaded = json.load(_fh)
                            generated_cases.append(_loaded)
                        # Always add ALL existing files to grid_spec_paths so the evaluate
                        # loop runs on every case (positive + negative/edge).
                        grid_spec_paths.append(os.path.join(grid_dir, _ef))
                        # Only count saved (non-positive) cases toward the quota when
                        # skip_positive_grids is True so we know how many more to generate.
                        if skip_positive_grids and _loaded.get("test_type") != "positive":
                            saved_count += 1
                        elif not skip_positive_grids:
                            saved_count += 1
                    except Exception as _e:
                        print(f"  Warning: could not preload existing grid {_ef}: {_e}")
            # Use the first grid to set grid_spec (for task_name extraction below)
            if grid_spec_paths and grid_spec is None:
                try:
                    with open(grid_spec_paths[0], "r", encoding="utf-8") as _fh:
                        grid_spec = json.load(_fh)
                except Exception as _e:
                    print(f"  Warning: could not load first preloaded grid {grid_spec_paths[0]}: {_e}")
        if saved_count >= num_grid_tests:
            print(f"[grid_generation] Already have {saved_count} saved cases for {func_name}; skipping generation.")
        _new_saved_count = 0  # count of newly written cases in this run (offset for filenames)
        while saved_count < num_grid_tests and total_iters < max_total_iters:
            total_iters += 1
            generated_count = _existing_count + _new_saved_count  # always append after existing files
            if dsl_round is not None:
                grid_filename = f"{safe_name}_dsl{dsl_round}_case{generated_count}.json"
            else:
                grid_filename = f"{safe_name}_case{generated_count}.json"
            grid_spec_path = os.path.join(grid_dir, grid_filename)
            grid_spec = ensure_function_grid_spec(
                func_name=func_name,
                description=description,
                recipes_path=recipes_path,
                output_path=grid_spec_path,
                shared_vllm=shared_vllm,
                default_task_name=default_task_name,
                prompt_path=grid_prompt_path,
                func_args=typed_args if has_args else "None",
                env_description=env_description,
                recipes_text=recipes_text,
                attempts=attempts_for_case,
                existing_cases=generated_cases if generated_cases else None,
                cfg_text=cfg,
                codebase_text=codebase_text,
                require_test_type=require_test_type,
                skip_positive_grids=skip_positive_grids,
                positive_grids=positive_grids,
                negative_grids=negative_grids,
                edge_grids=edge_grids,
            )
            if isinstance(grid_spec, dict):
                generated_cases.append(grid_spec)
                if skip_positive_grids and grid_spec.get('test_type') == 'positive':
                    print(f"[grid_generation] Skipping positive case for {func_name} (skip_positive_grids=True); using as LLM context only.")
                else:
                    grid_spec_paths.append(grid_spec_path)
                    saved_count += 1
                    _new_saved_count += 1
            else:
                print(f"[grid_generation] No valid grid for {func_name} case {generated_count} (see grid_generation logs above); will retry.")
        if total_iters >= max_total_iters and saved_count < num_grid_tests:
            print(f"[grid_generation] Warning: hit max iterations ({max_total_iters}) for {func_name}; only {saved_count}/{num_grid_tests} cases saved.")

    if not grid_spec_paths:
        raise ValueError(
            f"No grid specs available for {func_name}; shared_vllm unavailable and no reusable specs found."
        )

    grid_spec_path = grid_spec_paths[0]
    task_name_for_env = None
    if isinstance(grid_spec, dict):
        task_name_for_env = grid_spec.get("task_name") or None
    if not task_name_for_env:
        raise ValueError(f"Missing task_name in grid spec for {func_name}; LLM must supply a valid task.")

    
   
    if use_task_env:
        solve_func = craft_solve_template_task_env_basic(
            func_name=func_name,
            func_params=func_params,
            func_call_args=func_call_args,
        )
    else:
        solve_func = craft_solve_template_basic(
            func_name=func_name,
            func_params=func_params,
            func_call_args=func_call_args,
        )

    seed_body = _load_seed_body(experiment_dir, safe_name, dsl_round, func_evolution_round)
    if seed_body:
        seed_body = "\n".join([f"  {line}" if line.strip() else "" for line in seed_body.splitlines()])
    evolve_func = f'''@funsearch.evolve
def {safe_name}({func_params}):
  """
  {description}
  
  Args:
      env: The current environment instance.
  {args_docstring}  
      Returns: List[int]: A sequence of raw integer action codes accepted by env.step().

  """
'''
    
    # Generate evaluate() function - always use template (never generated by LLM)
    # Prepare environment setup code
    env_setup = craft_env_setup(
        recipes_path=recipes_path,
        hints_path=hints_path,
        task_name=task_name_for_env,
        grid_spec_path=grid_spec_path,
    )
        
    if env_setup:
        # Use extracted environment setup
        # Define function arguments directly from grid_spec arg_values.
        args_definitions = ""
        if has_args:
            # Split args if there are multiple (comma-separated)
            arg_list = [a.strip() for a in args.split(',')] if ',' in args else [args.strip()]

            args_def_lines = []
            args_def_lines.append('  arg_values = grid_spec["arg_values"] if isinstance(grid_spec, dict) else {}')
            for arg_name in arg_list:
                if not arg_name:
                    continue
                args_def_lines.append(f'  {arg_name} = arg_values["{arg_name}"]')
                args_def_lines.append(f'  if isinstance({arg_name}, str):')
                args_def_lines.append(f'    {arg_name} = {arg_name}.lower()')
        
            args_definitions = '\n'.join(args_def_lines) + '\n' if args_def_lines else ''
        
        eval_func = craft_evaluate_template(
            display_name=display_name,
            env_setup=env_setup,
            args_definitions=args_definitions,
            func_call_args=func_call_args,
            grid_spec_paths_var=repr(grid_spec_paths),
        )

    # Combine all functions (solve, evaluate, and evolve)
    if solve_func is None:
        raise ValueError(f"LLM generation failed for {func_name}. solve_func is None. Check LLM generation logs for errors.")
    prompt_content = solve_func + "\n" + eval_func + "\n" + evolve_func
    # print(prompt_content)
    with open(func_file, 'w', encoding='utf-8') as f:
        f.write(prompt_content)
    
    # Construct the function signature that will be used
    func_signature = f"def {safe_name}({func_params})"
    
    print(f"Generated function prompt: {func_file}")
    return func_file, func_signature


def generate_baseline_function_prompt(
    func_name: str,
    description: str,
    cfg: str,
    *,
    specification: str,
    experiment_dir: str,
    dsl_round: Optional[int],
    func_evolution_round: int,
    task_name: str,
    variant: str,
    shared_vllm=None,
    grid_prompt_path: str = "prompt_specifications/grid_prompt.txt",
    require_test_type: bool = True,
    skip_positive_grids: bool = False,
    positive_grids: int = 10,
    negative_grids: int = 4,
    edge_grids: int = 1,
) -> tuple[str, str]:
    """Baseline-specific wrapper around generate_function_prompt.

    Keeps baseline branching out of generic callers.
    - task_env: direct task-name env evaluation (no testcase grids)
    - testcase: current testcase-grid behavior
    - two_phase_seeded_random: same prompt generation behavior as testcase;
      phase orchestration is handled by src/baseline.py
    """
    mode = (variant or "").strip().lower()

    if mode == "task_env":
        return generate_function_prompt(
            func_name=func_name,
            description=description,
            cfg=cfg,
            specification=specification,
            experiment_dir=experiment_dir,
            dsl_round=dsl_round,
            func_evolution_round=func_evolution_round,
            shared_vllm=shared_vllm,
            forced_task_name=task_name,
            use_task_env=True,
            grid_prompt_path=grid_prompt_path,
            require_test_type=require_test_type,
            skip_positive_grids=skip_positive_grids,
            positive_grids=positive_grids,
            negative_grids=negative_grids,
            edge_grids=edge_grids,
        )

    if mode in {"testcase", "two_phase_seeded_random"}:
        return generate_function_prompt(
            func_name=func_name,
            description=description,
            cfg=cfg,
            specification=specification,
            experiment_dir=experiment_dir,
            dsl_round=dsl_round,
            func_evolution_round=func_evolution_round,
            shared_vllm=shared_vllm,
            forced_task_name=task_name,
            use_task_env=False,
            grid_prompt_path=grid_prompt_path,
            require_test_type=require_test_type,
            skip_positive_grids=skip_positive_grids,
            positive_grids=positive_grids,
            negative_grids=negative_grids,
            edge_grids=edge_grids,
        )

    raise ValueError(
        f"Unsupported baseline variant: {variant}. "
        "Expected one of: task_env, testcase, two_phase_seeded_random"
    )


def _generate_baseline_task_env_prompt(
    func_name: str,
    description: str,
    cfg: str,
    *,
    specification: str,
    experiment_dir: str,
    dsl_round: Optional[int],
    func_evolution_round: int,
    task_name: str,
) -> tuple[str, str]:
    """Baseline task-env prompt generation path (no testcase grids)."""
    func_dir = os.path.join(experiment_dir, "function_specific_prompts")
    os.makedirs(func_dir, exist_ok=True)

    base_name, args_list = parse_function_name_and_args(func_name)
    safe_name = sanitize_function_name(func_name)

    if args_list:
        args = ", ".join([a.strip().lower() for a in args_list if a.strip()])
    elif cfg and str(cfg).strip():
        args = extract_function_args(func_name, cfg)
    else:
        args = ""

    if (args == "arg" or not args) and cfg and str(cfg).strip():
        from src.pipeline.cfg_parser import CFGParser
        parser = CFGParser(cfg)
        for fname, fargs in parser.get_terminal_functions():
            if fname.strip().lower() == base_name.strip().lower():
                if fargs:
                    args_list = [a.strip().lower() for a in fargs if a.strip()]
                    args = ", ".join(args_list)
                break

    has_args = bool(args and args != "arg" and args.strip())
    display_name = base_name
    func_file = os.path.join(func_dir, f"{_versioned_name(safe_name, dsl_round, func_evolution_round)}.txt")

    if has_args:
        func_params = f"env, {args}"
        func_call_args = f"env, {args}"
        args_docstring = f"      {args} (str): Function-specific argument(s).\\n  "
    else:
        func_params = "env"
        func_call_args = "env"
        args_docstring = ""

    solve_func = craft_solve_template_task_env_basic(
        func_name=func_name,
        func_params=func_params,
        func_call_args=func_call_args,
    )

    evolve_func = f'''@funsearch.evolve
def {safe_name}({func_params}):
  """
  {description}

  Args:
      env: The current environment instance.
  {args_docstring}
      Returns: List[int]: A sequence of raw integer action codes accepted by env.step().

  """
'''

    eval_func = craft_baseline_evaluate_template(
        display_name=display_name,
        func_call_args=func_call_args,
        task_name=task_name,
        recipes_path="craft/resources/recipes.yaml",
        hints_path="craft/resources/hints.yaml",
        max_steps=400,
    )

    prompt_content = solve_func + "\n" + eval_func + "\n" + evolve_func
    with open(func_file, "w", encoding="utf-8") as f:
        f.write(prompt_content)

    func_signature = f"def {safe_name}({func_params})"
    print(f"Generated baseline task-env prompt: {func_file}")
    return func_file, func_signature


def generate_baseline_function_prompt(
    func_name: str,
    description: str,
    cfg: str,
    *,
    specification: str,
    experiment_dir: str,
    dsl_round: Optional[int],
    func_evolution_round: int,
    task_name: str,
    variant: str,
    shared_vllm=None,
    grid_prompt_path: str = "prompt_specifications/grid_prompt.txt",
    require_test_type: bool = True,
    skip_positive_grids: bool = False,
    positive_grids: int = 10,
    negative_grids: int = 4,
    edge_grids: int = 1,
) -> tuple[str, str]:
    """Baseline-specific entrypoint without changing generic prompt API."""
    normalized = (variant or "").strip().lower()

    if normalized == "task_env":
        return _generate_baseline_task_env_prompt(
            func_name=func_name,
            description=description,
            cfg=cfg,
            specification=specification,
            experiment_dir=experiment_dir,
            dsl_round=dsl_round,
            func_evolution_round=func_evolution_round,
            task_name=task_name,
        )

    if normalized in {"testcase", "two_phase_seeded_random"}:
        return generate_function_prompt(
            func_name=func_name,
            description=description,
            cfg=cfg,
            specification=specification,
            experiment_dir=experiment_dir,
            dsl_round=dsl_round,
            func_evolution_round=func_evolution_round,
            shared_vllm=shared_vllm,
            forced_task_name=task_name,
            grid_prompt_path=grid_prompt_path,
            require_test_type=require_test_type,
            skip_positive_grids=skip_positive_grids,
            positive_grids=positive_grids,
            negative_grids=negative_grids,
            edge_grids=edge_grids,
        )

    raise ValueError(
        f"Unsupported baseline variant: {variant}. "
        "Expected one of: task_env, testcase, two_phase_seeded_random"
    )


def generate_func_init(func_name: str, description: str, cfg: str = "", 
                       func_dir: str = "functions_generated",
                       experiment_dir: Optional[str] = None,
                       dsl_round: Optional[int] = None,
                       func_evolution_round: Optional[int] = None) -> str:
    """Generate a stub implementation file for func_init.
    
    Args:
        func_name: Name of the function
        description: Description of the function
        cfg: CFG string
        func_dir: Directory for func_init files
        experiment_dir: Experiment directory
        dsl_round: DSL evolution round number (0-indexed)
        func_evolution_round: Function evolution round number (0-indexed, None for initial)
    """
    # Use experiment directory if provided, otherwise use default
    if experiment_dir:
        func_dir = os.path.join(experiment_dir, "functions_generated")
    
    os.makedirs(func_dir, exist_ok=True)
    
    safe_name = sanitize_function_name(func_name)
    base_name, args_list = parse_function_name_and_args(func_name)
    if args_list:
        args = ", ".join([arg.strip().lower() for arg in args_list if arg.strip()])
    elif cfg and str(cfg).strip():
        args = extract_function_args(func_name, cfg)
    else:
        args = ""
    
    func_init_file = os.path.join(func_dir, f"{_versioned_name(safe_name, dsl_round, func_evolution_round)}_func_init.py")

    init_content = None
    if experiment_dir and dsl_round is not None and dsl_round > 0:
        prev_final = os.path.join(experiment_dir, "final_functions", f"{_versioned_name(safe_name, dsl_round - 1)}.py")
        if os.path.exists(prev_final):
            with open(prev_final, "r", encoding="utf-8") as f:
                init_content = f.read()
    
    if init_content is None:
        # Infer return type from description
        return_type, default_return = infer_return_type(description)
        
        # Generate stub implementation
        init_content = f'''def {safe_name}(env, {args}):
    return {default_return}
'''
    
    with open(func_init_file, 'w', encoding='utf-8') as f:
        f.write(init_content)
    
    print(f"Generated func_init: {func_init_file}")
    return func_init_file


def _normalize_statement_seq_example(text: str) -> str:
    """Prepare example text for CFG example programs.

    - Collapse duplicate separators (``;;``, ``; ;``, …) to a single ``;`` between statements.
    - Remove dangling leading/trailing separators to avoid parse failures at ``$END``.
    """
    s = text.strip()
    if not s:
        return s
    s = re.sub(r"(?:;\s*){2,}", "; ", s)
    s = re.sub(r"^\s*;\s*", "", s)
    s = re.sub(r"\s*;\s*$", "", s)
    s = s.strip()
    return s


def convert_tokenized_to_program_format(tokenized_example: str) -> str:
    """Convert tokenized example format to program format.
    
    Converts from: "MOVE LPAR UP RPAR; COLLECT LPAR WOOD RPAR"
    To: "MOVE(UP); COLLECT(WOOD)"
    
    Also handles: "MOVE LPAR UP RPAR SEMICOLON COLLECT LPAR WOOD RPAR"
    To: "MOVE(UP); COLLECT(WOOD)"
    
    Args:
        tokenized_example: Example in tokenized format with LPAR/RPAR/COMMA/SEMICOLON tokens
        
    Returns:
        Example in program format with actual parentheses and punctuation
    """
    if not tokenized_example:
        return tokenized_example
    
    # Replace token names with actual characters
    # Do this carefully to avoid replacing parts of other tokens
    result = tokenized_example
    
    # Replace tokens (with word boundaries to avoid partial matches)
    result = re.sub(r'\bLPAR\b', '(', result)
    result = re.sub(r'\bRPAR\b', ')', result)
    result = re.sub(r'\bCOMMA\b', ',', result)
    result = re.sub(r'\bSEMICOLON\b', ';', result)
    
    # Clean up extra spaces around parentheses and punctuation
    result = re.sub(r'\s*\(\s*', '(', result)  # Remove spaces around (
    result = re.sub(r'\s*\)\s*', ')', result)  # Remove spaces around )
    result = re.sub(r'\s*,\s*', ',', result)   # Remove spaces around ,
    result = re.sub(r'\s*;\s*', '; ', result)  # Normalize semicolons

    # Clean up multiple spaces
    result = re.sub(r'\s+', ' ', result)

    result = result.strip()
    result = _normalize_statement_seq_example(result)
    return result


def validate_cfg(cfg: str, example: Optional[str] = None) -> Tuple[bool, str]:
    """Validate that the CFG is parseable.
    
    Args:
        cfg: The CFG string in BNF format
        example: Optional example program to test parsing (can be in tokenized or program format)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not cfg or not cfg.strip():
        return False, "CFG is empty"
    
    # Basic validation: check for BNF structure
    lines = cfg.strip().split('\n')
    has_rules = False
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Check if line contains a rule (has ::=)
        if '::=' in line:
            has_rules = True
            # Validate rule structure
            parts = line.split('::=')
            if len(parts) != 2:
                return False, f"Invalid rule format: {line}"
            lhs = parts[0].strip()
            rhs = parts[1].strip()
            if not lhs:
                return False, f"Empty left-hand side in rule: {line}"
            if not rhs:
                return False, f"Empty right-hand side in rule: {line}"
    
    if not has_rules:
        return False, "CFG contains no valid rules (no '::=' found)"
    
    # Try to use CFGParser if available to validate grammar can be built
    try:
        # Import CFGParser from the prog_synth_pipeline
        from src.pipeline.cfg_parser import CFGParser
        
        # CFGParser now accepts CFG as a string directly
        # Try to parse the CFG (can now pass string directly)
        parser = CFGParser(cfg)
        
        # Always try to parse the example if provided
        if example:
            # Clean up the example program
            example_clean = example.strip()
            if not example_clean:
                return True, "CFG is valid and parseable (example program is empty, skipping validation)"
            
            # Check if example is in tokenized format (contains LPAR/RPAR as tokens)
            # If so, convert to program format
            # Check for LPAR or RPAR as separate tokens (not part of other words)
            if re.search(r'\bLPAR\b', example_clean) or re.search(r'\bRPAR\b', example_clean):
                example_clean = convert_tokenized_to_program_format(example_clean)

            # LLMs / tokenized output may produce ``;;``; collapse to single ``;`` and ensure one trailing ``;``.
            example_clean = _normalize_statement_seq_example(example_clean)
            
            try:
                # Attempt to parse the example program
                parser.parse(example_clean)
                return True, "CFG is valid and parseable. Example program parsed successfully."
            except Exception as e:
                error_msg = str(e)
                # Provide more detailed error information
                example_preview = example_clean[:100] + ('...' if len(example_clean) > 100 else '')
                return False, (f"CFG grammar is valid but cannot parse example program.\n"
                              f"Example: {example_preview}\n"
                              f"Parse error: {error_msg}")
        else:
            return True, "CFG is valid and parseable (no example program provided for validation)"
    except ImportError:
        # CFGParser not available, just do basic validation
        return True, "CFG structure is valid (advanced parsing validation skipped)"
    except Exception as e:
        return False, f"Error validating CFG: {e}"


def validate_terminal_descriptions(terminals: Dict[str, str]) -> Tuple[bool, str]:
    """Validate terminal descriptions do not reference other terminal names."""
    if not isinstance(terminals, dict) or not terminals:
        return False, "Terminals dictionary is missing or empty."

    terminal_names = [str(name).strip() for name in terminals.keys() if str(name).strip()]
    if not terminal_names:
        return False, "No terminal names found in terminals dictionary."

    upper_names = [name.upper() for name in terminal_names]
    for terminal_name, description in terminals.items():
        if not isinstance(description, str):
            return False, f"Terminal description for '{terminal_name}' must be a string."

        self_name_upper = str(terminal_name).strip().upper()
        desc_upper = description.upper()

        for candidate in upper_names:
            if candidate == self_name_upper:
                continue
            # Match full token to avoid partial-word false positives.
            pattern = rf"(?<![A-Z0-9_]){re.escape(candidate)}(?![A-Z0-9_])"
            if re.search(pattern, desc_upper):
                return (
                    False,
                    f"Terminal '{terminal_name}' description references terminal '{candidate}'. "
                    "Terminal descriptions must be independent and must not mention other terminals by name.",
                )

    return True, "Terminal descriptions are valid (no cross-terminal references)."


def determine_inputs(func_name: str, description: str, cfg: str) -> list:
    """Determine appropriate inputs for funsearch evaluation.
    
    Since inputs don't affect the evaluation logic, we just return a simple default.
    """
    # Return a simple default input - doesn't matter what it is
    return ["test_input"]


def find_funsearch_log_file(
    func_name: str,
    results_dir: str,
    dsl_round: Optional[int] = None,
    func_evolution_round: Optional[int] = None,
) -> Optional[str]:
    """Find the funsearch log file for a given function."""
    del func_evolution_round
    safe_name = sanitize_function_name(func_name)
    pattern = f"*{safe_name}*.log"
    log_files = glob.glob(os.path.join(results_dir, pattern))
    if log_files:
        context_matches = []
        if dsl_round is not None:
            # Match prompt token in log basename: move_dsl1.txt (new) or move_dsl1_func0.txt (legacy).
            context_pattern = re.compile(
                rf"{re.escape(safe_name)}_dsl{int(dsl_round)}(?:_func\d+)?\.txt"
            )
            context_matches = [
                path for path in log_files
                if context_pattern.search(os.path.basename(path))
            ]
        if context_matches:
            return max(context_matches, key=os.path.getmtime)
        if dsl_round is not None:
            print(
                f"  No FunSearch log for {func_name} at "
                f"dsl{dsl_round} under {results_dir}"
            )
            return None
        return max(log_files, key=os.path.getmtime)
    return None


def run_explicit_feedback_generation(func_name: str, results_dir: str, func_file: str,
                                     experiment_dir: str, explicit_feedback_dir: str,
                                     specification: str, k: int = 5,
                                     shared_vllm=None, func_signature: str = "",
                                     results_tracker=None, dsl_round: Optional[int] = None,
                                     func_evolution_round: Optional[int] = None,
                                     num_iterations: int = 1,
                                     log_file: Optional[str] = None) -> Optional[str]:
    """Run explicit feedback generation for a function using existing explicit_feedback_generation.py.
    
    Args:
        func_name: Name of the function
        results_dir: Directory containing funsearch log files
        func_file: Path to the function prompt file
        experiment_dir: Experiment directory
        explicit_feedback_dir: Directory to save explicit feedback results
        specification: Specification string
        k: Number of top functions to use for feedback generation
        shared_vllm: Optional shared vLLM instance
        log_file: Optional pre-resolved log path (e.g. llm_best_of_n / llm_chained logs)
        
    Returns:
        Final function code as string, or None if extraction failed
    """
    if results_tracker is None:
        results_tracker = ResultsTracker(experiment_dir)
        print(
            "[run_explicit_feedback_generation] ResultsTracker attached "
            "(experiment_dir → results_tracking/): explicit-feedback interaction counts persist."
        )
    # Import using package path so it works when project root is on sys.path
    from src.pipeline.explicit_feedback_generation import parse_log_file, response_gen
    
    # Find the log file (caller may pass llm_best_of_n / llm_chained log path directly)
    if log_file and os.path.isfile(log_file):
        print(f"  Using provided log file: {log_file}")
    else:
        log_file = find_funsearch_log_file(
            func_name,
            results_dir,
            dsl_round=dsl_round,
            func_evolution_round=func_evolution_round,
        )
    if not log_file:
        print(f"   Could not find log file for {func_name}")
        return None
    
    print(f"  Using log file: {log_file}")
    
    # Parse top k functions from log using existing function
    funcs = parse_log_file(log_file, k=k)
    if not funcs:
        print("   No functions found in log file")
        return None
    
    print(f"  Found {len(funcs)} top functions")
    
    # Create evaluation file for this function with versioning
    safe_name = sanitize_function_name(func_name)
    eval_file = os.path.join(explicit_feedback_dir, f"eval_{_versioned_name(safe_name, dsl_round, func_evolution_round)}.py")
    create_evaluation_file(func_file, eval_file)
    
    # Use the function signature passed from the pipeline
    # This is the signature that was created when generating the function prompt
    if not func_signature:
        raise ValueError(
            f"func_signature is required for explicit feedback on {func_name} "
            f"(dsl{dsl_round})"
        )
    print(f"  Using function signature: {func_signature}")
    
    # Use the updated response_gen from explicit_feedback_generation.py
    final_func = response_gen(
        funcs, k, eval_file, specification, func_signature, 
        explicit_feedback_dir, shared_vllm=shared_vllm, results_tracker=results_tracker,
        dsl_round=dsl_round, func_evolution_round=func_evolution_round,
        num_iterations=num_iterations,
    )
    
    return final_func


def create_evaluation_file(func_file: str, eval_file: str):
    """Create an evaluation file for explicit feedback generation.
    
    Args:
        func_file: Path to the function prompt file
        eval_file: Path where the evaluation file should be created
    """
    # Read the function prompt file to get solve and evaluate functions
    with open(func_file, 'r', encoding='utf-8') as f:
        func_content = f.read()
    
    # Remove @funsearch.run and @funsearch.evolve decorators that cause NameError
    # These decorators are only needed for funsearch, not for execution
    # Remove standalone decorator lines
    func_content = re.sub(r'^\s*@funsearch\.(run|evolve)\s*$', '', func_content, flags=re.MULTILINE)
    # Remove decorators on the same line as function definition
    func_content = re.sub(r'@funsearch\.(run|evolve)\s*\n\s*', '', func_content)
    func_content = re.sub(r'@funsearch\.(run|evolve)\s+', '', func_content)
    
    # The evaluation file should contain the solve and evaluate functions (without decorators)
    with open(eval_file, 'w', encoding='utf-8') as f:
        f.write(func_content)




def get_cfg(
    experiment_dir: str,
    skip_cfg_generation: bool = False,
    cfg_output_file: Optional[str] = None,
    max_cfg_retries: int = 10,
    nld_path: str = "prompt_specifications/nld.txt",
    recipes_path: Optional[str] = None,
    cfg_generator_prompt_path: str = "prompt_specifications/cfg_generator.txt",
    domain_context_template_path: Optional[str] = None,
    shared_vllm=None
) -> Tuple[str, Dict[str, str], Optional[str], bool]:
    """Get CFG either by loading from file or generating new one.
    
    Args:
        experiment_dir: Experiment directory
        skip_cfg_generation: If True, load from cfg_output_file
        cfg_output_file: File to load CFG from
        max_cfg_retries: Maximum retries for CFG generation
        nld_path: Path to natural language domain description
        recipes_path: Optional path to recipes/domain schema file
        cfg_generator_prompt_path: Path to CFG generator prompt template
        domain_context_template_path: Path to domain-context template
        shared_vllm: Optional shared vLLM instance
        
    Returns:
        Tuple of (cfg: str, terminals: Dict[str, str], example: Optional[str], success: bool)
    """
    print(f"\n{'='*80}")
    print("Getting CFG")
    print(f"{'='*80}")
    
    cfg_path = os.path.join(experiment_dir, "cfg", "cfg_output.json")
    
    if skip_cfg_generation and cfg_output_file and os.path.exists(cfg_output_file):
        print(f"\n[Loading CFG] Loading from {cfg_output_file}...")
        with open(cfg_output_file, 'r', encoding='utf-8') as f:
            cfg_data = json.load(f)
        cfg = cfg_data.get("cfg", "")
        terminals = cfg_data.get("terminals", {})
        example = cfg_data.get("example", None)
        
        # Validate loaded CFG
        print("\n[Validating CFG] Validating loaded CFG...")
        is_valid, validation_msg = validate_cfg(cfg, example)
        if not is_valid:
            print(f"ERROR: Loaded CFG validation failed: {validation_msg}", file=sys.stderr)
            return "", {}, None, False
        term_valid, term_msg = validate_terminal_descriptions(terminals)
        if not term_valid:
            print(f"ERROR: Loaded terminal descriptions failed validation: {term_msg}", file=sys.stderr)
            return "", {}, None, False
        else:
            print(f" {validation_msg}")
            print(f" {term_msg}")
            # Write to experiment's cfg_output.json and cfg_output_0.json so downstream stages can find it
            os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
            with open(cfg_path, 'w', encoding='utf-8') as f:
                json.dump(cfg_data, f, indent=2, ensure_ascii=False)
            print(f" Wrote CFG to {cfg_path}")
            versioned_path = os.path.join(os.path.dirname(cfg_path), "cfg_output_0.json")
            with open(versioned_path, 'w', encoding='utf-8') as f:
                json.dump(cfg_data, f, indent=2, ensure_ascii=False)
            print(f" Wrote versioned CFG to {versioned_path}")
            return cfg, terminals, example, True
    elif skip_cfg_generation and os.path.exists(cfg_path):
        print(f"\n[Loading CFG] Loading from {cfg_path}...")
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg_data = json.load(f)
        cfg = cfg_data.get("cfg", "")
        terminals = cfg_data.get("terminals", {})
        example = cfg_data.get("example", None)
        
        # Validate loaded CFG
        print("\n[Validating CFG] Validating loaded CFG...")
        is_valid, validation_msg = validate_cfg(cfg, example)
        if not is_valid:
            print(f"ERROR: Loaded CFG validation failed: {validation_msg}", file=sys.stderr)
            return "", {}, None, False
        term_valid, term_msg = validate_terminal_descriptions(terminals)
        if not term_valid:
            print(f"ERROR: Loaded terminal descriptions failed validation: {term_msg}", file=sys.stderr)
            return "", {}, None, False
        else:
            print(f" {validation_msg}")
            print(f" {term_msg}")
            return cfg, terminals, example, True
    else:
        print("\n[Generating CFG] Generating new CFG...")
        cfg = None
        terminals = None
        example = None
        
        # Retry loop: keep generating CFGs until we get a valid one
        for attempt in range(1, max_cfg_retries + 1):
            try:
                if attempt > 1:
                    print(f"\n[Generating CFG] Retry attempt {attempt}/{max_cfg_retries}...")
                
                from src.pipeline.getting_cfg import generate_and_parse_cfg
                cfg, terminals, example = generate_and_parse_cfg(
                    vllm_instance=shared_vllm,
                    nld_path=nld_path,
                    recipes_path=recipes_path,
                    prompt_template_path=cfg_generator_prompt_path,
                    domain_context_template_path=domain_context_template_path,
                )
                
                # Validate CFG is parseable
                print(f"\n[Validating CFG] Validating CFG (attempt {attempt})...")
                is_valid, validation_msg = validate_cfg(cfg, example)
                if is_valid:
                    print(f" {validation_msg}")
                    # Extract terminals using ensure_terminals_match_cfg which handles both
                    # functions with arguments and functions without arguments
                    try:
                        from src.pipeline.integrated_pipeline import ensure_terminals_match_cfg
                        # Use ensure_terminals_match_cfg to extract all terminals (with and without args)
                        # Pass shared_vllm if available for LLM-based description generation
                        terminals = ensure_terminals_match_cfg(cfg, terminals if terminals else {}, shared_vllm=shared_vllm)
                        
                        if not terminals:
                            print("   Warning: No terminal functions found in CFG after extraction")
                    except Exception as e:
                        # This should not happen if validation passed, but handle gracefully
                        print(f"   CRITICAL: Terminal extraction failed after validation passed: {e}", file=sys.stderr)
                        import traceback
                        traceback.print_exc()
                        # This is a bug - validation passed but terminal extraction failed
                        # Don't retry, but raise to surface the issue
                        raise RuntimeError(f"CFG validation passed but terminal extraction failed: {e}")

                    term_valid, term_msg = validate_terminal_descriptions(terminals)
                    if not term_valid:
                        print(f" Terminal description validation failed: {term_msg}")
                        if attempt < max_cfg_retries:
                            print("Retrying CFG generation...")
                            continue
                        print(f"\nERROR: Failed to generate valid terminal descriptions after {max_cfg_retries} attempts", file=sys.stderr)
                        print(f"Last terminal validation error: {term_msg}", file=sys.stderr)
                        return "", {}, None, False
                    print(f" {term_msg}")
                    
                    # Success! Save and return
                    cfg_data = {
                        "cfg": cfg,
                        "terminals": terminals,
                        "example": example
                    }
                    os.makedirs(os.path.join(experiment_dir, "cfg"), exist_ok=True)
                    
                    # Version existing file before writing new one
                    if os.path.exists(cfg_path):
                        version_file(cfg_path)
                    
                    with open(cfg_path, 'w', encoding='utf-8') as f:
                        json.dump(cfg_data, f, indent=2, ensure_ascii=False)
                    print(f" Saved CFG to {cfg_path}")

                    # Also save as cfg_output_0.json so stage_evolve_dsl.py can find it
                    # (convention: cfg_output_N.json = CFG at dsl_round N; cfg_output.json = latest)
                    versioned_path_0 = os.path.join(experiment_dir, "cfg", "cfg_output_0.json")
                    if not os.path.exists(versioned_path_0):
                        import shutil
                        shutil.copy2(cfg_path, versioned_path_0)
                        print(f" Also saved CFG as {versioned_path_0}")

                    return cfg, terminals, example, True
                else:
                    print(f" CFG validation failed: {validation_msg}")
                    if attempt < max_cfg_retries:
                        print("Retrying CFG generation...")
                        continue
                    else:
                        # Last attempt failed
                        print(f"\nERROR: Failed to generate valid CFG after {max_cfg_retries} attempts", file=sys.stderr)
                        print(f"Last validation error: {validation_msg}", file=sys.stderr)
                        return "", {}, None, False
                        
            except Exception as e:
                print(f" Error generating CFG (attempt {attempt}): {e}", file=sys.stderr)
                if attempt < max_cfg_retries:
                    print("Retrying CFG generation...")
                    continue
                else:
                    print(f"\nERROR: Failed to generate CFG after {max_cfg_retries} attempts", file=sys.stderr)
                    return "", {}, None, False


def _dsl_generator_project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _resolve_spec_asset_path(rel_or_abs: str) -> str:
    """Resolve a project-relative or absolute path to the NLD/codebase assets."""
    if not rel_or_abs:
        raise ValueError("Asset path must be non-empty")
    if os.path.isabs(rel_or_abs):
        return rel_or_abs
    return os.path.join(_dsl_generator_project_root(), rel_or_abs)


def apply_specification_template_placeholders(
    specification: str,
    *,
    cfg: Optional[str] = None,
    nld_path: Optional[str] = None,
    codebase_path: Optional[str] = None,
) -> str:
    """Expand <<NLD>>, <<CODEBASE>>, and optionally <<CFG>> in a specification template.

    Paths default to ``nld_path`` / ``codebase_path`` from :func:`load_config` (including
    ``EXPERIMENT_CONFIG`` / env overrides such as ``NLD_PATH`` / ``CODEBASE_PATH``).
    Explicit ``nld_path`` / ``codebase_path`` arguments override those defaults.
    """
    settings = load_config()
    nld_rel = nld_path if nld_path is not None else settings.get("nld_path", "prompt_specifications/nld.txt")
    codebase_rel = (
        codebase_path
        if codebase_path is not None
        else settings.get("codebase_path", "prompt_specifications/codebase.txt")
    )

    spec = specification
    if re.search(r'<<\s*NLD\s*>>', spec, flags=re.IGNORECASE):
        nld_abs = _resolve_spec_asset_path(nld_rel)
        if not os.path.isfile(nld_abs):
            raise FileNotFoundError(
                f"Specification contains <<NLD>> but NLD file not found: {nld_abs}"
            )
        with open(nld_abs, "r", encoding="utf-8") as f:
            nld_text = f.read()
        spec = replace_nld_placeholder_in_specification(spec, nld_text)
    if re.search(r'<<\s*CODEBASE\s*>>', spec, flags=re.IGNORECASE):
        codebase_abs = _resolve_spec_asset_path(codebase_rel)
        if not os.path.isfile(codebase_abs):
            raise FileNotFoundError(
                f"Specification contains <<CODEBASE>> but codebase file not found: {codebase_abs}"
            )
        with open(codebase_abs, "r", encoding="utf-8") as f:
            codebase_text = f.read()
        spec = replace_codebase_placeholder_in_specification(spec, codebase_text)
    if cfg:
        spec = replace_dsl_section_in_specification(spec, cfg)
    return spec


def replace_nld_placeholder_in_specification(specification: str, nld: str) -> str:
    """Replace <<NLD>> in the specification template with the natural language domain text."""
    return re.sub(r'<<\s*NLD\s*>>', nld, specification, flags=re.IGNORECASE)


def replace_codebase_placeholder_in_specification(specification: str, codebase: str) -> str:
    """Replace <<CODEBASE>> in the specification template with the codebase description text."""
    return re.sub(r'<<\s*CODEBASE\s*>>', codebase, specification, flags=re.IGNORECASE)


def replace_dsl_section_in_specification(specification: str, cfg: str) -> str:
    """Replace the DSL block in the specification with the current CFG."""

    # Work on a copy
    spec = specification

    spec = re.sub(r'<<\s*CFG\s*>>', cfg, spec, flags=re.IGNORECASE)

    # Final safety check: if placeholders remain, warn so callers can act
    if re.search(r'<<\s*CFG\s*>>', spec, flags=re.IGNORECASE) or re.search(r'(?m)^[ \t]*CFG[ \t]*$', spec):
        print("\n[Step 2.1] Warning: DSL placeholder remains in specification after replacement")

    return spec


def implement_cfg(
    cfg: str,
    terminals: Dict[str, str],
    example: Optional[str],
    spec_file: str,
    experiment_dir: str,
    model_type: str = "huggingface",
    shared_vllm=None,
    results_tracker=None,
    dsl_round: Optional[int] = None,
    func_evolution_round: Optional[int] = None,
    nld_path: Optional[str] = None,
    codebase_path: Optional[str] = None,
) -> Tuple[bool, Dict[str, str]]:
    """Implement CFG by generating prompts, running funsearch, and extracting final functions.
    
    This function implements steps 2-7 from the full pipeline:
    - Step 2: Generate function-specific prompts
    - Step 3: Generate func_init files
    - Step 4: Run funsearch for each terminal function
    - Step 5: Run explicit feedback generation
    - Step 6: Save final functions
    - Step 7: Test generated functions (optional)
    
    Args:
        cfg: CFG string
        terminals: Dictionary mapping terminal function names to descriptions
        example: Optional example program from CFG
        spec_file: Path to specification file for funsearch
        experiment_dir: Experiment directory
        model_type: Model type for funsearch ('huggingface', 'ollama', or 'gemini')
        shared_vllm: Optional shared vLLM instance
        nld_path: Optional path to NLD file (default: from experiment config)
        codebase_path: Optional path to codebase description (default: from experiment config)
        
    Returns:
        Tuple of (success: bool, final_functions: Dict[str, str])
        - success: True if all steps completed successfully
        - final_functions: Dictionary mapping function names to their final code
    """
    print(f"\n{'='*80}")
    print("Implementing CFG (Steps 2-7)")
    print(f"{'='*80}")
    del func_evolution_round

    if results_tracker is None:
        results_tracker = ResultsTracker(experiment_dir)
        print(
            "[implement_cfg] ResultsTracker attached "
            "(experiment_dir → results_tracking/): FunSearch + explicit-feedback interactions persist."
        )
    
    # Step 2: Generate function-specific prompts
    print("\n[Step 2] Generating function-specific prompts...")
    
    # Load specification early to extract environment setup
    specification = ""
    if os.path.exists(spec_file):
        with open(spec_file, 'r', encoding='utf-8') as f:
            specification = f.read()
        specification = apply_specification_template_placeholders(
            specification,
            cfg=cfg if cfg else None,
            nld_path=nld_path,
            codebase_path=codebase_path,
        )
    
    # Persisted spec includes NLD, CODEBASE, and CFG when cfg is set
    if cfg:
        print("\n[Step 2.1] Replaced template placeholders (NLD, CODEBASE, CFG) in specification")

        # Persist the replaced specification so downstream jobs/readers never see the original template
        try:
            spec_with_cfg_path = os.path.join(experiment_dir, "spec_with_cfg.txt")
            os.makedirs(os.path.dirname(spec_with_cfg_path), exist_ok=True)
            with open(spec_with_cfg_path, 'w', encoding='utf-8') as _f:
                _f.write(specification)

            # Verify no placeholder remains in the written file
            if re.search(r'<<\s*NLD\s*>>', specification, flags=re.IGNORECASE):
                print(f"\n[Step 2.1] ERROR: NLD placeholder remained in written spec {spec_with_cfg_path}", file=sys.stderr)
                raise RuntimeError("NLD placeholder remained after replacement")
            if re.search(r'<<\s*CODEBASE\s*>>', specification, flags=re.IGNORECASE):
                print(f"\n[Step 2.1] ERROR: CODEBASE placeholder remained in written spec {spec_with_cfg_path}", file=sys.stderr)
                raise RuntimeError("CODEBASE placeholder remained after replacement")
            if re.search(r'<<\s*CFG\s*>>', specification, flags=re.IGNORECASE) or re.search(r'(?m)^[ \t]*CFG[ \t]*$', specification):
                print(f"\n[Step 2.1] ERROR: Placeholder remained in written spec {spec_with_cfg_path}", file=sys.stderr)
                raise RuntimeError("DSL placeholder remained after replacement")

            # Use the written file for downstream components
            spec_file = spec_with_cfg_path
            print(f"\n[Step 2.1] Wrote replaced specification to {spec_with_cfg_path}")
        except Exception as e:
            print(f"\n[Step 2.1] ERROR writing replaced specification: {e}", file=sys.stderr)
            raise
    
    func_files = {}
    func_signatures = {}
    for func_name, description in terminals.items():
        func_file, func_signature = generate_function_prompt(func_name, description, cfg, specification, 
                                            experiment_dir=experiment_dir,
                                            dsl_round=dsl_round)
        func_files[func_name] = func_file
        func_signatures[func_name] = func_signature
    
    # Step 3: Generate func_init files
    print("\n[Step 3] Generating func_init files...")
    func_init_files = {}
    for func_name, description in terminals.items():
        func_init_file = generate_func_init(func_name, description, cfg, 
                                           experiment_dir=experiment_dir,
                                           dsl_round=dsl_round)
        func_init_files[func_name] = func_init_file
    
    # Step 4: Run funsearch for each terminal function
    print("\n[Step 4] Running funsearch for each terminal function...")
    
    # Specification should already be loaded and updated with CFG, but check again
    if not specification:
        if not os.path.exists(spec_file):
            print(f"Error: Specification file not found: {spec_file}", file=sys.stderr)
            return False, {}
        
        with open(spec_file, 'r', encoding='utf-8') as f:
            specification = f.read()
        specification = apply_specification_template_placeholders(
            specification,
            cfg=cfg if cfg else None,
            nld_path=nld_path,
            codebase_path=codebase_path,
        )
    

    # Set total_samples=1000 to ensure we get exactly 1000 samples total.
    regen_attempts = int(load_config().get("grid_regeneration_attempts", 5))
    config = config_lib.Config(
        **funsearch_grid_regen_kwargs_from_config(),
        num_samplers=1,  # Single sampler - generates samples_per_prompt samples per iteration
        num_evaluators=2,  # Match samples_per_prompt - each evaluator handles one sample
        samples_per_prompt=2,  # 2 samples per prompt
        total_samples=1000,  # Target 1000 total samples across all iterations
        programs_database=config_lib.ProgramsDatabaseConfig(),
        grid_regeneration_attempts=regen_attempts,
    )
    
    # Results directory within experiment directory
    results_dir = os.path.join(experiment_dir, "results", "funsearch")
    
    # Helper function to run FunSearch for a single function
    def run_funsearch_for_function(func_name, func_file, func_init_file, description):
        """Run FunSearch for a single function (used for parallelization)."""
        try:
            print(f"[{func_name}] Starting FunSearch...")
            # Create a new FunSearch instance for this function (shares vLLM)
            funsearch = FunSearch(model_type=model_type, shared_vllm=shared_vllm)
            funsearch.results_tracker = results_tracker
            inputs = determine_inputs(func_name, description, cfg)
            
            funsearch.run(
                specification=specification,
                inputs=inputs,
                config=config,
                function_to_implement=func_file,
                function_init=func_init_file,
                spec_file=spec_file,
                experiment_dir=results_dir
            )
            print(f"[{func_name}]  Completed FunSearch")
            return func_name, "success", None
        except Exception as e:
            error_msg = str(e)
            print(f"[{func_name}]  Error: {error_msg}", file=sys.stderr)
            return func_name, "error", error_msg
    
    # Run FunSearch in parallel for all functions
    print(f"\n[Step 4] Running FunSearch in parallel for {len(terminals)} functions...")
    # Run multiple functions concurrently.
    max_workers = min(len(terminals), 16)
    print(f"  Using {max_workers} parallel workers")
    
    results = {}
    errors = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all FunSearch tasks
        future_to_func = {
            executor.submit(
                run_funsearch_for_function,
                func_name,
                func_files[func_name],
                func_init_files[func_name],
                description
            ): func_name
            for func_name, description in terminals.items()
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_func):
            func_name, status, error = future.result()
            results[func_name] = status
            if error:
                errors[func_name] = error
    
    # Check for errors
    if errors:
        print(f"\n FunSearch failed for {len(errors)} function(s):")
        for func_name, error in errors.items():
            print(f"  - {func_name}: {error}")
        print(" Pipeline stopped due to FunSearch failures")
        raise RuntimeError(f"FunSearch failed for functions: {list(errors.keys())}")
    
    print(f"\n All {len(terminals)} functions completed FunSearch successfully")

    # Generate FunSearch plots for all successful functions in package mode.
    try:
        dsl_plot_dir = os.path.join(
            experiment_dir,
            "results_tracking",
            "funsearch",
            f"dsl{dsl_round}" if dsl_round is not None else "dsl0",
        )
        os.makedirs(dsl_plot_dir, exist_ok=True)

        plotted = 0
        for func_name in terminals.keys():
            log_file = find_funsearch_log_file(func_name, results_dir)
            if not log_file:
                print(f"   No FunSearch log found for plotting: {func_name}")
                continue

            try:
                out = plot_funsearch_reward_vs_interactions(
                    log_file=log_file,
                    output_dir=dsl_plot_dir,
                    function_name=func_name,
                )
                if out:
                    plotted += 1
            except Exception as plot_err:
                print(f"   Failed to plot FunSearch metrics for {func_name}: {plot_err}")

        print(f"   Generated FunSearch plots: {plotted}/{len(terminals)}")
    except Exception as e:
        print(f"   Warning: FunSearch plotting step failed: {e}")
    
    # Step 5: Run explicit feedback generation for each function (in parallel)
    print("\n[Step 5] Running explicit feedback generation for each function (in parallel)...")
    final_functions = {}
    dsl_folder = f"dsl{dsl_round}" if dsl_round is not None else "dsl_unknown"
    explicit_feedback_dir = os.path.join(experiment_dir, "explicit_feedback", dsl_folder)
    os.makedirs(explicit_feedback_dir, exist_ok=True)
    
    # Import explicit feedback functions from existing module
    sys.path.insert(0, os.path.dirname(__file__))
    
    # Helper function to run explicit feedback for a single function
    def run_explicit_feedback_for_function(func_name, func_file):
        """Run explicit feedback for a single function (used for parallelization)."""
        try:
            print(f"[{func_name}] Starting explicit feedback generation...")
            final_func = run_explicit_feedback_generation(
                func_name, results_dir, func_file, experiment_dir, explicit_feedback_dir,
                specification, k=5, shared_vllm=shared_vllm, func_signature=func_signatures.get(func_name, ""),
                results_tracker=results_tracker,
                dsl_round=dsl_round,
            )
            if final_func:
                print(f"[{func_name}]  Completed explicit feedback")
                return func_name, final_func, None
            else:
                print(f"[{func_name}]  No final function extracted")
                return func_name, None, None
        except Exception as e:
            error_msg = str(e)
            print(f"[{func_name}]  Error: {error_msg}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return func_name, None, error_msg
    
    successful_funcs = [func_name for func_name in terminals.keys() if results.get(func_name) == "success"]
    if successful_funcs:
        max_workers_ef = min(len(successful_funcs), 8)  # Parallel explicit feedback workers
        print(f"  Using {max_workers_ef} parallel workers for explicit feedback")
        
        with ThreadPoolExecutor(max_workers=max_workers_ef) as executor:
            future_to_func = {
                executor.submit(
                    run_explicit_feedback_for_function,
                    func_name,
                    func_files[func_name]
                ): func_name
                for func_name in successful_funcs
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_func):
                func_name, final_func, error = future.result()
                if final_func:
                    final_functions[func_name] = final_func
                elif error:
                    print(f"   Explicit feedback failed for {func_name}: {error}")
    
    # Step 6: Save final functions to files
    print("\n[Step 6] Saving final functions...")
    
    final_functions_dir = os.path.join(experiment_dir, "final_functions")
    os.makedirs(final_functions_dir, exist_ok=True)
    
    for func_name, func_code in final_functions.items():
        safe_name = sanitize_function_name(func_name)
        func_file = os.path.join(final_functions_dir, f"{_versioned_name(safe_name, dsl_round)}.py")
        with open(func_file, 'w', encoding='utf-8') as f:
            f.write(func_code)
        print(f"  Saved {func_name} to {os.path.basename(func_file)}")
    
    print(" Final functions saved. Use standalone cfg_evaluator.py for evaluation.")
    
    # Step 7: Test generated functions with example program from CFG (optional)
    if example and final_functions:
        print("\n[Step 7] Testing generated functions with example program from CFG...")
        try:
            from src.pipeline.cfg_evaluator import CFGEvaluator
            from craft import env_factory
            
            # Create evaluator with CFG and final functions directory
            evaluator = CFGEvaluator(
                cfg=cfg,
                final_functions_dir=final_functions_dir
            )
            
            # Convert example program if it's in tokenized format
            example_program = example.strip()
            if re.search(r'\bLPAR\b', example_program) or re.search(r'\bRPAR\b', example_program):
                example_program = convert_tokenized_to_program_format(example_program)
            
            print(f"  Example program: {example_program}")
            
            # Create a test environment
            recipes_path = "craft/resources/recipes.yaml"
            hints_path = "craft/resources/hints.yaml"
            env_sampler = env_factory.EnvironmentFactory(
                recipes_path, hints_path, 7, max_steps=300,
                reuse_environments=False, visualise=False
            )
            
            # Try to extract task name from example or use a default
            # Look for patterns like "make[item]" in the example or use default
            task_name = "make[goldarrow]"  # default
            task_match = re.search(r'make\[([^\]]+)\]', example_program, re.IGNORECASE)
            if task_match:
                task_name = f"make[{task_match.group(1)}]"
            
            test_env = env_sampler.sample_environment(task_name=task_name)
            test_env.reset()
            
            # Evaluate the example program
            print(f"  Testing with task: {task_name}")
            result = evaluator.evaluate_program(example_program, env=test_env, max_steps=300)
            
            # Check if program parsed successfully
            parse_success = evaluator.parse_program(example_program)
            
            print(f"\n  {'='*60}")
            print("  Test Results for Example Program")
            print(f"  {'='*60}")
            print(f"  Program: {example_program}")
            print(f"  Task: {task_name}")
            print("\n  Execution Status:")
            print(f"     Program parsed successfully: {parse_success}")
            
            if result.get('error'):
                print(f"     Execution error: {result.get('error')}")
                print("     Program execution failed")
            else:
                print("     Program executed without errors")
            
            # Check if task was solved (success = True and reward > 0)
            task_solved = result.get('success', False)
            total_reward = result.get('total_reward', 0.0)
            steps_taken = result.get('steps', 0)
            actions_taken = result.get('actions_taken', [])
            
            print("\n  Task Completion:")
            if task_solved and total_reward > 0:
                print("     Task SOLVED successfully!")
                print(f"     Total Reward: {total_reward}")
            elif task_solved:
                print("     Task completed but reward is 0 (may indicate partial success)")
                print(f"    Total Reward: {total_reward}")
            else:
                print("     Task NOT solved")
                print(f"    Total Reward: {total_reward}")
            
            print("\n  Execution Details:")
            print(f"    Steps taken: {steps_taken}")
            print(f"    Actions executed: {len(actions_taken)}")
            
            if actions_taken:
                print("\n  Actions Sequence:")
                # Show first 20 actions to avoid too much output
                display_actions = actions_taken[:20]
                for i, action in enumerate(display_actions, 1):
                    print(f"    Step {i}: Action {action}")
                if len(actions_taken) > 20:
                    print(f"    ... and {len(actions_taken) - 20} more actions")
            
            print("\n  Summary:")
            if parse_success and not result.get('error'):
                if task_solved and total_reward > 0:
                    print("     Example program executed and SOLVED the task!")
                elif task_solved:
                    print("     Example program executed (task completed with 0 reward)")
                else:
                    print("     Example program executed but did not solve the task")
            elif not parse_success:
                print("     Example program failed to parse")
            else:
                print("     Example program execution failed")
            print(f"  {'='*60}\n")
                
        except ImportError as e:
            print(f"   Could not import cfg_evaluator: {e}")
            print("  Skipping function testing")
        except Exception as e:
            print(f"   Error testing functions: {e}")
            import traceback
            traceback.print_exc()
    elif not example:
        print("\n[Step 7] Skipping function testing (no example program in CFG)")
    elif not final_functions:
        print("\n[Step 7] Skipping function testing (no final functions generated)")
    
    # Summary
    success = len(final_functions) > 0 and len(final_functions) == len(terminals)
    print(f"\n{'='*80}")
    print("CFG Implementation Summary")
    print(f"{'='*80}")
    print(f"Total terminal functions: {len(terminals)}")
    print(f"Successfully processed: {sum(1 for r in results.values() if r == 'success')}")
    print(f"Errors: {sum(1 for r in results.values() if r != 'success')}")
    print(f"Final functions extracted: {len(final_functions)}")
    print(f"Success: {success}")
    
    return success, final_functions


def run_pipeline(spec_file: str, model_type: str = "huggingface", 
                 skip_cfg_generation: bool = False, cfg_output_file: Optional[str] = None,
                 max_cfg_retries: int = 10, experiment_dir: Optional[str] = None,
                 nld_path: str = "prompt_specifications/nld.txt",
                 recipes_path: str = "craft/resources/recipes.yaml",
                 codebase_path: Optional[str] = None):
    """Main pipeline function.
    
    Args:
        spec_file: Path to specification file for funsearch
        model_type: Model type for funsearch ('huggingface', 'ollama', or 'gemini')
        skip_cfg_generation: If True, load CFG from cfg_output_file instead of generating
        cfg_output_file: File to save/load CFG output
        max_cfg_retries: Maximum number of attempts to generate a valid CFG (default: 10)
        experiment_dir: Optional experiment directory (if None, will create a new one)
        nld_path: Path to natural language domain description
        recipes_path: Path to recipes/domain file
        codebase_path: Optional path for <<CODEBASE>> in spec template (default: experiment config)
    
    Returns:
        0 on success, 1 on error
    """
    
    print("=" * 80)
    print("CFG to FunSearch Pipeline")
    print("=" * 80)
    
    # Create shared vLLM instance for reuse across CFG generation and explicit feedback
    shared_vllm = None
    if vLLM is not None:
        try:
            print("\n[Setup] Initializing shared vLLM instance...")
            shared_vllm = vLLM(model="/scratch/avani/gpt", tensor_parallel_size=4)
            print(" Shared vLLM instance created")
        except Exception as e:
            print(f" Warning: Could not create shared vLLM instance: {e}")
            print("  Will create individual instances as needed")
    
    # Create experiment directory if not provided
    if experiment_dir is None:
        experiment_dir = create_experiment_directory()
        print(f"\n[Setup] Created experiment directory: {experiment_dir}")
    else:
        # Ensure the directory structure exists
        os.makedirs(experiment_dir, exist_ok=True)
        os.makedirs(os.path.join(experiment_dir, "function_specific_prompts"), exist_ok=True)
        os.makedirs(os.path.join(experiment_dir, "functions_generated"), exist_ok=True)
        os.makedirs(os.path.join(experiment_dir, "results", "funsearch"), exist_ok=True)
        os.makedirs(os.path.join(experiment_dir, "cfg"), exist_ok=True)
        print(f"\n[Setup] Using experiment directory: {experiment_dir}")
    
    # Step 1: Get CFG (using reusable function)
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
    
    # Also save to user-specified location if provided and different from experiment dir
    if cfg_output_file and cfg_output_file != os.path.join(experiment_dir, "cfg", "cfg_output.json"):
        cfg_data = {
            "cfg": cfg,
            "terminals": terminals,
            "example": example
        }
        with open(cfg_output_file, 'w', encoding='utf-8') as f:
            json.dump(cfg_data, f, indent=2, ensure_ascii=False)
        print(f"Also saved CFG output to {cfg_output_file}")
    
    print(f"\nFound {len(terminals)} terminal functions:")
    for func_name, desc in terminals.items():
        print(f"  - {func_name}: {desc[:50]}...")
    
    if not terminals:
        print("No terminal functions found. Exiting.")
        return 1
    
    # Use the extracted implement_cfg function for steps 2-7
    success, final_functions = implement_cfg(
        cfg=cfg,
        terminals=terminals,
        example=example,
        spec_file=spec_file,
        experiment_dir=experiment_dir,
        model_type=model_type,
        shared_vllm=shared_vllm,
        nld_path=nld_path,
        codebase_path=codebase_path,
    )
    
    if not success:
        print("\n CFG implementation failed or incomplete")
        return 1
    
    # Summary
    print("\n" + "=" * 80)
    print("Pipeline Summary")
    print("=" * 80)
    print(f"Experiment directory: {experiment_dir}")
    print(f"Total terminal functions: {len(terminals)}")
    print(f"Final functions extracted: {len(final_functions)}")
    print(f"\nAll outputs saved to: {experiment_dir}")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline to generate CFG, create function prompts, and run funsearch"
    )
    parser.add_argument(
        '--spec_file',
        type=str,
        required=True,
        help='Path to specification file for funsearch'
    )
    parser.add_argument(
        '--model_type',
        type=str,
        choices=['huggingface', 'ollama', 'gemini', 'openai_compat'],
        default='huggingface',
        help='Model type for funsearch'
    )
    parser.add_argument(
        '--skip_cfg_generation',
        action='store_true',
        help='Skip CFG generation and load from file'
    )
    parser.add_argument(
        '--cfg_output_file',
        type=str,
        default='cfg_output.json',
        help='File to save/load CFG output (default: cfg_output.json)'
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
        default='prompt_specifications/nld.txt',
        help='Path to natural language domain description file'
    )
    parser.add_argument(
        '--codebase_path',
        type=str,
        default=None,
        help='Path to codebase description for <<CODEBASE>> in spec (default: codebase_path from experiment config)'
    )
    parser.add_argument(
        '--recipes_path',
        type=str,
        default='craft/resources/recipes.yaml',
        help='Path to recipes/domain file'
    )
    parser.add_argument(
        '--experiment_dir',
        type=str,
        default=None,
        help='Optional experiment directory (if not provided, will create a new dated/numbered one)'
    )
    
    args = parser.parse_args()
    
    return run_pipeline(
        spec_file=args.spec_file,
        model_type=args.model_type,
        skip_cfg_generation=args.skip_cfg_generation,
        cfg_output_file=args.cfg_output_file,
        max_cfg_retries=args.max_cfg_retries,
        experiment_dir=args.experiment_dir,
        nld_path=args.nld_path,
        recipes_path=args.recipes_path,
        codebase_path=args.codebase_path,
    )

if __name__ == "__main__":
    sys.exit(main())



