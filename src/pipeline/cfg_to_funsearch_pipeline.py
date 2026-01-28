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
import threading
from funsearch.implementation.funsearch import FunSearch
from funsearch.implementation import config as config_lib
from src.utils.file_utils import version_file
from src.pipeline.grid_generation import ensure_function_grid_spec, build_env_setup

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


def _load_seed_body(
    experiment_dir: Optional[str],
    safe_name: str,
    dsl_round: Optional[int],
    func_evolution_round: Optional[int],
) -> str:
    """Load previous round function body to seed evolve prompt."""
    if not experiment_dir:
        return ""
    prev_path = None
    if dsl_round is not None and func_evolution_round is not None and func_evolution_round > 0:
        prev_path = os.path.join(
            experiment_dir,
            "final_functions",
            f"{safe_name}_dsl{dsl_round}_func{func_evolution_round - 1}.py"
        )
    elif dsl_round is not None and dsl_round > 0:
        prev_path = os.path.join(
            experiment_dir,
            "final_functions",
            f"{safe_name}_dsl{dsl_round - 1}_func0.py"
        )
    elif dsl_round is None:
        prev_path = os.path.join(
            experiment_dir,
            "final_functions",
            f"{safe_name}.py"
        )
    if prev_path and os.path.exists(prev_path):
        try:
            with open(prev_path, "r", encoding="utf-8") as f:
                code = f.read()
            return _extract_function_body(code, safe_name)
        except Exception:
            return ""
    return ""


def resolve_to_terminal_value(symbol: str, cfg: str, visited: set = None) -> Optional[str]:
    """Recursively resolve a non-terminal symbol to a terminal value.
    
    Args:
        symbol: The symbol to resolve (e.g., "TOOL", "ITEM")
        cfg: The CFG string
        visited: Set of already visited symbols to prevent infinite recursion
        
    Returns:
        A terminal value (string) or None if no terminal found
    """
    if visited is None:
        visited = set()
    
    # Prevent infinite recursion
    if symbol in visited:
        return None
    visited.add(symbol)
    
    # Look for the rule definition for this symbol
    rule_line_pattern = rf"^{re.escape(symbol)}\s*::=\s*(.+)$"
    rule_match = re.search(rule_line_pattern, cfg, re.IGNORECASE | re.MULTILINE)
    
    if not rule_match:
        # No rule found - this might be a terminal or doesn't exist
        return None
    
    values_str = rule_match.group(1).strip()
    rule_start_pos = rule_match.end()
    
    # Also handle continuation lines (lines starting with |)
    lines = cfg[rule_start_pos:].split('\n')
    for line in lines[:20]:  # Limit to first 20 continuation lines
        line = line.strip()
        if not line:
            break  # Empty line means end of rule
        if '::=' in line:
            break  # New rule definition means end of previous rule
        if line.startswith('|'):
            # This is a continuation line
            cont_values = line[1:].strip()  # Remove leading |
            values_str += " | " + cont_values
    
    # Extract individual values by splitting on |
    values = [v.strip() for v in values_str.split('|')]
    
    # Filter out grammar symbols and clean values
    clean_values = []
    for v in values:
        # Remove any newlines and normalize whitespace
        v = ' '.join(v.split()).strip().strip('"').strip("'")
        # Skip if it contains rule syntax (::=), is empty, or is a grammar symbol
        if (v and '::=' not in v and 
            v.upper() not in ['LPAR', 'RPAR', 'COMMA', 'SEMI', 'LBRACKET', 'RBRACKET'] and
            not v.startswith('::=')):
            clean_values.append(v)
    
    if not clean_values:
        return None
    
    # Check if the first value is a non-terminal (has a rule definition)
    first_value = clean_values[0]
    
    # Check if this value has its own rule definition (is a non-terminal)
    check_pattern = rf"^{re.escape(first_value)}\s*::="
    if re.search(check_pattern, cfg, re.IGNORECASE | re.MULTILINE):
        # It's a non-terminal, recursively resolve it
        resolved = resolve_to_terminal_value(first_value, cfg, visited)
        if resolved:
            return resolved
        # If recursion failed, try the next value
        for v in clean_values[1:]:
            check_pattern_v = rf"^{re.escape(v)}\s*::="
            if not re.search(check_pattern_v, cfg, re.IGNORECASE | re.MULTILINE):
                # This is a terminal, use it
                return v
            # It's also a non-terminal, try to resolve
            resolved = resolve_to_terminal_value(v, cfg, visited)
            if resolved:
                return resolved
    else:
        # It's a terminal value, return it
        return first_value
    
    # If we get here, couldn't find a terminal
    return None


def extract_function_args(func_name: str, cfg: str) -> str:
    """Extract function arguments from function name or CFG dynamically.
    
    Returns a comma-separated string of all arguments (e.g., "tool, item").
    """
    # First, try to parse arguments directly from the function name
    # (e.g., "TURN(DIR)" or "USE(TOOL, OBSTACLE)")
    base_name, args_from_name = parse_function_name_and_args(func_name)
    
    if args_from_name:
        # Return all arguments as comma-separated string
        return ", ".join([arg.lower() for arg in args_from_name])
    
    # If no args in function name, try to extract from CFG
    if not cfg:
        return "arg"
    
    # Use base_name for CFG search (without parentheses)
    func_name_for_search = base_name
    
    # Try patterns to extract ALL arguments from CFG
    # Pattern 1: FUNC_NAME LPAR ARG1 COMMA ARG2 COMMA ... RPAR (handles any number of args)
    # Match: FUNC_NAME LPAR ... RPAR and extract everything between LPAR and RPAR
    pattern_with_lpar = rf"{re.escape(func_name_for_search)}\s+LPAR\s+(.*?)\s+RPAR"
    match = re.search(pattern_with_lpar, cfg, re.IGNORECASE | re.MULTILINE | re.DOTALL)
    if match:
        args_content = match.group(1).strip()
        # Split by COMMA to get individual arguments
        # Filter out COMMA tokens and extract actual argument names
        args_list = []
        # Split by COMMA (as a token, not literal comma)
        parts = re.split(r'\s+COMMA\s+', args_content, flags=re.IGNORECASE)
        for part in parts:
            part = part.strip()
            # Extract the argument name (should be a single word/non-terminal)
            # Filter out LPAR, RPAR, COMMA tokens
            if part and part.upper() not in ['LPAR', 'RPAR', 'COMMA', 'SEMI', 'SEMICOLON']:
                # Take the first word as the argument name
                arg_match = re.match(r'^(\w+)', part)
                if arg_match:
                    arg = arg_match.group(1).strip().lower()
                    if arg and arg not in ['lpar', 'rpar', 'comma', 'semi', 'lparen', 'rparen']:
                        args_list.append(arg)
        
        if args_list:
            return ", ".join(args_list)
    
    # Pattern 2: FUNC_NAME LPAR ARG RPAR (single argument)
    single_arg_pattern = rf"{re.escape(func_name_for_search)}\s+LPAR\s+(\w+)\s+RPAR"
    match = re.search(single_arg_pattern, cfg, re.IGNORECASE | re.MULTILINE)
    if match:
        arg = match.group(1).strip().lower()
        if arg and arg not in ['lpar', 'rpar', 'comma', 'semi', 'lparen', 'rparen', '(', ')']:
            return arg
    
    # Pattern 3: FUNC_NAME(ARG1, ARG2, ...) with literal parentheses (handles any number)
    pattern_literal = rf"{re.escape(func_name_for_search)}\s*\(\s*([^)]+)\s*\)"
    match = re.search(pattern_literal, cfg, re.IGNORECASE | re.MULTILINE)
    if match:
        args_content = match.group(1).strip()
        # Split by comma
        args_list = []
        for arg in args_content.split(','):
            arg = arg.strip()
            if arg and arg.upper() not in ['LPAR', 'RPAR', 'COMMA', 'SEMI']:
                args_list.append(arg.lower())
        if args_list:
            return ", ".join(args_list)
    
    # Pattern 4: FUNC_NAME(ARG) single arg with literal parentheses
    single_arg_pattern2 = rf"{re.escape(func_name_for_search)}\s*\(\s*(\w+)\s*\)"
    match = re.search(single_arg_pattern2, cfg, re.IGNORECASE | re.MULTILINE)
    if match:
        arg = match.group(1).strip().lower()
        if arg and arg not in ['lpar', 'rpar', 'comma', 'semi']:
            return arg
    
    # Last resort: return generic default
    return "arg"


def infer_return_type(description: str) -> tuple[str, str]:
    """Infer return type and default return value from function description."""
    desc_lower = description.lower()
    
    # Check for boolean return indicators
    if any(phrase in desc_lower for phrase in ["returns true", "returns false", "returns bool", "boolean"]):
        return "bool", "False"
    
    # Check for integer return indicators
    if any(phrase in desc_lower for phrase in ["returns int", "returns number", "action number", "returns -1", "returns 0"]):
        return "int", "-1"
    
    # Default to list of actions
    return "list[int]", "[]"


def infer_argument_type(arg_name: str, cfg: str, description: str = "") -> str:
    """Infer the type of a function argument from CFG or description.
    
    Args:
        arg_name: Name of the argument (e.g., "dir", "item")
        cfg: The CFG string
        description: Optional function description for additional hints
        
    Returns:
        Type string: "str", "int", or "float"
    """
    arg_name_lower = arg_name.lower()
    desc_lower = description.lower() if description else ""
    
    # Check CFG for enumeration rules (likely strings)
    if cfg:
        # Look for enumeration rule: ARG_TYPE ::= VALUE1 | VALUE2 | VALUE3
        rule_pattern = rf"{re.escape(arg_name)}\s*::=\s*([^|]+(?:\s*\|\s*[^|]+)*)"
        rule_match = re.search(rule_pattern, cfg, re.IGNORECASE | re.MULTILINE)
        if rule_match:
            # If it's an enumeration with multiple string values, it's likely a string
            values_str = rule_match.group(1)
            values = [v.strip() for v in values_str.split('|')]
            # Check if values look like strings (uppercase identifiers, not numbers)
            if values:
                # If all values are uppercase identifiers (not numeric), it's a string
                all_string_like = all(
                    re.match(r'^[A-Z_][A-Z0-9_]*$', v.strip().strip('"').strip("'")) 
                    for v in values if v.strip()
                )
                if all_string_like:
                    return "str"
    
    # Check description for type hints
    if description:
        # Look for numeric indicators
        if any(phrase in desc_lower for phrase in [
            "integer", "int", "whole number", "count", "number of", 
            "index", "position", "coordinate", "distance"
        ]):
            return "int"
        
        # Look for float indicators
        if any(phrase in desc_lower for phrase in [
            "float", "decimal", "real number", "percentage", "ratio", 
            "probability", "weight", "score"
        ]):
            return "float"
        
        # Look for string indicators
        if any(phrase in desc_lower for phrase in [
            "string", "text", "name", "label", "identifier", "direction",
            "direction", "item", "type", "category", "option"
        ]):
            return "str"
    
    # Check argument name for common patterns
    if any(pattern in arg_name_lower for pattern in [
        "dir", "direction", "item", "type", "name", "label", "id", 
        "option", "choice", "category", "kind", "mode"
    ]):
        return "str"
    
    if any(pattern in arg_name_lower for pattern in [
        "count", "num", "index", "pos", "x", "y", "z", "width", "height",
        "size", "length", "distance", "step", "level"
    ]):
        # Could be int or float, default to int
        return "int"
    
    # Default to string (most common for DSL arguments)
    return "str"


def extract_env_setup_from_spec(specification: str) -> str:
    """Extract environment setup code from specification file."""
    if not specification:
        return None
    
    # Look for common environment setup patterns
    # Pattern 1: Multi-line environment factory initialization
    pattern1 = r'env_sampler\s*=\s*[^\n]+EnvironmentFactory\s*\([^)]+\)'
    match1 = re.search(pattern1, specification, re.MULTILINE | re.DOTALL)
    if match1:
        setup = match1.group(0).strip()
        # Try to also get the sample_environment call if nearby
        remaining = specification[match1.end():match1.end()+200]
        env_match = re.search(r'env\s*=\s*[^\n]+sample_environment\s*\([^)]+\)', remaining, re.MULTILINE)
        if env_match:
            return setup + "\n  " + env_match.group(0).strip()
        return setup
    
    # Pattern 2: Direct environment creation
    pattern2 = r'env\s*=\s*[^\n]+EnvironmentFactory\s*\([^)]+\)'
    match2 = re.search(pattern2, specification, re.MULTILINE | re.DOTALL)
    if match2:
        return match2.group(0).strip()
    
    # Pattern 3: Generic factory pattern
    pattern3 = r'[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*[a-zA-Z_][a-zA-Z0-9_]*\.EnvironmentFactory\s*\([^)]+\)'
    match3 = re.search(pattern3, specification, re.MULTILINE | re.DOTALL)
    if match3:
        return match3.group(0).strip()
    
    # Return None if nothing found - will use generic template
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
                             use_llm_evaluation: bool = True,
                             shared_vllm=None) -> tuple[str, str]:
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
        use_llm_evaluation: If True, use LLM to generate custom solve/evaluate functions
        shared_vllm: Optional shared vLLM instance for LLM-based evaluation generation
    """
    # Use experiment directory if provided, otherwise use default
    if experiment_dir:
        func_dir = os.path.join(experiment_dir, "function_specific_prompts")
    
    os.makedirs(func_dir, exist_ok=True)
    
    # Parse function name to get base name and arguments
    base_name, args_list = parse_function_name_and_args(func_name)
    
    # Get sanitized name for file naming
    safe_name = sanitize_function_name(func_name)
    
    # Extract argument name(s) - prefer from function name, fallback to CFG
    args = extract_function_args(func_name, cfg)
    
    # Check if function has arguments (check args_list first, then args)
    # args is now a comma-separated string like "tool, item" or just "tool"
    has_args = len(args_list) > 0 or (args and args != "arg" and args.strip())
    
    # For display purposes, use the base function name
    display_name = base_name
    
    # Build filename with DSL and function evolution round numbers
    # Always include func number, use func0 for initial round
    if dsl_round is not None:
        if func_evolution_round is not None:
            func_file = os.path.join(func_dir, f"{safe_name}_dsl{dsl_round}_func{func_evolution_round}.txt")
        else:
            # Initial round: use func0
            func_file = os.path.join(func_dir, f"{safe_name}_dsl{dsl_round}_func0.txt")
    else:
        # Fallback to old naming if rounds not provided
        func_file = os.path.join(func_dir, f"{safe_name}.txt")
    
    # Infer return type from description
    return_type, default_return = infer_return_type(description)
    
    # Infer argument type if function has arguments
    arg_type = "str"  # default
    if has_args:
        arg_type = infer_argument_type(args, cfg, description)
    
    # Build function signature parts
    if has_args:
        func_params = f"env, {args}"
        func_call_args = f"env, {args}" 
        args_docstring = f"      {args} ({arg_type}): Function-specific argument(s).\n  "
    else:
        func_params = "env"
        func_call_args = "env"
        args_docstring = ""

    default_task_name = "make[goldarrow]"
    recipes_path = "craft/resources/recipes.yaml"
    hints_path = "craft/resources/hints.yaml"
    grid_prompt_path = "prompt_specifications/grid_prompt.txt"
    grid_dir = os.path.join(experiment_dir, "grids") if experiment_dir else "grids"
    os.makedirs(grid_dir, exist_ok=True)
    if dsl_round is not None:
        grid_filename = f"{safe_name}_dsl{dsl_round}.json"
    else:
        grid_filename = f"{safe_name}.json"
    grid_spec_path = os.path.join(grid_dir, grid_filename)
    grid_spec = None
    if shared_vllm is not None:
        try:
            grid_spec = ensure_function_grid_spec(
                func_name=func_name,
                description=description,
                recipes_path=recipes_path,
                output_path=grid_spec_path,
                shared_vllm=shared_vllm,
                default_task_name=default_task_name,
                prompt_path=grid_prompt_path,
            )
        except Exception as e:
            print(f"  Warning: Grid generation failed for {func_name}: {e}")
    if grid_spec is None and os.path.exists(grid_spec_path):
        try:
            with open(grid_spec_path, "r", encoding="utf-8") as f:
                grid_spec = json.load(f)
        except Exception:
            grid_spec = None
    task_name_for_env = default_task_name
    if isinstance(grid_spec, dict):
        task_name_for_env = grid_spec.get("task_name", default_task_name) or default_task_name

    
    # Try to use LLM to generate custom solve/evaluate functions if enabled
    solve_func_custom = None
    evaluate_func_custom = None
    
    if use_llm_evaluation and shared_vllm is not None:
        try:
            from src.pipeline.generate_evaluation_functions import generate_custom_evaluation_functions
            
            # Prepare environment setup code
            env_setup = build_env_setup(
                recipes_path=recipes_path,
                hints_path=hints_path,
                task_name=task_name_for_env,
                grid_spec_path=grid_spec_path,
            )
            
            # Try to load recipes for better argument selection
            recipes = None
            task_name = task_name_for_env
            try:
                from src.pipeline.evaluation_helpers import load_recipes
                recipes = load_recipes(recipes_path)
            except Exception as e:
                print(f"  Note: Could not load recipes: {e}")
            
            print(f"  [LLM] Generating custom solve() function for {func_name}...")
            solve_func_custom = generate_custom_evaluation_functions(
                func_name=display_name,
                description=description,
                func_signature=f"def {safe_name}({func_params})",
                return_type=return_type,
                args=args if has_args else "",
                cfg=cfg,
                specification=specification,
                shared_vllm=shared_vllm,
                recipes=recipes,
                task_name=task_name
            )
            print(f"  [LLM] ✓ Generated custom solve() function for {func_name}")
            # evaluate_func_custom will be generated separately using template below
            evaluate_func_custom = None
        except Exception as e:
            print(f"  [LLM] ⚠ Failed to generate custom evaluation functions: {e}")
            import traceback
            traceback.print_exc()
            print(f"  [LLM] Falling back to template-based evaluation")
            solve_func_custom = None
            evaluate_func_custom = None
    

       
    solve_func = f'''def solve({func_params}, visualise=False):
  """Runs the environment with a {safe_name} function that returns list of actions to take and returns total reward."""
  # Capture grid state before function execution (with agent position)
  grid_before = None
  try:
    if hasattr(env, '_current_state') and hasattr(env._current_state, 'grid'):
      try:
        from test import grid_to_markdown
        # Get agent position for grid representation - ensure it's a tuple
        agent_pos = None
        if hasattr(env._current_state, 'pos'):
          pos = env._current_state.pos
          # Convert to tuple if it's a numpy array or list
          if hasattr(pos, '__iter__') and not isinstance(pos, str):
            agent_pos = tuple(pos) if len(pos) == 2 else None
          elif isinstance(pos, tuple):
            agent_pos = pos
        grid_before = grid_to_markdown(env._current_state.grid, env.world.cookbook, agent_pos)
      except (ImportError, AttributeError) as e:
        agent_pos = None
        if hasattr(env._current_state, 'pos'):
          pos = env._current_state.pos
          if hasattr(pos, '__iter__') and not isinstance(pos, str):
            agent_pos = tuple(pos) if len(pos) == 2 else None
        grid_before = f"Grid shape: {{env._current_state.grid.shape if hasattr(env._current_state.grid, 'shape') else 'N/A'}}\\nAgent position: {{agent_pos}}"
  except Exception as e:
    pass
  
  # Execute function to get actions using a deepcopy
  env_for_func = env
  try:
    import copy
    env_for_func = copy.deepcopy(env)
  except Exception:
    pass
  actions_to_take = {safe_name}({func_call_args.replace("env", "env_for_func", 1)})
  # Ensure actions_to_take is a list (handle None case)
  if actions_to_take is None:
    actions_to_take = []
  total_reward = 0.0
  actions_count = len(actions_to_take)

  # Execute actions
  for t in range(len(actions_to_take)):
    action = actions_to_take[t]
    reward, done, observations = env.step(action)
    total_reward += reward
    if done:
      break

  # Capture grid state after function execution (with agent position)
  grid_after = None
  try:
    if hasattr(env, '_current_state') and hasattr(env._current_state, 'grid'):
      try:
        from test import grid_to_markdown
        # Get agent position for grid representation - ensure it's a tuple
        agent_pos = None
        if hasattr(env._current_state, 'pos'):
          pos = env._current_state.pos
          # Convert to tuple if it's a numpy array or list
          if hasattr(pos, '__iter__') and not isinstance(pos, str):
            agent_pos = tuple(pos) if len(pos) == 2 else None
          elif isinstance(pos, tuple):
            agent_pos = pos
        grid_after = grid_to_markdown(env._current_state.grid, env.world.cookbook, agent_pos)
      except (ImportError, AttributeError) as e:
        agent_pos = None
        if hasattr(env._current_state, 'pos'):
          pos = env._current_state.pos
          if hasattr(pos, '__iter__') and not isinstance(pos, str):
            agent_pos = tuple(pos) if len(pos) == 2 else None
        grid_after = f"Grid shape: {{env._current_state.grid.shape if hasattr(env._current_state.grid, 'shape') else 'N/A'}}\\nAgent position: {{agent_pos}}"
  except Exception as e:
    pass

  # Return [total_reward, actions_count, grid_before, grid_after]
  return [total_reward, actions_count, grid_before, grid_after]
'''

    seed_body = _load_seed_body(experiment_dir, safe_name, dsl_round, func_evolution_round)
    if seed_body:
        seed_body = "\n".join([f"  {line}" if line.strip() else "" for line in seed_body.splitlines()])
    evolve_func = f'''@funsearch.evolve
def {safe_name}({func_params}):
  """
  {description}
  
  Args:
      env: The current environment instance.
{args_docstring}  Returns:
      List[int]: A sequence of encoded actions the agent should execute.
  """
{seed_body}
'''
    
    # Generate solve() function - use LLM-generated only (no fallback)
    if solve_func_custom:
        solve_func = solve_func_custom
    else:
        # No fallback - if LLM generation fails, solve_func will be None
        solve_func = None
    
    # Generate evaluate() function - always use template (never generated by LLM)
    # Prepare environment setup code
    env_setup = build_env_setup(
        recipes_path=recipes_path,
        hints_path=hints_path,
        task_name=task_name_for_env,
        grid_spec_path=grid_spec_path,
    )
        
    if env_setup:
        # Use extracted environment setup
        # Define function arguments before calling solve - extract from CFG
        args_definitions = ""
        if has_args:
            # Split args if there are multiple (comma-separated)
            arg_list = [a.strip() for a in args.split(',')] if ',' in args else [args.strip()]
        
            # Try to create a test environment to check what's on the grid
            test_env = None
            recipes = None
            try:
                from src.pipeline.evaluation_helpers import (
                    load_recipes, create_test_environment_for_item, get_valid_test_argument
                )
                from craft import env_factory
                
                recipes_path = "craft/resources/recipes.yaml"
                hints_path = "craft/resources/hints.yaml"
                recipes = load_recipes(recipes_path)
                
                # Try to extract task name from env_setup or use default
                task_name = "make[goldarrow]"  # default
                import re
                task_match = re.search(r"task_name\s*=\s*['\"]([^'\"]+)['\"]", env_setup)
                if task_match:
                    task_name = task_match.group(1)
                
                # Create a test environment to check what's on the grid
                test_env = create_test_environment_for_item(
                    task_name.replace("make[", "").replace("]", "").replace("get[", ""),
                    recipes_path, hints_path, env_factory
                )
            except Exception as e:
                print(f"  Note: Could not create test environment for grid checking: {e}")
                test_env = None
                recipes = None
            
            args_def_lines = []
            if has_args:  # Only process args if we have them
                for arg_name in arg_list:
                    if not arg_name:
                        continue
                    
                    # Use helper to get valid test argument that's on the grid
                    try:
                        from src.pipeline.evaluation_helpers import get_valid_test_argument
                        arg_value, explanation = get_valid_test_argument(
                            arg_name, arg_type, cfg, recipes, test_env, task_name
                        )
                        args_def_lines.append(f'  {arg_name} = {arg_value}  {explanation}')
                    except Exception as e:
                        # Fallback to old logic if helper fails
                        print(f"  Warning: Could not use grid-aware argument selection: {e}")
                        # Try to get valid values from CFG for this argument
                        arg_value = None
                        if cfg:
                            arg_value = resolve_to_terminal_value(arg_name.upper(), cfg)
                        
                        # If no CFG value found, use type-based defaults
                        if arg_value is None:
                            if arg_type == "str":
                                arg_value = '"test_input"'
                            elif arg_type == "int":
                                arg_value = "0"
                            elif arg_type == "float":
                                arg_value = "0.0"
                            else:
                                arg_value = '"test_input"'
                        else:
                            # Check if value is already quoted
                            is_quoted = (isinstance(arg_value, str) and 
                                        len(arg_value) >= 2 and 
                                        arg_value.startswith('"') and arg_value.endswith('"'))
                            
                            # If not already quoted, check if it's a number or string
                            if not is_quoted:
                                try:
                                    int(arg_value)
                                except ValueError:
                                    try:
                                        float(arg_value)
                                    except ValueError:
                                        arg_value = f'"{arg_value}"'
                        
                        args_def_lines.append(f'  {arg_name} = {arg_value}  # Argument value from CFG')
        
            args_definitions = '\n'.join(args_def_lines) + '\n' if args_def_lines else ''
        
        eval_func = f'''@funsearch.run
def evaluate():
  """Evaluates {display_name} behavior in a sample environment."""
  visualise = False
  {env_setup}
  env.reset()
{args_definitions}  result = solve({func_call_args}, visualise=visualise)
  # Return as list: [total_reward, actions_count, grid_before, grid_after]
  return result
'''
    else:
        # Generic template - user should customize based on their domain
        # Define function arguments before calling solve - extract from CFG
        args_definitions = ""
        if has_args:
            # Split args if there are multiple (comma-separated)
            arg_list = [a.strip() for a in args.split(',')] if ',' in args else [args.strip()]
            
            # Try to use grid-aware argument selection if possible
            test_env = None
            recipes = None
            task_name = None
            try:
                from src.pipeline.evaluation_helpers import (
                    load_recipes, get_valid_test_argument
                )
                recipes_path = "craft/resources/recipes.yaml"
                recipes = load_recipes(recipes_path)
            except Exception:
                recipes = None
            
            args_def_lines = []
            for arg_name in arg_list:
                if not arg_name:
                    continue
                
                # Try grid-aware selection first
                try:
                    from src.pipeline.evaluation_helpers import get_valid_test_argument
                    arg_value, explanation = get_valid_test_argument(
                        arg_name, arg_type, cfg, recipes, test_env, task_name
                    )
                    args_def_lines.append(f'  {arg_name} = {arg_value}  {explanation}')
                except Exception:
                    # Fallback to CFG-based selection
                    arg_value = None
                    if cfg:
                        arg_value = resolve_to_terminal_value(arg_name.upper(), cfg)
                    
                    if arg_value is None:
                        if arg_type == "str":
                            arg_value = '"test_input"'
                        elif arg_type == "int":
                            arg_value = "0"
                        elif arg_type == "float":
                            arg_value = "0.0"
                        else:
                            arg_value = '"test_input"'
                    else:
                        if arg_type == "str":
                            arg_value = f'"{arg_value}"'
                    
                    args_def_lines.append(f'  {arg_name} = {arg_value}  # Argument value from CFG')
            
            args_definitions = '\n'.join(args_def_lines) + '\n' if args_def_lines else ''
        
        eval_func = f'''@funsearch.run
def evaluate():
  """Evaluates {display_name} behavior in a sample environment."""
  visualise = False
  {env_setup}
  env.reset()
{args_definitions}  result = solve({func_call_args}, visualise=visualise)
  # Return as list: [total_reward, actions_count, grid_before, grid_after]
  return result
'''
    
    # Combine all functions (solve, evaluate, and evolve)
    if solve_func is None:
        raise ValueError(f"LLM generation failed for {func_name}. solve_func is None. Check LLM generation logs for errors.")
    prompt_content = solve_func + "\n" + eval_func + "\n" + evolve_func
    print(prompt_content)
    with open(func_file, 'w', encoding='utf-8') as f:
        f.write(prompt_content)
    
    # Construct the function signature that will be used
    func_signature = f"def {safe_name}({func_params})"
    
    print(f"Generated function prompt: {func_file}")
    return func_file, func_signature


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
    args = extract_function_args(func_name, cfg)
    
    # Build filename with DSL and function evolution round numbers to match prompt files
    # Always include func number, use func0 for initial round
    if dsl_round is not None:
        if func_evolution_round is not None:
            func_init_file = os.path.join(func_dir, f"{safe_name}_dsl{dsl_round}_func{func_evolution_round}_func_init.py")
        else:
            # Initial round: use func0
            func_init_file = os.path.join(func_dir, f"{safe_name}_dsl{dsl_round}_func0_func_init.py")
    else:
        # Fallback to old naming if rounds not provided
        func_init_file = os.path.join(func_dir, f"{safe_name}_func_init.py")
    
    init_content = None
    if experiment_dir and func_evolution_round is not None and func_evolution_round > 0:
        if dsl_round is not None:
            prev_final = os.path.join(
                experiment_dir,
                "final_functions",
                f"{safe_name}_dsl{dsl_round}_func{func_evolution_round - 1}.py"
            )
        else:
            prev_final = os.path.join(
                experiment_dir,
                "final_functions",
                f"{safe_name}.py"
            )
        if os.path.exists(prev_final):
            try:
                with open(prev_final, "r", encoding="utf-8") as f:
                    init_content = f.read()
            except Exception:
                init_content = None
    
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
    
    # Ensure it ends with semicolon
    result = result.strip()
    if result and not result.endswith(';'):
        result += ';'
    
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


def determine_inputs(func_name: str, description: str, cfg: str) -> list:
    """Determine appropriate inputs for funsearch evaluation.
    
    Since inputs don't affect the evaluation logic, we just return a simple default.
    """
    # Return a simple default input - doesn't matter what it is
    return ["test_input"]


def find_funsearch_log_file(func_name: str, results_dir: str) -> Optional[str]:
    """Find the funsearch log file for a given function.
    
    Args:
        func_name: Name of the function
        results_dir: Directory containing funsearch results
        
    Returns:
        Path to the log file, or None if not found
    """
    safe_name = sanitize_function_name(func_name)
    # Log files have pattern: model_q2.5_function_name_func_init_spec_timestamp.log
    pattern = f"*{safe_name}*.log"
    import glob
    log_files = glob.glob(os.path.join(results_dir, pattern))
    if log_files:
        # Return the most recent one
        return max(log_files, key=os.path.getmtime)
    return None


def run_explicit_feedback_generation(func_name: str, results_dir: str, func_file: str,
                                     experiment_dir: str, explicit_feedback_dir: str,
                                     specification: str, k: int = 5,
                                     shared_vllm=None, func_signature: str = "",
                                     results_tracker=None, dsl_round: Optional[int] = None,
                                     func_evolution_round: Optional[int] = None) -> Optional[str]:
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
        
    Returns:
        Final function code as string, or None if extraction failed
    """
    # Import using package path so it works when project root is on sys.path
    from src.pipeline.explicit_feedback_generation import parse_log_file, response_gen
    
    # Find the log file
    log_file = find_funsearch_log_file(func_name, results_dir)
    if not log_file:
        print(f"  ⚠ Could not find log file for {func_name}")
        return None
    
    print(f"  Using log file: {log_file}")
    
    # Parse top k functions from log using existing function
    funcs = parse_log_file(log_file, k=k)
    if not funcs:
        print(f"  ⚠ No functions found in log file")
        return None
    
    print(f"  Found {len(funcs)} top functions")
    
    # Create evaluation file for this function with versioning
    safe_name = sanitize_function_name(func_name)  # Always define safe_name for file naming
    if dsl_round is not None:
        if func_evolution_round is not None:
            eval_file = os.path.join(explicit_feedback_dir, f"eval_{safe_name}_dsl{dsl_round}_func{func_evolution_round}.py")
        else:
            eval_file = os.path.join(explicit_feedback_dir, f"eval_{safe_name}_dsl{dsl_round}_func0.py")
    else:
        eval_file = os.path.join(explicit_feedback_dir, f"eval_{safe_name}.py")
    create_evaluation_file(func_file, eval_file)
    
    # Use the function signature passed from the pipeline
    # This is the signature that was created when generating the function prompt
    if not func_signature:
        # Fallback: construct from function name if not provided
        base_name, args_list = parse_function_name_and_args(func_name)
        if args_list:
            args_str = ", ".join([a.lower() for a in args_list])
            func_signature = f"def {safe_name}(env, {args_str})"
        else:
            func_signature = f"def {safe_name}(env)"
        print(f"  Using fallback signature: {func_signature}")
    else:
        print(f"  Using function signature: {func_signature}")
    
    # Use the updated response_gen from explicit_feedback_generation.py
    final_func = response_gen(
        funcs, k, eval_file, specification, func_signature, 
        explicit_feedback_dir, shared_vllm=shared_vllm, results_tracker=results_tracker,
        dsl_round=dsl_round, func_evolution_round=func_evolution_round
    )
    
    # # Save final function
    # if final_func:
    #     final_func_file = os.path.join(explicit_feedback_dir, f"{safe_name}_final.py")
    #     with open(final_func_file, 'w', encoding='utf-8') as f:
    #         f.write(final_func)
    #     print(f"  Saved final function to: {final_func_file}")
    
    return final_func


def create_evaluation_file(func_file: str, eval_file: str):
    """Create an evaluation file for explicit feedback generation.
    
    Args:
        func_file: Path to the function prompt file
        eval_file: Path where the evaluation file should be created
    """
    import re
    
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
    shared_vllm=None
) -> Tuple[str, Dict[str, str], Optional[str], bool]:
    """Get CFG either by loading from file or generating new one.
    
    Args:
        experiment_dir: Experiment directory
        skip_cfg_generation: If True, load from cfg_output_file
        cfg_output_file: File to load CFG from
        max_cfg_retries: Maximum retries for CFG generation
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
        else:
            print(f"✓ {validation_msg}")
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
        else:
            print(f"✓ {validation_msg}")
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
                cfg, terminals, example = generate_and_parse_cfg(vllm_instance=shared_vllm)
                
                # Validate CFG is parseable
                print(f"\n[Validating CFG] Validating CFG (attempt {attempt})...")
                is_valid, validation_msg = validate_cfg(cfg, example)
                
                if is_valid:
                    print(f"✓ {validation_msg}")
                    # Extract terminals using ensure_terminals_match_cfg which handles both
                    # functions with arguments and functions without arguments
                    try:
                        from src.pipeline.integrated_pipeline import ensure_terminals_match_cfg
                        # Use ensure_terminals_match_cfg to extract all terminals (with and without args)
                        # Pass shared_vllm if available for LLM-based description generation
                        terminals = ensure_terminals_match_cfg(cfg, terminals if terminals else {}, shared_vllm=shared_vllm)
                        
                        if not terminals:
                            print("  ⚠ Warning: No terminal functions found in CFG after extraction")
                    except Exception as e:
                        # This should not happen if validation passed, but handle gracefully
                        print(f"  ✗ CRITICAL: Terminal extraction failed after validation passed: {e}", file=sys.stderr)
                        import traceback
                        traceback.print_exc()
                        # This is a bug - validation passed but terminal extraction failed
                        # Don't retry, but raise to surface the issue
                        raise RuntimeError(f"CFG validation passed but terminal extraction failed: {e}")
                    
                    # Success! Save and return
                    cfg_data = {
                        "cfg": cfg,
                        "terminals": terminals,
                        "example": example
                    }
                    os.makedirs(os.path.join(experiment_dir, "cfg"), exist_ok=True)
                    
                    # Version existing file before writing new one
                    if os.path.exists(cfg_path):
                        version_file(cfg_path, keep_original=False)
                    
                    with open(cfg_path, 'w', encoding='utf-8') as f:
                        json.dump(cfg_data, f, indent=2, ensure_ascii=False)
                    print(f"✓ Saved CFG to {cfg_path}")
                    return cfg, terminals, example, True
                else:
                    print(f"✗ CFG validation failed: {validation_msg}")
                    if attempt < max_cfg_retries:
                        print(f"Retrying CFG generation...")
                        continue
                    else:
                        # Last attempt failed
                        print(f"\nERROR: Failed to generate valid CFG after {max_cfg_retries} attempts", file=sys.stderr)
                        print(f"Last validation error: {validation_msg}", file=sys.stderr)
                        return "", {}, None, False
                        
            except Exception as e:
                print(f"✗ Error generating CFG (attempt {attempt}): {e}", file=sys.stderr)
                if attempt < max_cfg_retries:
                    print(f"Retrying CFG generation...")
                    continue
                else:
                    print(f"\nERROR: Failed to generate CFG after {max_cfg_retries} attempts", file=sys.stderr)
                    return "", {}, None, False


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
    func_evolution_round: Optional[int] = None
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
        
    Returns:
        Tuple of (success: bool, final_functions: Dict[str, str])
        - success: True if all steps completed successfully
        - final_functions: Dictionary mapping function names to their final code
    """
    print(f"\n{'='*80}")
    print("Implementing CFG (Steps 2-7)")
    print(f"{'='*80}")
    
    # Step 2: Generate function-specific prompts
    print("\n[Step 2] Generating function-specific prompts...")
    
    # Load specification early to extract environment setup
    specification = ""
    if os.path.exists(spec_file):
        with open(spec_file, 'r', encoding='utf-8') as f:
            specification = f.read()
    
    # Replace DSL section in specification with current CFG
    if cfg:
        # Pattern to match the DSL section (from "## DSL" to the closing """ on its own line)
        dsl_pattern = r'(## DSL[^\n]*\n.*?"""\n)(.*?)(\n"""\n)'
        # Find the DSL section
        dsl_match = re.search(dsl_pattern, specification, re.DOTALL)
        if dsl_match:
            # Replace the CFG content between the triple quotes
            header = dsl_match.group(1)  # "## DSL" + description + opening """
            footer = dsl_match.group(3)  # closing """
            # Replace with new CFG section
            cfg_section = header + cfg + footer
            specification = re.sub(dsl_pattern, cfg_section, specification, flags=re.DOTALL)
            print("\n[Step 2.1] Replaced DSL section in specification with current CFG")
        else:
            # Try a simpler pattern - just match from "## DSL" to the next """
            dsl_pattern_simple = r'(## DSL[^\n]*\n"""\n)(.*?)(\n"""\n)'
            dsl_match_simple = re.search(dsl_pattern_simple, specification, re.DOTALL)
            if dsl_match_simple:
                header = dsl_match_simple.group(1)
                footer = dsl_match_simple.group(3)
                cfg_section = header + cfg + footer
                specification = re.sub(dsl_pattern_simple, cfg_section, specification, flags=re.DOTALL)
                print("\n[Step 2.1] Replaced DSL section in specification with current CFG (simple pattern)")
            else:
                print("\n[Step 2.1] Warning: Could not find DSL section in specification to replace")
    
    func_files = {}
    func_signatures = {}
    for func_name, description in terminals.items():
        func_file, func_signature = generate_function_prompt(func_name, description, cfg, specification, 
                                            experiment_dir=experiment_dir,
                                            dsl_round=dsl_round,
                                            func_evolution_round=func_evolution_round,
                                            use_llm_evaluation=True,
                                            shared_vllm=shared_vllm)
        func_files[func_name] = func_file
        func_signatures[func_name] = func_signature
    
    # Step 3: Generate func_init files
    print("\n[Step 3] Generating func_init files...")
    func_init_files = {}
    for func_name, description in terminals.items():
        func_init_file = generate_func_init(func_name, description, cfg, 
                                           experiment_dir=experiment_dir,
                                           dsl_round=dsl_round,
                                           func_evolution_round=func_evolution_round)
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
        
        # Replace DSL section if CFG is available
        if cfg:
            # Pattern to match the DSL section (from "## DSL" to the closing """ on its own line)
            dsl_pattern = r'(## DSL[^\n]*\n.*?"""\n)(.*?)(\n"""\n)'
            dsl_match = re.search(dsl_pattern, specification, re.DOTALL)
            if dsl_match:
                header = dsl_match.group(1)
                footer = dsl_match.group(3)
                cfg_section = header + cfg + footer
                specification = re.sub(dsl_pattern, cfg_section, specification, flags=re.DOTALL)
            else:
                # Try simpler pattern
                dsl_pattern_simple = r'(## DSL[^\n]*\n"""\n)(.*?)(\n"""\n)'
                dsl_match_simple = re.search(dsl_pattern_simple, specification, re.DOTALL)
                if dsl_match_simple:
                    header = dsl_match_simple.group(1)
                    footer = dsl_match_simple.group(3)
                    cfg_section = header + cfg + footer
                    specification = re.sub(dsl_pattern_simple, cfg_section, specification, flags=re.DOTALL)
    
    # Configure FunSearch with parallelization
    # Match evaluators to samples_per_prompt for clean parallelization
    # Set total_samples=1000 to ensure we get exactly 1000 samples total
    config = config_lib.Config(
        num_samplers=1,  # Single sampler - generates samples_per_prompt samples per iteration
        num_evaluators=2,  # Match samples_per_prompt - each evaluator handles one sample
        samples_per_prompt=2,  # 2 samples per prompt
        total_samples=1000,  # Target 1000 total samples across all iterations
        programs_database=config_lib.ProgramsDatabaseConfig()
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
            # Pass results_tracker to funsearch so it can pass it to evaluators
            if results_tracker is not None:
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
            print(f"[{func_name}] ✓ Completed FunSearch")
            return func_name, "success", None
        except Exception as e:
            error_msg = str(e)
            print(f"[{func_name}] ✗ Error: {error_msg}", file=sys.stderr)
            return func_name, "error", error_msg
    
    # Run FunSearch in parallel for all functions
    print(f"\n[Step 4] Running FunSearch in parallel for {len(terminals)} functions...")
    # Increase max_workers - with shared vLLM, we can handle more concurrent FunSearch runs
    # Each FunSearch run uses 4 samplers + 4 evaluators internally, so we can run more functions in parallel
    max_workers = min(len(terminals), 16)  # Increased from 8 to 16 for better parallelization
    print(f"  Using {max_workers} parallel workers (each with 4 samplers + 4 evaluators)")
    
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
        print(f"\n✗ FunSearch failed for {len(errors)} function(s):")
        for func_name, error in errors.items():
            print(f"  - {func_name}: {error}")
        print("✗ Pipeline stopped due to FunSearch failures")
        raise RuntimeError(f"FunSearch failed for functions: {list(errors.keys())}")
    
    print(f"\n✓ All {len(terminals)} functions completed FunSearch successfully")
    
    # Step 5: Run explicit feedback generation for each function (in parallel)
    print("\n[Step 5] Running explicit feedback generation for each function (in parallel)...")
    final_functions = {}
    explicit_feedback_dir = os.path.join(experiment_dir, "explicit_feedback")
    os.makedirs(explicit_feedback_dir, exist_ok=True)
    
    # Import explicit feedback functions from existing module
    sys.path.insert(0, os.path.dirname(__file__))
    from explicit_feedback_generation import parse_log_file, eval as explicit_eval
    
    # Helper function to run explicit feedback for a single function
    def run_explicit_feedback_for_function(func_name, func_file):
        """Run explicit feedback for a single function (used for parallelization)."""
        try:
            print(f"[{func_name}] Starting explicit feedback generation...")
            final_func = run_explicit_feedback_generation(
                func_name, results_dir, func_file, experiment_dir, explicit_feedback_dir,
                specification, k=5, shared_vllm=shared_vllm, func_signature=func_signatures.get(func_name, ""),
                results_tracker=results_tracker,
                dsl_round=dsl_round, func_evolution_round=func_evolution_round
            )
            if final_func:
                print(f"[{func_name}] ✓ Completed explicit feedback")
                return func_name, final_func, None
            else:
                print(f"[{func_name}] ⚠ No final function extracted")
                return func_name, None, None
        except Exception as e:
            error_msg = str(e)
            print(f"[{func_name}] ✗ Error: {error_msg}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return func_name, None, error_msg
    
    # Run explicit feedback in parallel
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
                    print(f"  ✗ Explicit feedback failed for {func_name}: {error}")
    
    # Step 6: Save final functions to files
    print("\n[Step 6] Saving final functions...")
    
    # Save final functions to individual files for easy loading
    final_functions_dir = os.path.join(experiment_dir, "final_functions")
    os.makedirs(final_functions_dir, exist_ok=True)
    
    for func_name, func_code in final_functions.items():
        safe_name = sanitize_function_name(func_name)
        
        # Build filename with DSL and function evolution round numbers to match prompt files
        # Initial functions (no func_evolution_round) should be func0
        if dsl_round is not None:
            if func_evolution_round is not None:
                func_file = os.path.join(final_functions_dir, f"{safe_name}_dsl{dsl_round}_func{func_evolution_round}.py")
            else:
                # Initial functions should be func0
                func_file = os.path.join(final_functions_dir, f"{safe_name}_dsl{dsl_round}_func0.py")
        else:
            # Fallback to old naming if rounds not provided
            func_file = os.path.join(final_functions_dir, f"{safe_name}.py")
        
        with open(func_file, 'w', encoding='utf-8') as f:
            f.write(func_code)
        print(f"  Saved {func_name} to {os.path.basename(func_file)}")
    
    print(f"✓ Final functions saved. Use standalone cfg_evaluator.py for evaluation.")
    
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
            print(f"  Test Results for Example Program")
            print(f"  {'='*60}")
            print(f"  Program: {example_program}")
            print(f"  Task: {task_name}")
            print(f"\n  Execution Status:")
            print(f"    ✓ Program parsed successfully: {parse_success}")
            
            if result.get('error'):
                print(f"    ✗ Execution error: {result.get('error')}")
                print(f"    ✗ Program execution failed")
            else:
                print(f"    ✓ Program executed without errors")
            
            # Check if task was solved (success = True and reward > 0)
            task_solved = result.get('success', False)
            total_reward = result.get('total_reward', 0.0)
            steps_taken = result.get('steps', 0)
            actions_taken = result.get('actions_taken', [])
            
            print(f"\n  Task Completion:")
            if task_solved and total_reward > 0:
                print(f"    ✓ Task SOLVED successfully!")
                print(f"    ✓ Total Reward: {total_reward}")
            elif task_solved:
                print(f"    ⚠ Task completed but reward is 0 (may indicate partial success)")
                print(f"    Total Reward: {total_reward}")
            else:
                print(f"    ✗ Task NOT solved")
                print(f"    Total Reward: {total_reward}")
            
            print(f"\n  Execution Details:")
            print(f"    Steps taken: {steps_taken}")
            print(f"    Actions executed: {len(actions_taken)}")
            
            if actions_taken:
                print(f"\n  Actions Sequence:")
                # Show first 20 actions to avoid too much output
                display_actions = actions_taken[:20]
                for i, action in enumerate(display_actions, 1):
                    print(f"    Step {i}: Action {action}")
                if len(actions_taken) > 20:
                    print(f"    ... and {len(actions_taken) - 20} more actions")
            
            print(f"\n  Summary:")
            if parse_success and not result.get('error'):
                if task_solved and total_reward > 0:
                    print(f"    ✓✓ Example program executed and SOLVED the task!")
                elif task_solved:
                    print(f"    ✓ Example program executed (task completed with 0 reward)")
                else:
                    print(f"    ✓ Example program executed but did not solve the task")
            elif not parse_success:
                print(f"    ✗ Example program failed to parse")
            else:
                print(f"    ✗ Example program execution failed")
            print(f"  {'='*60}\n")
                
        except ImportError as e:
            print(f"  ⚠ Could not import cfg_evaluator: {e}")
            print(f"  Skipping function testing")
        except Exception as e:
            print(f"  ⚠ Error testing functions: {e}")
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
                 max_cfg_retries: int = 10, experiment_dir: Optional[str] = None):
    """Main pipeline function.
    
    Args:
        spec_file: Path to specification file for funsearch
        model_type: Model type for funsearch ('huggingface', 'ollama', or 'gemini')
        skip_cfg_generation: If True, load CFG from cfg_output_file instead of generating
        cfg_output_file: File to save/load CFG output
        max_cfg_retries: Maximum number of attempts to generate a valid CFG (default: 10)
        experiment_dir: Optional experiment directory (if None, will create a new one)
    
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
            print("✓ Shared vLLM instance created")
        except Exception as e:
            print(f"⚠ Warning: Could not create shared vLLM instance: {e}")
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
        shared_vllm=shared_vllm
    )
    
    if not success or not cfg or not terminals:
        print("✗ Failed to get valid CFG. Cannot proceed.")
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
        shared_vllm=shared_vllm
    )
    
    if not success:
        print("\n✗ CFG implementation failed or incomplete")
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
        choices=['huggingface', 'ollama', 'gemini'],
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
        experiment_dir=args.experiment_dir
    )


if __name__ == "__main__":
    sys.exit(main())

