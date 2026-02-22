"""
Integrated pipeline that:
1. Checks if final_functions exist (if not, runs explicit feedback generation)
2. Synthesizes programs using DSL and tests if all tasks are solved
3. If tasks fail:
   - Evolves functions using failing tasks in funsearch evaluate (up to 3 turns)
   - If 3 turns don't work, evolves DSL and restarts the whole pipeline
"""

import os
import sys
import re
import json
import argparse
import glob
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path (go up to project root)
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _project_root)

from src.pipeline.cfg_to_funsearch_pipeline import (
    run_pipeline, sanitize_function_name, parse_function_name_and_args, implement_cfg,
    extract_function_args, resolve_to_terminal_value
)
from src.utils.file_utils import version_file
# Removed dependency on got120dsl_program_synthesis.py
# Import grid_to_markdown from test.py instead
from src.utils.test import grid_to_markdown
from craft import env_factory
from vllm import LLM, SamplingParams

# Import vLLM for shared instance
try:
    from vllm import LLM as vLLM
except ImportError:
    vLLM = None

# Import CFG evaluator
try:
    from src.pipeline.cfg_evaluator import CFGEvaluator
    CFG_EVALUATOR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import CFGEvaluator: {e}")
    CFG_EVALUATOR_AVAILABLE = False


def _generate_description_from_name(func_name: str, cfg: str = "", terminals: Dict[str, str] = None, shared_vllm=None) -> str:
    """Generate a description for a terminal function.
    
    Uses LLM if available, otherwise falls back to pattern-based generation.
    
    Args:
        func_name: Name of the terminal function
        cfg: Optional CFG string for context
        shared_vllm: Optional shared vLLM instance for LLM-based generation
        
    Returns:
        Description string for the function
    """
    # Try LLM-based generation if available
    if shared_vllm is not None:
        try:
            from vllm import SamplingParams
            
            # Build prompt with CFG and existing terminals as context
            prompt = f"""Given a terminal function name from a domain-specific language (DSL), generate a clear, concise description of what this function does.

Function name: {func_name}
"""
            if cfg:
                # Include CFG context (limit to avoid token limits)
                prompt += f"""
Context-free grammar (CFG):
{cfg[:800]}
"""
            
            if terminals:
                # Include ALL existing terminal descriptions for context (both with and without arguments)
                # This helps the LLM understand the domain and style
                existing_descriptions = []
                for name, desc in terminals.items():
                    # Determine if function has arguments by checking CFG
                    has_args = False
                    if cfg:
                        # Check if function appears with LPAR in CFG (has arguments)
                        import re
                        # Pattern: FUNC_NAME LPAR ... (function with args)
                        pattern_with_args = rf'\b{re.escape(name)}\s+LPAR'
                        # Pattern: FUNC_NAME (standalone, no args)
                        pattern_without_args = rf'\b{re.escape(name)}\s*(?:SEMICOLON|$|\|)'
                        if re.search(pattern_with_args, cfg):
                            has_args = True
                    
                    arg_info = " (with arguments)" if has_args else " (no arguments)"
                    existing_descriptions.append(f"- {name}{arg_info}: {desc[:100]}")
                
                # Limit to avoid token limits, but show mix of with/without args
                if len(existing_descriptions) > 15:
                    # Take first 15 to show variety
                    existing_descriptions = existing_descriptions[:15]
                
                if existing_descriptions:
                    prompt += f"""
Existing terminal functions and their descriptions (for reference - includes both functions with and without arguments):
{chr(10).join(existing_descriptions)}
"""
            
            prompt += """
Generate a single-sentence description that explains the PURPOSE and INTENT of this function (what it achieves), not how it is implemented. Do NOT make assumptions about environment mechanics, specific state changes, or low-level execution details. Be specific and domain-appropriate. The description should be consistent with the style and domain of the existing terminal functions.

Return only the description, no additional text, explanations, or formatting.

Description:"""

            params = SamplingParams(temperature=0.7, max_tokens=200)
            output = shared_vllm.generate([prompt], sampling_params=params)
            description = output[0].outputs[0].text.strip()
            
            # Clean up the response (remove quotes, extra whitespace, etc.)
            description = description.strip('"\'')
            description = description.split('\n')[0].strip()  # Take first line only
            # Remove common prefixes like "Description:" or "The function"
            description = re.sub(r'^(Description|The function|This function)[:\s]*', '', description, flags=re.IGNORECASE)
            description = description.strip()
            
            if description and len(description) > 10:  # Valid description
                return description
        except Exception as e:
            print(f"  ⚠ LLM description generation failed for {func_name}: {e}, using fallback")
    
    # Fallback: pattern-based generation
    # Convert function name to a readable verb form
    # Handle common patterns like SNAKE_CASE or PascalCase
    func_lower = func_name.lower()
    
    # Split on underscores or camelCase boundaries
    if '_' in func_name:
        words = func_name.lower().split('_')
    else:
        # Simple camelCase detection: split on capital letters
        import re
        words = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', func_name)
        words = [w.lower() for w in words]
    
    # Get the primary verb (usually the first word)
    primary_verb = words[0] if words else func_lower
    
    # Generate a generic description based on the verb
    # Use simple present tense form
    verb_form = primary_verb
    if verb_form.endswith('ed'):
        verb_form = verb_form[:-2]  # Remove past tense
    elif verb_form.endswith('ing'):
        verb_form = verb_form[:-3]  # Remove gerund
    
    # Create a generic description
    if len(words) > 1:
        # Multi-word function: "USE_TOOL" -> "Use tool with the specified parameters"
        action = ' '.join(words)
        return f"Execute {action} with the specified parameters"
    else:
        # Single word function: "MOVE" -> "Execute move with the specified parameters"
        return f"Execute {verb_form} with the specified parameters"


def _filter_example_program(example_program: str, terminals: Dict[str, str]) -> str:
    """Filter example program to only include functions that exist in terminals dictionary.
    
    Args:
        example_program: Example program string like "MOVE(UP); COLLECT(WOOD); CRAFT(STICK);"
        terminals: Dictionary of terminal function names to descriptions
        
    Returns:
        Filtered example program with only functions that exist in terminals
    """
    if not example_program or not terminals:
        return example_program
    
    # Get set of available function names (case-insensitive matching)
    available_funcs = {name.upper() for name in terminals.keys()}
    
    # Split program into statements (separated by semicolons)
    statements = [s.strip() for s in example_program.split(';') if s.strip()]
    filtered_statements = []
    
    for statement in statements:
        # Extract function name from statement (e.g., "MOVE(UP)" -> "MOVE")
        # Pattern: FUNC_NAME(ARGS) or FUNC_NAME
        match = re.match(r'^([A-Z_][A-Z0-9_]*)\s*\(', statement)
        if match:
            func_name = match.group(1).upper()
            if func_name in available_funcs:
                filtered_statements.append(statement)
        else:
            # No parentheses - check if it's just a function name
            func_name = statement.strip().upper()
            if func_name in available_funcs:
                filtered_statements.append(statement)
    
    # Join filtered statements back with semicolons
    if filtered_statements:
        return '; '.join(filtered_statements) + (';' if example_program.rstrip().endswith(';') else '')
    else:
        # If no statements remain, return empty string (will trigger fallback generation)
        return ""


def _generate_example_from_terminal(func_name: str, cfg: str) -> str:
    """Generate an example program statement from a terminal function and CFG.
    
    Args:
        func_name: Terminal function name (e.g., "COLLECT", "MOVE")
        cfg: CFG string
        
    Returns:
        Example program statement like "COLLECT(WOOD);" or "OBSERVE();"
    """
    if not cfg:
        return f"{func_name}();"
    
    # Extract function arguments from CFG
    args_str = extract_function_args(func_name, cfg)
    
    if not args_str or args_str == "arg":
        # No arguments or generic arg - try to check CFG for actual signature
        # Check if function has LPAR in CFG
        func_pattern = rf"{re.escape(func_name)}\s+LPAR"
        if re.search(func_pattern, cfg, re.IGNORECASE):
            # Has arguments, try to find them
            args_str = extract_function_args(func_name, cfg)
        else:
            # No arguments
            return f"{func_name}();"
    
    # Parse arguments (comma-separated)
    arg_list = [arg.strip() for arg in args_str.split(',') if arg.strip()]
    
    if not arg_list:
        return f"{func_name}();"
    
    # For each argument, try to resolve to a terminal value from CFG
    arg_values = []
    for arg in arg_list:
        # Try to resolve to terminal value (e.g., PRIMITIVE -> WOOD, DIRECTION -> NORTH)
        terminal_value = resolve_to_terminal_value(arg.upper(), cfg)
        if terminal_value:
            arg_values.append(terminal_value)
        else:
            # Fallback: use the argument name itself (uppercase)
            arg_values.append(arg.upper())
    
    # Construct example: FUNC_NAME(VALUE1, VALUE2, ...);
    if len(arg_values) == 1:
        return f"{func_name}({arg_values[0]});"
    else:
        return f"{func_name}({', '.join(arg_values)});"


def ensure_terminals_match_cfg(cfg: str, terminals: Dict[str, str], old_terminals: Optional[Dict[str, str]] = None, shared_vllm=None) -> Dict[str, str]:
    """Ensure terminals dictionary includes ALL terminal functions from the CFG.
    
    Includes both functions with arguments (from get_terminal_functions()) and functions
    without arguments (extracted from statement productions).
    
    Args:
        cfg: CFG string in BNF format
        terminals: Current terminals dictionary (may be incomplete)
        old_terminals: Optional previous terminals dictionary to preserve descriptions from
        
    Returns:
        Updated terminals dictionary with all functions from CFG
    """
    try:
        from src.pipeline.cfg_parser import CFGParser
        import re
        cfg_parser = CFGParser(cfg)
        terminal_funcs = cfg_parser.get_terminal_functions()
        
        # Get functions with arguments
        func_names_with_args = [func_name for func_name, _ in terminal_funcs] if terminal_funcs else []
        
        # Extract functions without arguments from statement productions
        # Look for statement rules and find the first symbol in each alternative
        func_names_without_args = set()
        statement_rules = ['statement', 'action', 'command', 'instruction']  # Common names for statement rules
        
        for rule_name, productions in cfg_parser.rules.items():
            # Check if this is a statement rule (case-insensitive)
            if rule_name.lower() in statement_rules or 'statement' in rule_name.lower():
                for production in productions:
                    production_stripped = production.strip()
                    if not production_stripped:
                        continue
                    
                    # Check if production is just a single uppercase identifier (function without args)
                    # Pattern 1: Just the identifier itself (e.g., "SCAN", "WAIT")
                    standalone_match = re.match(r'^\s*([A-Z_][A-Z0-9_]*)\s*$', production_stripped)
                    if standalone_match:
                        first_symbol = standalone_match.group(1)
                        # Exclude special symbols and non-terminals
                        if (first_symbol not in cfg_parser.special_symbols and 
                            first_symbol not in cfg_parser.keywords and
                            first_symbol not in cfg_parser.non_terminals and
                            first_symbol not in func_names_with_args):
                            func_names_without_args.add(first_symbol)
                            continue
                    
                    # Pattern 2: Identifier followed by space and then something else (but not LPAR)
                    # This catches functions that might be at the start but we need to check they don't have args
                    match = re.match(r'^\s*([A-Z_][A-Z0-9_]*)(?:\s|$)', production_stripped)
                    if match:
                        first_symbol = match.group(1)
                        # Check if this production has LPAR (meaning it has arguments) - if so, skip
                        if 'LPAR' in production_stripped or '(' in production_stripped:
                            # This function has arguments, skip it (already handled by get_terminal_functions)
                            continue
                        # Exclude special symbols and non-terminals
                        if (first_symbol not in cfg_parser.special_symbols and 
                            first_symbol not in cfg_parser.keywords and
                            first_symbol not in cfg_parser.non_terminals and
                            first_symbol not in func_names_with_args):
                            func_names_without_args.add(first_symbol)
        
        # Additional fallback: check all terminals from CFG parser for function terminals
        # that might have been missed (filter out enum values, special symbols, etc.)
        special_symbols_set = set(cfg_parser.special_symbols.keys())
        keywords_set = set(cfg_parser.keywords.keys())
        # Common enum terminal names that should be excluded
        enum_patterns = {'DIRECTION', 'TURN_DIR', 'ITEM', 'TOOL', 'OBSTACLE', 'WORKSHOP', 
                        'NORTH', 'SOUTH', 'EAST', 'WEST', 'LEFT', 'RIGHT',
                        'IRON', 'GRASS', 'WOOD', 'ROCK', 'GOLD', 'GEM', 'STICK', 'AXE',
                        'SLINGSHOT', 'ARROW', 'GOLDARROW', 'BRIDGE', 'BUNDLE', 'HAMMER',
                        'KNIFE', 'BED', 'SHEARS', 'LADDER', 'BOW', 'BENCH', 'FLAG', 'PLANK',
                        'CLOTH', 'ROPE', 'PICKAXE', 'SHOVEL', 'WATER', 'STONE', 'BOUNDARY',
                        'WORKSHOP0', 'WORKSHOP1', 'WORKSHOP2'}
        
        # Check all terminals and find potential function terminals
        for terminal in cfg_parser.terminals:
            # Skip if already found or if it's a special symbol/keyword/enum
            if (terminal in func_names_with_args or 
                terminal in func_names_without_args or
                terminal in special_symbols_set or
                terminal in keywords_set or
                terminal in enum_patterns or
                terminal in cfg_parser.non_terminals):
                continue
            
            # Check if this terminal appears in statement rules as a standalone production
            # This is a function terminal if it appears as a complete production in a statement rule
            for rule_name, productions in cfg_parser.rules.items():
                if rule_name.lower() in statement_rules or 'statement' in rule_name.lower():
                    for production in productions:
                        production_stripped = production.strip()
                        # Check if this terminal is the entire production or starts the production
                        if (production_stripped == terminal or
                            production_stripped.startswith(terminal + ' ') or
                            production_stripped.startswith(terminal + '\t')):
                            # Make sure it's not followed by LPAR (which would mean it has args)
                            if 'LPAR' not in production_stripped and '(' not in production_stripped:
                                func_names_without_args.add(terminal)
                                break
        
        # Combine all function names (with and without args)
        all_func_names = set(func_names_with_args) | func_names_without_args
        
        if all_func_names:
            # Start with existing terminals (preserve descriptions)
            updated_terminals = terminals.copy()
            
            # Collect functions that need descriptions
            functions_needing_descriptions = []
            for func_name in all_func_names:
                if func_name not in updated_terminals:
                    # Try to get description from old_terminals first (preserve across evolutions)
                    if old_terminals and func_name in old_terminals:
                        updated_terminals[func_name] = old_terminals[func_name]
                        print(f"  ✓ Preserved description for {func_name} from previous CFG")
                    else:
                        functions_needing_descriptions.append(func_name)
            
            # Generate descriptions for missing functions using LLM if available
            if functions_needing_descriptions and shared_vllm is not None:
                print(f"  Generating LLM descriptions for {len(functions_needing_descriptions)} functions...")
                # Generate descriptions one by one, passing CFG and existing terminals as context
                for func_name in functions_needing_descriptions:
                    try:
                        description = _generate_description_from_name(
                            func_name, 
                            cfg=cfg, 
                            terminals=updated_terminals,  # Pass existing terminals for context
                            shared_vllm=shared_vllm
                        )
                        updated_terminals[func_name] = description
                        print(f"  ✓ Generated LLM description for {func_name}")
                    except Exception as e:
                        print(f"  ⚠ LLM description generation failed for {func_name}: {e}, using pattern-based fallback")
                        # Fall back to pattern-based generation
                        description = _generate_description_from_name(func_name, cfg=cfg, shared_vllm=None)
                        updated_terminals[func_name] = description
                        print(f"  ✓ Generated pattern-based description for {func_name}")
            else:
                # Use pattern-based generation (no LLM available or no functions needing descriptions)
                for func_name in functions_needing_descriptions:
                    description = _generate_description_from_name(func_name, cfg=cfg, shared_vllm=None)
                    updated_terminals[func_name] = description
                    print(f"  ✓ Generated pattern-based description for {func_name}")
            
            # Remove any terminals that aren't in the CFG (cleanup)
            terminals_to_remove = [name for name in updated_terminals.keys() if name not in all_func_names]
            for name in terminals_to_remove:
                del updated_terminals[name]
                print(f"  ✓ Removed {name} (not in CFG)")
            
            return updated_terminals
        else:
            # If no functions found, return terminals as-is
            return terminals
    except Exception as e:
        print(f"  ⚠ Warning: Could not extract terminals from CFG: {e}")
        return terminals


def extract_and_save_cfg(output_text, cfg_dir="cfg"):
    """
    Extracts three sections from a model output:
      1. Failure Analysis
      2. Updated CFG (BNF)
      3. Terminal Functions

    Returns:
      (filepath, cfg_text, term_text, failure_text, cfg_explanation)

    - If CFG not found, returns (None, None, None, failure_text, None)
    - Also saves the CFG to a timestamped file if found.
    """
    # --- Extract Failure Analysis ---
    failure_match = re.search(
        r"\*\*Failure Analysis\*\*(.*?)(?:\n---|\Z)",
        output_text,
        re.DOTALL | re.IGNORECASE
    )

    failure_text = failure_match.group(1).strip() if failure_match else ""

    # --- Extract CFG block ---
    cfg_match = re.search(
        r"(?:[#*]+\s*)?(?:Updated\s+CFG\s*\(BNF\))[:\-]*\s*(?:[#*]+\s*)?"
        r"(?:```(?:bnf)?\s*([\s\S]*?)```|([\s\S]*?))"
        r"(?=\n\s*(?:---|[#*]+\s*|\bChanges in CFG\b|\bUpdated CFG Explanation\b|\bTerminal Functions\b|\Z))",
        output_text,
        re.IGNORECASE,
    )
    cfg_explanation = re.search(
        r"(?:[#*]+\s*)?(?:Updated\s+CFG\s+Explanation)[:\-]*\s*(?:[#*]+\s*)?"
        r"([\s\S]*?)(?:\n---|\Z)",
        output_text,
        re.IGNORECASE,
    )
    # --- Extract Terminal Functions block ---
    term_match = re.search(
        r"(?:[#*]+\s*)?Terminal Functions\*\*(.*?)(?:\n---|\Z)",
        output_text,
        re.DOTALL | re.IGNORECASE
    )

    # Handle missing CFG block gracefully
    if not cfg_match:
        print(" No CFG block found in output_text.")
        if failure_text:
            print("\n Extracted Failure Analysis:\n")
            print(failure_text)
        return None, None, None, failure_text, None

    if cfg_match:
        # The regex has two capture groups: one for code blocks, one for non-code blocks
        cfg_text = cfg_match.group(1) if cfg_match.group(1) else (cfg_match.group(2) if len(cfg_match.groups()) > 1 and cfg_match.group(2) else "")
    else:
        cfg_text = ""
    cfg_explanation = cfg_explanation.group(1).strip() if cfg_explanation else ""
    term_text = term_match.group(1).strip() if term_match else ""

    # --- Save CFG to file ---
    os.makedirs(cfg_dir, exist_ok=True)
    filename = f"cfg_updated.txt"
    filepath = os.path.join(cfg_dir, filename)
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(cfg_text)
        f.write(cfg_explanation)
        f.write(failure_text)
        f.write(term_text)

    return None, cfg_text, term_text, failure_text, cfg_explanation


def check_final_functions_exist(experiment_dir: str, terminals: Dict[str, str], 
                                dsl_round: Optional[int] = None,
                                func_evolution_round: Optional[int] = None) -> Tuple[bool, Dict[str, str], List[str]]:
    """Check if all required final functions exist and are valid.
    
    Args:
        experiment_dir: Path to experiment directory
        terminals: Dictionary mapping terminal function names to descriptions
        dsl_round: DSL evolution round number (0-indexed)
        func_evolution_round: Function evolution round number (0-indexed, None for initial)
        
    Returns:
        (all_exist: bool, missing_functions: Dict[str, str], empty_functions: List[str])
        - all_exist: True if all functions exist and are non-empty
        - missing_functions: Dict of functions that don't exist
        - empty_functions: List of function names that exist but are empty or invalid
    """
    final_functions_dir = os.path.join(experiment_dir, "final_functions")
    if not os.path.exists(final_functions_dir):
        return False, terminals.copy(), []
    
    missing = {}
    empty_or_invalid = []
    
    for func_name, description in terminals.items():
        safe_name = sanitize_function_name(func_name)
        
        # Try to find file with new naming scheme first, then fall back to old naming
        func_file = None
        if dsl_round is not None:
            if func_evolution_round is not None:
                func_file = os.path.join(final_functions_dir, f"{safe_name}_dsl{dsl_round}_func{func_evolution_round}.py")
                if not os.path.exists(func_file):
                    func_file = None
            if func_file is None:
                # Try func0 for initial functions
                func_file = os.path.join(final_functions_dir, f"{safe_name}_dsl{dsl_round}_func0.py")
                if not os.path.exists(func_file):
                    # Fallback to old naming without func suffix
                    func_file = os.path.join(final_functions_dir, f"{safe_name}_dsl{dsl_round}.py")
                    if not os.path.exists(func_file):
                        func_file = None
        
        # Fall back to old naming
        if func_file is None:
            func_file = os.path.join(final_functions_dir, f"{safe_name}.py")
        
        if not os.path.exists(func_file):
            missing[func_name] = description
            continue
        
        # Check if file is empty or doesn't contain a function definition
        try:
            with open(func_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if not content:
                empty_or_invalid.append(func_name)
                continue

            base_name, _ = parse_function_name_and_args(func_name)
            base_name_lower = base_name.lower()
            
            # Check for function definition
            has_function_def = (
                f"def {safe_name}(" in content or
                f"def {base_name_lower}(" in content or
                f"def {base_name}(" in content
            )
            
            if not has_function_def:
                empty_or_invalid.append(func_name)
                continue

            lines = [line.strip() for line in content.split('\n') 
                    if line.strip() and not line.strip().startswith('#') 
                    and not (line.strip().startswith('"""') or line.strip().startswith("'''"))]
            
            if len(lines) < 3:
                empty_or_invalid.append(func_name)
                
        except Exception as e:
            print(f"  ⚠ Error reading {func_file}: {e}")
            empty_or_invalid.append(func_name)
    
    all_exist = len(missing) == 0 and len(empty_or_invalid) == 0
    return all_exist, missing, empty_or_invalid


def load_final_functions(experiment_dir: str, terminals: Optional[Dict[str, str]] = None,
                        dsl_round: Optional[int] = None,
                        func_evolution_round: Optional[int] = None) -> Dict[str, str]:
    """Load all final function implementations from directory.
    
    Args:
        experiment_dir: Path to experiment directory
        terminals: Optional dict of terminal functions to validate against
        dsl_round: DSL evolution round number (0-indexed)
        func_evolution_round: Function evolution round number (0-indexed, None for initial)
        
    Returns:
        Dictionary mapping function names (sanitized) to their code
    """
    final_functions_dir = os.path.join(experiment_dir, "final_functions")
    if not os.path.exists(final_functions_dir):
        return {}
    
    functions = {}
    loaded_files = set()
    
    # If terminals provided, only load those functions
    if terminals:
        # Debug: Print what parameters we're using
        print(f"  [DEBUG] load_final_functions called with dsl_round={dsl_round}, func_evolution_round={func_evolution_round}")
        
        for func_name, _ in terminals.items():
            safe_name = sanitize_function_name(func_name)
            
            # Try to find file with exact dsl_round and func_evolution_round
            func_file = None
            if dsl_round is not None:
                # Try specific func_evolution_round if provided and > 0
                if func_evolution_round is not None and func_evolution_round > 0:
                    func_file = os.path.join(final_functions_dir, f"{safe_name}_dsl{dsl_round}_func{func_evolution_round}.py")
                    if os.path.exists(func_file):
                        print(f"  [DEBUG] {func_name}: Found {os.path.basename(func_file)}")
                    else:
                        print(f"  [DEBUG] {func_name}: Tried {os.path.basename(func_file)} - NOT FOUND")
                        func_file = None
                # Try func0 for initial functions (when func_evolution_round is None or 0)
                if func_file is None:
                    func0_file = os.path.join(final_functions_dir, f"{safe_name}_dsl{dsl_round}_func0.py")
                    if os.path.exists(func0_file):
                        func_file = func0_file
                        print(f"  [DEBUG] {func_name}: Found {os.path.basename(func_file)}")
                    else:
                        print(f"  [DEBUG] {func_name}: Tried {os.path.basename(func0_file)} - NOT FOUND")
                # Try dsl{dsl_round} without func suffix (legacy naming)
                if func_file is None:
                    func_file = os.path.join(final_functions_dir, f"{safe_name}_dsl{dsl_round}.py")
                    if os.path.exists(func_file):
                        print(f"  [DEBUG] {func_name}: Found {os.path.basename(func_file)} (legacy naming)")
                    else:
                        print(f"  [DEBUG] {func_name}: Tried {os.path.basename(func_file)} - NOT FOUND")
                        func_file = None
            else:
                print(f"  [DEBUG] {func_name}: dsl_round is None, trying old naming")
            
            # Fall back to old naming (no version suffix)
            if func_file is None:
                func_file = os.path.join(final_functions_dir, f"{safe_name}.py")
                if os.path.exists(func_file):
                    print(f"  [DEBUG] {func_name}: Found {os.path.basename(func_file)} (old naming)")
                else:
                    print(f"  [DEBUG] {func_name}: Tried {os.path.basename(func_file)} - NOT FOUND")
            
            if os.path.exists(func_file):
                try:
                    with open(func_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:  # Only add non-empty files
                            functions[safe_name] = content
                            loaded_files.add(safe_name)
                        else:
                            print(f"  [DEBUG] {func_name}: File {os.path.basename(func_file)} is empty")
                except Exception as e:
                    print(f"  ⚠ Error loading {func_file}: {e}")
    else:
        # Load all .py files in the directory
        for func_file in glob.glob(os.path.join(final_functions_dir, "*.py")):
            func_name = os.path.basename(func_file).replace(".py", "")
            try:
                with open(func_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:  # Only add non-empty files
                        functions[func_name] = content
                        loaded_files.add(func_name)
            except Exception as e:
                print(f"  ⚠ Error loading {func_file}: {e}")
    
    if terminals and len(functions) < len(terminals):
        missing = set(sanitize_function_name(f) for f in terminals.keys()) - loaded_files
        if missing:
            error_msg = (
                f"\n✗ CRITICAL ERROR: Could not load {len(missing)} required function files:\n"
                f"  Missing functions: {sorted(missing)}\n"
                f"  Expected dsl_round={dsl_round}, func_evolution_round={func_evolution_round}\n"
                f"  Final functions directory: {final_functions_dir}\n"
                f"  Pipeline stopped - cannot continue without all required functions."
            )
            print(error_msg)
            raise FileNotFoundError(error_msg)
    
    return functions


def synthesize_and_test_programs(
    experiment_dir: str,
    tasks: List[str],
    cfg_path: str = None,
    terminals: Optional[Dict[str, str]] = None,
    max_attempts: int = 1,
    recipes_path: str = "craft/resources/recipes.yaml",
    hints_path: str = "craft/resources/hints.yaml",
    shared_vllm=None,
    results_tracker=None,
    cfg_version: int = 0,
    func_evolution_round: Optional[int] = None
) -> Dict[str, bool]:
    """Synthesize programs for each task and test if they solve the task.
    
    Returns:
        Dict mapping task_name -> success (bool)
    """
    print(f"\n{'='*80}")
    print("Synthesizing and Testing Programs")
    print(f"{'='*80}")
    
    # Load CFG
    cfg_example = None
    if cfg_path is None:
        cfg_path = os.path.join(experiment_dir, "cfg", "cfg_output.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, 'r') as f:
                cfg_data = json.load(f)
                cfg = cfg_data.get("cfg", "")
                cfg_example = cfg_data.get("example", None)
                # Ensure terminals match CFG (add missing functions from CFG)
                if terminals is None:
                    terminals = cfg_data.get("terminals", {})
                terminals = ensure_terminals_match_cfg(cfg, terminals, shared_vllm=shared_vllm)
        else:
            # Try to find cfg.txt
            cfg_txt_path = os.path.join(experiment_dir, "cfg", "cfg.txt")
            if os.path.exists(cfg_txt_path):
                with open(cfg_txt_path, 'r') as f:
                    cfg = f.read()
            else:
                print("⚠ Warning: Could not find CFG file, using default")
                cfg = ""
    else:
        # Check if it's a JSON file
        if cfg_path.endswith('.json'):
            with open(cfg_path, 'r') as f:
                cfg_data = json.load(f)
                cfg = cfg_data.get("cfg", "")
                cfg_example = cfg_data.get("example", None)
                # Ensure terminals match CFG (add missing functions from CFG)
                if terminals is None:
                    terminals = cfg_data.get("terminals", {})
                terminals = ensure_terminals_match_cfg(cfg, terminals, shared_vllm=shared_vllm)
        else:
            with open(cfg_path, 'r') as f:
                cfg = f.read()
    
    # Load final functions (validate against terminals if provided)
    # Pass dsl_round (cfg_version) and func_evolution_round to load correct version
    # This will raise FileNotFoundError if any required functions are missing
    try:
        final_functions = load_final_functions(
            experiment_dir, 
            terminals=terminals,
            dsl_round=cfg_version,
            func_evolution_round=func_evolution_round
        )
    except FileNotFoundError as e:
        # Re-raise to stop the pipeline
        raise
    
    if not final_functions:
        error_msg = "✗ CRITICAL ERROR: No final functions found! Cannot synthesize programs."
        print(error_msg)
        raise FileNotFoundError(error_msg)
    
    print(f"Loaded {len(final_functions)} final functions")
    
    # Load recipes
    with open(recipes_path, 'r') as f:
        recipes = f.read()
    
    # Use shared vLLM instance if provided, otherwise create new one
    if shared_vllm is not None:
        llm = shared_vllm
        params = SamplingParams(temperature=0.7, max_tokens=25000)
    else:
        try:
            llm = LLM(model="/scratch/avani/gpt", tensor_parallel_size=4)
            params = SamplingParams(temperature=0.7, max_tokens=25000)
        except Exception as e:
            print(f"✗ Error initializing LLM: {e}")
            return {task: False for task in tasks}
    
    def _synthesize_single_task(task: str) -> Tuple[str, bool]:
        """Synthesize and test a program for a single task.
        
        Returns:
            Tuple of (task_name, success)
        """
        print(f"[{task}] Starting program synthesis...")
        
        # Create environment sampler for this task (thread-safe - each task gets its own)
        task_env_sampler = env_factory.EnvironmentFactory(
            recipes_path, hints_path, 7, max_steps=300,
            reuse_environments=True, visualise=False
        )
        test_env = task_env_sampler.sample_environment(task_name=task)
        test_env.reset()
        
        # Get initial grid state
        try:
            markdown = grid_to_markdown(
                test_env._current_state.grid, 
                test_env.world.cookbook, 
                test_env._current_state.pos
            )
        except Exception as e:
            print(f"[{task}] ⚠ Could not generate grid markdown: {e}")
            markdown = "Grid state unavailable"
        
        # Try to synthesize a program that solves the task
        success = False
        programs_tried = []
        
        for attempt in range(max_attempts):
            # first get the env then markdown goes to prompt
            test_env = task_env_sampler.sample_environment(task_name=task)
            test_env.reset()
                
            try:
                markdown = grid_to_markdown(
                    test_env._current_state.grid, 
                    test_env.world.cookbook, 
                    test_env._current_state.pos
                )
            except Exception as e:
                print(f"[{task}] ⚠ Could not generate grid markdown: {e}")
                markdown = "Grid state unavailable"
            try:
                import hashlib
                grid_md5 = hashlib.md5(test_env._current_state.grid.tobytes()).hexdigest()
                print("=== GRID HASH ===")
                print(f"[{task}] grid_md5={grid_md5}", flush=True)
                # Persist grid hash + initial state snapshot for later debugging
                grid_log_path = os.path.join(experiment_dir, "grid_hashes.log")
                os.makedirs(os.path.dirname(grid_log_path), exist_ok=True)
                state = test_env._current_state
                pos = tuple(state.pos) if hasattr(state, "pos") else None
                direction = int(state.dir) if hasattr(state, "dir") else None
                inventory = getattr(state, "inventory", None)
                if inventory is not None:
                    inv_nonzero = [
                        (test_env.world.cookbook.index.get(i, str(i)), float(v))
                        for i, v in enumerate(inventory) if v
                    ]
                else:
                    inv_nonzero = None
                with open(grid_log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"[{task}] attempt={attempt+1} grid_md5={grid_md5} "
                        f"pos={pos} dir={direction} inv={inv_nonzero}\n"
                    )
                    f.write(markdown + "\n\n")
            except Exception as e:
                print(f"[{task}] ⚠ Could not hash grid: {e}")
            # Prepare example program for prompt
            # If no example provided, generate a simple example using available functions
            if cfg_example:
                # Filter example to only include functions that are actually implemented
                example_program = _filter_example_program(cfg_example, terminals)
                # If filtering removed all statements, generate a new example
                if not example_program or not example_program.strip():
                    if terminals and len(terminals) > 0:
                        first_func = list(terminals.keys())[0]
                        example_program = _generate_example_from_terminal(first_func, cfg)
                    else:
                        example_program = " "
            else:
                # Generate a simple example using the first available function from terminals
                if terminals and len(terminals) > 0:
                    first_func = list(terminals.keys())[0]
                    # Generate example from CFG using actual terminal values
                    example_program = _generate_example_from_terminal(first_func, cfg)
                else:
                    # Fallback: use a generic example without hardcoding MOVE
                    example_program = " "
            # if "knife" in task:
            print("markdown", markdown)
            programs_str = "\n".join(programs_tried)
            prompt = f"""
You are a Domain Specific Language (DSL) program generator for the Craft domain. 

### Start State
{markdown}

## Natural Language Description
Craft is a single-agent game in a pre-specified environment. 
The environment of craft is a grid world of size n * n. Each cell can be empty, contain an item, or part of natural terrain or functional structures. When the cell is nonempty, it is considered as blocked. A agent can move around the environment freely through empty cells. At each step, the agent can either move or perform a specific actions, such as collect or craft, towards the immediate cell that it is facing towards. 
At the beginning of each episode, the agent is placed at a starting cell and a distribution of items across the grid is initialized. The agent's tasks involve either collecting primitives (raw resources) or crafting items. A item can only be crafted at the specific workshop mentioned in the recipes. 
The item to be craft are produced from primitives (or other crafted items) by following recipes. Each recipe specifies which items are required and at which workshop the crafting must occur. A primitive item might not need to be crafted but just collected. More complex items, such as arrow, bridge, hammer, axe or flag, require intermediate items along with primitives. This all is specified in the recipe file of the environment. Please note a item can only be crafted at the specific workshop mentioned in the recipes. 
In this domain, primitives may sometimes be blocked by obstacles. Obstacles are entities that are part of the recipe but are not primitives, workshops, or boundaries. To reach the blocked primitives, the agent must identify and use appropriate tools to remove or bypass these obstacles.
The correspondence between tools and obstacles is not predefined or known a priori. It cannot be inferred from real-world knowledge or semantic associations. Instead, the correct relationships must be discovered empirically through exploration and interaction within the environment, by observing which tools succeed or fail when applied to different obstacles.
Primitives used to craft an item has no relation to it being the tool that helps pass an obstacle.

## Available Recipes
Here are the recipes for the domain:
{recipes}

## Context Free Grammar (CFG)
Here is the context-free grammar (CFG) that defines the DSL. Strictly follow this CFG when synthesising programs:

{cfg}

## Task
Generate a program that solves the following task:

**{task}**

## Output Format Instructions
Return ONLY the program string delimited by $ signs. Do not include any explanations, comments, or additional text outside the $ delimiters.
Example output ->
${example_program}$

## Previous programs that FAILED to solve the task:
{programs_str}

These programs are syntactically correct but did not solve the task. When generating a new program, avoid repeating the mistakes made in these failed programs, and generate semantically different programs.

Also always ensure that the the information provided in this prompt is facts and always correct and cannot be changed so please adhere to it strictly.
##Return a program that is able to solve the task that is different from the previous failed programs.
"""
            
            try:
                conversation = [{"role": "user", "content": prompt, "chat_template_kwargs": {"reasoning_effort": "high"}}]
                output = llm.chat(conversation, params)
                response = output[0].outputs[0].text
                
                # Extract program from response
                marker_match = re.search(r'assistantfinal', response, re.IGNORECASE)
                search_target = response
                if marker_match:
                    search_target = response[marker_match.end():]
                
                program_match = re.search(r'\$(.*?)\$', search_target, re.DOTALL)
                if not program_match:
                    continue
                
                program = program_match.group(1).strip()
                

                # Use CFG evaluator to test the program
                if CFG_EVALUATOR_AVAILABLE:
                    # Create a temporary directory with only the correct version functions
                    # This ensures CFGEvaluator loads the correct versions
                    import tempfile
                    import shutil
                    temp_func_dir = tempfile.mkdtemp(prefix="test_functions_")
                    final_functions_dir = os.path.join(experiment_dir, "final_functions")
                    
                    # Copy only the correct version files to temp directory
                    for func_name, func_code in final_functions.items():
                        # Write the function code to a temporary file
                        safe_name = sanitize_function_name(func_name)
                        temp_func_file = os.path.join(temp_func_dir, f"{safe_name}.py")
                        with open(temp_func_file, 'w', encoding='utf-8') as f:
                            f.write(func_code)
                    
                    try:
                        evaluator = CFGEvaluator(
                            cfg=cfg,
                            final_functions_dir=temp_func_dir
                        )
                    finally:
                          print()
                    #     # Clean up temporary directory
                    #     shutil.rmtree(temp_func_dir, ignore_errors=True)
                    
                    # Evaluate program with environment passed directly
                    result = evaluator.evaluate_program(program, env=test_env, max_steps=300)
                    print(f"[{task}] Result: {result}")
                    success = result.get("success", False)
                    reward = result.get("total_reward", 0.0)
                    steps = result.get("steps", 0)  # Number of environment steps taken
                else:
                    # Fallback: simple test
                    print(f"[{task}] ⚠ CFGEvaluator not available, using fallback")
                    success = False
                    reward = 0.0
                    steps = 0
                
                # Track result if tracker is available
                if results_tracker is not None:
                    results_tracker.add_program_synthesis_result(
                        task=task,
                        cfg_version=cfg_version,
                        program=program,
                        reward=reward,
                        steps=steps,
                        func_evolution_round=func_evolution_round,
                        success=success
                    )
                
                if success:
                    print(f"[{task}] ✓ Task solved with program: {program}")
                    return (task, True)
                else:
                    programs_tried.append(program)
                    print(f"[{task}] Attempt {attempt + 1}: Program failed - {program}")
                    
            except Exception as e:
                print(f"[{task}] ⚠ Error in attempt {attempt + 1}: {e}")
                continue
        
        if not success:
            print(f"[{task}] ✗ Could not be solved after {max_attempts} attempts")
            return (task, False)
    
    # Run program synthesis sequentially for all tasks
    print(f"\nRunning program synthesis sequentially for {len(tasks)} tasks...")
    
    task_results = {}
    
    # Process tasks sequentially (one at a time)
    for task in tasks:
        task_name, success = _synthesize_single_task(task)
        task_results[task_name] = success
    
    return task_results


def normalize_to_two_spaces(body_lines):

    merged = "\n".join(body_lines)

    # Remove common leading indentation
    dedented = textwrap.dedent(merged)

    # Re-indent with exactly 2 spaces
    return textwrap.indent(dedented, "  ")
def skip_docstring(lines):
    if not lines:
        return lines
    
    line = lines[0].strip()
    
    # Detect starting docstring
    if line.startswith('"""') or line.startswith("'''"):
        quote = line[:3]  # Either """ or '''
        
        # Case 1: single-line docstring  """text"""
        if line.count(quote) >= 2:
            return lines[1:]  # Skip the first line only
        
        # Case 2: multi-line docstring
        for i in range(1, len(lines)):
            if quote in lines[i]:
                return lines[i+1:]  # Skip through closing line
    
        # Edge case: docstring never closed (invalid Python)
        return []  

    return lines 



def evolve_functions_with_failing_tasks(
    experiment_dir: str,
    failing_tasks: List[str],
    terminals: Dict[str, str],
    specification: str,
    spec_file: str = "",
    cfg: str = "",
    max_evolutions: int = 1,
    shared_vllm=None,
    dsl_round: Optional[int] = None,
    func_evolution_round: Optional[int] = None,
    total_samples: int = 1000
) -> bool:
    """Evolve functions by reusing the first round's domain-template prompt and seeding
    funsearch with the final function from the previous round.
    
    Instead of modifying evaluate() to test on failing tasks, this simply:
    1. Reuses the first round's prompt file (e.g., craft_dsl0_func0.txt) which already
       has the domain template with grid specs and pass_checks
    2. Updates the func_init file with the body from the last round's final function
    3. Runs funsearch + explicit feedback with this combination
    
    Args:
        experiment_dir: Path to experiment directory
        failing_tasks: List of task names that failed (kept for API compat, logged but not used)
        terminals: Dictionary of terminal functions
        specification: Specification string for funsearch
        cfg: CFG string
        max_evolutions: Number of evolution rounds (currently 1 per call)
        shared_vllm: Optional shared vLLM instance
        dsl_round: DSL evolution round number
        func_evolution_round: Function evolution round number
        total_samples: Total samples for funsearch
        
    Returns:
        True if evolution succeeded and new functions were generated, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"Evolving Functions (domain template + last round's final function)")
    print(f"{'='*80}")
    print(f"Failing tasks (for reference): {failing_tasks}")
    
    # Import necessary functions from cfg_to_funsearch_pipeline
    from src.pipeline.cfg_to_funsearch_pipeline import (
        generate_func_init, determine_inputs,
        run_explicit_feedback_generation, sanitize_function_name, parse_function_name_and_args,
        extract_function_args
    )
    from funsearch.implementation.funsearch import FunSearch
    from funsearch.implementation import config as config_lib
    
    func_prompts_dir = os.path.join(experiment_dir, "function_specific_prompts")
    if not os.path.exists(func_prompts_dir):
        print("  ✗ Function prompts directory not found")
        return False
    
    # Step 1: Reuse the first round's prompt file (domain template with grid specs)
    print(f"\n  [Step 1] Locating first round prompt files to reuse...")
    updated_prompts = []
    func_files = {}
    func_init_files = {}
    
    for func_name, description in terminals.items():
        safe_name = sanitize_function_name(func_name)
        
        # Find the first round's prompt file (func0)
        prompt_file = None
        if dsl_round is not None:
            # Try func0 first (the initial domain-template-based prompt)
            func0_file = os.path.join(func_prompts_dir, f"{safe_name}_dsl{dsl_round}_func0.txt")
            if os.path.exists(func0_file):
                prompt_file = func0_file
            else:
                # Try without func suffix
                dsl_only = os.path.join(func_prompts_dir, f"{safe_name}_dsl{dsl_round}.txt")
                if os.path.exists(dsl_only):
                    prompt_file = dsl_only
        
        # Fallback to old naming
        if prompt_file is None:
            old_name = os.path.join(func_prompts_dir, f"{safe_name}.txt")
            if os.path.exists(old_name):
                prompt_file = old_name
        
        if prompt_file is None:
            print(f"    ✗ No prompt file found for {func_name}, skipping")
            continue
        
        func_files[func_name] = prompt_file
        updated_prompts.append(func_name)
        print(f"    ✓ Reusing prompt: {os.path.basename(prompt_file)}")
    
    if not updated_prompts:
        print("  ✗ No prompt files found")
        return False
    
    print(f"\n  ✓ Found {len(updated_prompts)} prompt files to reuse")
    
    # Step 2: Load final functions from previous round to use as func_init seed
    print(f"\n  [Step 2] Loading final functions from previous round as seed...")
    final_functions_dir = os.path.join(experiment_dir, "final_functions")
    current_final_functions = {}
    
    for func_name in updated_prompts:
        safe_name = sanitize_function_name(func_name)
        
        # Try to find the best final function from previous rounds
        final_func_file = None
        if dsl_round is not None:
            # Try previous func evolution round
            if func_evolution_round is not None and func_evolution_round > 0:
                prev_file = os.path.join(final_functions_dir, f"{safe_name}_dsl{dsl_round}_func{func_evolution_round - 1}.py")
                if os.path.exists(prev_file):
                    final_func_file = prev_file
            # Try func0 (initial round)
            if final_func_file is None:
                func0_file = os.path.join(final_functions_dir, f"{safe_name}_dsl{dsl_round}_func0.py")
                if os.path.exists(func0_file):
                    final_func_file = func0_file
        
        # Fallback to old naming
        if final_func_file is None:
            old_file = os.path.join(final_functions_dir, f"{safe_name}.py")
            if os.path.exists(old_file):
                final_func_file = old_file
        
        if final_func_file and os.path.exists(final_func_file):
            with open(final_func_file, 'r', encoding='utf-8') as f:
                final_func_content = f.read().strip()
            if final_func_content:
                current_final_functions[func_name] = final_func_content
                print(f"    ✓ Loaded previous implementation for {func_name} from {os.path.basename(final_func_file)}")
        else:
            print(f"    ⚠ No previous implementation found for {func_name}")
    
    # Step 3: Create func_init files seeded with previous round's final function
    print(f"\n  [Step 3] Creating func_init files with previous round's implementation...")
    
    for func_name in updated_prompts:
        safe_name = sanitize_function_name(func_name)
        base_name, _ = parse_function_name_and_args(func_name)
        base_name_lower = base_name.lower()
        
        if dsl_round is not None and func_evolution_round is not None:
            func_init_file = os.path.join(experiment_dir, "functions_generated",
                                          f"{safe_name}_dsl{dsl_round}_func{func_evolution_round}_func_init.py")
        else:
            func_init_file = os.path.join(experiment_dir, "functions_generated",
                                          f"{safe_name}_func_init.py")
        
        if func_name in current_final_functions:
            final_func_content = current_final_functions[func_name]
            func_lines = final_func_content.split('\n')
            
            # Find function definition line
            func_start_idx = None
            for i, line in enumerate(func_lines):
                if (f"def {safe_name}(" in line or
                    f"def {base_name_lower}(" in line or
                    f"def {base_name}(" in line):
                    func_start_idx = i
                    break
            
            if func_start_idx is not None:
                # Find colon
                colon_idx = None
                for i in range(func_start_idx, len(func_lines)):
                    if ':' in func_lines[i]:
                        colon_idx = i
                        break
                
                if colon_idx is not None:
                    body_start_idx = colon_idx + 1
                    func_indent = len(func_lines[func_start_idx]) - len(func_lines[func_start_idx].lstrip())
                    body_end_idx = len(func_lines)
                    
                    for i in range(body_start_idx, len(func_lines)):
                        line = func_lines[i]
                        if line.strip():
                            line_indent = len(line) - len(line.lstrip())
                            if ((line.strip().startswith('def ') or line.strip().startswith('class '))
                                and line_indent <= func_indent):
                                body_end_idx = i
                                break
                    
                    body_lines = func_lines[body_start_idx:body_end_idx]
                    body_lines = skip_docstring(body_lines)
                    if not body_lines:
                        body_lines = ["  pass"]
                    body_str = normalize_to_two_spaces(body_lines)
                    
                    # Read existing func_init to preserve signature
                    existing_init = None
                    for candidate in [
                        os.path.join(experiment_dir, "functions_generated", f"{safe_name}_dsl{dsl_round}_func0_func_init.py"),
                        os.path.join(experiment_dir, "functions_generated", f"{safe_name}_dsl{dsl_round}_func_init.py"),
                    ]:
                        if os.path.exists(candidate):
                            with open(candidate, 'r', encoding='utf-8') as f:
                                existing_init = f.read().split('\n')
                            break
                    
                    if existing_init:
                        first_line = existing_init[0]
                        updated_content = first_line + '\n' + body_str
                    else:
                        updated_content = func_lines[func_start_idx] + '\n' + body_str
                    
                    os.makedirs(os.path.dirname(func_init_file), exist_ok=True)
                    with open(func_init_file, 'w', encoding='utf-8') as f:
                        f.write(updated_content)
                    print(f"    ✓ Created func_init for {func_name} seeded with previous implementation")
                else:
                    func_init_file = generate_func_init(func_name, terminals[func_name], cfg,
                                                        experiment_dir=experiment_dir,
                                                        dsl_round=dsl_round, func_evolution_round=func_evolution_round)
                    print(f"    ⚠ Could not parse function, generated stub for {func_name}")
            else:
                func_init_file = generate_func_init(func_name, terminals[func_name], cfg,
                                                    experiment_dir=experiment_dir,
                                                    dsl_round=dsl_round, func_evolution_round=func_evolution_round)
                print(f"    ⚠ Could not find function def, generated stub for {func_name}")
        else:
            # No previous implementation, generate stub
            func_init_file = generate_func_init(func_name, terminals[func_name], cfg,
                                                experiment_dir=experiment_dir,
                                                dsl_round=dsl_round, func_evolution_round=func_evolution_round)
            print(f"    ⚠ No previous implementation, generated stub for {func_name}")
        
        func_init_files[func_name] = func_init_file
    
    # Step 4: Run FunSearch
    print(f"\n  [Step 4] Running funsearch for {len(updated_prompts)} functions...")
    
    # Use shared vLLM instance if provided, otherwise create new one
    # Track if we need to clean up between FunSearch and explicit feedback
    funsearch_vllm_instance = None
    if shared_vllm is None:
        try:
            if vLLM is not None:
                print("  Creating vLLM instance for funsearch (will be cleaned up before explicit feedback)...")
                shared_vllm = vLLM(
                    model="/scratch/avani/gpt", 
                    tensor_parallel_size=4,
                    gpu_memory_utilization=0.6  # Reduced to 60% to handle parallel jobs
                )
                funsearch_vllm_instance = shared_vllm  # Track this instance for cleanup
                print("  ✓ Created vLLM instance for funsearch")
            else:
                print("  ⚠ vLLM not available, funsearch will create its own instance")
        except Exception as e:
            print(f"  ⚠ Could not create vLLM instance: {e}")
            shared_vllm = None
    else:
        print("  ✓ Using provided shared vLLM instance for funsearch")
    
    # Configure FunSearch with parallelization
    # Match evaluators to samples_per_prompt for clean parallelization
    regen_attempts = int(os.environ.get("GRID_REGENERATION_ATTEMPTS", 5))
    config = config_lib.Config(
        num_samplers=1,  # Single sampler - generates samples_per_prompt samples per iteration
        num_evaluators=2,  # Match samples_per_prompt - each evaluator handles one sample
        samples_per_prompt=2,  # 2 samples per prompt
        total_samples=total_samples,  # Use provided total_samples parameter (default: 1000)
        programs_database=config_lib.ProgramsDatabaseConfig(),
        grid_regeneration_attempts=regen_attempts,
    )
    
    results_dir = os.path.join(experiment_dir, "results", "funsearch")
    os.makedirs(results_dir, exist_ok=True)
    
    funsearch_results = {}
    # Remove "LOOK" from updated_prompts if present (exclude from re-running funsearch)
    if "LOOK" in updated_prompts:
        updated_prompts.remove("LOOK")
    elif "look" in updated_prompts:
        updated_prompts.remove("look")
    
    # (func_init files already prepared in Step 3 above - old duplicate block removed)
    _skip_old_block = True
    for func_name in []:  # OLD LOOP - SKIP
        safe_name = sanitize_function_name(func_name)
        if dsl_round is not None:
            if func_evolution_round is not None:
                func_init_file = os.path.join(experiment_dir, "functions_generated", f"{safe_name}_dsl{dsl_round}_func{func_evolution_round}_func_init.py")
            else:
                # Initial round: use func0
                func_init_file = os.path.join(experiment_dir, "functions_generated", f"{safe_name}_dsl{dsl_round}_func0_func_init.py")
        else:
            # Fallback to old naming if rounds not provided
            func_init_file = os.path.join(experiment_dir, "functions_generated", f"{safe_name}_func_init.py")
        base_name, _ = parse_function_name_and_args(func_name)
        base_name_lower = base_name.lower()
        
        if func_name in current_final_functions:
            # Extract function definition from final function file
            final_func_content = current_final_functions[func_name]
            
            try:
                func_lines = final_func_content.split('\n')
                
                # Find function definition line
                func_start_idx = None
                for i, line in enumerate(func_lines):
                    if (f"def {safe_name}(" in line or 
                        f"def {base_name_lower}(" in line or 
                        f"def {base_name}(" in line):
                        func_start_idx = i
                        break
                
                if func_start_idx is None:
                    raise ValueError(f"Could not find function definition for {func_name}")
                
                # Find the colon after function signature
                colon_idx = None
                for i in range(func_start_idx, len(func_lines)):
                    if ':' in func_lines[i]:
                        colon_idx = i
                        break
                
                if colon_idx is None:
                    raise ValueError(f"Could not find colon after function signature")
                
                # Body starts after the colon
                body_start_idx = colon_idx + 1

                
                # Find the end of the function by looking for next def/class at same or less indentation
                func_indent = len(func_lines[func_start_idx]) - len(func_lines[func_start_idx].lstrip())
                body_end_idx = len(func_lines)
                
                for i in range(body_start_idx, len(func_lines)):
                    line = func_lines[i]
                    if line.strip():
                        line_indent = len(line) - len(line.lstrip())
                        # If we find a def or class at same or less indentation, we've reached the end
                        if ((line.strip().startswith('def ') or line.strip().startswith('class ')) 
                            and line_indent <= func_indent):
                            body_end_idx = i
                            break
                        # If we find a line at less indentation than function, we're done
                        if line_indent < func_indent:
                            body_end_idx = i
                            break
                
                # Extract body lines (from after colon to end of function)

                body_lines = func_lines[body_start_idx:body_end_idx]
                body_lines = skip_docstring(body_lines)
            
                if not body_lines:
                    body_lines = ["  pass"]
                
                body_lines = normalize_to_two_spaces(body_lines)

                
                # Read existing func_init file to preserve signature
                # Try to find existing file with previous round's naming if needed
                existing_func_init_file = func_init_file
                if not os.path.exists(existing_func_init_file) and dsl_round is not None:
                    if func_evolution_round is not None and func_evolution_round > 0:
                        # Try previous round's file
                        prev_func_init_file = os.path.join(experiment_dir, "functions_generated", f"{safe_name}_dsl{dsl_round}_func{func_evolution_round - 1}_func_init.py")
                        if os.path.exists(prev_func_init_file):
                            existing_func_init_file = prev_func_init_file
                    # Try func0 for initial round (if current file doesn't exist)
                    if not os.path.exists(existing_func_init_file):
                        func0_file = os.path.join(experiment_dir, "functions_generated", f"{safe_name}_dsl{dsl_round}_func0_func_init.py")
                        if os.path.exists(func0_file):
                            existing_func_init_file = func0_file
                    # Fallback to old naming without func suffix
                    if not os.path.exists(existing_func_init_file):
                        old_naming_file = os.path.join(experiment_dir, "functions_generated", f"{safe_name}_dsl{dsl_round}_func_init.py")
                        if os.path.exists(old_naming_file):
                            existing_func_init_file = old_naming_file
                
                if os.path.exists(existing_func_init_file):
                    with open(existing_func_init_file, 'r', encoding='utf-8') as f:
                        existing_lines = f.read().split('\n')
                    
                    # Keep the first line (function signature) and append body lines
                    # body_lines is a list, need to join with newlines
                    body_lines_str = '\n'.join(body_lines) if isinstance(body_lines, list) else str(body_lines)
                    
                    if existing_lines:
                        first_line = existing_lines[0]
                        updated_content = first_line + '\n' + body_lines_str
                    else:
                        # No existing file, use body_lines directly (but need function signature)
                        # Extract signature from func_lines
                        func_signature_line = func_lines[func_start_idx]
                        updated_content = func_signature_line + '\n' + body_lines_str
                    
                    # Verify the content before writing
                    if not updated_content or updated_content.strip() == '':
                        print(f"    ⚠ WARNING: Updated content is empty for {func_name}!")
                        raise ValueError("Updated content is empty")
                    
                    print(f"    ✓ Prepared func_init content ({len(updated_content)} chars)")
                    
                    # Ensure directory exists
                    os.makedirs(os.path.dirname(func_init_file), exist_ok=True)
                    
                    # If we found a previous round's file, we'll create a new one with current round's name
                    # Otherwise, version the existing file
                    if existing_func_init_file != func_init_file:
                        # Create new file with current round's name
                        with open(func_init_file, 'w', encoding='utf-8') as f:
                            f.write(updated_content)
                        
                        # Verify the file was written correctly - STOP PIPELINE if not updated
                        with open(func_init_file, 'r', encoding='utf-8') as f:
                            written_content = f.read()
                            if written_content.strip() == '':
                                error_msg = f"✗ CRITICAL: func_init file for {func_name} is empty after write! Pipeline stopped."
                                print(f"    {error_msg}")
                                raise RuntimeError(error_msg)
                            elif 'return []' in written_content and len(written_content.strip()) < 50:
                                error_msg = f"✗ CRITICAL: func_init file for {func_name} only contains stub (return [])! File was not updated correctly. Pipeline stopped."
                                print(f"    {error_msg}")
                                print(f"    File content: {repr(written_content[:200])}")
                                raise RuntimeError(error_msg)
                            else:
                                print(f"    ✓ Verified: File contains implementation ({len(written_content)} chars)")
                        
                        print(f"    ✓ Created func_init for {func_name} with current implementation (from previous round)")
                    else:
                        # Version the file before updating
                        version_file(func_init_file, keep_original=False)
                        
                        # Write updated function to func_init file
                        with open(func_init_file, 'w', encoding='utf-8') as f:
                            f.write(updated_content)
                        
                        # Verify the file was written correctly - STOP PIPELINE if not updated
                        with open(func_init_file, 'r', encoding='utf-8') as f:
                            written_content = f.read()
                            if written_content.strip() == '':
                                error_msg = f"✗ CRITICAL: func_init file for {func_name} is empty after write! Pipeline stopped."
                                print(f"    {error_msg}")
                                raise RuntimeError(error_msg)
                            elif 'return []' in written_content and len(written_content.strip()) < 50:
                                error_msg = f"✗ CRITICAL: func_init file for {func_name} only contains stub (return [])! File was not updated correctly. Pipeline stopped."
                                print(f"    {error_msg}")
                                print(f"    File content: {repr(written_content[:200])}")
                                raise RuntimeError(error_msg)
                            else:
                                print(f"    ✓ Verified: File contains implementation ({len(written_content)} chars)")
                        
                        print(f"    ✓ Updated func_init body for {func_name} (preserved signature, previous version saved)")
                else:
                    # No existing file, write full function (skip docstrings)
                    func_code_lines = func_lines[func_start_idx:body_end_idx]
                    func_code = '\n'.join(func_code_lines)
                    
                    # Verify content before writing
                    if not func_code or func_code.strip() == '':
                        print(f"    ⚠ WARNING: Extracted function code is empty for {func_name}!")
                        raise ValueError("Extracted function code is empty")
                    
                    # Ensure directory exists
                    os.makedirs(os.path.dirname(func_init_file), exist_ok=True)
                    
                    with open(func_init_file, 'w', encoding='utf-8') as f:
                        f.write(func_code)
                    
                    # Verify the file was written correctly - STOP PIPELINE if not updated
                    with open(func_init_file, 'r', encoding='utf-8') as f:
                        written_content = f.read()
                        if written_content.strip() == '':
                            error_msg = f"✗ CRITICAL: func_init file for {func_name} is empty after write! Pipeline stopped."
                            print(f"    {error_msg}")
                            raise RuntimeError(error_msg)
                        elif 'return []' in written_content and len(written_content.strip()) < 50:
                            error_msg = f"✗ CRITICAL: func_init file for {func_name} only contains stub (return [])! File was not updated correctly. Pipeline stopped."
                            print(f"    {error_msg}")
                            print(f"    File content: {repr(written_content[:200])}")
                            raise RuntimeError(error_msg)
                        else:
                            print(f"    ✓ Verified: File contains function implementation ({len(written_content)} chars)")
                    
                    print(f"    ✓ Created func_init for {func_name} with current implementation")
            except Exception as e:
                print(f"    ⚠ Error extracting function definition for {func_name}: {e}")
                # Fallback: generate stub
                func_init_file = generate_func_init(func_name, terminals[func_name], cfg, experiment_dir=experiment_dir,
                                                   dsl_round=dsl_round, func_evolution_round=func_evolution_round)
        else:
            # No current implementation found, generate stub if file doesn't exist
            if not os.path.exists(func_init_file):
                func_init_file = generate_func_init(func_name, terminals[func_name], cfg, experiment_dir=experiment_dir,
                                                   dsl_round=dsl_round, func_evolution_round=func_evolution_round)
                print(f"    ⚠ No current implementation found, generated stub for {func_name}")
            else:
                print(f"    ⚠ Using existing func_init for {func_name} (no update available)")
        
        func_init_files[func_name] = func_init_file
    
    # Run FunSearch in parallel for all functions
    print(f"\n  [Step 4.1] Running FunSearch in parallel for {len(updated_prompts)} functions...")
    max_workers = min(len(updated_prompts), 16)
    print(f"    Using {max_workers} parallel workers")
    
    # Helper function to run FunSearch for a single function
    def run_funsearch_for_function_evolution(func_name, func_file, func_init_file):
        """Run FunSearch for a single function during evolution."""
        try:
            print(f"    [{func_name}] Starting FunSearch evolution...")
            funsearch = FunSearch(model_type="huggingface", shared_vllm=shared_vllm)
            inputs = determine_inputs(func_name, terminals[func_name], cfg)
            
            funsearch.run(
                specification=specification,
                inputs=inputs,
                config=config,
                function_to_implement=func_file,
                function_init=func_init_file,
                spec_file=spec_file if spec_file else None,
                experiment_dir=results_dir
            )
            print(f"    [{func_name}] ✓ Completed FunSearch evolution")
            return func_name, "success", None
        except Exception as e:
            error_msg = str(e)
            print(f"    [{func_name}] ✗ Error: {error_msg}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return func_name, "error", error_msg
    
    # Run FunSearch in parallel
    errors = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_func = {
            executor.submit(
                run_funsearch_for_function_evolution,
                func_name,
                func_files[func_name],
                func_init_files[func_name]
            ): func_name
            for func_name in updated_prompts
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_func):
            func_name, status, error = future.result()
            funsearch_results[func_name] = status
            if error:
                errors[func_name] = error
    
    # Check for errors
    if errors:
        print(f"\n  ✗ FunSearch failed for {len(errors)} function(s):")
        for func_name, error in errors.items():
            print(f"    - {func_name}: {error}")
        print("  ✗ Pipeline stopped due to FunSearch failures")
        raise RuntimeError(f"FunSearch failed for functions: {list(errors.keys())}")
    
    print(f"\n  ✓ All {len(updated_prompts)} functions completed FunSearch successfully")
    
    # If we created a separate instance for FunSearch, clean it up before explicit feedback
    # This ensures we free GPU memory before explicit feedback creates its own instance
    if funsearch_vllm_instance is not None and funsearch_vllm_instance == shared_vllm:
        try:
            print("\n  [Cleanup] Freeing FunSearch vLLM instance before explicit feedback...")
            # Store reference before clearing
            temp_vllm = shared_vllm
            shared_vllm = None  # Clear shared reference so explicit feedback creates its own
            del temp_vllm
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            print("  ✓ FunSearch vLLM instance freed - explicit feedback will create its own")
        except Exception as cleanup_error:
            print(f"  ⚠ Warning: Error cleaning up FunSearch instance: {cleanup_error}")
    
    # Re-run explicit feedback generation for successfully updated functions (30 iterations)
    print(f"\n  [Step 4.2] Re-running explicit feedback generation (30 iterations)...")
    
    explicit_feedback_dir = os.path.join(experiment_dir, "explicit_feedback")
    os.makedirs(explicit_feedback_dir, exist_ok=True)
    
    final_functions = {}
    func_signatures = {}
    
    # Get function signatures (reconstruct from function names)
    for func_name in updated_prompts:
        base_name, args_list = parse_function_name_and_args(func_name)
        safe_name = sanitize_function_name(func_name)
        args = extract_function_args(func_name, cfg)
        if args and args != "arg":
            func_signatures[func_name] = f"def {safe_name}(env, {args})"
        else:
            func_signatures[func_name] = f"def {safe_name}(env)"
    
    NUM_EXPLICIT_FEEDBACK_ITERATIONS = 30
    
    for func_name in updated_prompts:
        if funsearch_results.get(func_name) != "success":
            print(f"    Skipping explicit feedback for {func_name} (funsearch failed)")
            continue
        
        print(f"\n    --- Re-running explicit feedback for {func_name} (30 iterations) ---")
        func_file = func_files[func_name]
        current_func_file = func_file
        
        try:
            # Run explicit feedback iteratively (30 times)
            current_func_code = None
            temp_files_to_cleanup = []
            
            # Read initial function code
            with open(func_file, 'r', encoding='utf-8') as f:
                current_func_code = f.read()
            
            for iteration in range(NUM_EXPLICIT_FEEDBACK_ITERATIONS):
                print(f"      Explicit feedback iteration {iteration + 1}/{NUM_EXPLICIT_FEEDBACK_ITERATIONS} for {func_name}...")
                
                # Use temporary file for this iteration
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp_file:
                    tmp_file.write(current_func_code)
                    tmp_file_path = tmp_file.name
                    temp_files_to_cleanup.append(tmp_file_path)
                
                try:
                    final_func = run_explicit_feedback_generation(
                        func_name, results_dir, func_file, experiment_dir, explicit_feedback_dir,
                        specification, k=5, shared_vllm=shared_vllm, 
                        func_signature=func_signatures.get(func_name, ""),
                        dsl_round=dsl_round, func_evolution_round=func_evolution_round
                    )
                    if final_func:
                        current_func_code = final_func  # Update for next iteration
                    else:
                        print(f"        ⚠ No improvement in iteration {iteration + 1}")
                finally:
                    # Clean up temporary file immediately
                    try:
                        os.remove(tmp_file_path)
                        if tmp_file_path in temp_files_to_cleanup:
                            temp_files_to_cleanup.remove(tmp_file_path)
                    except OSError:
                        pass
            
            if final_func:
                final_functions[func_name] = final_func
                print(f"    ✓ Completed {NUM_EXPLICIT_FEEDBACK_ITERATIONS} iterations of explicit feedback for {func_name}")
            else:
                print(f"    ⚠ No final function extracted for {func_name} after {NUM_EXPLICIT_FEEDBACK_ITERATIONS} iterations")
        except Exception as e:
            print(f"    ✗ Error running explicit feedback for {func_name}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
        finally:
            # Clean up any remaining temporary files
            for tmp_file_path in temp_files_to_cleanup:
                try:
                    os.remove(tmp_file_path)
                except OSError:
                    pass
    
    # Save updated final functions with versioning
    if final_functions:
        print(f"\n  [Step 4.3] Saving {len(final_functions)} updated final functions...")
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
            
            # Version existing file if it exists (only if using old naming)
            if dsl_round is None and os.path.exists(func_file):
                version_file(func_file, keep_original=False)
            
            with open(func_file, 'w', encoding='utf-8') as f:
                f.write(func_code)
            print(f"    ✓ Saved {func_name} to {os.path.basename(func_file)}")
        
        print(f"\n  ✓ Function evolution completed: {len(final_functions)} functions updated")
        return True
    else:
        print(f"\n  ✗ No final functions were generated from evolution")
        return False


def test_cfg_on_tasks(
    experiment_dir: str,
    tasks: List[str],
    cfg: str,
    terminals: Dict[str, str],
    recipes_path: str = "craft/resources/recipes.yaml",
    hints_path: str = "craft/resources/hints.yaml",
    max_attempts: int = 1,
    shared_vllm=None,
    results_tracker=None,
    cfg_version: int = 0,
    func_evolution_round: Optional[int] = None
) -> Dict[str, bool]:
    """Test the current CFG and functions on the given tasks.
    
    Args:
        experiment_dir: Experiment directory
        tasks: List of tasks to test
        cfg: CFG string
        terminals: Dictionary of terminal functions
        recipes_path: Path to recipes file
        hints_path: Path to hints file
        shared_vllm: Optional shared vLLM instance
        
    Returns:
        Dictionary mapping task_name -> success (bool)
    """
    print(f"\n{'='*80}")
    print("Testing CFG on Tasks")
    print(f"{'='*80}")
    
    cfg_path = os.path.join(experiment_dir, "cfg", "cfg_output.json")
    
    # Test programs using synthesize_and_test_programs
    task_results = synthesize_and_test_programs(
        experiment_dir, tasks, cfg_path=cfg_path, terminals=terminals,
        recipes_path=recipes_path, hints_path=hints_path, max_attempts=max_attempts,
        shared_vllm=shared_vllm, results_tracker=results_tracker, cfg_version=cfg_version,
        func_evolution_round=func_evolution_round
    )
    
    print(f"\nTask Results:")
    for task, success in task_results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {task}")
    
    return task_results


def evolve_dsl(
    experiment_dir: str,
    failing_tasks: List[str],
    cfg: str,
    recipes: str,
    terminals: Dict[str, str],
    shared_vllm=None
) -> Tuple[str, Dict[str, str], bool]:
    """Evolve the DSL based on failing tasks (without implementing).
    
    This function only evolves the DSL and returns the new CFG and terminals.
    It does NOT implement the CFG - that should be done separately.
    
    Args:
        experiment_dir: Experiment directory
        failing_tasks: List of tasks that failed
        cfg: Current CFG string
        recipes: Recipes string
        terminals: Current terminal functions
        shared_vllm: Optional shared vLLM instance
        
    Returns:
        Tuple of (new_cfg: str, new_terminals: Dict[str, str], success: bool)
    """
    print(f"\n{'='*80}")
    print("Evolving DSL Based on Failing Tasks")
    print(f"{'='*80}")
    print(f"Failing tasks: {failing_tasks}")
    
    # Use shared vLLM instance if provided, otherwise create new one
    if shared_vllm is not None:
        llm = shared_vllm
        params = SamplingParams(temperature=0.7, max_tokens=25000)
    else:
        try:
            llm = LLM(model="/scratch/avani/gpt", tensor_parallel_size=4)
            params = SamplingParams(temperature=0.7, max_tokens=25000)
        except Exception as e:
            print(f"✗ Error initializing LLM: {e}")
            return cfg, terminals, False
    
    # Get failure analysis
    failure_analysis_prompt = f"""
Here are the tasks that failed to be solved:
{failing_tasks}

Give me top three reasons why the DSL might be failing to solve these tasks in bullet points.
"""
    
    try:
        conversation = [{"role": "user", "content": failure_analysis_prompt, "chat_template_kwargs": {"reasoning_effort": "high"}}]
        output = llm.chat(conversation, params)
        failure_analysis = output[0].outputs[0].text
        
        marker_match = re.search(r'assistantfinal', failure_analysis, re.IGNORECASE)
        if marker_match:
            failure_analysis = failure_analysis[marker_match.end():]
        
        print("Failure analysis:", failure_analysis)
        
        # Evolve CFG
        cfg_evolution_prompt = f"""
The following is the failure analysis for the unsuccessful DSL programs:

{failure_analysis}

Use this failure analysis to improve the current CFG for the DSL in order to synthesise better programs that can solve the tasks: {failing_tasks}.
---

### Current CFG (Context-Free Grammar) for the current DSL:
{cfg}

### Here are the recipes for the domain, only these items can be used in the programs. You cannot propose any new items that are not in the recipes:
{recipes}

---

## CRITICAL CFG FORMAT RULES - Follow Exactly:

1. **ALL SYMBOLS MUST BE UPPERCASE**: Use UPPERCASE for ALL terminal functions, terminal symbols, and non-terminals. NEVER use lowercase or mixed case (except for the start symbol `program`).

2. **Terminal Functions**:
   - Terminal functions are actions that appear directly in productions (e.g., ACTION1, ACTION2, ACTION3)
   - Use UPPERCASE names for all terminal functions
   - NEVER create rules like `ACTION1 ::= 'action1'` or `ACTION1 ::= 'ACTION1'` - terminal functions appear directly in productions, not as separate rules

3. **Function Arguments Format**:
   - Use space-separated format: `FUNC LPAR ARG RPAR`
   - Example: `ACTION1 LPAR PARAM RPAR` (correct)
   - Example: `ACTION2 LPAR PARAM1 COMMA PARAM2 RPAR` (correct for multiple args)
   - NEVER use literal parentheses like `ACTION1(PARAM)` - always use `ACTION1 LPAR PARAM RPAR`

4. **Special Symbols** (single characters only):
   - Define punctuation as: `SYMBOL ::= 'char'` (single character in single quotes)
   - Examples: `SEMICOLON ::= ';'`, `LPAR ::= '('`, `RPAR ::= ')'`, `COMMA ::= ','`
   - These are the ONLY terminals that should have quoted character definitions

5. **Enumeration Rules** (for parameter values):
   - Use: `PARAM ::= VALUE1 | VALUE2 | VALUE3`
   - Example: `PARAM ::= OPTION1 | OPTION2 | OPTION3 | OPTION4`
   - DO NOT create individual rules like `VALUE1 ::= 'VALUE1'` - the enumeration is sufficient
   - All values in enumerations must be UPPERCASE
   - NEVER use regex syntax like `(?:VALUE1|VALUE2|)` - use simple BNF: `PARAM ::= VALUE1 | VALUE2`
   - NEVER create empty alternatives (zero-width terminals) - every alternative must have at least one value

6. **Start Symbol**: Use lowercase `program` as the top-level non-terminal

7. **Rule Format**: One rule per line, use `|` for alternatives:
```
program        ::= statement_seq

statement_seq  ::= statement
                |  statement SEMICOLON statement_seq

statement      ::= ACTION1 LPAR PARAM RPAR
                |  ACTION2 LPAR PARAM RPAR
                |  ACTION3 PARAM
                |  ACTION4 LPAR PARAM1 COMMA PARAM2 RPAR

PARAM          ::= VALUE1 | VALUE2 | VALUE3 | VALUE4
PARAM1         ::= OPTION1 | OPTION2
PARAM2         ::= CHOICE1 | CHOICE2

SEMICOLON      ::= ';'
LPAR           ::= '('
RPAR           ::= ')'
COMMA          ::= ','
```

8. **What NOT to do** (COMMON MISTAKES THAT CAUSE VALIDATION ERRORS):
   - ❌ NEVER use regex syntax: `(?:VALUE1|VALUE2|)` - use BNF: `PARAM ::= VALUE1 | VALUE2`
   - ❌ NEVER create empty alternatives: `PARAM ::= VALUE1 | | VALUE2` - remove empty alternatives
   - ❌ NEVER put non-terminals inside terminal definitions: `ITEM ::= NonTerminal('items')` - use: `ITEM ::= VALUE1 | VALUE2`
   - ❌ NEVER create recursive terminals: `NUMBER ::= NUMBER DIGIT` - use non-terminals for recursion
   - ❌ NEVER create undefined rules - every non-terminal used must be defined
   - ❌ NEVER create `ACTION1 ::= 'action1'` or any lowercase definitions
   - ❌ NEVER create `ACTION1 ::= 'ACTION1'` - terminal functions appear directly in productions
   - ❌ NEVER use lowercase terminal function names
   - ❌ NEVER use literal parentheses in productions (always use LPAR/RPAR)
   - ❌ NEVER create circular rules like `X ::= X`

9. **Validation Requirements**:
   - The CFG must be parseable by a strict BNF parser
   - All non-terminals used in rules must be defined
   - All terminals must be properly formatted (either as quoted characters or as enumeration values)
   - The example program must be parseable by the generated CFG

---

### Context

You are an expert in **DSL and program synthesis**.

Your task is to analyze these failures and propose **targeted improvements** while maintaining strict BNF format compliance.

You need to:
1. Identify **gaps or weaknesses** in the current CFG.  
2. Suggest **specific additions or removal of terminal functions** to the CFG that would enable better synthesis results.  
   - Terminal functions are the action functions that appear directly in statement productions (e.g., in rules like `statement ::= ACTION LPAR PARAM RPAR`, the terminal function is ACTION, not PARAM)
   - Terminal functions are the actual executable actions in programs
   - Grammar symbols that represent parameter types or special symbols are NOT terminal functions
3. Provide **solution explanation** where the proposed changes are justified based on the failure analysis.
4. **CRITICALLY**: Ensure your CFG follows ALL format rules above - validation will fail if you use regex syntax, empty alternatives, or other invalid patterns.

---

### Output Format

Your response must strictly follow this structure:

Updated CFG(BNF)
<return the full updated CFG in BNF format here - MUST follow all format rules above>

Updated CFG Explanation
<Write a comprehensive, standalone explanation of the entire CFG shown above. 
Do NOT list changes, revisions, differences, deltas, or what was "added" or "modified" 
compared to previous grammars. Instead, explain the grammar from scratch as if the 
reader has never seen any earlier version.>

Changes in CFG
(bullet point list of specific changes made to the CFG)

Terminal Functions
FUNCTION_NAME(args): description of purpose and usage

**CRITICAL - READ CAREFULLY**: In the "Terminal Functions" section, you MUST list ALL actual terminal function names that appear directly in statement productions. Every function that appears at the start of a statement production alternative MUST be included with a clear description. Do not omit any terminal functions - completeness is essential for the system to work correctly.

**What ARE terminal functions:**
- Terminal functions are the ACTION WORDS that appear at the START of statement rules
- These are the functions that can be CALLED in programs
- They appear as the FIRST symbol in statement production alternatives
- They are the executable actions/operations in your DSL

**What are NOT terminal functions (DO NOT LIST THESE):**
- Grammar symbols that represent parameter types (e.g., PARAM, ARG, TYPE, VALUE, OPTION, etc.)
- Enumeration symbols (symbols that define sets of values like `SYMBOL ::= VALUE1 | VALUE2 | VALUE3`)
- Special symbols like LPAR, RPAR, COMMA, SEMICOLON, LBRACK, RBRACK
- Any symbol that appears AFTER a function name in a production rule
- Any symbol that is used as a parameter/argument to a function

**How to identify terminal functions:**
1. Look at your statement rules (e.g., `statement ::= ...`)
2. Find the FIRST UPPERCASE WORD in each alternative (before any parentheses, brackets, or parameters)
3. That FIRST WORD is the terminal function
4. Everything after it (parameters, arguments, etc.) are NOT functions

**Concrete Example:**
If your CFG has:
```
statement ::= ACTION1 LPAR PARAM RPAR
           |  ACTION2 LPAR PARAM RPAR
           |  ACTION3 PARAM
           |  ACTION4 LPAR PARAM1 COMMA PARAM2 RPAR
```

Then your Terminal Functions section MUST list:
```
ACTION1: Description of what ACTION1 does
ACTION2: Description of what ACTION2 does
ACTION3: Description of what ACTION3 does
ACTION4: Description of what ACTION4 does
```

You MUST NOT list:
- PARAM (this is a parameter type, not a function)
- PARAM1, PARAM2 (these are parameter types, not functions)
- Any symbol that appears in enumeration rules (e.g., `PARAM ::= VALUE1 | VALUE2 | VALUE3`)

**Remember**: Terminal functions are the ACTION WORDS that start each statement alternative. They are what you CALL in programs. Parameters are what you PASS TO functions, not functions themselves. If a symbol appears in a rule like `SYMBOL ::= VALUE1 | VALUE2`, it is NOT a terminal function - it's a parameter type.

If the current CFG is already sufficient, restate it under "Updated CFG (BNF)" and note that no changes are required.
---
"""
        
        conversation = [{"role": "user", "content": cfg_evolution_prompt, "chat_template_kwargs": {"reasoning_effort": "high"}}]
        output = llm.chat(conversation, params)
        response = output[0].outputs[0].text
        
        marker_match = re.search(r'assistantfinal', response, re.IGNORECASE)
        if marker_match:
            response = response[marker_match.end():]
        
        # Extract CFG from response
        filepath, cfg_text, term_text, failure_text, cfg_explanation = extract_and_save_cfg(response)
        
        if cfg_text:
            print("✓ Generated new CFG")
            
            # Validate the evolved CFG
            print("\n[Validating Evolved CFG] Checking CFG validity...")
            from src.pipeline.cfg_to_funsearch_pipeline import validate_cfg
            
            # Get example program if available for validation
            cfg_path = os.path.join(experiment_dir, "cfg", "cfg_output.json")
            example = None
            if os.path.exists(cfg_path):
                with open(cfg_path, 'r') as f:
                    cfg_data = json.load(f)
                    example = cfg_data.get("example", None)
            
            is_valid, validation_msg = validate_cfg(cfg_text, example=example)
            if not is_valid:
                print(f"✗ Evolved CFG validation failed: {validation_msg}")
                print("  Rejecting evolved CFG, using original")
                return cfg, terminals, False
            else:
                print(f"✓ {validation_msg}")
            
            # Extract terminal functions directly from CFG text using CFGParser
            new_terminals = {}
            try:
                from src.pipeline.cfg_parser import CFGParser
                # Parse the CFG to extract actual terminal functions
                cfg_parser = CFGParser(cfg_text)
                terminal_funcs = cfg_parser.get_terminal_functions()
                
                if terminal_funcs:
                    # Extract function names (first element of each tuple)
                    func_names = [func_name for func_name, _ in terminal_funcs]
                    print(f"  ✓ Extracted {len(func_names)} terminal functions from CFG: {func_names}")
                    
                    # Try to get descriptions from term_text if available
                    term_descriptions = {}
                    if term_text:
                        # First try using the robust parser from getting_cfg
                        try:
                            from src.pipeline.getting_cfg import parse_generated_output
                            # Create a combined text that mimics the expected format
                            combined_text = f"```bnf\n{cfg_text}\n```\n\nTerminal Functions:\n{term_text}"
                            _, parsed_terminals, _ = parse_generated_output(combined_text)
                            if parsed_terminals:
                                # Use parsed terminals, but only for functions that are actually in the CFG
                                for func_name in func_names:
                                    if func_name in parsed_terminals:
                                        term_descriptions[func_name] = parsed_terminals[func_name]
                        except Exception as e:
                            print(f"  ⚠ Could not parse terminal descriptions using parse_generated_output: {e}")
                        
                        # Fallback: try regex patterns if parser didn't work
                        if not term_descriptions:
                            # Pattern: FUNCTION_NAME(args): description or FUNCTION_NAME: description
                            # Try multiple patterns to catch different formats
                            patterns = [
                                r'([A-Z_][A-Z0-9_()]*)\s*[:\-–]\s*(.+?)(?=\n|$)',  # FUNCTION_NAME: description
                                r'([A-Z_][A-Z0-9_()]*)\s*\([^)]*\)\s*[:\-–]\s*(.+?)(?=\n|$)',  # FUNCTION_NAME(args): description
                            ]
                            for pattern in patterns:
                                for match in re.finditer(pattern, term_text, re.MULTILINE):
                                    func_name = match.group(1).strip()
                                    description = match.group(2).strip().rstrip(';.')
                                    # Only add if it's an actual terminal function (not a grammar symbol)
                                    if func_name in func_names and func_name not in term_descriptions:
                                        term_descriptions[func_name] = description
                    
                    # Build terminals dict: use descriptions from term_text if available, otherwise use old terminals or generic
                    for func_name in func_names:
                        if func_name in term_descriptions:
                            new_terminals[func_name] = term_descriptions[func_name]
                        elif func_name in terminals:
                            # Use description from old terminals if available
                            new_terminals[func_name] = terminals[func_name]
                        else:
                            # Use a generic description without hardcoding specific patterns
                            new_terminals[func_name] = f"Terminal function: {func_name}"
                else:
                    print("  ⚠ CFGParser found no terminal functions, falling back to term_text parsing")
                    raise ValueError("No terminal functions found in CFG")
            except Exception as e:
                print(f"  ⚠ Could not parse terminal functions from CFG: {e}")
                # Fallback: try to parse from term_text
                if term_text:
                    try:
                        from src.pipeline.getting_cfg import parse_generated_output
                        combined_text = f"```bnf\n{cfg_text}\n```\n\nTerminal Functions:\n{term_text}"
                        _, parsed_terminals, _ = parse_generated_output(combined_text)
                        if parsed_terminals:
                            new_terminals = parsed_terminals
                            print(f"  ✓ Parsed {len(new_terminals)} terminal functions from term_text")
                    except Exception as e2:
                        print(f"  ⚠ Could not parse terminals from term_text: {e2}")
                        # Final fallback: extract from term_text manually
                        term_pattern = r'([A-Z_][A-Z0-9_()]*)\s*[:\-–]\s*(.+?)(?=\n|$)'
                        for match in re.finditer(term_pattern, term_text, re.MULTILINE):
                            func_name = match.group(1).strip()
                            description = match.group(2).strip().rstrip(';.')
                            new_terminals[func_name] = description
                        if new_terminals:
                            print(f"  ✓ Extracted {len(new_terminals)} terminal functions manually")
            
            # If still no terminals extracted, try to use the old terminals (may need updating)
            if not new_terminals:
                print("  ⚠ No terminals extracted, using previous terminals")
                new_terminals = terminals.copy()
            
            # Check if the new CFG is actually different from the old one
            if cfg_text == cfg:
                print("  ⚠ Evolved CFG is identical to original, will retry evolution")
                return cfg, terminals, False  # Return False to trigger retry in calling code
            
            # Save the new CFG to JSON file
            cfg_path = os.path.join(experiment_dir, "cfg", "cfg_output.json")
            example = None
            if os.path.exists(cfg_path):
                # Read example from existing file before versioning
                with open(cfg_path, 'r') as f:
                    cfg_data = json.load(f)
                    example = cfg_data.get("example", None)
            
            # Version file before writing new CFG data (only if CFG is different)
            if os.path.exists(cfg_path):
                try:
                    version_file(cfg_path, keep_original=False)
                    print(f"  ✓ Versioned previous CFG file")
                except Exception as e:
                    print(f"  ⚠ Warning: Failed to version CFG file: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Ensure terminals dictionary includes ALL terminal functions from CFG
            # Pass old terminals to preserve descriptions across evolutions
            new_terminals = ensure_terminals_match_cfg(cfg_text, new_terminals, old_terminals=terminals, shared_vllm=shared_vllm)
            
            # Save new CFG data
            cfg_data = {
                "cfg": cfg_text,
                "terminals": new_terminals,
                "example": example
            }
            with open(cfg_path, 'w', encoding='utf-8') as f:
                json.dump(cfg_data, f, indent=2, ensure_ascii=False)
            print(f"  ✓ Saved new CFG to {cfg_path}")
            print(f"  ✓ New CFG has {len(new_terminals)} terminal functions: {list(new_terminals.keys())}")
            
            return cfg_text, new_terminals, True
        else:
            print("⚠ Could not extract new CFG, using original")
            return cfg, terminals, False
        
    except Exception as e:
        print(f"✗ Error evolving DSL: {e}")
        import traceback
        traceback.print_exc()
        return cfg, terminals, False


def evolve_dsl_and_restart(
    experiment_dir: str,
    failing_tasks: List[str],
    cfg: str,
    recipes: str,
    spec_file: str,
    terminals: Dict[str, str],
    shared_vllm=None,
    model_type: str = "huggingface",
    max_retries: int = 10
) -> Tuple[str, Dict[str, str], bool]:
    """Evolve the DSL based on failing tasks, implement the new CFG, and return results.
    
    This function combines evolve_dsl() and implement_cfg() for convenience.
    Retries DSL evolution if the evolved CFG is the same as the original.
    
    Args:
        max_retries: Maximum number of retry attempts if CFG is same (default: 10)
    
    Returns:
        Tuple of (new_cfg: str, terminals: Dict[str, str], success: bool)
        - new_cfg: The evolved CFG string
        - terminals: Dictionary of terminal functions from the new CFG
        - success: True if CFG was successfully evolved and implemented
    """
    # Retry DSL evolution if CFG is same as original
    dsl_success = False
    new_cfg = cfg
    new_terminals = terminals
    
    for dsl_attempt in range(1, max_retries + 1):
        if dsl_attempt > 1:
            print(f"\n[DSL Evolution Retry] Attempt {dsl_attempt}/{max_retries}")
        
    # Use evolve_dsl function (reusable)
        new_cfg, new_terminals, attempt_success = evolve_dsl(
        experiment_dir=experiment_dir,
        failing_tasks=failing_tasks,
        cfg=cfg,
        recipes=recipes,
        terminals=terminals,
        shared_vllm=shared_vllm
    )
        
        # Check if evolution was successful and CFG is different
        if attempt_success and new_cfg != cfg:
            dsl_success = True
            print(f"\n✓ DSL evolved successfully on attempt {dsl_attempt}")
            break
        else:
            if attempt_success:
                print(f"  ⚠ Attempt {dsl_attempt}: Evolved CFG is same as original, retrying...")
            else:
                print(f"  ⚠ Attempt {dsl_attempt}: DSL evolution failed, retrying...")
    
    if not dsl_success or new_cfg == cfg:
        print(f"\n✗ Could not evolve DSL after {max_retries} attempts")
        return cfg, terminals, False
    
    # Implement the new CFG (steps 2-7)
    cfg_path = os.path.join(experiment_dir, "cfg", "cfg_output.json")
    example = None
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r') as f:
            cfg_data = json.load(f)
            example = cfg_data.get("example", None)
    
    print("\n[Implementing new CFG] Running implementation steps...")
    print(f"  New CFG has {len(new_terminals)} terminal functions: {list(new_terminals.keys())}")
    print(f"  Experiment directory: {experiment_dir}")
    
    success, final_functions = implement_cfg(
        cfg=new_cfg,
        terminals=new_terminals,
        example=example,
        spec_file=spec_file,
        experiment_dir=experiment_dir,
        model_type=model_type,
        shared_vllm=shared_vllm,
        dsl_round=None,  # DSL round not available in this context
        func_evolution_round=None  # Initial implementation after DSL evolution
    )
    
    if success:
        print(f"\n✓ New CFG successfully implemented")
        print(f"  Generated {len(final_functions)} final functions:")
        for func_name in final_functions.keys():
            print(f"    - {func_name}")
        
        # Verify files were created
        final_functions_dir = os.path.join(experiment_dir, "final_functions")
        if os.path.exists(final_functions_dir):
            generated_files = [f for f in os.listdir(final_functions_dir) if f.endswith('.py')]
            print(f"  Final function files created: {len(generated_files)} files in {final_functions_dir}")
        
        return new_cfg, new_terminals, True
    else:
        print("\n⚠ New CFG implemented but some functions may be missing")
        if final_functions:
            print(f"  Generated {len(final_functions)} final functions: {list(final_functions.keys())}")
        return new_cfg, new_terminals, False


def run_integrated_pipeline(
    experiment_dir: str,
    spec_file: str,
    tasks: List[str],
    max_function_evolutions: int = 3,
    max_dsl_evolutions: int = 2,
    recipes_path: str = "craft/resources/recipes.yaml",
    hints_path: str = "craft/resources/hints.yaml",
    max_attempts: int = 1,
    shared_vllm=None
) -> int:
    """Run the complete integrated pipeline.
    
    Returns:
        0 on success, 1 on failure
    """
    print(f"\n{'='*80}")
    print("INTEGRATED PIPELINE")
    print(f"{'='*80}")
    print(f"Experiment directory: {experiment_dir}")
    print(f"Tasks to solve: {tasks}")
    
    # Create experiment directory structure if it doesn't exist
    if not os.path.exists(experiment_dir):
        print(f"\n[Setup] Creating experiment directory: {experiment_dir}")
        os.makedirs(experiment_dir, exist_ok=True)
        os.makedirs(os.path.join(experiment_dir, "function_specific_prompts"), exist_ok=True)
        os.makedirs(os.path.join(experiment_dir, "functions_generated"), exist_ok=True)
        os.makedirs(os.path.join(experiment_dir, "results", "funsearch"), exist_ok=True)
        os.makedirs(os.path.join(experiment_dir, "cfg"), exist_ok=True)
        os.makedirs(os.path.join(experiment_dir, "final_functions"), exist_ok=True)
        os.makedirs(os.path.join(experiment_dir, "explicit_feedback"), exist_ok=True)
        print(f"✓ Created experiment directory structure")
    else:
        # Ensure subdirectories exist even if main directory does
        os.makedirs(os.path.join(experiment_dir, "function_specific_prompts"), exist_ok=True)
        os.makedirs(os.path.join(experiment_dir, "functions_generated"), exist_ok=True)
        os.makedirs(os.path.join(experiment_dir, "results", "funsearch"), exist_ok=True)
        os.makedirs(os.path.join(experiment_dir, "cfg"), exist_ok=True)
        os.makedirs(os.path.join(experiment_dir, "final_functions"), exist_ok=True)
        os.makedirs(os.path.join(experiment_dir, "explicit_feedback"), exist_ok=True)
        print(f"\n[Setup] Using existing experiment directory: {experiment_dir}")
    
    # Create shared vLLM instance if not provided
    if shared_vllm is None:
        if vLLM is not None:
            try:
                print("\n[Setup] Initializing shared vLLM instance...")
                shared_vllm = vLLM(model="/scratch/avani/gpt", tensor_parallel_size=4)
                print("✓ Shared vLLM instance created")
            except Exception as e:
                print(f"⚠ Warning: Could not create shared vLLM instance: {e}")
                print("  Will create individual instances as needed")
                shared_vllm = None
        else:
            print("\n[Setup] vLLM not available, will use regular LLM instances")
    else:
        print("\n[Setup] Using provided shared vLLM instance")
    
    # Step 1: Check if final_functions exist
    print(f"\n[Step 1] Checking final functions...")
    
    # Load CFG to get terminals
    cfg_path = os.path.join(experiment_dir, "cfg", "cfg_output.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r') as f:
            cfg_data = json.load(f)
            terminals = cfg_data.get("terminals", {})
            cfg = cfg_data.get("cfg", "")
    else:
        print("⚠ Could not find CFG, need to run initial pipeline")
        terminals = {}
        cfg = ""
    
    if not terminals:
        print("✗ No terminal functions found in CFG. Cannot proceed.")
        return 1
    
    print(f"  Checking {len(terminals)} terminal functions:")
    for func_name in terminals.keys():
        print(f"    - {func_name}")
    
    all_exist, missing, empty_or_invalid = check_final_functions_exist(experiment_dir, terminals)
    
    if not all_exist:
        if missing:
            print(f"\n  ✗ Missing final functions ({len(missing)}):")
            for func_name, description in missing.items():
                safe_name = sanitize_function_name(func_name)
                func_file = os.path.join(experiment_dir, "final_functions", f"{safe_name}.py")
                print(f"    - {func_name} (expected: {func_file})")
        
        if empty_or_invalid:
            print(f"\n  ✗ Empty or invalid final functions ({len(empty_or_invalid)}):")
            for func_name in empty_or_invalid:
                safe_name = sanitize_function_name(func_name)
                func_file = os.path.join(experiment_dir, "final_functions", f"{safe_name}.py")
                print(f"    - {func_name} (file exists but is empty or invalid: {func_file})")
        
        print(f"\n  ⚠ Total missing/invalid: {len(missing) + len(empty_or_invalid)}/{len(terminals)}")
        print("  Running explicit feedback generation for missing/invalid functions...")
        
        # Run explicit feedback generation for missing functions
        # This would require calling the explicit feedback generation
        # For now, we'll assume the user needs to run it manually
        print("  ⚠ Please run explicit feedback generation manually or")
        print("    re-run the full pipeline to generate missing functions")
        print(f"    Missing functions: {list(missing.keys())}")
        if empty_or_invalid:
            print(f"    Invalid functions: {empty_or_invalid}")
        return 1
    
    print(f"\n  ✓ All {len(terminals)} terminal functions exist and are valid")
    
    # Step 2: Synthesize and test programs
    print(f"\n[Step 2] Synthesizing and testing programs...")
    task_results = synthesize_and_test_programs(
        experiment_dir, tasks, cfg_path=cfg_path, terminals=terminals,
        recipes_path=recipes_path, hints_path=hints_path, max_attempts=max_attempts,
        shared_vllm=shared_vllm
    )
    
    all_solved = all(task_results.values())
    failing_tasks = [task for task, success in task_results.items() if not success]
    
    print(f"\nTask Results:")
    for task, success in task_results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {task}")
    
    if all_solved:
        print("\n✓ All tasks solved successfully!")
        return 0
    
    # Step 3: Evolve functions (up to max_function_evolutions times)
    print(f"\n[Step 3] Some tasks failed. Attempting function evolution...")
    
    for evolution_round in range(max_function_evolutions):
        print(f"\n  Evolution round {evolution_round + 1}/{max_function_evolutions}")
        
        # Load specification
        if os.path.exists(spec_file):
            with open(spec_file, 'r') as f:
                specification = f.read()
        else:
            print("  ⚠ Specification file not found")
            specification = ""
        
        # Load CFG if not already loaded
        if not cfg:
            cfg_path = os.path.join(experiment_dir, "cfg", "cfg_output.json")
            if os.path.exists(cfg_path):
                with open(cfg_path, 'r') as f:
                    cfg_data = json.load(f)
                    cfg = cfg_data.get("cfg", "")
        
        evolved = evolve_functions_with_failing_tasks(
            experiment_dir, failing_tasks, terminals, specification, 
            spec_file=spec_file, cfg=cfg, max_evolutions=1, shared_vllm=shared_vllm
        )
        
        if evolved:
            # Re-test tasks
            task_results = synthesize_and_test_programs(
                experiment_dir, failing_tasks, cfg_path=cfg_path, terminals=terminals,
                recipes_path=recipes_path, hints_path=hints_path, max_attempts=max_attempts,
                shared_vllm=shared_vllm
            )
            
            all_solved = all(task_results.values())
            failing_tasks = [task for task, success in task_results.items() if not success]
            
            if all_solved:
                print("\n✓ All tasks solved after function evolution!")
                return 0
    
    # Step 4: Evolve DSL and implement (up to max_dsl_evolutions times)
    print(f"\n[Step 4] Function evolution did not solve all tasks. Evolving DSL...")
    
    with open(recipes_path, 'r') as f:
        recipes = f.read()
    
    current_cfg = cfg
    current_terminals = terminals
    
    for dsl_evolution_round in range(max_dsl_evolutions):
        print(f"\n  DSL Evolution round {dsl_evolution_round + 1}/{max_dsl_evolutions}")
        
        # Evolve DSL and implement
        new_cfg, new_terminals, implementation_success = evolve_dsl_and_restart(
            experiment_dir, failing_tasks, current_cfg, recipes, 
            spec_file=spec_file, terminals=current_terminals,
            shared_vllm=shared_vllm, model_type="huggingface"
        )
        
        if new_cfg == current_cfg:
            print("\n  ✗ Could not evolve DSL or evolution produced same CFG")
            if dsl_evolution_round < max_dsl_evolutions - 1:
                print("  Continuing to next evolution round...")
                continue
            else:
                print("\n✗ Reached maximum DSL evolution rounds without success")
                return 1
        
        if not implementation_success:
            print("\n  ⚠ DSL evolved but implementation had issues")
            if dsl_evolution_round < max_dsl_evolutions - 1:
                print("  Continuing to next evolution round...")
                current_cfg = new_cfg
                current_terminals = new_terminals
                continue
            else:
                print("\n✗ Reached maximum DSL evolution rounds")
                return 1
        
        print("\n  ✓ DSL evolved and implemented successfully")
        
        # Update current CFG and terminals
        current_cfg = new_cfg
        current_terminals = new_terminals
        
        # Update CFG path for re-testing
        cfg_path = os.path.join(experiment_dir, "cfg", "cfg_output.json")
        
        # Re-test tasks with new CFG and functions
        print(f"\n  Re-testing tasks with evolved DSL...")
        task_results = synthesize_and_test_programs(
            experiment_dir, failing_tasks, cfg_path=cfg_path, terminals=current_terminals,
            recipes_path=recipes_path, hints_path=hints_path, max_attempts=max_attempts,
            shared_vllm=shared_vllm
        )
        
        all_solved = all(task_results.values())
        failing_tasks = [task for task, success in task_results.items() if not success]
        
        print(f"\n  Task Results after DSL evolution round {dsl_evolution_round + 1}:")
        for task, success in task_results.items():
            status = "✓" if success else "✗"
            print(f"    {status} {task}")
        
        if all_solved:
            print(f"\n✓ All tasks solved after DSL evolution round {dsl_evolution_round + 1}!")
            return 0
        
        # If not all solved and we have more rounds, continue
        if dsl_evolution_round < max_dsl_evolutions - 1:
            print(f"\n  Some tasks still failing. Continuing to next DSL evolution round...")
        else:
            print(f"\n✗ Reached maximum DSL evolution rounds ({max_dsl_evolutions})")
            print(f"  Remaining failing tasks: {failing_tasks}")
            return 1
    
    # Should not reach here, but just in case
    print("\n✗ DSL evolution loop completed without solving all tasks")
    return 1


def main():
    parser = argparse.ArgumentParser(
        description="Integrated pipeline: generate functions, synthesize programs, and evolve as needed"
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
        '--max_function_evolutions',
        type=int,
        default=3,
        help='Maximum number of function evolution rounds (default: 3)'
    )
    parser.add_argument(
        '--max_dsl_evolutions',
        type=int,
        default=2,
        help='Maximum number of DSL evolution rounds (default: 2)'
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
            print(f"✗ Error: Tasks file not found: {tasks_file}")
            return 1
    
    return run_integrated_pipeline(
        experiment_dir=args.experiment_dir,
        spec_file=args.spec_file,
        tasks=tasks,
        max_function_evolutions=args.max_function_evolutions,
        max_dsl_evolutions=args.max_dsl_evolutions,
        recipes_path=args.recipes_path,
        hints_path=args.hints_path,
        max_attempts=args.max_attempts
    )


if __name__ == "__main__":
    sys.exit(main())

