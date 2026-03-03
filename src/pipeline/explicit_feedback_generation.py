import os
import json
import argparse
import random
import subprocess
import re
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional
import time
import ast
import textwrap
from vllm import SamplingParams
from vllm import LLM as vLLM

def get_end_score(scores: Dict[str, Any]) -> Optional[float]:
    if not isinstance(scores, dict) or not scores:
        return None
    try:
        step_keys = [int(k) for k in scores.keys()]
    except (ValueError, TypeError):
        # Fallback: if keys are not numeric, just take any deterministic "last" by insertion order
        try:
            # Python 3.7+ preserves insertion order
            last_key = next(reversed(scores))
            return float(scores[last_key])
        except Exception:
            return None
    last_step = max(step_keys)
    value = scores.get(str(last_step))
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def _find_function_end(lines: List[str], start_idx: int) -> int:
    """Find the end line index of a function starting at start_idx.
    
    Args:
        lines: List of code lines
        start_idx: Index of the function definition line
        
    Returns:
        Index of the line after the function ends
    """
    if start_idx >= len(lines):
        return start_idx + 1
    
    line = lines[start_idx]
    indent = len(line) - len(line.lstrip())
    end_idx = start_idx + 1
    
    while end_idx < len(lines):
        next_line = lines[end_idx]
        # Skip empty lines and comments
        if not next_line.strip() or next_line.strip().startswith('#'):
            end_idx += 1
            continue
        
        next_indent = len(next_line) - len(next_line.lstrip())
        # If we hit a line at same or lower indent that's a def/class, we're done
        if next_indent <= indent:
            if re.match(r'^\s*(def|class)\s+', next_line):
                break
        end_idx += 1
    
    return end_idx


def parse_log_file(path: str, k: int = 1) -> List[Tuple[float, str]]:
    scored_funcs: List[Tuple[float, str]] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            scores = record.get("scores")
            function_body = record.get("function_body")

            # Skip if function_body is None, empty string, or only whitespace
            if function_body is None or scores is None:
                continue
            
            # Filter out empty or whitespace-only function bodies
            if not function_body or not function_body.strip():
                continue

            end_score = get_end_score(scores)
            if end_score is None:
                continue

            scored_funcs.append((end_score, function_body))

    if not scored_funcs:
        return []

    # Sort by score (descending), then by function body length (descending) for tie-breaking
    # This ensures non-empty functions are preferred when scores are equal
    scored_funcs.sort(key=lambda x: (x[0], len(x[1])), reverse=True)

    # Find cutoff score if ties go beyond k
    cutoff = scored_funcs[k-1][0] if len(scored_funcs) >= k else scored_funcs[-1][0]

    # Keep all functions with score >= cutoff
    top_candidates = [(s, f) for (s, f) in scored_funcs if s >= cutoff]

    if len(top_candidates) > k:
        # Too many due to ties → take first k (already sorted by score, then length)
        # This ensures consistent ordering and prefers longer (more complete) functions
        return top_candidates[:k]
    else:
        return top_candidates

def eval(res, file, specification=None, func_signature=None, function_name=None, results_tracker=None):
    # with tempfile.TemporaryDirectory() as temp_dir:
    # Create unique filename using function name, process ID and timestamp
    # Include function name to prevent collisions when parallel funsearch workers run
    temp_dir = os.getcwd()
    unique_id = f"{os.getpid()}_{int(time.time() * 1000000)}"
    if function_name:
        # Sanitize function name for use in filename
        safe_func_name = function_name.replace('/', '_').replace('\\', '_').replace(' ', '_')
        script_path = f'explicit_generated_code_{safe_func_name}_{unique_id}.py'
    else:
        script_path = f'explicit_generated_code_{unique_id}.py'
    script_path = os.path.join(temp_dir, script_path)

    with open(file,"r") as f:
        full_program = f.read()
    
    # Extract imports from the specification (they should be at the bottom of specification file)
    import_lines = []
    seen_imports = set()
    
    if specification:
        # Extract imports from specification string
        for line in specification.split('\n'):
            stripped = line.strip()
            if stripped.startswith('import ') or stripped.startswith('from '):
                # Avoid duplicate imports
                if stripped not in seen_imports:
                    import_lines.append(line)
                    seen_imports.add(stripped)
    else:
        # Fallback: try to extract from the evaluation file if specification not provided
        for line in full_program.split('\n'):
            stripped = line.strip()
            if stripped.startswith('import ') or stripped.startswith('from '):
                if stripped not in seen_imports:
                    import_lines.append(line)
                    seen_imports.add(stripped)
    
    imports_section = '\n'.join(import_lines) + '\n\n' if import_lines else ''
    
    # Remove @funsearch.run and @funsearch.evolve decorators that cause NameError
    # These decorators are only needed for funsearch, not for execution
    # Remove standalone decorator lines
    full_program = re.sub(r'^\s*@funsearch\.(run|evolve)\s*$', '', full_program, flags=re.MULTILINE)
    # Remove decorators on the same line as function definition
    full_program = re.sub(r'@funsearch\.(run|evolve)\s*\n\s*', '', full_program)
    full_program = re.sub(r'@funsearch\.(run|evolve)\s+', '', full_program)
    
    # Normalize Unicode quotes before parsing
    def _normalize_unicode_quotes(text: str) -> str:
        """Normalize Unicode quotes and other problematic characters to ASCII equivalents."""
        replacements = {
            '\u2018': "'",  # Left single quotation mark
            '\u2019': "'",  # Right single quotation mark
            '\u201A': "'",  # Single low-9 quotation mark
            '\u201B': "'",  # Single high-reversed-9 quotation mark
            '\u201C': '"',  # Left double quotation mark
            '\u201D': '"',  # Right double quotation mark
            '\u201E': '"',  # Double low-9 quotation mark
            '\u201F': '"',  # Double high-reversed-9 quotation mark
            '\u2032': "'",  # Prime
            '\u2033': '"',  # Double prime
        }
        result = text
        for unicode_char, ascii_char in replacements.items():
            result = result.replace(unicode_char, ascii_char)
        return result
    
    # Remove duplicate function definitions - keep only the last occurrence of each function
    # Also remove incomplete function definitions (just signature/docstring, no body)
    # Use AST to properly parse function boundaries
    try:
        import ast
        full_program = _normalize_unicode_quotes(full_program)
        tree = ast.parse(full_program)
        
        # Find all function definitions with their line ranges
        # Skip incomplete functions (only docstring, no body)
        function_defs = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                start_line = node.lineno - 1  # Convert to 0-based
                end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line
                
                # Check if function has actual body (not just docstring)
                # A function with only a docstring will have body length <= 1
                # A function with actual code will have body length > 1 or contain non-docstring statements
                has_real_body = False
                if len(node.body) > 1:
                    has_real_body = True
                elif len(node.body) == 1:
                    # Check if the single statement is more than just a docstring
                    stmt = node.body[0]
                    if not (isinstance(stmt, ast.Expr) and 
                           isinstance(stmt.value, (ast.Str, ast.Constant))):
                        has_real_body = True
                
                # Only include functions with actual body (skip incomplete stubs)
                if has_real_body:
                    function_defs.append((node.name, start_line, end_line))
        
        # Group by function name and keep only last occurrence
        functions_by_name = {}
        for func_name, start, end in function_defs:
            if func_name not in functions_by_name:
                functions_by_name[func_name] = []
            functions_by_name[func_name].append((start, end))
        
        # Determine which lines to keep
        lines = full_program.split('\n')
        lines_to_keep = set(range(len(lines)))  # Start by keeping all lines
        
        # Remove earlier occurrences of duplicate functions
        for func_name, occurrences in functions_by_name.items():
            if len(occurrences) > 1:
                # Keep only the last occurrence
                occurrences.sort(key=lambda x: x[0])  # Sort by start line
                last_start, last_end = occurrences[-1]
                
                # Remove all earlier occurrences
                for start, end in occurrences[:-1]:
                    for i in range(start, end):
                        lines_to_keep.discard(i)
        
        # Remove incomplete function definitions (functions not in function_defs)
        # These are functions that only have docstrings, no actual body
        # IMPORTANT: Always preserve 'solve' and 'evaluate' functions as they are essential for evaluation
        essential_functions = {'solve', 'evaluate'}
        all_valid_function_ranges = set()
        for func_name, start, end in function_defs:
            for i in range(start, end):
                all_valid_function_ranges.add(i)
        
        # Find and remove incomplete functions
        i = 0
        while i < len(lines):
            line = lines[i]
            func_match = re.match(r'^\s*def\s+(\w+)\s*\(', line)
            if func_match:
                func_name = func_match.group(1)
                # Always preserve essential evaluation functions (solve, evaluate)
                if func_name in essential_functions:
                    # Keep this function even if AST parsing didn't find it
                    # Find its end and ensure all lines are kept
                    end_idx = _find_function_end(lines, i)
                    for j in range(i, end_idx):
                        lines_to_keep.add(j)
                    i = end_idx
                    continue
                # Check if this function is in our valid list
                if i not in all_valid_function_ranges:
                    # This is an incomplete function, find its end and remove it
                    end_idx = _find_function_end(lines, i)
                    # Remove this incomplete function
                    for j in range(i, end_idx):
                        lines_to_keep.discard(j)
                    i = end_idx
                    continue
            i += 1
        
        # Build cleaned program
        cleaned_lines = [lines[i] for i in sorted(lines_to_keep)]
        full_program = '\n'.join(cleaned_lines)
        
    except (SyntaxError, ValueError, AttributeError) as e:
        # Fallback: simple approach - find function definitions and keep last occurrence
        print(f"Warning: Could not parse code with AST, using fallback method: {e}")
        lines = full_program.split('\n')
        
        # Find all function definitions
        function_positions = []
        i = 0
        while i < len(lines):
            line = lines[i]
            func_match = re.match(r'^(\s*)def\s+(\w+)\s*\(', line)
            if func_match:
                func_name = func_match.group(2)
                start = i
                
                # Find end of function
                end = _find_function_end(lines, i)
                
                function_positions.append((func_name, start, end))
                i = end
            else:
                i += 1
        
        # Keep only last occurrence of each function
        # IMPORTANT: Always preserve 'solve' and 'evaluate' functions as they are essential for evaluation
        essential_functions = {'solve', 'evaluate'}
        last_occurrence = {}
        for func_name, start, end in function_positions:
            if func_name not in last_occurrence or start > last_occurrence[func_name][0]:
                last_occurrence[func_name] = (start, end)
        
        # Build set of lines to keep
        keep_lines = set()
        for func_name, (start, end) in last_occurrence.items():
            for j in range(start, end):
                keep_lines.add(j)
        
        # Also preserve all essential functions even if they appear multiple times
        for func_name, start, end in function_positions:
            if func_name in essential_functions:
                for j in range(start, end):
                    keep_lines.add(j)
        
        # Also keep all non-function lines
        function_ranges = [(start, end) for _, start, end in function_positions]
        cleaned_lines = []
        for i, line in enumerate(lines):
            in_function = any(start <= i < end for start, end in function_ranges)
            if not in_function or i in keep_lines:
                cleaned_lines.append(line)
        
        full_program = '\n'.join(cleaned_lines)

    # Check if res has a function signature, if not prepend it
    if func_signature and not re.search(r'^\s*def\s+\w+\s*\(', res.strip(), re.MULTILINE):
        # Function body is missing signature, prepend it
        # Remove return type annotation if present in signature for consistency
        sig_clean = re.sub(r'\s*->\s*[^:]+', '', func_signature).strip()
        if not sig_clean.endswith(':'):
            sig_clean += ':'
        res = f"{sig_clean}\n{res}"
        print(f"   Function body missing signature, prepended: {sig_clean}")
    
    # Before inserting the new function, remove any existing function with the same name
    # Extract function name from res
    # IMPORTANT: Never remove 'solve' or 'evaluate' functions as they are essential for evaluation
    essential_functions = {'solve', 'evaluate'}
    func_name_match = re.search(r'def\s+(\w+)\s*\(', res)
    if func_name_match:
        func_name = func_name_match.group(1)
        # Only remove if it's not an essential function
        if func_name not in essential_functions:
            # Remove any existing function with this name from full_program
            # This ensures the new function replaces any old one
            lines = full_program.split('\n')
            cleaned_lines = []
            i = 0
            while i < len(lines):
                line = lines[i]
                # Check if this is the function definition we want to remove
                existing_func_match = re.match(rf'^\s*def\s+{re.escape(func_name)}\s*\(', line)
                if existing_func_match:
                    # Skip this function - find its end
                    end_idx = _find_function_end(lines, i)
                    i = end_idx
                    continue
                cleaned_lines.append(line)
                i += 1
            full_program = '\n'.join(cleaned_lines)
    
    full_program = f"""
{imports_section}{full_program}
{res}
print(evaluate())
"""
    print(full_program)
    # Normalize Unicode quotes before writing to file (Python will execute this file)
    full_program = _normalize_unicode_quotes(full_program)
    with open(script_path, 'w') as f:
        f.write(full_program.strip())

    try:
                result = subprocess.run(
                            ['python', script_path],
                            capture_output=True,
                            text=True,
                            timeout=300, #this is in seconds
                            check=True,
                            encoding='utf-8'
                        )
                        # Try to parse numerical output
                output = result.stdout.strip()
                # Convert numpy types to native Python types and parse as JSON
                output = output.replace("np.float64", "")
                output = output.replace("np.float32", "")
                output = output.replace("(", "").replace(")", "")

                # # Access the values
                # output, actions_count = result[0], result[1]
                # print("output ", output)
                parsed = ast.literal_eval(output)

                # Expect [total_reward, actions_count, grid_markdown]
                if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
                    total_reward = parsed[0]
                    actions_count = parsed[1]
                    grid_markdown = parsed[2] if len(parsed) > 2 else None
                else:
                    raise ValueError(f"Unexpected parsed format: {parsed}")

                print("total_reward:", total_reward)
                print("actions_count:", actions_count)
                # if grid_markdown:
                if actions_count > 0 :
                  total_reward +=actions_count *0.01
                #     print("grid_markdown:\n", grid_markdown)

                # Track explicit feedback interactions if tracker is available
                if results_tracker is not None and actions_count > 0:
                    results_tracker.add_explicit_feedback_interactions(actions_count)
                
                try:
                    return float(total_reward), True , actions_count
                except ValueError:
                    print("Output is not a float.")
                    return -1, True, 0
                    
    except subprocess.TimeoutExpired:
                    return -1, False, 0
    except subprocess.CalledProcessError as e:
                    print(f"Process Error: Command failed with exit code {e.returncode}")
                    print(f"Command: {e.cmd}")
                    print(f"Output: {e.stdout}")
                    print(f"Error: {e.stderr}")
                    return -1, False, 0
    except Exception as e:
                    print(f"Unexpected Error: {e}")
                    return -1, False, 0 
    finally:
                # Clean up the temporary file
                if os.path.exists(script_path):
                    # print("ugh")
                    os.remove(script_path)
        

def response_gen(funcs: List[Tuple[float, str]], k: int, file: str, 
                 specification: str, func_signature: str, 
                 output_dir: str, shared_vllm=None, results_tracker=None,
                 dsl_round: Optional[int] = None, func_evolution_round: Optional[int] = None) -> Optional[str]:
    """Generate feedback and extract the best improved function.
    
    Args:
        funcs: List of (score, function_body) tuples
        k: Number of functions used
        file: Path to evaluation file
        specification: Specification string
        func_signature: Function signature string (e.g., "def collect(env, primitive) -> list[int]:")
        output_dir: Directory to save results
        shared_vllm: Optional shared vLLM instance
        
    Returns:
        Best function code as string, or None if extraction failed
    """
    funcs_text = "\n\n".join(
        [f"### Score: {score}\n```python\n{body}\n```" for score, body in funcs]
    )
    print(f"\n  Function signature: {func_signature}")
    print(f"  Adding {len(funcs)} functions to prompt:")
    for i, (score, body) in enumerate(funcs, 1):
        body_preview = body[:200] + "..." if len(body) > 200 else body
        print(f"    Function {i} (score: {score:.4f}):")
        print(f"      {repr(body_preview)}")
        print(f"      Length: {len(body)} chars")
    
    prompt = (
        specification
        + f"\n\nHere are different implementations of `{func_signature}`\n"
        + funcs_text
        + "\n\nAnalyse the functions and give natural language feedback in bullet points."
    )

    # Use shared vLLM if available
    # Note: If shared_vllm is None, we create a new one. However, in stages that explicitly
    # want to share an instance (like stage_implement_cfg_single), they should create it first
    # and pass it here to ensure only ONE instance is created and shared.
    if shared_vllm is not None:
        llm = shared_vllm
    else:
        # Fallback: create new instance if shared one not provided
        # This should only happen if the calling stage didn't create a shared instance
        # Use lower memory utilization since multiple jobs may run in parallel
        print("   Warning: No shared vLLM instance provided, creating new one with reduced memory settings")
        llm = vLLM(
            model="/scratch/avani/gpt", 
            tensor_parallel_size=4,
            gpu_memory_utilization=0.6  # Reduced to 60% to handle parallel jobs
        )
    params = SamplingParams(temperature=0.7, max_tokens=35000)
    
    output = llm.generate([prompt], sampling_params=params)
    response = output[0].outputs[0]
    feedback = response.text
    print(f"\n  Generated feedback (length: {len(feedback)} chars)")
    print(f"  Using same {len(funcs)} functions in correction prompt")
    correction_prompt = (
      specification
      + "\n\nFeedback:\n"
      + feedback
      + f"\n\nHere are the candidate functions for `{func_signature}`\n"
      + funcs_text
      + "\n\nReturn a corrected and improved version of the function and ."
    )
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Extract function name from signature for filename
    func_name_match = re.search(r'def\s+(\w+)', func_signature)
    func_name = func_name_match.group(1) if func_name_match else "function"
    
    # build deterministic filename with versioning (no timestamp); we will append entries
    os.makedirs(output_dir, exist_ok=True)
    if dsl_round is not None:
        if func_evolution_round is not None:
            log_filename = os.path.join(output_dir, f"feedback_{func_name}_dsl{dsl_round}_func{func_evolution_round}.json")
        else:
            log_filename = os.path.join(output_dir, f"feedback_{func_name}_dsl{dsl_round}_func0.json")
    else:
        log_filename = os.path.join(output_dir, f"feedback_{func_name}.json")
    
    best_func = None
    best_score = float('-inf')
    best_runs_ok = False
    
    # Store best function from log as fallback (will evaluate only if LLM generation fails)
    best_log_func = None
    best_log_score = float('-inf')
    best_log_runs_ok = False
    
    # Track feedback per iteration for logging
    feedback_entries = []
    
    # Now try to generate improved functions via LLM (single iteration)
    i = 0 # number of iterations
    max_iterations = 1
    while i < max_iterations:
        output = llm.generate([correction_prompt], sampling_params=params)
        response = output[0].outputs[0]
        feedback = response.text
        b = response.text
        try:
            # Extract function code from markdown code block
            if "```python" in b:
                b = b[b.index("```python")+len("```python"):]
                b = b[:b.index("```")].strip()
            elif "```" in b:
                b = b[b.index("```")+3:]
                b = b[:b.index("```")].strip()
                if b.startswith("python"):
                    b = b[6:].strip()
            else:
                # Try to extract function definition directly
                func_match = re.search(rf'def\s+{re.escape(func_name)}\s*\([^)]+\)[^:]*:.*?(?=\n\ndef|\nclass|\Z)', 
                                     b, re.DOTALL)
                if func_match:
                    b = func_match.group(0)
                else:
                    i += 1
                    continue
            
            # Replace the function signature with the exact one from func_signature
            # This ensures it matches what funsearch expects
            if func_signature:
                # Extract just the signature part (def name(params))
                sig_match = re.search(rf'def\s+{re.escape(func_name)}\s*\([^)]+\)', func_signature)
                if sig_match:
                    expected_sig = sig_match.group(0)
                    # Replace the signature in the extracted function
                    b = re.sub(rf'def\s+{re.escape(func_name)}\s*\([^)]+\)', expected_sig, b, count=1)
                else:
                    # If func_signature doesn't match, try to extract params from it
                    sig_params_match = re.search(rf'def\s+\w+\s*\(([^)]+)\)', func_signature)
                    if sig_params_match:
                        expected_params = sig_params_match.group(1)
                        # Replace the function signature with correct params
                        b = re.sub(rf'def\s+{re.escape(func_name)}\s*\([^)]+\)', 
                                 f'def {func_name}({expected_params})', b, count=1)
        except:
            i += 1
            continue
        
        i += 1
        # print(f"Generated function {i}:")
        # print(b[:200] + "..." if len(b) > 200 else b)
        
        # Evaluate the function (pass specification for imports, func_signature, and function_name)
        score, runs_ok, actions_count = eval(
            b,
            file,
            specification=specification,
            func_signature=func_signature,
            function_name=func_name,
            results_tracker=results_tracker,
        )
        
        # Skip writing per-iteration feedback logs
        
        # Track best function (only consider functions that run successfully)
        # Use >= to allow updating when scores are equal (prefer later/better implementations)
        if runs_ok and (best_func is None or score >= best_score):
            best_score = score
            best_func = b
            best_runs_ok = runs_ok
        
        # Record this iteration's output
        feedback_entries.append({
            "iteration": i,
            "feedback": feedback,
            "score": score,
            "runs_ok": runs_ok,
            "env_interactions": int(actions_count) if actions_count is not None else 0,
            "actions_count": int(actions_count) if actions_count is not None else 0,
            "function": b,
        })
    
    # Decide which function to return:
    # 1. Prefer generated function if it runs successfully
    # 2. Fall back to best function from log if generation failed
    # 3. If nothing works, return a stub function that returns []
    
    if best_runs_ok:
        # Generated function works - use it
        print(f"   Using generated function (score: {best_score:.4f})")
        # Ensure the function has a signature before using it
        # Extract function name from signature to check if it's the main function
        func_name_match = re.search(r'def\s+(\w+)', func_signature) if func_signature else None
        func_name = func_name_match.group(1) if func_name_match else None
        
        # Check if the function body contains a COMPLETE function definition with the expected name
        # If it does, we should NOT prepend - the function definition already exists
        # A complete definition must have: def func_name(...): followed by indented code
        has_main_def = False
        if func_name and best_func:
            stripped = best_func.strip()
            # Look for a COMPLETE function definition (def func_name(...): followed by indented body)
            # Pattern: def func_name(...): followed by whitespace and then indented code (starts with space/tab)
            # This ensures we only match complete functions, not just signatures
            complete_func_pattern = rf'def\s+{re.escape(func_name)}\s*\([^)]*\)[^:]*:\s*\n\s+[^\n]'
            if re.search(complete_func_pattern, stripped, re.MULTILINE):
                # Found a complete function definition with this name - don't prepend
                has_main_def = True
            else:
                # Also check if it's at the start and has indented content immediately after
                start_pattern = rf'^\s*def\s+{re.escape(func_name)}\s*\([^)]*\)[^:]*:\s*\n\s+[^\n]'
                if re.search(start_pattern, stripped, re.MULTILINE):
                    has_main_def = True
        
        if func_signature and best_func and not has_main_def:
            # Function body is missing signature, prepend it
            # No need to remove existing definitions since has_main_def=False means none exist
            sig_clean = re.sub(r'\s*->\s*[^:]+', '', func_signature).strip()
            if not sig_clean.endswith(':'):
                sig_clean += ':'
            
            # Indent the function body properly (normalize to 4 spaces)
            # Preserve relative indentation by dedenting to minimum level first
            body_lines = best_func.split('\n')
            
            # Find minimum indentation (excluding empty lines)
            min_indent = None
            for line in body_lines:
                if line.strip():  # Non-empty line
                    indent = len(line) - len(line.lstrip())
                    if min_indent is None or indent < min_indent:
                        min_indent = indent
            
            # Dedent all lines by minimum indent, then add 4-space base indent
            # This preserves relative indentation structure
            indented_body_lines = []
            for line in body_lines:
                if line.strip():  # Non-empty line
                    # Calculate current indent and relative indent
                    current_indent = len(line) - len(line.lstrip())
                    relative_indent = current_indent - (min_indent if min_indent is not None else 0)
                    stripped_line = line.lstrip()
                    # Add 4-space base indent + preserve relative indent
                    indented_body_lines.append('    ' + (' ' * relative_indent) + stripped_line)
                else:
                    indented_body_lines.append(line)  # Keep empty lines as-is
            
            indented_body = '\n'.join(indented_body_lines)
            best_func = f"{sig_clean}\n{indented_body}"
            print(f"    Prepended signature to generated function (with indentation)")
        final_func = best_func
    else:
        # Generated function failed, use best function from log as fallback
        # Functions in log are already evaluated, so we can use their scores directly
        # Filter out functions with score -1 (runs_ok=False) and use the best one
        print(f"   Generated function failed, using best function from log as fallback...")
        
        # Find the best function from log based on existing scores
        # funcs is already sorted by score (highest first) from parse_log_file
        # Filter out functions with score -1 (these had runs_ok=False)
        working_funcs = [(score, func) for score, func in funcs if score > -1]
        
        if working_funcs:
            # Use the function with the highest score (first in the list)
            best_log_score, best_log_func = working_funcs[0]
            
            # Ensure the function has a signature before using it
            # Extract function name from signature to check if it's the main function
            func_name_match = re.search(r'def\s+(\w+)', func_signature) if func_signature else None
            func_name = func_name_match.group(1) if func_name_match else None
            
            # Check if the function body contains a COMPLETE function definition with the expected name
            # If it does, we should NOT prepend - the function definition already exists
            # A complete definition must have: def func_name(...): followed by indented code
            has_main_def = False
            if func_name and best_log_func:
                stripped = best_log_func.strip()
                # Look for a COMPLETE function definition (def func_name(...): followed by indented body)
                # Pattern: def func_name(...): followed by whitespace and then indented code (starts with space/tab)
                # This ensures we only match complete functions, not just signatures
                complete_func_pattern = rf'def\s+{re.escape(func_name)}\s*\([^)]*\)[^:]*:\s*\n\s+[^\n]'
                if re.search(complete_func_pattern, stripped, re.MULTILINE):
                    # Found a complete function definition with this name - don't prepend
                    has_main_def = True
                else:
                    # Also check if it's at the start and has indented content immediately after
                    start_pattern = rf'^\s*def\s+{re.escape(func_name)}\s*\([^)]*\)[^:]*:\s*\n\s+[^\n]'
                    if re.search(start_pattern, stripped, re.MULTILINE):
                        has_main_def = True
            
            if func_signature and not has_main_def:
                # Function body is missing signature, prepend it
                # No need to remove existing definitions since has_main_def=False means none exist
                sig_clean = re.sub(r'\s*->\s*[^:]+', '', func_signature).strip()
                if not sig_clean.endswith(':'):
                    sig_clean += ':'
                
                # Indent the function body properly (normalize to 4 spaces)
                # Preserve relative indentation by dedenting to minimum level first
                body_lines = best_log_func.split('\n')
                
                # Find minimum indentation (excluding empty lines)
                min_indent = None
                for line in body_lines:
                    if line.strip():  # Non-empty line
                        indent = len(line) - len(line.lstrip())
                        if min_indent is None or indent < min_indent:
                            min_indent = indent
                
                # Dedent all lines by minimum indent, then add 4-space base indent
                # This preserves relative indentation structure
                indented_body_lines = []
                for line in body_lines:
                    if line.strip():  # Non-empty line
                        # Calculate current indent and relative indent
                        current_indent = len(line) - len(line.lstrip())
                        relative_indent = current_indent - (min_indent if min_indent is not None else 0)
                        stripped_line = line.lstrip()
                        # Add 4-space base indent + preserve relative indent
                        indented_body_lines.append('    ' + (' ' * relative_indent) + stripped_line)
                    else:
                        indented_body_lines.append(line)  # Keep empty lines as-is
                
                indented_body = '\n'.join(indented_body_lines)
                best_log_func = f"{sig_clean}\n{indented_body}"
                print(f"    Prepended signature to log function (with indentation)")
            
            print(f"   Using best function from log (score: {best_log_score:.4f}, already evaluated)")
            final_func = best_log_func
            # Update best_score and best_runs_ok to reflect the fallback function
            best_score = best_log_score
            best_runs_ok = True  # Functions from log that pass the filter (score > -1) are working
        else:
            
            print(f"   No working functions found. Creating stub function that returns []")
            # Extract function name and parameters from signature
            func_name_match = re.search(r'def\s+(\w+)', func_signature)
            func_name = func_name_match.group(1) if func_name_match else "function"
            
            # Extract parameters from signature
            params_match = re.search(r'def\s+\w+\s*\(([^)]*)\)', func_signature)
            params = params_match.group(1) if params_match else ""
            
            # Create stub function
            final_func = f"""def {func_name}({params}):
    \"\"\"
    Stub implementation - no working function found.
    Returns empty list as fallback.
    \"\"\"
    return []
"""
            print(f"   Created stub function: {func_name}({params}) -> []")
            # Update best_score and best_runs_ok for stub function
            best_score = -1.0  # Stub function has score -1
            best_runs_ok = False  # Stub function doesn't actually run
        
    # Use final_func for the rest of the processing
    best_func = final_func
    
    # Extract imports from specification and add them to the final function
    if best_func:
        import_lines = []
        seen_imports = set()  # Track normalized import strings to prevent duplicates
        
        def normalize_import(import_line):
            """Normalize import line for comparison (remove extra whitespace)."""
            stripped = import_line.strip()
            # Normalize whitespace - collapse multiple spaces into single space
            normalized = ' '.join(stripped.split())
            return normalized
        
        if specification:
            # Extract imports from specification string
            for line in specification.split('\n'):
                stripped = line.strip()
                if stripped.startswith('import ') or stripped.startswith('from '):
                    normalized = normalize_import(stripped)
                    # Avoid duplicate imports by checking normalized version
                    if normalized not in seen_imports:
                        import_lines.append(line)
                        seen_imports.add(normalized)
        
        # Also check the eval file for imports (only top-level imports, not inside functions)
        if os.path.exists(file):
            with open(file, 'r', encoding='utf-8') as f:
                eval_content = f.read()
            for line in eval_content.split('\n'):
                stripped = line.strip()
                # Only extract imports that are at the module level (no leading indentation)
                # This excludes imports inside functions, try blocks, etc.
                if (stripped.startswith('import ') or stripped.startswith('from ')) and not line.startswith((' ', '\t')):
                    normalized = normalize_import(stripped)
                    if normalized not in seen_imports:
                        import_lines.append(line)
                        seen_imports.add(normalized)
        
        # Remove any duplicate imports that might already be in best_func
        # Check if best_func already has imports at the top
        best_func_lines = best_func.split('\n')
        func_imports_start = 0
        func_imports_end = 0
        
        # Find where imports end in best_func (if any)
        for i, line in enumerate(best_func_lines):
            stripped = line.strip()
            if stripped.startswith('import ') or stripped.startswith('from '):
                if func_imports_start == 0:
                    func_imports_start = i
                func_imports_end = i + 1
            elif stripped and func_imports_end > 0:
                # We've hit a non-import, non-empty line, so imports have ended
                break
        
        # Remove duplicate imports from import_lines that already exist in best_func
        if func_imports_end > 0:
            existing_imports = [normalize_import(line) for line in best_func_lines[func_imports_start:func_imports_end]]
            # Filter out imports that already exist in best_func
            import_lines = [line for line in import_lines 
                          if normalize_import(line) not in existing_imports]
        
        # Prepend imports to the final function if any were found
        if import_lines:
            imports_section = '\n'.join(import_lines) + '\n\n'
            best_func = imports_section + best_func
        
        # Format score for display (handle -inf case)
        if best_score == float('-inf'):
            score_str = "-inf"
        else:
            score_str = f"{best_score:.4f}"
        print(f"   Final function ready (score: {score_str}, runs_ok: {best_runs_ok})")
    
    # Append entries to deterministic feedback file (only iteration entries)
    try:
        existing_entries = []
        if os.path.exists(log_filename):
            with open(log_filename, "r", encoding="utf-8") as log_file:
                try:
                    data = json.load(log_file)
                    if isinstance(data, list):
                        existing_entries = data
                    elif isinstance(data, dict) and "entries" in data and isinstance(data["entries"], list):
                        existing_entries = data["entries"]
                except json.JSONDecodeError:
                    pass
        combined_entries = existing_entries + feedback_entries
        with open(log_filename, "w", encoding="utf-8") as log_file:
            json.dump(combined_entries, log_file, indent=2)
    except Exception as e:
        print(f"   Failed to write feedback log: {e}")
    
    return best_func
    

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--logfile", type=str, required=True,
    #                     help="Path to the feedback_sampling.json log file")
    # parser.add_argument("--k", type=int, default=5,
    #                     help="Number of top functions to extract (ties included)")
    # parser.add_argument("--file", type=str, default="evaluation_scripts/eval_collect.py",
    #                     help="Number of top functions to extract (ties included)")
    # args = parser.parse_args()

    with open("config_explicit_feedback.json", "r") as f:
        config = json.load(f)
    print(config.get("logfile"))
    funcs = parse_log_file(config.get("logfile"), k=config.get("k", 5))

    if not funcs:
        print("No functions found.")
        return

    response_gen(funcs, config.get("k"), config.get("file"))

if __name__ == "__main__":
    main()
