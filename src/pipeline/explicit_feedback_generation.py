import os
import json
import subprocess
import re
import requests as _requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import List, Tuple, Dict, Any, Optional
import time

try:
    from vllm import SamplingParams
    from vllm import LLM as vLLM
except ImportError:
    vLLM = None
    SamplingParams = None


class _TextOutput:
    def __init__(self, text: str):
        self.text = text


class _CompletionOutput:
    def __init__(self, text: str):
        self.outputs = [_TextOutput(text)]


class OpenAICompatLLMWrapper:
    """Drop-in replacement for a vLLM instance that calls an OpenAI-compatible API.

    Implements the subset of vLLM's ``generate()`` interface used by
    ``response_gen``:
        outputs = llm.generate([prompt], sampling_params=params)
        text    = outputs[0].outputs[0].text
    """

    def __init__(self, key_file: Optional[str] = None) -> None:
        from src.utils.openai_compat_key import resolve_openai_compat_api_key
        self._api_key = resolve_openai_compat_api_key(key_file)
        self._base_url = os.environ.get(
            "OPENAI_COMPAT_BASE_URL", "https://llm.vulcan.alliancecan.ca"
        ).rstrip("/")
        self._model = os.environ.get("OPENAI_COMPAT_MODEL", "qwen3-235b").strip()
        self._timeout_seconds = float(os.environ.get("OPENAI_COMPAT_HTTP_TIMEOUT", "500"))
        self._retry_total = int(os.environ.get("OPENAI_COMPAT_MAX_RETRIES", "4"))
        self._backoff_factor = float(os.environ.get("OPENAI_COMPAT_BACKOFF_FACTOR", "1.0"))
        chat_path = os.environ.get(
            "OPENAI_COMPAT_CHAT_PATH", "/api/chat/completions"
        ).strip()
        if chat_path.startswith("http://") or chat_path.startswith("https://"):
            self._endpoint = chat_path
        else:
            self._endpoint = f"{self._base_url}/{chat_path.lstrip('/')}"

    def generate(self, prompts: list, sampling_params=None) -> list:
        temperature = 0.7
        max_tokens = 35000
        if sampling_params is not None:
            temperature = getattr(sampling_params, "temperature", temperature)
            max_tokens = getattr(sampling_params, "max_tokens", max_tokens)

        results = []
        for prompt in prompts:
            text = self._call(prompt, temperature, max_tokens)
            results.append(_CompletionOutput(text))
        return results

    def _call(self, prompt: str, temperature: float, max_tokens: int) -> str:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        retry = Retry(
            total=self._retry_total,
            connect=self._retry_total,
            read=self._retry_total,
            status=self._retry_total,
            backoff_factor=self._backoff_factor,
            status_forcelist=(408, 429, 500, 502, 503, 504),
            allowed_methods=frozenset(["POST"]),
            raise_on_status=False,
        )
        session = _requests.Session()
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        response = session.post(
            self._endpoint, headers=headers, json=payload, timeout=self._timeout_seconds
        )
        if response.status_code >= 400:
            print(
                f"[OpenAICompatLLMWrapper] API error {response.status_code}: {response.text}"
            )
            return ""
        body = response.json()
        choices = body.get("choices", [])
        if not choices:
            print("[OpenAICompatLLMWrapper] API returned no choices")
            return ""
        first_choice = choices[0]
        message = first_choice.get("message")
        if isinstance(message, dict):
            content = message.get("content", "")
        elif isinstance(first_choice.get("text"), str):
            content = first_choice.get("text", "")
        else:
            content = ""
        if not isinstance(content, str):
            content = str(content)
        return content

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
                 dsl_round: Optional[int] = None, func_evolution_round: Optional[int] = None,
                 num_iterations: int = 1) -> Optional[str]:
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
    funcs = [(float(score), body) for score, body in funcs if body and body.strip()]
    funcs.sort(key=lambda x: x[0], reverse=True)
    pool: List[Tuple[float, str]] = funcs[:max(1, int(k))]

    funcs_text = "\n\n".join(
        [f"### Score: {score}\n```python\n{body}\n```" for score, body in pool]
    )
    print(f"\n  Function signature: {func_signature}")
    print(f"  Starting pool size: {len(pool)} (k={k})")
    for i, (score, body) in enumerate(pool, 1):
        body_preview = body[:200] + "..." if len(body) > 200 else body
        print(f"    Function {i} (score: {score:.4f}):")
        print(f"      {repr(body_preview)}")
        print(f"      Length: {len(body)} chars")

    # Use shared vLLM if available
    # Note: If shared_vllm is None, we create a new one. However, in stages that explicitly
    # want to share an instance (like stage_implement_cfg_single), they should create it first
    # and pass it here to ensure only ONE instance is created and shared.
    if shared_vllm is not None:
        llm = shared_vllm
    elif vLLM is not None:
        print("   Warning: No shared vLLM instance provided, creating new one with reduced memory settings")
        llm = vLLM(
            model="/scratch/avani/gpt",
            tensor_parallel_size=4,
            gpu_memory_utilization=0.6,
        )
    else:
        print("   vLLM not available — using OpenAI-compatible API for explicit feedback")
        llm = OpenAICompatLLMWrapper()
    params = SamplingParams(temperature=0.7, max_tokens=35000)
    
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

    def _ensure_signature(code: str, signature: str) -> str:
        """Ensure returned function code includes the expected def signature."""
        if not code:
            return code
        if re.search(r'^\s*def\s+\w+\s*\(', code, re.MULTILINE):
            return code

        sig_clean = re.sub(r'\s*->\s*[^:]+', '', (signature or '').strip())
        if not sig_clean:
            return code
        if not sig_clean.endswith(':'):
            sig_clean += ':'

        body = code.strip('\n')
        indented = "\n".join((f"    {ln}" if ln.strip() else "") for ln in body.splitlines())
        return f"{sig_clean}\n{indented}\n"
    
    best_func = pool[0][1] if pool else None
    best_score = pool[0][0] if pool else float('-inf')
    best_runs_ok = bool(pool)
    
    # Track feedback per iteration for logging
    feedback_entries = []
    
    # Iterate pool update loop: critique pool -> generate candidate -> evaluate -> optionally replace worst.
    i = 0
    max_iterations = max(1, int(num_iterations))
    while i < max_iterations:
        funcs_text = "\n\n".join(
            [f"### Score: {score}\n```python\n{body}\n```" for score, body in pool]
        )
        prompt = (
            specification
            + f"\n\nHere are different implementations of `{func_signature}`\n"
            + funcs_text
            + "\n\nAnalyse the functions and give natural language feedback in bullet points."
        )

        output = llm.generate([prompt], sampling_params=params)
        response = output[0].outputs[0]
        feedback = response.text
        print(f"\n  Iteration {i + 1}/{max_iterations}: generated feedback (length: {len(feedback)} chars)")
        correction_prompt = (
          specification
          + "\n\nFeedback:\n"
          + feedback
          + f"\n\nHere are the candidate functions for `{func_signature}`\n"
          + funcs_text
          + "\n\nReturn a corrected and improved version of the function and ."
        )

        output = llm.generate([correction_prompt], sampling_params=params)
        response = output[0].outputs[0]
        feedback = response.text
        b = response.text
        extracted_ok = False
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
                    feedback_entries.append({
                        "iteration": i + 1,
                        "feedback": feedback,
                        "score": None,
                        "runs_ok": False,
                        "pool_best_score_before": pool[0][0] if pool else None,
                        "pool_worst_score_before": pool[-1][0] if pool else None,
                        "inserted_into_pool": False,
                        "reason": "candidate_extraction_failed",
                    })
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
                    sig_params_match = re.search(r'def\s+\w+\s*\(([^)]+)\)', func_signature)
                    if sig_params_match:
                        expected_params = sig_params_match.group(1)
                        # Replace the function signature with correct params
                        b = re.sub(rf'def\s+{re.escape(func_name)}\s*\([^)]+\)', 
                                 f'def {func_name}({expected_params})', b, count=1)
            extracted_ok = True
        except:
            feedback_entries.append({
                "iteration": i + 1,
                "feedback": feedback,
                "score": None,
                "runs_ok": False,
                "pool_best_score_before": pool[0][0] if pool else None,
                "pool_worst_score_before": pool[-1][0] if pool else None,
                "inserted_into_pool": False,
                "reason": "candidate_parse_exception",
            })
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
        
        inserted_into_pool = False
        replaced_score = None
        if extracted_ok and runs_ok:
            is_duplicate = any(existing_body.strip() == b.strip() for _, existing_body in pool)
            if not is_duplicate:
                if len(pool) < max(1, int(k)):
                    pool.append((float(score), b))
                    inserted_into_pool = True
                else:
                    worst_idx = min(range(len(pool)), key=lambda idx: pool[idx][0])
                    worst_score = pool[worst_idx][0]
                    if float(score) > float(worst_score):
                        replaced_score = float(worst_score)
                        pool[worst_idx] = (float(score), b)
                        inserted_into_pool = True
            pool.sort(key=lambda x: x[0], reverse=True)

        if pool:
            best_score = float(pool[0][0])
            best_func = pool[0][1]
            best_runs_ok = True
        
        # Record this iteration's output
        feedback_entries.append({
            "iteration": i,
            "feedback": feedback,
            "score": score,
            "runs_ok": runs_ok,
            "env_interactions": int(actions_count) if actions_count is not None else 0,
            "actions_count": int(actions_count) if actions_count is not None else 0,
            "function": b,
            "pool_size_after": len(pool),
            "pool_best_score_after": pool[0][0] if pool else None,
            "pool_worst_score_after": pool[-1][0] if pool else None,
            "inserted_into_pool": inserted_into_pool,
            "replaced_score": replaced_score,
        })

        print(
            f"  Iteration {i}/{max_iterations}: candidate_score={score:.4f} runs_ok={runs_ok} "
            f"inserted={inserted_into_pool} pool_best={pool[0][0] if pool else 'NA'}"
        )
    
    # Decide which function to return:
    # 1. Prefer generated function if it runs successfully
    # 2. Fall back to best function from log if generation failed
    # 3. If nothing works, return a stub function that returns []
    
    if pool:
        final_func = pool[0][1]
        best_score = float(pool[0][0])
        best_runs_ok = True
        print(f"   Final pool best selected (score: {best_score:.4f}, pool_size={len(pool)})")
    else:
        print("   No functions in pool. Creating stub function that returns []")
        func_name_match = re.search(r'def\s+(\w+)', func_signature)
        func_name = func_name_match.group(1) if func_name_match else "function"
        params_match = re.search(r'def\s+\w+\s*\(([^)]*)\)', func_signature)
        params = params_match.group(1) if params_match else ""
        final_func = f"""def {func_name}({params}):
    \"\"\"
    Stub implementation - no working function found.
    Returns empty list as fallback.
    \"\"\"
    return []
"""
        best_score = -1.0
        best_runs_ok = False
        
    # Use final_func for the rest of the processing and ensure a valid signature is present.
    best_func = _ensure_signature(final_func, func_signature)
    
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
