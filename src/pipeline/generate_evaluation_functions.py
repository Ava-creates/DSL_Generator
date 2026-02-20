#!/usr/bin/env python3
"""
Module for generating custom solve() function using LLM (reward logic only).
This allows function-specific evaluation logic that doesn't rely solely on environment rewards.
NOTE: evaluate() function is generated separately by the caller using templates.
"""

import ast
import os
import re
import textwrap
from typing import Optional, Tuple, Dict

from src.pipeline.domain_templates import craft_solve_template_for_prompt

try:
    from vllm import LLM as vLLM, SamplingParams
except ImportError:
    vLLM = None
    SamplingParams = None



def generate_custom_evaluation_functions(
    func_name: str,
    description: str,
    func_signature: str,
    return_type: str,
    args: str,
    cfg: str,
    specification: str,
    shared_vllm=None,
    recipes: Optional[Dict] = None,
    task_name: Optional[str] = None
) -> str:
    """Generate custom solve() function using LLM.
    
    The LLM generates function-specific reward computation logic that checks if the function
    actually works correctly, rather than relying solely on environment rewards.
    This is important for functions like WAIT or TURN that don't give rewards.
    
    NOTE: This function ONLY generates solve() function. evaluate() is generated separately
    by the caller using templates.
    
    Args:
        func_name: Name of the function
        description: Description of what the function should do
        func_signature: Function signature (e.g., "def function_name(env, arg)")
        return_type: Return type ("bool", "int", or "list[int]")
        args: Function arguments as comma-separated string (e.g., "arg1, arg2")
        cfg: CFG string for context
        specification: Specification file content (contains domain and environment information)
        shared_vllm: Optional shared vLLM instance for generating evaluation functions
        recipes: Optional recipes dictionary (for finding related items/primitives)
        task_name: Optional task name (e.g., "make[goldarrow]") for context
        
    Returns:
        solve_func_code (only solve() function, not evaluate())
    """
    
    # Build prompt for LLM to generate custom evaluation functions
    prompt = f"""You are generating custom evaluation functions for a domain-specific language (DSL) function.

## Function Information:
- Function name: {func_name}
- Description: {description}
- Signature: {func_signature}
- Return type: {return_type}
- Arguments: {args if args else "None"}

## Context:
{specification[:2000] if specification else "No specification provided"}
"""
    
    # Add task context if available
    if task_name:
        prompt += f"""
## Task Context:
- Task: {task_name}
- For this task, related items/primitives may be needed as function arguments
- Ensure test arguments are items that will be present on the grid
"""
    
    # Add recipe information if available
    if recipes:
        primitives_str = ', '.join(recipes.get('primitives', []))
        recipes_str = ', '.join(recipes.get('recipes', {}).keys())
        prompt += f"""
## Recipe Information:
- Primitives: {primitives_str}
- Available recipes: {recipes_str}
- When testing functions with item/primitive arguments, use items from this list that are on the grid
"""
    
    # Generate the full solve template for the prompt
    solve_template = craft_solve_template_for_prompt(func_name, args)
    
    prompt += f"""

## Task:
Generate ONLY the **reward computation logic** for a solve() function.

**CRITICAL: You are ONLY generating reward logic that goes inside solve() function!**
- You are ONLY generating the reward computation logic that goes inside `solve()`
- The template will handle: state capture before, function call, action execution (accumulating `total_reward` from env.step() rewards), state capture after, grid capture, and `new_reward = 0.0` initialization
- You ONLY need to generate the logic that computes `new_reward` based on whether the function worked correctly
- Read the function description to understand what success means
- Compare state before and after to determine if the function achieved its intended effect
- Do NOT include state capture, function call, action execution, grid capture, or `new_reward = 0.0` - those are templated
- The final return will be `total_reward + new_reward` (environment rewards + your evaluation)

**How to determine if function worked:**
- Read the function description: "{description}"
- Identify what state changes should occur if the function works correctly
- Compare the relevant state attributes before and after executing the function
- Set `new_reward = 5.0` if the intended effect occurred, `0.0` otherwise
- Use partial scores (e.g., 2.5) for partial success if appropriate

**Available state variables (captured by template):**
- `pos_before`, `pos_after` - position before/after (if available)
- `inventory_before`, `inventory_after` - inventory before/after (if available)  
- `dir_before`, `dir_after` - direction before/after (if available)
- You can also access `env._current_state` directly if needed for other attributes

**Complete solve() Template (for context):**
Here is the complete solve() function template. Your task is to generate ONLY the code that goes where `<--- YOUR new_reward LOGIC GOES HERE --->` is marked:

```python
{solve_template}
```

**Your task:** Generate ONLY the code that goes where `<--- YOUR new_reward LOGIC GOES HERE --->` is marked above. This code should update `new_reward` based on whether the function worked correctly.

## Evaluation Guidelines (Domain-Agnostic):

### General Evaluation Approach:
1. **Understand the function's intended effect from its description:**
   - Read the function description: "{description}"
   - Identify what state changes should occur if the function works correctly
   - The function description tells you what success means

2. **State is captured before and after (by template):**
   - Common state attributes are available: pos_before/after, inventory_before/after, dir_before/after
   - You can also access `env._current_state` for other attributes if needed

3. **Actions are executed (by template):**
   - The function is called to get actions
   - Actions are executed in the environment
   - State is captured after execution

4. **Your task: Compute reward by comparing BEFORE and AFTER:**
   - Compare the relevant state attributes before and after
   - Based on the function description, determine if the intended effect occurred
   - Score: 5.0 if the intended effect occurred, 0.0 otherwise
   - Use partial scores (e.g., 2.5) for partial success if appropriate

## Important Guidelines:
- **DO NOT rely solely on env.step() rewards** - many functions may give 0 reward even when they work correctly
- **Check the ACTUAL EFFECT** by comparing environment state before and after executing the function
- **Read the function description** to understand what the function should accomplish
- **The function description is:** "{description}"
- Use the state variables provided (pos_before/after, inventory_before/after, dir_before/after) to check if function worked
- You can also access `env._current_state` directly if you need other state attributes
- Use `try/except` to handle cases where state attributes don't exist
- Use `hasattr()` to check if state attributes exist before accessing them
- Return meaningful scores: 5.0 if function worked perfectly, 0.0 if it failed, partial scores (e.g., 2.5) for partial success
- The evaluation should be based on the function's description and intended effect, not hardcoded assumptions about specific function types


## Function Signature:
```python
{func_signature}
```

**IMPORTANT:** The solve() function MUST use this exact signature pattern:
- For functions with arguments: `def solve(env, {args}, visualise=False):`
- For functions without arguments: `def solve(env, visualise=False):`
- The solve() function MUST call the function using: `{func_name.lower()}(env, {args})` or `{func_name.lower()}(env)` if no args
- The solve() function MUST return: `[total_reward, actions_count, grid_before, grid_after]`

Generate ONLY the reward computation logic.

**CRITICAL INSTRUCTIONS:**
- Return ONLY Python code - NO explanations, NO commentary, NO reasoning, NO descriptions
- Do NOT include any text before or after the code
- Do NOT include phrases like "Function description:", "Implementation:", "Need to", "We'll", "Might", "Could", etc.
- Do NOT include any instructions, notes, or explanations
- Do NOT write commentary about what the code does - just write the code
- Do NOT include function signature or return statement - those are templated
- **ONLY generate the reward computation logic** that updates `new_reward` based on state comparison
- **CRITICAL: Use proper Python indentation (2 or 4 spaces per level). Nested blocks (if, for, try, etc.) must be indented relative to their parent blocks.**

**Your response should be ONLY Python code wrapped in markers - NO explanations, NO commentary, NO reasoning:**

Wrap your reward logic code in these exact markers:
$$$

Inside the markers, provide ONLY the reward computation logic (how to check if the function worked).

**DO NOT include:**
- State capture (pos_before, inventory_before, etc.) - that's templated
- Function call (actions_to_take = ...) - that's templated
- Action execution (for loop with env.step) - that's templated
- State capture after (pos_after, inventory_after, etc.) - that's templated
- `new_reward = 0.0` initialization - that's templated

**ONLY include:**
- The logic to compute new_reward based on comparing state before and after
- Read the function description to understand what success means
- Compare relevant state attributes to determine if the function achieved its intended effect

Example format (note proper indentation - nested blocks must be indented relative to their parents):
$$$
# Compute additional reward based on whether function worked correctly
if pos_before is not None and pos_after is not None:
    # Check if the function achieved its intended effect based on description
    if pos_before != pos_after:
        new_reward = 5.0
    else:
        new_reward = 0.0
else:
    new_reward = 0.0
$$$

**CRITICAL:** 
- Wrap your code in `$$$` markers (start with `$$$` and end with `$$$`)
- Return ONLY the code between the markers
- NO text before or after the markers
- NO explanations or commentary

**Remember:**
- The template already initializes `new_reward = 0.0` right before your code
- Your code will be inserted deterministically right before the return statement
- You only need to update `new_reward` based on your evaluation logic
- Do NOT include `new_reward = 0.0` - it's already in the template
- Do NOT include the return statement - it's already in the template
"""

    # Use shared vLLM if available, otherwise create new one
    if shared_vllm is not None:
        llm = shared_vllm
    else:
        if vLLM is None:
            raise ValueError("vLLM not available and no shared_vllm provided")
        llm = vLLM(model="/scratch/avani/gpt", tensor_parallel_size=4)
    
    if SamplingParams is None:
        raise ValueError("vLLM SamplingParams not available")
    # Use lower temperature for more focused, code-only responses without commentary
    params = SamplingParams(temperature=0.3, max_tokens=4000)
    output = llm.generate([prompt], sampling_params=params)
    response = output[0].outputs[0].text
    
    # Extract only the core evaluation logic from response (reward computation logic)
    core_eval_logic = _extract_core_evaluation_logic(response, func_name, args)
    
    # Generate solve() function from template with grid capture, inserting LLM-generated reward logic
    solve_func =_generate_solve_template (
        func_name=func_name,
        args=args,
        return_type=return_type,
    )
    
    return solve_func


def _extract_core_evaluation_logic(response: str, func_name: str, args: str) -> str:
    """Extract only the reward computation logic from LLM response.
    
    Args:
        response: LLM response text
        func_name: Name of the function
        args: Function arguments
        
    Returns:
        Reward computation logic code (ONLY the code that updates new_reward)
    """
    # First, try to extract from code blocks (most reliable)
    code_block_pattern = r'```python\s*\n(.*?)\n```'
    code_blocks = re.findall(code_block_pattern, response, re.DOTALL)
    
    if code_blocks:
        # Use the first/largest code block
        full_code = code_blocks[0] if code_blocks else ""
        lines = full_code.split('\n')
    else:
        # Use response directly (we'll extract from markers or filter commentary)
        lines = response.split('\n')
    
    core_lines = []
    in_core_section = False
    found_function_call = False
    found_code = False
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        
        # Skip function definitions, decorators, grid capture, and clear commentary
        if (re.match(r'^def\s+', line_stripped) or
            re.match(r'^@funsearch\.(run|evolve)', line_stripped) or
            re.search(r'grid_(before|after)\s*=', line, re.IGNORECASE) or
            re.search(r'#\s*Capture.*grid', line, re.IGNORECASE) or
            re.search(r'from test import grid_to_markdown', line, re.IGNORECASE) or
            # Skip obvious commentary
            re.match(r'^(Function description|Implementation|Need to|We\'ll|Might|Could|Also|Maybe):', line_stripped, re.IGNORECASE)):
            continue
        
        # Detect start of core section (state capture before or calling function)
        if (re.search(r'state_before|pos_before|inventory_before|dir_before', line, re.IGNORECASE) or
            re.search(r'actions_to_take\s*=\s*', line, re.IGNORECASE) or
            re.search(r'actions\s*=\s*', line, re.IGNORECASE) or
            re.search(rf'{func_name.lower()}\s*\(', line, re.IGNORECASE) or
            (re.search(r'#\s*(Execute|Call|Run|Capture)', line, re.IGNORECASE) and 
             not re.search(r'grid', line, re.IGNORECASE))):
            in_core_section = True
            found_code = True
            if re.search(rf'{func_name.lower()}\s*\(', line, re.IGNORECASE):
                found_function_call = True
        
        if in_core_section:
            # Stop at return statement (but don't include it) or grid capture
            if (re.match(r'^\s*return\s+', line) or
                re.search(r'grid_(before|after)\s*=', line, re.IGNORECASE) or
                re.search(r'#\s*Capture.*grid', line, re.IGNORECASE)):
                break
            
            # Skip commentary lines (not code)
            if (line_stripped and 
                not line_stripped.startswith('#') and  # Not a comment
                not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*\s*[=:]', line_stripped) and  # Not assignment
                not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*\s*\(', line_stripped) and  # Not function call
                not re.match(r'^(if|for|while|try|except|return|pass|break|continue|with|import|from)', line_stripped) and  # Not control flow
                (len(line_stripped) > 50 or  # Long lines are likely commentary
                 any(word in line_stripped.lower() for word in ['function', 'description', 'implementation', 'need to', "we'll", 'might', 'could', 'also', 'maybe']))):  # Commentary keywords
                continue  # Skip this commentary line
            
            # Only include actual code lines
            core_lines.append(line)
    
    core_logic = '\n'.join(core_lines).strip()
    
    # Extract from $$$ markers
    # Try format with newlines first: $$$
    marker_pattern = r'\$\$\$\s*\n(.*?)\n\$\$\$'
    marker_match = re.search(marker_pattern, response, re.DOTALL)
    
    # If not found, try without requiring newlines: just $$$
    if not marker_match:
        marker_pattern = r'\$\$\$(.*?)\$\$\$'
        marker_match = re.search(marker_pattern, response, re.DOTALL)
    
    if marker_match:
        marked_logic = marker_match.group(1).strip()
        # Clean up any remaining commentary
        lines = marked_logic.split('\n')
        cleaned_lines = []
        for line in lines:
            line_stripped = line.strip()
            # Skip obvious commentary
            if (line_stripped and 
                not line_stripped.startswith('#') and
                not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*\s*[=:]', line_stripped) and
                not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*\s*\(', line_stripped) and
                not re.match(r'^(if|for|while|try|except|return|pass|break|continue|with|import|from)', line_stripped) and
                len(line_stripped) > 50):
                continue
            cleaned_lines.append(line)
        marked_logic = '\n'.join(cleaned_lines).strip()
        if marked_logic:
            return marked_logic
    
    # If we didn't find proper code, provide a minimal fallback reward logic
    if not core_logic or not found_function_call or not found_code:
        # Provide a basic fallback that checks if state changed
        core_logic = '''# Compute additional reward based on whether function worked correctly
# NOTE: This is a fallback - LLM should generate proper reward logic
# Basic check: if position or inventory changed, give partial credit
if pos_before is not None and pos_after is not None and pos_before != pos_after:
  new_reward = 0.5
elif inventory_before is not None and inventory_after is not None and inventory_before != inventory_after:
  new_reward = 0.5'''
    
    return core_logic


def _generate_solve_template(
    func_name: str,
    args: str,
    return_type: str,
    core_eval_logic: str
) -> str:
    """Generate solve() function from template with grid capture, inserting LLM-generated reward logic.
    
    Args:
        func_name: Name of the function
        args: Function arguments as comma-separated string
        return_type: Return type
        core_eval_logic: LLM-generated core evaluation logic (calling function, executing actions, computing reward)
        
    Returns:
        Complete solve() function code
    """
    # Build function parameters
    if args:
        func_params = f"env, {args}, visualise=False"
        func_call_args = f"env, {args}"
    else:
        func_params = "env, visualise=False"
        func_call_args = "env"
    
    # Get safe function name
    safe_name = func_name.lower().replace('-', '_')
    
    def _build_solve_func(core_logic: str) -> str:
        # Generate solve function with template grid capture and reward logic
        return f'''def solve({func_params}):
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
  
  # Capture state before (for reward computation)
  state_before = {{}}
  if hasattr(env, '_current_state'):
    state = env._current_state
    if hasattr(state, 'pos'):
      state_before['pos'] = tuple(state.pos) if hasattr(state.pos, '__iter__') and not isinstance(state.pos, str) else state.pos
    if hasattr(state, 'inventory'):
      state_before['inventory'] = state.inventory.copy() if hasattr(state.inventory, 'copy') else state.inventory
    if hasattr(state, 'dir'):
      state_before['dir'] = state.dir
    # Also store individual variables for convenience
    pos_before = state_before.get('pos')
    inventory_before = state_before.get('inventory')
    dir_before = state_before.get('dir')
  else:
    pos_before = None
    inventory_before = None
    dir_before = None
  
  # Call function to get actions
  actions_to_take = {safe_name}({func_call_args})
  if actions_to_take is None:
    actions_to_take = []
  
  # Execute actions and accumulate environment rewards
  actions_count = 0
  total_reward = 0.0
  for action in actions_to_take:
    reward, done, obs = env.step(action)
    total_reward += reward
    actions_count += 1
    if done:
      break
  
  # Capture state after (for reward computation)
  state_after = {{}}
  if hasattr(env, '_current_state'):
    state = env._current_state
    if hasattr(state, 'pos'):
      state_after['pos'] = tuple(state.pos) if hasattr(state.pos, '__iter__') and not isinstance(state.pos, str) else state.pos
    if hasattr(state, 'inventory'):
      state_after['inventory'] = state.inventory.copy() if hasattr(state.inventory, 'copy') else state.inventory
    if hasattr(state, 'dir'):
      state_after['dir'] = state.dir
    # Also store individual variables for convenience
    pos_after = state_after.get('pos')
    inventory_after = state_after.get('inventory')
    dir_after = state_after.get('dir')
  else:
    pos_after = None
    inventory_after = None
    dir_after = None
  
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

  # Compute additional reward based on whether function worked correctly (LLM-generated logic)
  new_reward = 0.0
  
  total_reward += new_reward

  # Return [total_reward, actions_count, grid_before, grid_after]
  return [total_reward, actions_count, grid_before, grid_after]
'''

    def _fallback_core_eval_logic() -> str:
        return '''# Fallback reward logic (LLM output failed to parse)
if pos_before is not None and pos_after is not None and pos_before != pos_after:
  new_reward = 0.5
elif inventory_before is not None and inventory_after is not None and inventory_before != inventory_after:
  new_reward = 0.5'''

    solve_func = _build_solve_func(core_eval_logic)
    try:
        ast.parse(solve_func)
        return solve_func
    except SyntaxError:
        fallback_logic = _fallback_core_eval_logic()
        solve_func = _build_solve_func(fallback_logic)
        try:
            ast.parse(solve_func)
        except SyntaxError:
            # Last resort: keep template valid with no-op logic
            solve_func = _build_solve_func("pass")
        return solve_func


def _indent_code(code: str, indent_level: int) -> str:
    """Indent code by a given number of spaces.
    
    Preserves the code's relative indentation by:
    1. Dedenting to the minimum common indentation
    2. Applying the requested base indentation
    
    Args:
        code: Code to indent
        indent_level: Number of spaces to indent
        
    Returns:
        Indented code
    """
    if not code:
        return " " * indent_level + "# No evaluation logic generated"
    
    dedented = textwrap.dedent(code).split('\n')
    if not dedented:
        return ""
    
    base = " " * indent_level
    indented_lines = []
    for line in dedented:
        if not line.strip():
            indented_lines.append("")
        else:
            indented_lines.append(base + line)
    return '\n'.join(indented_lines)




