from datetime import date
import textwrap
from typing import List
from .cfg_parser import CFGParser
import matplotlib.pyplot as plt
# ProgramEvaluator no longer used - replaced with CFGEvaluator
try:
    from cfg_evaluator import CFGEvaluator
    CFG_EVALUATOR_AVAILABLE = True
except ImportError:
    CFGEvaluator = None
    CFG_EVALUATOR_AVAILABLE = False 
# import concurrent.futures
import time
from multiprocessing import Pool, cpu_count
from craft import env_factory
import json
import sys
# from google import genai
import ast
import re
# from funsearch.implementation.funsearch import FunSearch
# from funsearch.implementation import config as config_lib
import pandas as pd
import os
import subprocess
from vllm import LLM, SamplingParams
import glob


def eval(res):
    # with tempfile.TemporaryDirectory() as temp_dir:
    # Create unique filename using process ID and timestamp
    temp_dir = os.getcwd()
    unique_id = f"{os.getpid()}_{int(time.time() * 1000000)}"
    script_path = f'explicit_generated_code_{unique_id}.py'
    script_path = os.path.join(temp_dir, script_path)


    full_program = f'''
import numpy as np
import time
import collections
from craft import craft, env, env_factory
import random
def solve(env, visualise=False) -> float:
  """Runs the environment with a collect function that returns list of actions to take and returns total reward."""
  actions_to_take = make_stick(env)
  total_reward = 0.0

  for t in range(len(actions_to_take)):
    action = actions_to_take[t]
    reward, done, observations = env.step(action)
    total_reward += reward
    if done:
      break
  return [total_reward, len(actions_to_take)]

def evaluate() -> float:
  """Evaluates a collecting policy on a set of sample tasks."""
  #max reward is 4
  visualise = False
  recipes_path = "craft/resources/recipes.yaml"
  hints_path = "craft/resources/hints.yaml"
  reward = 0 

  ##easy make stick env 

  env_sampler = env_factory.EnvironmentFactory(
  recipes_path, hints_path, 0, max_steps=300, reuse_environments=False,
            visualise=visualise)

  env = env_sampler.sample_environment(task_name= 'make[stick]')
  reward = solve(env,  visualise=visualise)
  return reward
{res}
print(evaluate())
'''
    # print(full_program)
    with open(script_path, 'w') as f:
        f.write(full_program.strip())

    try:
        result = subprocess.run(
                    ['python', script_path],
                    capture_output=True,
                    text=True,
                    timeout=400, #this is in seconds
                    check=True,
                    encoding='utf-8',
                    errors='replace'
                )
                # Try to parse numerical output
        output = result.stdout.strip()
        print(output)
        # Convert numpy types to native Python types and parse as JSON
        output = output.replace("np.float64", "")
        output = output.replace("np.float32", "")
        output = output.replace("(", "").replace(")", "")
        result = ast.literal_eval(output)

        # Access the values
        output, actions_count = result[0], result[1]
        print("output ", output)
        try:
            print(output)
            return float(output), True, actions_count, None
        except ValueError:
            return -1, True, 0 , None
    except subprocess.TimeoutExpired:
        return -1, False, 0, None
    except subprocess.CalledProcessError as e:
        print(f"Process Error: Command failed with exit code {e.returncode}")
        print(f"Command: {e.cmd}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return -1, False, 0 , e.stderr
    except Exception as e:
        print(f"Some other error occurred: {e}")
        return -1, False, 0 , "Some error"
    finally:
        # Clean up the temporary file
        if os.path.exists(script_path):
            os.remove(script_path)

def _get_final_function_descriptions(experiment_dir: str, dsl_round: int = None, func_evolution_round: int = None) -> str:
    """Load final function source code from experiment_dir/final_functions for prompt context.
    
    Args:
        experiment_dir: Path to experiment directory
        dsl_round: DSL evolution round number (optional)
        func_evolution_round: Function evolution round number (optional)
        
    Raises:
        ValueError: If experiment_dir is not provided
        FileNotFoundError: If final_functions directory doesn't exist
        FileNotFoundError: If specific versioned functions are requested but not found
    """
    if not experiment_dir:
        raise ValueError("experiment_dir must be provided")
    
    final_functions_dir = os.path.join(experiment_dir, "final_functions")
    if not os.path.isdir(final_functions_dir):
        raise FileNotFoundError(f"final_functions directory not found: {final_functions_dir}")
    
    parts = []
    
    # If specific rounds are provided, look for exact versioned files
    if dsl_round is not None and func_evolution_round is not None:
        # Look for files with specific DSL and function rounds
        pattern = f"*_dsl{dsl_round}_func{func_evolution_round}.py"
        versioned_files = glob.glob(os.path.join(final_functions_dir, pattern))
        
        if not versioned_files:
            available_files = [os.path.basename(f) for f in glob.glob(os.path.join(final_functions_dir, "*.py")) if not os.path.basename(f).startswith("__")]
            raise FileNotFoundError(
                f"No final functions found for DSL round {dsl_round}, function round {func_evolution_round}. "
                f"Pattern searched: {pattern}\n"
                f"Available files: {available_files}"
            )
        
        for path in sorted(versioned_files):
            name = os.path.basename(path)
            if name.startswith("__"):
                continue
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            parts.append(f"## {name}\n```python\n{content}\n```")
    
    # If only DSL round is provided, look for func0 files
    elif dsl_round is not None and func_evolution_round is None:
        pattern = f"*_dsl{dsl_round}_func0.py"
        dsl_files = glob.glob(os.path.join(final_functions_dir, pattern))
        
        if not dsl_files:
            available_files = [os.path.basename(f) for f in glob.glob(os.path.join(final_functions_dir, "*.py")) if not os.path.basename(f).startswith("__")]
            raise FileNotFoundError(
                f"No final functions found for DSL round {dsl_round} (func0). "
                f"Pattern searched: {pattern}\n"
                f"Available files: {available_files}"
            )
        
        for path in sorted(dsl_files):
            name = os.path.basename(path)
            if name.startswith("__"):
                continue
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            parts.append(f"## {name}\n```python\n{content}\n```")
    
    # If no rounds specified, load all .py files
    else:
        for path in sorted(glob.glob(os.path.join(final_functions_dir, "*.py"))):
            name = os.path.basename(path)
            if name.startswith("__"):
                continue
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            parts.append(f"## {name}\n```python\n{content}\n```")
    
    if not parts:
        raise FileNotFoundError(f"No valid Python files found in {final_functions_dir}")
    
    return "\n\n".join(parts)

def grid_to_markdown(grid, cookbook, agent_pos=None) -> str:
    width, height, n_kinds = grid.shape
    inv_index = cookbook.index.reverse_contents  # index -> item name

    table = []
    for y in range(height):  # row by row
        row = []
        for x in range(width):
            cell_items = [inv_index[k] for k in range(1, n_kinds) if grid[x, y, k] == 1]
            cell_repr = ",".join(cell_items) if cell_items else "."
            # Mark the agent start location
            if agent_pos and (x, y) == agent_pos:
                cell_repr = f"Agent({cell_repr})" if cell_repr != "." else "Agent"
            row.append(cell_repr)
        table.append(row)

    df = pd.DataFrame(table)
    return df.to_markdown(index=False, headers=[])


def generate_funsearch_function(term_text, func_dir="function_specific_prompts"):
    """
    Extract terminal functions from term_text and generate FunSearch-compatible files.
    """
    os.makedirs(func_dir, exist_ok=True)
    # print(term_text)
    funcs = re.findall(
        r"-\s*\*\*(\w+)\((.*?)\)\*\*\s*:?\s*(?:\n\s*)?(.*?)(?=\n-\s*\*\*|\Z)",
        term_text,
        flags=re.DOTALL | re.MULTILINE,
    )
    # Normalize whitespace and strip each capture
    funcs = [
        (name.strip(), args.strip(), re.sub(r"\s+\n\s+", "\n", desc).strip())
        for name, args, desc in funcs
    ]

    if not funcs:
        print("No terminal functions detected.")
        return
    print(funcs)
    for func_name, func_args, func_desc in funcs:
        func_file = os.path.join(func_dir, f"{func_name.lower()}.txt")
        timestamp = date.today().strftime("%Y%m%d_%H%M")

        # sanitize function name for a valid Python identifier
        safe_name = re.sub(r'\W|^(?=\d)', '_', func_name.lower())
        args = func_args.strip() or "env"
        with open("prompt_specifications/codebase.txt", "r") as f:
            codebase = f.read()
        with open("craft/resources/recipes.yaml", "r") as f:
            recipes = f.read()
        content = f'''"""
    You are a Funsearch module generator. You fill in the <TODO> tags in the template below.

    ###Here is the Craft codebase you can use to help with your TODO tasks:
    {codebase}

    ### Recipes for the codebase:
    {recipes}

    Return the template exactly  as it is under this with the only changes in the <TODO></TODO> tags. Only have the template before in the final answer. 
    """
    def solve(env, {args}):
    """Runs the environment with a collect function that returns list of actions to take and returns total reward."""
    actions_to_take = {safe_name}(env, {args})
    <TODO>
    #You can update the reward design here if needed as per the function you are implementing 
    total_reward = 0.0

    for t in range(len(actions_to_take)):
        action = actions_to_take[t]
        reward, done, observations = env.step(action)
        total_reward += reward
        if done:
        break

    return total_reward
    </TODO>

    @funsearch.run
    def evaluate():
        \"\"\"Evaluates {func_name} behavior in a Craft sample environment.\"\"\"
        visualise = False
        recipes_path = "craft/resources/recipes.yaml"
        hints_path = "craft/resources/hints.yaml"

        env_sampler = env_factory.EnvironmentFactory(
        recipes_path, hints_path, max_steps=300, reuse_environments=False,
        visualise=visualise
        )

        env = env_sampler.sample_environment(task_name='get[gem]')

        <TODO>
        # Example placeholder logic for evaluation — modify as needed.
        # For example, test whether the function correctly changes environment state.
        </TODO>
        <TODO
        return 0.0 # Replace with actual reward calculation
        </TODO>

    @funsearch.evolve
    def {safe_name}(env, {args}):
        \"\"\"{func_desc}\"\"\"
        return []

    '''
        
        conversation  = [{"role": "user", "content": content,  "chat_template_kwargs": {"reasoning_effort": "high"}}]
        response = llm.chat(conversation, params)
        # print(response[0].outputs[0].text)
        response = response[0].outputs[0].text
        print(type(response))
        print(response)
        marker_match = re.search(r'assistantfinal```python', response, re.IGNORECASE)
        search_target = response
        if marker_match:
                # take the text after the "assistantfinal" marker
            search_target = response[marker_match.end():]

        

        code = textwrap.dedent(search_target)
        with open(func_file, "w") as f:
            f.write(code + "\n")

    return funcs

def generate_funsearch_function_using_another_prompt(term_text, func_dir="function_specific_prompts"):
    """
    Extract terminal functions from term_text and generate FunSearch-compatible files.
    """
    os.makedirs(func_dir, exist_ok=True)
    # print(term_text)
    funcs = re.findall(
        r"-\s*\*\*(\w+)\((.*?)\)\*\*\s*:?\s*(?:\n\s*)?(.*?)(?=\n-\s*\*\*|\Z)",
        term_text,
        flags=re.DOTALL | re.MULTILINE,
    )
    # Normalize whitespace and strip each capture
    funcs = [
        (name.strip(), args.strip(), re.sub(r"\s+\n\s+", "\n", desc).strip())
        for name, args, desc in funcs
    ]

    if not funcs:
        print("No terminal functions detected.")
        return
    print(funcs)
    for func_name, func_args, func_desc in funcs:
        func_file = os.path.join(func_dir, f"{func_name.lower()}.txt")
        timestamp = date.today().strftime("%Y%m%d_%H%M")

        # sanitize function name for a valid Python identifier
        safe_name = re.sub(r'\W|^(?=\d)', '_', func_name.lower())
        args = func_args.strip() or "env"
        with open("prompt_specifications/codebase.txt", "r") as f:
            codebase = f.read()
        with open("craft/resources/recipes.yaml", "r") as f:
            recipes = f.read()
        content = f'''"""
### Here is the Craft codebase you can use for context:
{codebase}

### Recipes for the codebase:
{recipes}

### Given:
- Function name: {func_name}
- Description: {func_desc}
- Arguments: {args}

Please consult the codebase and recipes to write the complete Python function definition, including only:
1. The function signature (starting with "def {func_name}(...):")
2. A detailed docstring describing the function’s purpose, arguments, and list as return value as we always return list of actions.
Do not include the implementation body.
"""
'''
        
        conversation  = [{"role": "user", "content": content,  "chat_template_kwargs": {"reasoning_effort": "high"}}]
        response = llm.chat(conversation, params)
        # print(response[0].outputs[0].text)
        response = response[0].outputs[0].text
        print(type(response))
        print(response)

        # with open(func_file, "w") as f:
        #     f.write(code + "\n")

def extract_and_save_cfg(output_text, cfg_dir="cfg"):
    """
    Extracts three sections from a model output:
      1. Failure Analysis
      2. Updated CFG (BNF)
      3. Terminal Functions

    Returns:
      (filepath, cfg_text, term_text, failure_text)

    - If CFG not found, returns (None, None, None, failure_text)
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
        return None, None, None, failure_text

    cfg_text = cfg_match.group(1) if cfg_match else ""
    cfg_explanation = cfg_explanation.group(1).strip() if cfg_explanation else ""
    term_text = term_match.group(1).strip() if term_match else ""

    # --- Save CFG to file ---
    
    os.makedirs(cfg_dir, exist_ok=True)
    filename = "cfg_updated.txt"
    filepath = os.path.join(cfg_dir, filename)

    with open(filepath, "a", encoding="utf-8") as f:
            f.write(cfg_text)
            f.write(cfg_explanation)
            f.write(failure_text)
            f.write(term_text)

    # # --- Print diagnostics ---
    # if failure_text:
    #     print("\n Extracted Failure Analysis:\n")
    #     print(failure_text)

    # if term_text:
    #     print("\n Extracted Terminal Functions:\n")
    #     print(term_text)

    return None, cfg_text, term_text, failure_text, cfg_explanation


# Example usage:
# output = """ your assistant response here """
# extract_and_save_cfg(output)
llm = LLM(model="/scratch/avani/gpt",     tensor_parallel_size=4 )
params = SamplingParams(temperature=0.7, max_tokens=25000)
final =[]
recipes_path = "craft/resources/recipes.yaml"
hints_path = "craft/resources/hints.yaml"
# evaluator will be created in synthesis_llm() where CFG is available
evaluator = None
env_sampler = env_factory.EnvironmentFactory(
            recipes_path, hints_path, 7, max_steps=400, 
            reuse_environments=False, visualise=False)


with open("src/prog_synth_pipeline/task_config.json", "r") as f:
        config = json.load(f)
        tasks = config["tasks"]
        time_limits = config["time"]

def is_terminal(symbol: str, cfg: CFGParser) -> bool:
    return symbol not in cfg.non_terminals

def evaluate_program_with_evaluator(evaluator, program_str: str, env, max_steps=400) -> int:
    """
    Evaluate a program using CFGEvaluator.
    """
    if evaluator is None:
        raise ValueError("Evaluator is None. Cannot evaluate program.")
    result = evaluator.evaluate_program(program_str, env=env, max_steps=max_steps)
    # Map CFGEvaluator result format to expected format
    actions = result.get("actions_taken", [])
    success = result.get("success", False)
    total_reward = result.get("total_reward", 0.0)
    evaluation_time = result.get("evaluation_time", 0.0)
    
    # Check if program ran out of steps
    max_steps_reached = result.get("max_steps_reached", False)
    steps_taken = result.get("steps_taken", len(actions))
    failure_reason = None
    
    if not success:
        if max_steps_reached or steps_taken >= max_steps:
            failure_reason = f"Program ran out of steps (exceeded max_steps limit of {max_steps})"
        else:
            failure_reason = result.get("failure_reason", "Program failed to complete task")
    
    # CFGEvaluator doesn't provide these, so use defaults
    func = None
    interactions = [len(actions)] if actions else [0]
    rewards = [total_reward] if success else [0.0]
    return actions, success, total_reward, evaluation_time, func, interactions, rewards, failure_reason

def plot_watermark(data, task):
    # print(data)
    # print("here")
    x_vals = []
    y_vals = []

    cumulative_x = 0
    max_y = float('-inf')

    for a, b in data:
        if a != -1:
            cumulative_x += a  # sum of first values ignoring -1
        max_y = max(max_y, b)  # running max of second values
        x_vals.append(cumulative_x)
        y_vals.append(max_y)
    # print(len(y_vals))
    plt.figure(figsize=(8,4))
    plt.plot(x_vals, y_vals, marker='o', linestyle='-', alpha=0.8)
    plt.title('Reward vs Interactions')
    plt.xlabel('Number of Interactions')
    plt.ylabel('Reward')
    plt.grid()
    plt.savefig(f'results/plots/plot_{date.today().isoformat()}_{task}.png')
    plt.close()

def plot_interactions_rewards(interactions, rewards, task):
        plt.figure(figsize=(10, 5))
        # plt.plot(interactions, label='Interactions', marker='o')
        plt.plot(rewards, label='Rewards', marker='x')
        plt.title(f'Interactions and Rewards for Task: {task}')
        plt.xlabel('Interactions')
        plt.ylabel('Cummulative Reward')
        plt.legend()
        plt.grid()
        plt.savefig(f'plot_{task}.png')
        plt.close()

def format_program(tokens: List[str]) -> str:
    result = []
    i = 0
    while i < len(tokens):
        if tokens[i] == "MOVE_FUNC":
            result.append(f"MOVE_FUNC({tokens[i+2]})")
            i += 4
        elif tokens[i] == "CRAFT_FUNC":
            result.append(f"CRAFT_FUNC({tokens[i+2]})")
            i += 4
        elif tokens[i] == "COLLECT_FUNC":
            result.append(f"COLLECT_FUNC({tokens[i+2]})")
            i += 4
        elif tokens[i] == "if":
            # Skip LPAR and RPAR, use the actual item
            item = tokens[i+2]
            if item == "LPAR":
                item = tokens[i+3]
            if tokens[i+4] == "RPAR":
                i += 1  # Skip RPAR
            result.append(f"if has({item})")
            i += 4
        elif tokens[i] == "then":
            result.append("then")
            i += 1
        elif tokens[i] == "SEMI":
            result.append(";")
            i += 1
        else:
            result.append(tokens[i])
            i += 1
    return " ".join(result)

def tokenize_rhs(rhs: str) -> List[List[str]]:
    alternatives = [alt.strip().split() for alt in rhs.split('|')]
    return alternatives

def evaluate(program_str, task , env, inter, reward):
    results = set()
    if evaluator is None:
        # Fallback: return failure if evaluator not initialized
        return [], program_str, {0}, False, 0.0, 0.0, task, None, [0], [0.0], None
    result = evaluator.evaluate_program(program_str, env=env, max_steps=400)
    results.add(1 if result.get("success", False) else 0)
    # Map CFGEvaluator result format to expected format
    actions = result.get("actions_taken", [])
    success = result.get("success", False)
    total_reward = result.get("total_reward", 0.0)
    evaluation_time = result.get("evaluation_time", 0.0)
    
    # Check if program ran out of steps
    max_steps_reached = result.get("max_steps_reached", False)
    steps_taken = result.get("steps_taken", len(actions))
    failure_reason = None
    
    if not success:
        if max_steps_reached or steps_taken >= 400:
            failure_reason = "Program ran out of steps (exceeded max_steps limit)"
        else:
            failure_reason = result.get("failure_reason", "Program failed to complete task")
    
    # CFGEvaluator doesn't provide these, so use defaults
    func = None
    interactions = [len(actions)] if actions else [0]
    rewards = [total_reward] if success else [0.0]
    return actions, program_str, results, success, total_reward, evaluation_time, task, func, interactions, rewards, failure_reason
        
def eval_pll(programs, num_workers=None):
    if num_workers is None:
        num_workers = cpu_count()  # use all available cores
        print(num_workers)

    results = {}
    with Pool(processes=num_workers) as pool:
        for prog, res, s, r, eval_time, task_name, func in pool.map(evaluate, programs):
            results[prog] = res
            print(func)
            with open("solutions.txt", "a") as f:              
                    f.write(f"{task_name}: {prog}, solution: {s}, reward: {r}, evaluation_time: {eval_time:.4f}s\n")
    return results


def synthesis_llm(experiment_dir: str = None, dsl_round: int = None, func_evolution_round: int = None):
    global evaluator
    # llm = LLM(model="/scratch/avani/gpt",     tensor_parallel_size=4 )
    # params = SamplingParams(temperature=0.7, max_tokens=15000)
    with open("cfg/cfg.txt") as f:
        cfg = f.read()
    with open("craft/resources/recipes.yaml") as f:
        recipes = f.read()
    
    # Try to create CFGEvaluator if available and final_functions_dir exists
    if CFG_EVALUATOR_AVAILABLE:
        # Try to find final_functions directory
        possible_dirs = [
            "final_functions",
            "experiment_*/final_functions",
            "../final_functions"
        ]
        final_functions_dir = None
        for pattern in possible_dirs:
            import glob
            matches = glob.glob(pattern)
            if matches:
                final_functions_dir = matches[0]
                break
        
        if final_functions_dir and os.path.exists(final_functions_dir):
            try:
                evaluator = CFGEvaluator(cfg=cfg, final_functions_dir=final_functions_dir)
                print(f" Created CFGEvaluator with functions from {final_functions_dir}")
            except Exception as e:
                print(f" Warning: Could not create CFGEvaluator: {e}")
                evaluator = None
        else:
            print(" Warning: final_functions directory not found. Evaluator will not be available.")
            evaluator = None
    else:
        print(" Warning: CFGEvaluator not available. Evaluation will not work.")
        evaluator = None
    extra_body={"reasoning_effort": "high"}
#    client = genai.Client()
    cfg_explanation = (
        "The CFG defines the DSL for the Craft domain with three primary primitives: "
        "MOVE_FUNC(direction), COLLECT_FUNC(item) and CRAFT_FUNC(item). "
        "MOVE_FUNC moves the agent one cell in the given direction. "
        "COLLECT_FUNC makes the agent pathfind to and pick up a primitive resource. If the resource is blocked and the tool that can get through the obstacle blocking the resource is in the inventory then collect will be able to get through the obstacle and collect the primitive."
        "to overcome obstacles when necessary. "
        "CRAFT_FUNC moves the agent to the required workshop and crafts the item if the needed ingredients "
        "are present in the agent's inventory. "
    )
    first_failing_funcs = []
    programs = []
    tasks = ["get[gem]", "get[gem]"]
    reasoning = {}
    for task in tasks:
        plot =[]
        # print(task)
        plot.append((0,0))
        env = env_sampler.sample_environment(task_name=task)
        inter, reward, = 0, 0 
        interactions = []
        rewards = []
        markdown = grid_to_markdown(env._current_state.grid, env.world.cookbook, env._current_state.pos)
        # print(markdown)
        s = False
        program = "MOVE_FUNC(UP) ;"
        # print(prompt)
        # break
        programs= []
        programs.append(program)
        failed_programs_info = []  # Store programs with their failure reasons
        
        # Load final function implementations if experiment_dir is provided
        final_functions_descriptions = ""
        if experiment_dir:
            final_functions_descriptions = _get_final_function_descriptions(experiment_dir, dsl_round, func_evolution_round)
        
        #change to 128 
        for i in range(30):
            # Create detailed failed programs string with failure reasons
            if failed_programs_info:
                programs_str = "\n".join([f"Program: {prog}\nFailure reason: {reason}" for prog, reason in failed_programs_info])
            else:
                programs_str = "\n".join(programs)  
            prompt = f"""
    You are a Domain Specific Language (DSL) program generator for the Craft domain. 

    ### Start State
    {markdown}

    ## Natural Language Description
    Craft is a single-agent game in a pre-specified environment. 
    The environment of craft is a grid world of size n * n. Each cell can be empty, contain an item, or part of natural terrain or functional structures. When the cell is nonempty, it is considered as blocked. A agent can move around the environment freely through empty cells. At each step, the agent can either move or perform a specific actions, such as collect or craft, towards the immediate cell that it is facing towards. 
    At the beginning of each episode, the agent is placed at a starting cell and a distribution of items across the grid is initialized. The agent’s tasks involve either collecting primitives (raw resources) or crafting items. A item can only be crafted at the specific workshop mentioned in the recipes. 
    The item to be craft are produced from primitives (or other crafted items) by following recipes. Each recipe specifies which items are required and at which workshop the crafting must occur. A primitive item might not need to be crafted but just collected. More complex items, such as arrow, bridge, hammer, axe or flag, require intermediate items along with primitives. This all is specified in the recipe file of the environment. Please note a item can only be crafted at the specific workshop mentioned in the recipes. 
    In this domain, primitives may sometimes be blocked by obstacles. Obstacles are entities that are part of the recipe but are not primitives, workshops, or boundaries. To reach the blocked primitives, the agent must identify and use appropriate tools to remove or bypass these obstacles.
    The correspondence between tools and obstacles is not predefined or known a priori. It cannot be inferred from real-world knowledge or semantic associations. Instead, the correct relationships must be discovered empirically through exploration and interaction within the environment, by observing which tools succeed or fail when applied to different obstacles.
    Primitives used to craft an item has no relation to it being the tool that helps pass an obstacle.
    This is the schema of the recipes:

    recipes:
        item:
        primtive: count of primtive
        _at: at what workshop does the primitve needs to be crafted


    ## Available Recipes
    Here are the recipes for the domain:
    {recipes}
    ## Context Free Grammar (CFG)
    Here is the context-free grammar (CFG) that defines the DSL. Strictly follow this CFG when synthesising programs :

    {cfg}

    {cfg_explanation}

    ## Final Function Implementations
    Here are the current implementations of the terminal functions used in the DSL:

    {final_functions_descriptions}

    ## Example Programs
    Here are examples of programs written in this DSL:

    COLLECT_FUNC(WOOD) ; MOVE_FUNC(RIGHT) ; CRAFT_FUNC(STICK) ;
    COLLECT_FUNC(GRASS) ; MOVE_FUNC(RIGHT) ;

    ## Domain Context
    This DSL is used to solve tasks in the Craft domain. Tasks typically look like:
    - get(wood)
    - make(stick)

    The goal is to write programs in the context free grammar (CFG) provided that can complete these tasks using the available functions and recipes.

    ## Task
    Generate a program that solves the following task :

    **{task}**

    ## Output Format Instructions
    Return ONLY the program string delimited by $ signs. Do not include any explanations, comments, or additional text outside the $ delimiters.
    Example output ->
    $MOVE_FUNC(UP) ;$

    ##Previous programs that FAILED to solve the task:
    {programs_str}
    These programs are syntactically correct but did not solve the task. When generating a new program, avoid repeating the mistakes made in these failed programs, and generate semantically different programs. Try to explore the environment in different ways.
    
    IMPORTANT: Some programs may have failed because they ran out of steps (exceeded the maximum step limit of 400). If this is the case, try to generate more efficient programs that can complete the task in fewer steps.


    Also always ensure that the the information provided in this prompt is facts and always correct and cannot be changed so please adhere to it strictly.
    ##Return a program that is able to solve the task that is different from the previous failed programs.
    
    """
            # response = client.models.generate_content(
            #                 model="gemini-2.5-pro", contents = prompt
            #             )

            # payload = {
            #   "model": "gpt-oss:latest", 
            #   "prompt": prompt, 
            #   "template": "{{.Prompt}}",
            #   "stream": False, 
            #   "options": {
            #     "num_ctx": 4096, 
            #     # "stop": self.stop_tokens
            #   }
            # }
            # api_url = "http://129.128.243.184:11434/api/generate"
            # headers = {"Content-Type": "application/json"}
            # response = requests.post(api_url, headers=headers, json=payload, timeout=300)
            # response = response.json()["response"]
            conversation = [{"role": "user", "content": prompt,  "chat_template_kwargs": {"reasoning_effort": "high"}}]
            output = llm.chat(conversation, params)
            # output = llm.generate([prompt], params, extra_body)
            response = output[0].outputs[0].text
            # print("Raw response:", response)
            # response = response.text for gemini
            # b = response.strip('$ ')
            marker_match = re.search(r'assistantfinal', response, re.IGNORECASE)
            search_target = response
            if marker_match:
                # take the text after the "assistantfinal" marker
                search_target = response[marker_match.end():]
            b = re.search(r'\$(.*?)\$', search_target, re.DOTALL)
            os.makedirs("results/program_synthesis", exist_ok=True)
            print(b)
            if b:
                b = b.group(1)
            else:
                continue
            print(b)
            # Check if all alphabetic characters in b are lowercase
            is_all_upper = all((not ch.isalpha()) or ch.isupper() for ch in b)
            if not is_all_upper:
                continue
            program = b
            # print(b)
            # b= "COLLECT_FUNC(WOOD) ;  COLLECT_FUNC(IRON) ; CRAFT_FUNC(BRIDGE) ;"
            programs.append(b)
            a, program_str, results, s, r, eval_time, task, funcs, interact, rewa, failure_reason = evaluate(b, task ,env, inter, reward)
            # a, program_str, results, s, r, eval_time, task, funcs, interact, rewa = None, b, None, False, None, None, task, None, [0], 0
            # s = input("Enter s (True/False): ").strip().lower() in ("true", "t", "1", "yes", "y")
            # rewa = [float(input("Enter rewa: "))]
            
            # Print feedback about program execution
            if not s and failure_reason:
                print(f"Program failed: {program_str}")
                print(f"Failure reason: {failure_reason}")
                if "ran out of steps" in failure_reason:
                    print("⚠️  This program exceeded the maximum step limit - consider generating more efficient programs")
            interactions += interact
            rewards += rewa
            inter+= interactions[-1] if interactions else 0
            reward+= rewards[-1] if rewards else 0
            if not s:
                reasoning[program_str] = {
                    "generation_reasoning": response,
                    "failure_reason": failure_reason
                }
            os.makedirs(f"results/program_synthesis/{task}", exist_ok=True)
            record = {
                "task": task,
                "program": program_str,
                "success": bool(s),
                "total_reward": r,
                "eval_time": eval_time,
                "interactions": interact,
                "rewards": rewa,
                "failure_reason": failure_reason if not s else None,
            }
            json_path = os.path.join(f"results/program_synthesis/{task}", f"programs_results_date_{date.today().isoformat()}.jsonl")

            with open(json_path, "a", encoding="utf-8") as jf:
#                jf.write(json.dumps(record) + "\n")
                 jf.write(json.dumps(record, default=lambda o: float(o) if hasattr(o, 'dtype') else str(o)) + "\n")
               
            plot.append((len(interact), r))
            # print(a, program_str, results, s, r, eval_time, task, funcs )
            if s :
                break
            else:
                program = b
                # Add failed program with its failure reason to the list
                failed_programs_info.append((b, failure_reason if failure_reason else "Unknown failure reason"))

                # find_bad_func(funcs, task)
        # # plot_interactions_rewards(interactions, rewards, task)
        # plot_watermark(plot, task)
            # programs = programs_str
        # Removed dead code: redundant if True block
                "role": "user",
                "content": f"""
            The following is the failure analysis for the unsuccessful DSL programs:

            {failure_analysis}
            
            Previusly failed programs:
            {programs_str}
            Use this failure analysis along with the previous failed programs to improve the current CFG for the DSL in order to synthesise better programs that can solve the task: {task}.
            ---

            ### Current CFG (Context-Free Grammar) for the current DSL:
            {cfg}
            ### Current CFG explanation:
            {cfg_explanation}
            ### Here are the recipes for the domain, only these items can be used in the programs. You cannot propose any new items that are not in the recipes:
            {recipes}

            ---

            ### Context

            You are an expert in **DSL and program synthesis** for the **Craft** environment.

            Each program above was generated using the current CFG but failed to complete the task.  
            Your task is to analyze these failures and propose **targeted improvements**.

            You need to:
            1. Identify **gaps or weaknesses** in the current CFG.  
            2. Suggest **specific additions or removal of terminal functions** to the CFG that would enable better synthesis results.  
            3. Provide **solution explpanation** where the proposed changes are justified based on the failure analysis.
            ---

            ### Output Format

            Your response must strictly follow this structure:

            Updated CFG(BNF)
            <return the full updated CFG in BNF format here>

            Updated CFG Explanation
            <Write a comprehensive, standalone explanation of the entire CFG shown above. 
            Do NOT list changes, revisions, differences, deltas, or what was “added” or “modified” 
            compared to previous grammars. Instead, explain the grammar from scratch as if the 
            reader has never seen any earlier version.

            Your explanation should:

            - Describe the full DSL defined by the CFG at a high level.
            - Walk through each major nonterminal (e.g., program structure, tasks, movement, 
            crafting, collecting, breaking, turning, conditionals, items, primitives).
            - Explain the semantics of each task (e.g., what MOVE_FUNC does, what BREAK_FUNC 
            does, how direction arguments work, what crafting items represent, etc.).
            - Describe how programs are structured and executed in this DSL.
            - Provide a coherent domain-level overview, similar in style to:

            “The CFG defines the DSL for the Craft domain with primitives such as MOVE_FUNC(direction), 
            COLLECT_FUNC(item), and CRAFT_FUNC(item). MOVE_FUNC moves the agent one cell in the given direction… 
            [etc.]”

            - Be detailed, cohesive, and descriptive—NOT a change log.

            The explanation should read like documentation for the DSL, not a summary of modifications.>

            Changes in CFG
            (bullet point list of specific changes made to the CFG)

            Terminal Functions
            FUNCTION_NAME(args): description of purpose and usage

            If the current CFG is already sufficient, restate it under “CFG Changes (BNF)” and note that no changes are required.

            ---
            """
            }]
            output = llm.chat(conversation, params)
            output = output[0].outputs[0].text
            print(output)
            filepath, cfg_text, term_text, failure_text, cfg_explanation = extract_and_save_cfg(output)
            # print(cfg_text)
            cfg = cfg_text
            print("Updated CFG:\n", cfg)
            # print(cfg_explanation)
            # if term_text:
            #     # Extract terminal functions and ask the LLM to produce a detailed spec + a concrete Python function
            #     funcs = generate_funsearch_function_using_another_prompt(term_text)



    return programs


if __name__ == "__main__":
    
    # Check if arguments are provided
    if len(sys.argv) < 2:
        print("Usage: python program_synthesis.py <json_file_path> [experiment_dir] [dsl_round] [func_evolution_round]")
        print("Example: python program_synthesis.py task_config.json experiments/experiment_20260223_163251_4042520 1 2")
        sys.exit(1)
    
    json_file = sys.argv[1]
    experiment_dir = sys.argv[2] if len(sys.argv) > 2 else None
    dsl_round = int(sys.argv[3]) if len(sys.argv) > 3 else None
    func_evolution_round = int(sys.argv[4]) if len(sys.argv) > 4 else None
    
    cfg_parser = CFGParser("cfg/cfg.txt")
    start_symbol = "s"
    print(f"Start symbol: {start_symbol}")
    print(f"Using JSON config file: {json_file}")
    if experiment_dir:
        print(f"Using experiment directory: {experiment_dir}")
    if dsl_round is not None:
        print(f"DSL round: {dsl_round}")
    if func_evolution_round is not None:
        print(f"Function evolution round: {func_evolution_round}")
    print("\nGenerating programs (worklist)...")
    synthesis_llm(experiment_dir, dsl_round, func_evolution_round)

    # synthesis_baseline()
