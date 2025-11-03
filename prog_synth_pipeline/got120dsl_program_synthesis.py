from datetime import date
import textwrap
from typing import List
from collections import deque
from cfg_parser import CFGParser
import numpy as np
import itertools
import matplotlib.pyplot as plt
from program_evaluator import ProgramEvaluator 
import heapq
# import concurrent.futures
import time
from multiprocessing import Pool, cpu_count
from craft import env_factory
import json
import sys
# from google import genai
import ast
import re
import requests
# from funsearch.implementation.funsearch import FunSearch
# from funsearch.implementation import config as config_lib
import pandas as pd
import os
import random
import subprocess
from vllm import LLM, SamplingParams


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
  recipes_path, hints_path, 0, max_steps=100, reuse_environments=False,
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
                    timeout=300, #this is in seconds
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
        recipes_path, hints_path, max_steps=100, reuse_environments=False,
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
        cleaned = re.sub(r"```[a-zA-Z]*", "", response)
        cleaned = cleaned.replace("```", "").strip()
        
        # Find the section starting with 'def solve'
        match = re.search(r"def\s+solve\s*\(.*", cleaned, re.DOTALL)
        if not match:
            raise ValueError("No 'def solve' function found in the input text.")
        
        # Extract from def solve onward
        code = cleaned[match.start():].strip()
        
        # Stop before any trailing text like "assistantfinal" or explanations
        code = re.split(r"(assistantfinal|Thus final answer|###|---|\Z)", code)[0].strip()

        # Dedent to normalize indentation
        code = textwrap.dedent(code)
        with open(func_file, "w") as f:
            f.write(code + "\n")

def extract_and_save_cfg(output_text, cfg_dir="cfg"):
    # Extract CFG and Terminal Functions sections
    cfg_match = re.search(r"\*\*CFG Changes \(BNF\)\*\*.*?```bnf(.*?)```", output_text, re.DOTALL)
    term_match = re.search(r"\*\*Terminal Functions\*\*(.*?)(?:\n---|\Z)", output_text, re.DOTALL)

    if not cfg_match:
        print("No CFG block found.")
        return None

    cfg_text = cfg_match.group(1).strip()
    term_text = term_match.group(1).strip() if term_match else ""

    # Create cfg directory if missing
    os.makedirs(cfg_dir, exist_ok=True)

    # Name file as cfg_YYYYMMDD_HHMM_updated.txt
    timestamp = date.today().strftime("%Y%m%d_%H%M")
    filename = f"cfg_{timestamp}_updated.txt"
    filepath = os.path.join(cfg_dir, filename)

    # Write to file
    with open(filepath, "w") as f:
        f.write(cfg_text)

    if term_text:
        print("\nExtracted Terminal Functions:\n")
        print(term_text)
        ##call funsearch with new terminal function to do that we will first need to create a file in functions_generated foder with name of the new terminal function -> with def solve def evaluate, def terminal_function 

    return filepath, cfg_text, term_text


# Example usage:
# output = """ your assistant response here """
# extract_and_save_cfg(output)
llm = LLM(model="/scratch/avani/gpt",     tensor_parallel_size=4 )
params = SamplingParams(temperature=0.7, max_tokens=25000)
final =[]
evaluator = ProgramEvaluator()
recipes_path = "craft/resources/recipes.yaml"
hints_path = "craft/resources/hints.yaml"
env_sampler = env_factory.EnvironmentFactory(
            recipes_path, hints_path, 7, max_steps=100, 
            reuse_environments=False, visualise=False)


with open("prog_synth_pipeline/task_config.json", "r") as f:
        config = json.load(f)
        tasks = config["tasks"]
        time_limits = config["time"]

def is_terminal(symbol: str, cfg: CFGParser) -> bool:
    return symbol not in cfg.non_terminals

def evaluate_program_with_evaluator(evaluator, program_str: str, env, time) -> int:
    """
    Evaluate a program using your ProgramEvaluator.
    """
    # try:
    result = evaluator.evaluate_program(program_str, env, time)
    return result["actions"], result["success"], result['total_reward'], result['evaluation_time'] , result["func"], result["interactions"], result["rewards"]
    # except Exception as e:
    #     print(E)
    #     return False, float('-inf'), 0.0

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
    result = evaluator.evaluate_program(program_str, env, time, inter, reward)
    results.add(1 if result["success"] else 0)
    return result["actions"], program_str, results, result["success"], result['total_reward'], result['evaluation_time'] ,task, result["func"], result["interactions"], result["rewards"]
        
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


def synthesis_llm():
    # llm = LLM(model="/scratch/avani/gpt",     tensor_parallel_size=4 )
    # params = SamplingParams(temperature=0.7, max_tokens=15000)
    with open("cfg/cfg.txt") as f:
        cfg = f.read()
    with open("craft/resources/recipes.yaml") as f:
        recipes = f.read()
    extra_body={"reasoning_effort": "high"}
#    client = genai.Client()
    cfg_explanation = (
        "The CFG defines the DSL for the Craft domain with three primary primitives: "
        "MOVE_FUNC(direction), COLLECT_FUNC(item) and CRAFT_FUNC(item). "
        "MOVE_FUNC moves the agent one cell in the given direction. "
        "COLLECT_FUNC makes the agent pathfind to and pick up a primitive resource. If the resource is blocked and the tool that can get through the obstacle blocking the resource is inthe inventory then collect will be able to get through the obstacle and collect the primitive."
        "to overcome obstacles when necessary. "
        "CRAFT_FUNC moves the agent to the required workshop and crafts the item if the needed ingredients "
        "are present in the agent's inventory. "
    )
    first_failing_funcs = []
    programs = []
    tasks = ["get[gem]"]
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
        print(markdown)
        s = False
        program = "MOVE_FUNC(UP) ;"
        # print(prompt)
        # break
        programs= []
        programs.append(program)
        for i in range(5):
            programs_str = "\n".join(programs)  
            prompt = f"""
    You are a Domain Specific Language (DSL) program generator for the Craft domain. 

    ### Start State
    {markdown}

    ## Natural Language Description
    Craft is a single-agent game in a pre-specified environment. 
    The environment of craft is a grid world of size n * n. Each cell can be empty, contain an item, or part of natural terrain or functional structures. When the cell is nonempty, it is considered as blocked. A agent can move around the environment freely through empty cells. At each step, the agent can either move or perform a specific actions, such as collect or craft, towards the immediate cell that it is facing towards. 
    At the beginning of each episode, the agent is placed at a starting cell and a distribution of items across the grid is initialized. The agent’s tasks involve either collecting primitives (raw resources) or crafting items. A item can only be crafted at the specific workshop mentioned in the recipes. 
    The item to be craft are produced from primitives (or other crafted items) by following recipes. Each recipe specifies which items are required and at which workshop the crafting must occur. A primitive item might not need to be crafted but just collected. More complex items, such as axe, or flag, require intermediate items along with primitives. This all is specified in the recipe file of the environment. Please note a item can only be crafted at the specific workshop mentioned in the recipes. 

    This is the schema of the recipes:

    recipes:
        item:
        primtive: count of primtive
        _at: at what workshop does the primitve needs to be crafted

    Sometimes, primitive can be blocked by obstacles like trees, water, etc. and needs the player to use a tool to pass the obstacle in order to reach and collect the primitive.
    ## Context Free Grammar (CFG)
    Here is the context-free grammar (CFG) that defines the DSL. Strictly follow this CFG when synthesising programs :

    {cfg}

    {cfg_explanation}

    ## Example Programs
    Here are examples of programs written in this DSL:

    COLLECT_FUNC(WOOD) ; MOVE_FUNC(RIGHT) ; CRAFT_FUNC(STICK) ;
    COLLECT_FUNC(GRASS) ; MOVE_FUNC(RIGHT) ;

    ## Domain Context
    This DSL is used to solve tasks in the Craft domain. Tasks typically look like:
    - get(wood)
    - make(stick)

    The goal is to write programs in the context free grammar (CFG) provided that can complete these tasks using the available functions and recipes.

    ## Available Recipes
    Here are the recipes for the domain:

    {recipes}

    ## Task
    Generate a program that solves the following task :

    **{task}**

    ## Output Format Instructions
    Return ONLY the program string delimited by $ signs. Do not include any explanations, comments, or additional text outside the $ delimiters.
    Example output ->
    $MOVE_FUNC(UP) ;$

    ##Previous programs that FAILED to solve the task:
    {programs_str}
    These programs are syntactically correct but did not solve the task. When generating a new program, avoid repeating the mistakes made in these failed programs, and generate semantically different programs.


    Also always ensure that the the information provided in this prompt is facts and always correct and cannot be changed so please adhere to it strictly.
    ##Return a program that is able to solve the task
    
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
            a, program_str, results, s, r, eval_time, task, funcs, interact, rewa = evaluate(b, task ,env, inter, reward)
            interactions += interact
            rewards += rewa
            inter+= interactions[-1] if interactions else 0
            reward+= rewards[-1] if rewards else 0
            reasoning[program_str]= response
            os.makedirs(f"results/program_synthesis/{task}", exist_ok=True)
            record = {
                "task": task,
                "program": program_str,
                "success": bool(s),
                "total_reward": r,
                "eval_time": eval_time,
                "interactions": interact,
                "rewards": rewa,
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
                # find_bad_func(funcs, task)
        # # plot_interactions_rewards(interactions, rewards, task)
        # plot_watermark(plot, task)
            # programs = programs_str
        if not s:
            print("Failed to find a solution for task:", task)
            print(programs)
            conversation = [{
                "role": "user",
                "content": f"""
            The following Domain-Specific Language (DSL) programs failed to solve the task **{task}** and this is the list of these programs and the reasoning traces from them:

            {reasoning}

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
            1. Analyze **why** these programs might have failed.  
            2. Identify **gaps or weaknesses** in the current CFG.  
            3. Suggest **specific additions or removal of terminal functions** to the CFG that would enable better synthesis results.  
            4. Output an **updated CFG** in **BNF format**, showing only changed or newly added rules (if possible).  
            5. Propose any **new terminal functions**, with short, clear descriptions of their purpose and behavior.  
            ---

            ### Output Format

            Your response must strictly follow this structure:

            Failure Analysis
            (bullet point explanations for why previous programs failed)

            CFG Changes (BNF)
            <only include modified or newly added rules; restate unchanged ones if necessary please adhere to a format I can parse though this re.findall(r"\*\*(\w+)\((.*?)\)\*\*"))>

            Terminal Functions
            FUNCTION_NAME(args): description of purpose and usage

            If the current CFG is already sufficient, restate it under “CFG Changes (BNF)” and note that no changes are required.

            ---
            """
            }]
            output = llm.chat(conversation, params)
            output = output[0].outputs[0].text
            print(output)
            cfg_path, cfg_text, term_text = extract_and_save_cfg(output)
            if term_text:
                generate_funsearch_function(term_text)
    return programs


if __name__ == "__main__":
    
    # Check if JSON file path is provided as command line argument
    if len(sys.argv) != 2:
        print("Usage: python program_synthesis.py <json_file_path>")
        print("Example: python program_synthesis.py task_config.json")
        sys.exit(1)
    
    json_file = sys.argv[1]
    
    cfg_parser = CFGParser("cfg/cfg.txt")
    start_symbol = "s"
    print(f"Start symbol: {start_symbol}")
    print(f"Using JSON config file: {json_file}")
    print("\nGenerating programs (worklist)...")
    synthesis_llm()

    # synthesis_baseline()
