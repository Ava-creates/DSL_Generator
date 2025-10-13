from typing import List
from collections import deque
from cfg_parser import CFGParser
import numpy as np
import itertools
import matplotlib.pyplot as plt
from program_evaluator import ProgramEvaluator 
import heapq
import concurrent.futures
import time
from multiprocessing import Pool, cpu_count
from craft import env_factory
import json
import sys
from google import genai
import ast
import re
import requests
from funsearch.implementation.funsearch import FunSearch
from funsearch.implementation import config as config_lib
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

  env_sampler = env_factory.EnvironmentFactory(
  recipes_path, hints_path, 7, max_steps=100, reuse_environments=False,
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
        # print(output)
        result = ast.literal_eval(output)

        # Access the values
        output, actions_count = result[0], result[1]
        # print("output ", output)
        try:
            print(output)
            float(np.float64(output)), True, actions_count, None
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

def grid_to_markdown(grid, cookbook):
    width, height, n_kinds = grid.shape
    inv_index = cookbook.index.reverse_contents  # index -> item name

    table = []
    for y in range(height):  # row by row
        row = []
        for x in range(width):
            cell_items = [inv_index[k] for k in range(1, n_kinds) if grid[x, y, k] == 1]
            row.append(",".join(cell_items) if cell_items else ".")
        table.append(row)

    df = pd.DataFrame(table)
    return df.to_markdown(index=False, headers=[])


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
    if len(data) < 2:
        return

    x_values = [sum(point[0] for point in data[:i+1]) for i in range(len(data))]
    y_values = [max(point[1] for point in data[:i+1]) for i in range(len(data))]

    plt.plot(x_values, y_values, marker='o')
    plt.title('Reward vs Interactions')
    plt.xlabel('Number of Interactions')
    plt.ylabel('Reward')
    plt.grid()
    plt.savefig(f'results/plots/plot_{task}.png')
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

def find_bad_func(funcs, task):

    first_failing_funcs = []
    # print(funcs)
    if funcs:
            actions_up_to_failure = []
            for i, (func_name, reward, func_actions) in enumerate(funcs):
                if reward <= 0:
                    # Collect actions from all functions up to this failing one
                    for j in range(i):
                        actions_up_to_failure.append(funcs[j][2])  
                    first_failing_funcs.append((func_name, reward, actions_up_to_failure, task))
                
    # print(first_failing_funcs)
    # Run FunSearch for each failing function

def synthesis_baseline():
    llm = LLM(model="/scratch/avani/gpt",     tensor_parallel_size=4 )
    params = SamplingParams(temperature=0.7, max_tokens=5000)
    with open("cfg/cfg.txt") as f:
        cfg = f.read()
    with open("craft/resources/recipes.yaml") as f:
        recipes = f.read()

    client = genai.Client()
    first_failing_funcs = []
    programs = []
    
    for task in tasks:
        plot =[]
        env = env_sampler.sample_environment(task_name=task)
        markdown = grid_to_markdown(env._current_state.grid, env.world.cookbook)
        with open("prompt_specifications/specification_with_updated_nld_baseline.txt", "r") as f1, open("function_specific_prompts/make_stick_base.txt", "r") as f2:
            first_file_content = f1.read()
            second_file_content = f2.read()
        
        prompt = (
                f"{first_file_content}\n"
                f"{second_file_content}\n"
                + "\n\n"
                "Your task:\n"
                "Return a **correct implementation** of the `make_stick` function in Python.\n\n"
                "Formatting Requirements (do NOT ignore):\n"
                "1. Your response MUST begin exactly like this:\n"
                "   ```python\n"
                "   def make_stick(env):\n"
                "2. Only output the complete function implementation inside the code block.\n\n"
                "Example of correct response format:\n"
                "```python\n"
                "def make_stick(env):\n"
                "    # your implementation here\n"
                "```\n"
                "Now return only the correct implementation of `make_stick` following these rules."
            )
        data = []
        data.append((0, 0))
        failed_programs = []
        for i in range(50):
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

            #vllm gpt 120b

            output = llm.generate([prompt], params)
            # print(output[0].outputs[0].text)
            try:
                # response = requests.post(api_url, headers=headers, json=payload, timeout=300)

                # response = response.json()["response"]
                response = output[0].outputs[0].text
                # print(response)
                response = response[response.index("```python")+len("```python"):]
                response = response[:response.index("```")]
                # print(response)
            except:
                continue 
            a, b, c, d= eval(response)
            print(a, b, c, d)
            if a== -1 :
                failed_programs.append(response + "\nError:\n"+ d)
                continue
            else:
                data.append((c, a))
            selected_failed_programs = random.sample(failed_programs, min(4, len(failed_programs)))
            prompt = (
                f"{first_file_content}\n"
                f"{second_file_content}\n"
                "Previous Failed Programs:\n"
                + "\n".join(selected_failed_programs)
                + "\n\n"
                "Your task:\n"
                "Return a **correct implementation** of the `make_stick` function in Python.\n\n"
                "Formatting Requirements (do NOT ignore):\n"
                "1. Your response MUST begin exactly like this:\n"
                "   ```python\n"
                "   def make_stick(env):\n"
                "2. Only output the complete function implementation inside the code block.\n\n"
                "Example of correct response format:\n"
                "```python\n"
                "def make_stick(env):\n"
                "    # your implementation here\n"
                "```\n"
                "Now return only the correct implementation of `make_stick` following these rules."
            )
        plot_watermark(data, "make[stick]")
        return 0






















def synthesis_llm():
    with open("cfg/cfg.txt") as f:
        cfg = f.read()
    with open("craft/resources/recipes.yaml") as f:
        recipes = f.read()

    client = genai.Client()
    first_failing_funcs = []
    programs = []
    
    for task in tasks:
        plot =[]
        plot.append((0,0))
        env = env_sampler.sample_environment(task_name=task)
        inter, reward, = 0, 0 
        interactions = []
        rewards = []
        markdown = grid_to_markdown(env._current_state.grid, env.world.cookbook)
        # print(markdown)
        program = "$MOVE_FUNC(UP) ;"
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
    $program$

    ##Previous program that did not solve the task:
    {program}

    ##Return a program that is able to solve the task
    
    """
        print(prompt)
        break
        for i in range(10):
            # response = client.models.generate_content(
            #                 model="gemini-2.5-pro", contents = prompt
            #             )

            payload = {
              "model": "gpt-oss:latest", 
              "prompt": prompt, 
              "template": "{{.Prompt}}",
              "stream": False, 
              "options": {
                "num_ctx": 4096, 
                # "stop": self.stop_tokens
              }
            }
            api_url = "http://129.128.243.184:11434/api/generate"
            headers = {"Content-Type": "application/json"}
            response = requests.post(api_url, headers=headers, json=payload, timeout=300)
            response = response.json()["response"]
            print(response)
            # response = response.text for gemini
            # b = response.strip('$ ')
            response = "Here is $some text$ and more"
            b = re.search(r'\$(.*?)\$\s*', response)
            if b:
                b = b.group(1)
            else:
                continue
            print(b)
            # b= "COLLECT_FUNC(WOOD) ;  COLLECT_FUNC(IRON) ; CRAFT_FUNC(BRIDGE) ;"
            programs.append(b)
            a, program_str, results, s, r, eval_time, task, funcs, interact, rewa = evaluate(b, task ,env, inter, reward)
            # print(a)
            # print(len(interact))
            interactions += interact
            rewards += rewa
            inter+= interactions[-1] if interactions else 0
            reward+= rewards[-1] if rewards else 0
            print(interact, rewa)
            plot.append((len(interact), r))
            # print(a, program_str, results, s, r, eval_time, task, funcs )
            if s :
                with open("program_for_tasks.log", 'a') as f:
                    ans = program_str + "," +task +","+"True,"+str(r)+","+ str(eval_time)+"\n"
                    f.write(ans)
                break
            else:
                find_bad_func(funcs, task)
        plot_interactions_rewards(interactions, rewards, task)
        plot_watermark(plot, task)

                
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
    # synthesis_llm()

    synthesis_baseline()
