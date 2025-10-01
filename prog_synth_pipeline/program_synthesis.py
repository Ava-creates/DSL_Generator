from typing import List
from collections import deque
from cfg_parser import CFGParser
import itertools
from program_evaluator import ProgramEvaluator 
import heapq
import concurrent.futures
import time
from multiprocessing import Pool, cpu_count
import env_factory
import json
import sys
from google import genai
import requests
from funsearch.implementation.funsearch import FunSearch
from funsearch.implementation import config as config_lib
import pandas as pd

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
recipes_path = "resources/recipes.yaml"
hints_path = "resources/hints.yaml"
env_sampler = env_factory.EnvironmentFactory(
            recipes_path, hints_path, 7, max_steps=100, 
            reuse_environments=False, visualise=False)


with open("prog_synth_pipeline/task_config.json", "r") as f:
        config = json.load(f)
        tasks = config["tasks"]
        time_limits = config["time"]


tasks = [tasks[-1]]
print(tasks)
def is_terminal(symbol: str, cfg: CFGParser) -> bool:
    return symbol not in cfg.non_terminals

def evaluate_program_with_evaluator(evaluator, program_str: str, env, time) -> int:
    """
    Evaluate a program using your ProgramEvaluator.
    """
    # try:
    result = evaluator.evaluate_program(program_str, env, time)
    if(result['success']):
            final.append(program_str)
            with open("final2.txt", "a") as f:
                for program in final:
                    f.write(program + "\n")
    return result["actions"], result["success"], result['total_reward'], result['evaluation_time'] , result["func"]
    # except Exception as e:
    #     print(E)
    #     return False, float('-inf'), 0.0


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

def evaluate(program_str, task , env):
    
    # for task in tasks:
    
    results = set()
    # for ind in range(len(envs)):
    a, s, r, eval_time, funcs = evaluate_program_with_evaluator(evaluator, program_str, env, 60)
    results.add(1 if s else 0)
   
    
    return a, program_str, results, s, r, eval_time, task,funcs

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
                
    print(first_failing_funcs)
    # Run FunSearch for each failing function

    

        
def synthesis_llm():
    with open("cfg/cfg.txt") as f:
        cfg = f.read()
    with open("resources/recipes.yaml") as f:
        recipes = f.read()

    client = genai.Client()
    first_failing_funcs = []
    programs = []
    
    for task in tasks:
        env = env_sampler.sample_environment(task_name=task)

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
        for i in range(10):
            response = client.models.generate_content(
                            model="gemini-2.5-pro", contents = prompt
                        )
            b = response.text.strip('$')
            # print(b)
            # b= "COLLECT_FUNC(WOOD) ;  CRAFT_FUNC(STICK) ; COLLECT_FUNC(IRON) ; CRAFT_FUNC(AXE) ; COLLECT_FUNC(GEM)"
            programs.append(b)
            a, program_str, results, s, r, eval_time, task, funcs = evaluate(b, task ,env)
            print(a, program_str, results, s, r, eval_time, task, funcs )
            if s :
                with open("program_for_tasks.log", 'a') as f:
                    ans = program_str + "," +task +","+"True,"+str(r)+","+ str(eval_time)+"\n"
                    f.write(ans)
                break

            else:
                find_bad_func(funcs, task)

                
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