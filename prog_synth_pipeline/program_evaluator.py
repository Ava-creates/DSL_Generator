from typing import List, Dict, Any
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import env_factory
import time
import re
import json
import os
import io
import numpy as np
import contextlib
from craft_func import craft
from has_func import has
from move_func import move
from collect_func import collect
import multiprocessing
def run_funcs(queue, func_name, args_1, env):
        w = args_1[0]
        if func_name == "move":
            result = move(env, w)
        elif func_name == "craft":
            result = craft(env, w)
        elif func_name == "collect":
            result = collect(env, w)
        elif func_name == "has":
            result = has(env, w) 

        return result
    #     queue.put(result)
    # except Exception as e:
    #     queue.put(e)

def run_with_timeout(func_name, args_1, env, timeout):

        return run_funcs([], func_name, args_1, env)
        # queue_obj = multiprocessing.Queue()
        # p = multiprocessing.Process(target=run_funcs, args=(queue_obj, func_name, args_1, env))
        # p.start()
        # p.join(timeout)
        # if p.is_alive():
        #     # print("Evaluation timed out.")
        #     p.terminate()
        #     p.join()
        #     return -1
        # if not queue_obj.empty():
        #     result = queue_obj.get()
        #     if isinstance(result, Exception):
        #         print("Error evaluating:", result)
        #         return -1
        #     return result
        # else:
        #     print("No result returned.")
        #     return -1
class ProgramEvaluator:
    def __init__(self, recipes_path: str = "resources/recipes.yaml", 
                 hints_path: str = "resources/hints.yaml",
                 visualise: bool = True):

        self.item_map =item_id_map = { 
                                "WOOD": 9,
                                "IRON": 7,
                                "GRASS": 8,
                                "ROCK": 10,
                                "ROPE": 13,
                                "KNIFE": 14,                             
                                "SLINGSHOT": 15,
                                "ARROW": 16,
                                "GOLDARROW": 17
                            }

    def parse_program(self, program, env, timeout) -> List[int]:
        """Convert a program string into a list of actions."""
        start_time = time.time()  # Start timing
        actions = []
        tokens = program.split()
        # print("tokens", tokens)
        i = 0
        reward = 0
        func=[]
        d = False
        while i < len(tokens):
            if len(tokens[i]) > 10 and tokens[i][:9] == "MOVE_FUNC":
                dir_str = tokens[i].split('(')[1].strip(') ;')

                result = run_with_timeout( "move", [dir_str], env, timeout)
                if(result == -1):
                    print("Evaluation timed out in move")
                    return [], reward, False
                # print("action in move, ", result)
                r, done, observations = env.step(result)
                if done:
                    d = True
                reward += r
                i += 1
                func.append(("MOVE_FUNC", r, result))
                
            elif len(tokens[i]) > 11 and tokens[i][:10] == "CRAFT_FUNC":
                # print(tokens[i])
                item = tokens[i].split('(')[1].strip(') ;').lower()      
                # print(item)      
                result = run_with_timeout( "craft", [item], env, timeout)
                if(result == -1):
                    print("Evaluation timed out in craft")
                    return [], reward, False
                r = -2
                for j in result:
                    r, done, observations = env.step(j)
                    if done:
                        d = True
                    reward += r 
                func.append((tokens[i][:10], r, result))
                i += 1

            elif len(tokens[i]) > 13 and tokens[i][:12] == "COLLECT_FUNC":
                primitive = tokens[i].split('(')[1].strip(') ;').lower()
                # print("primitive", primitive+"space test")
                # primtive = primitive.strip()
                result = run_with_timeout( "collect", [primitive], env, timeout)
                r = -2
                if(result == -1):
                    print("Evaluation timed out in collect")
                    return [], reward, False
                # print(result)
                for j in result:
                    r, done, observations = env.step(j)
                    if done:
                        d = True
                    reward += r 
                func.append((tokens[i][:12], r, result))
                i += 1
            elif tokens[i] == "if" and i + 4 < len(tokens):
                # print(i)
                condition = tokens[i + 1]
                then_token = tokens[i + 2]
                then_action = tokens[i + 3]

                if condition.startswith("has(") and condition.endswith(")"):
                    item = condition[4:-1]  # Extract "GOLDARROW"    
                    # print("item", item)
                    item = int(self.item_map[item])
                    result = run_with_timeout("has", [item], env, timeout)
                    if(result == -1):
                        print("Evaluation timed out in has")
                        return [], reward, False
                    # print("Captured print:", printed_output.strip())
                    if(result == False):
                        i+=3
                    else:
                        i+=3
                        
                else:
                    raise ValueError(f"Unsupported if condition: {condition}")

            elif tokens[i] == ";":
                i += 1

            elif tokens[i] == "":
                i += 1  

            else:
                # print("Unknown token", tokens[i])
                evaluation_time = time.time() - start_time  # Calculate evaluation time
                return [], reward, False, evaluation_time

        evaluation_time = time.time() - start_time  # Calculate evaluation time
        return  reward, d, evaluation_time , func

    def evaluate_program(self, program: str, env, timeout) -> Dict[str, Any]:
        """Evaluate a program in the craft environment."""
        env.reset()
        total_reward, d, evaluation_time, func = self.parse_program(program, env, timeout)
        return {
            "actions": "just ignore",
            "total_reward": total_reward,
            "success": d and total_reward > 0,
            "evaluation_time": evaluation_time,
            "func":func
        }

def main():
    evaluator = ProgramEvaluator(visualise=True)
    # flag  = "CRAFT_FUNC(HAMMER) ; CRAFT_FUNC(WOOD) ; CRAFT_FUNC(IRON) ; CRAFT_FUNC(BENCH) ;"
    flag ="CRAFT_FUNC(ROPE) ; CRAFT_FUNC(BUNDLE) ; CRAFT_FUNC(BOW) ;"
    program = " COLLECT_FUNC(ROCK) ; COLLECT_FUNC(IRON) ; CRAFT_FUNC(KNIFE) "
    recipes_path = "resources/recipes_for_synth.yaml"
    hints_path = "resources/hints.yaml"
    env_sampler = env_factory.EnvironmentFactory(
            recipes_path, hints_path, 6, max_steps=100, 
            reuse_environments=False, visualise=False)
    env = env_sampler.sample_environment(task_name="make[knife]")
    print("VDFS \n", env.world.cookbook.index, "\n")

    result = evaluator.evaluate_program(program, env, 300)
    print("\nEvaluation Results:")
    print(f"Total Reward: {result['total_reward']}")
    print(f"Success: {result['success']}")

if __name__ == "__main__":
    main() 