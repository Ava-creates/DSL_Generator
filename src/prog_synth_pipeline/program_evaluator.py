from typing import List, Dict, Any
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from craft import env_factory
import time
import re
import json
import os
import io
import numpy as np
import contextlib
from functions_generated.craft_func import craft
from functions_generated.has_func import has
from functions_generated.move_func import move
from functions_generated.collect_func_old_and_good import collect
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


def run_with_timeout(func_name, args_1, env, timeout):
        return run_funcs([], func_name, args_1, env)

class ProgramEvaluator:
    def __init__(self, recipes_path: str = "craft/resources/recipes.yaml", 
                 hints_path: str = "craft/resources/hints.yaml",
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

    def parse_program(self, program, env, timeout, inter, reward) -> List[int]:
        interactions =[]
        rewards =[]
        """Convert a program string into a list of actions."""
        start_time = time.time()  # Start timing
        actions = []
        tokens = program.split()
        # print("tokens", tokens)
        i = 0
        func=[]
        d = False
        while i < len(tokens):
            if len(tokens[i]) > 10 and tokens[i][:9] == "MOVE_FUNC":
                dir_str = tokens[i].split('(')[1].strip(') ;')
                result = run_with_timeout( "move", [dir_str], env, timeout)
                if(result == -1):
                    print("Evaluation timed out in move")
                    return reward, d, evaluation_time , func, interactions, rewards

                r, done, observations = env.step(result)
                inter+=1
                interactions.append(inter)
                if done:
                    d = True
                reward += r
                rewards.append(reward)
                i += 1
                func.append(("MOVE_FUNC", r, result))
                
            elif len(tokens[i]) > 11 and tokens[i][:10] == "CRAFT_FUNC":
                item = tokens[i].split('(')[1].strip(') ;').lower()      
                result = run_with_timeout( "craft", [item], env, timeout)
                if(result == -1):
                    print("Evaluation timed out in craft")
                    return reward, d, evaluation_time , func, interactions, rewards

                r = -2
                for j in result:
                    r, done, observations = env.step(j)
                    inter+=1
                    if done:
                        d = True
                    reward += r 
                    interactions.append(inter)
                    rewards.append(reward)
                func.append((tokens[i][:10], r, result))
                i += 1

            elif len(tokens[i]) > 13 and tokens[i][:12] == "COLLECT_FUNC":
                primitive = tokens[i].split('(')[1].strip(') ;').lower()
                result = run_with_timeout( "collect", [primitive], env, timeout)
                r = -2
                if(result == -1):
                    print("Evaluation timed out in collect")
                    return reward, d, evaluation_time , func, interactions, rewards

                # print(result)
                for j in result:
                    inter+=1
                    r, done, observations = env.step(j)
                    if done:
                        d = True
                    reward += r 
                    rewards.append(reward)
                    interactions.append(inter)
                func.append((tokens[i][:12], r, result))
                i += 1
            elif tokens[i] == "if" and i + 4 < len(tokens):
                # print(i)
                condition = tokens[i + 1]
                then_token = tokens[i + 2]
                then_action = tokens[i + 3]

                if condition.startswith("HAS(") and condition.endswith(")"):
                    item = condition[4:-1]  # Extract "GOLDARROW"    
                    # print("item", item)
                    item = int(self.item_map[item])
                    result = run_with_timeout("has", [item], env, timeout)
                    if(result == -1):
                        print("Evaluation timed out in has")
                        return reward, d, evaluation_time , func, interactions, rewards

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
                return reward, d, evaluation_time , func, interactions, rewards

        evaluation_time = time.time() - start_time  # Calculate evaluation time
        return  reward, d, evaluation_time , func, interactions, rewards

    def evaluate_program(self, program: str, env, timeout, inter, reward) -> Dict[str, Any]:
        """Evaluate a program in the craft environment."""
        env.reset()
        total_reward, d, evaluation_time, func, interactions, rewards = self.parse_program(program, env, timeout, inter, reward)
        return {
            "actions": "just ignore",
            "total_reward": total_reward,
            "success": d and total_reward > 0,
            "evaluation_time": evaluation_time,
            "func":func,
            "interactions": interactions,
            "rewards": rewards
        }

def main():
    evaluator = ProgramEvaluator(visualise=True)
    # flag  = "CRAFT_FUNC(HAMMER) ; CRAFT_FUNC(WOOD) ; CRAFT_FUNC(IRON) ; CRAFT_FUNC(BENCH) ;"
    flag ="CRAFT_FUNC(ROPE) ; CRAFT_FUNC(BUNDLE) ; CRAFT_FUNC(BOW) ;"
    program = " COLLECT_FUNC(ROCK) ; COLLECT_FUNC(IRON) ; CRAFT_FUNC(KNIFE) "
    recipes_path = "craft/resources/recipes_for_synth.yaml"
    hints_path = "carft/resources/hints.yaml"
    env_sampler = env_factory.EnvironmentFactory(
            recipes_path, hints_path, 6, max_steps=300, 
            reuse_environments=False, visualise=False)
    env = env_sampler.sample_environment(task_name="make[knife]")
    print("VDFS \n", env.world.cookbook.index, "\n")

    result = evaluator.evaluate_program(program, env, 300)
    print("\nEvaluation Results:")
    print(f"Total Reward: {result['total_reward']}")
    print(f"Success: {result['success']}")

if __name__ == "__main__":
    main() 
