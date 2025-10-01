import os
import json
import argparse
import random
import subprocess
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional

from google import genai

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

            if function_body is None or scores is None:
                continue

            end_score = get_end_score(scores)
            if end_score is None:
                continue

            scored_funcs.append((end_score, function_body))

    if not scored_funcs:
        return []

    scored_funcs.sort(key=lambda x: x[0], reverse=True)

    # Find cutoff score if ties go beyond k
    cutoff = scored_funcs[k-1][0] if len(scored_funcs) >= k else scored_funcs[-1][0]

    # Keep all functions with score >= cutoff
    top_candidates = [(s, f) for (s, f) in scored_funcs if s >= cutoff]

    if len(top_candidates) > k:
        # Too many due to ties → sample exactly k at random
        return random.sample(top_candidates, k)
    else:
        return top_candidates

def eval(res):
            # with tempfile.TemporaryDirectory() as temp_dir:
     # Create unique filename using process ID and timestamp
    temp_dir = os.getcwd()
    unique_id = f"{os.getpid()}_{int(time.time() * 1000000)}"
    script_path = f'explicit_generated_code_{unique_id}.py'
    script_path = os.path.join(temp_dir, script_path)

    # script_path = os.path.join(temp_dir, 'generated_code.py')
    full_program_collect=f'''
import numpy as np
import time
import collections
from craft import craft, env, env_factory

def solve(env, primitive, visualise=False) -> float:
  """Runs the environment with a collect function that returns list of actions to take and returns total reward."""
  actions_to_take = collect(env, primitive)
  total_reward = 0.0

  for t in range(len(actions_to_take)):
    action = actions_to_take[t]
    reward, done, observations = env.step(action)
    total_reward += reward
    if done:
      break

  if total_reward>0.5:
    return 0.2

  return total_reward


def evaluate() -> float:
  """Evaluates a crafting policy on a sample task."""
  visualise = False
  recipes_path = "craft/resources/recipes.yaml"
  hints_path = "craft/resources/hints.yaml"
  reward = 0 
  for i in range(10):
    if(i == 0):
      p = "wood"
      env_sampler = env_factory.EnvironmentFactory(
      recipes_path, hints_path, 0, max_steps=100, reuse_environments=False,
            visualise=visualise)

      env = env_sampler.sample_environment(task_name= 'make[stick]')
      env.reset()
    
    elif(i==1):
      p = "iron"
      env_sampler = env_factory.EnvironmentFactory(
      recipes_path, hints_path, 1, max_steps=100, reuse_environments=False,
            visualise=visualise)

      env = env_sampler.sample_environment(task_name= 'make[bridge]')
      env.reset()
      
    elif(i==2):
      p = "wood"
      env_sampler = env_factory.EnvironmentFactory(
      recipes_path, hints_path, 1, max_steps=100, reuse_environments=False,
            visualise=visualise)

      env = env_sampler.sample_environment(task_name= 'make[bridge]')
      env.reset()

    elif(i==3): #grass not present onthe grid should return empty list
      p = "grass"
      env_sampler = env_factory.EnvironmentFactory(
      recipes_path, hints_path, 1, max_steps=100, reuse_environments=False,
            visualise=visualise)

      env = env_sampler.sample_environment(task_name= 'make[bridge]')
      env.reset()

    elif(i==4):
      p = "wood"
      env_sampler = env_factory.EnvironmentFactory(
      recipes_path, hints_path, 2, max_steps=100, reuse_environments=False,
            visualise=visualise)

      env = env_sampler.sample_environment(task_name= 'make[plank]')
      env.reset()
      #env.step(1)
      #env.step(4)

    elif(i==5):
      p = "grass"
      env_sampler = env_factory.EnvironmentFactory(
      recipes_path, hints_path, 3, max_steps=100, reuse_environments=False,
            visualise=visualise)

      env = env_sampler.sample_environment(task_name= 'make[cloth]')
      env.reset()
      #env.step(1)
      #env.step(4)


    elif(i==6):
      p = "grass"
      env_sampler = env_factory.EnvironmentFactory(
      recipes_path, hints_path, 4, max_steps=100, reuse_environments=False,
            visualise=visualise)

      env = env_sampler.sample_environment(task_name= 'make[rope]')
      env.reset()
      #env.step(0)
      #env.step(0)
      #env.step(4)

    elif(i==7):
      p = "grass"
      env_sampler = env_factory.EnvironmentFactory(
      recipes_path, hints_path, 5, max_steps=100, reuse_environments=False,
            visualise=visualise)

      env = env_sampler.sample_environment(task_name= 'make[bundle]')
      env.reset()
      #env.step(0)
      #env.step(0)
      #env.step(4)
      #env.step(0)
      #env.step(4)

    elif(i==8):
      p = "wood"
      env_sampler = env_factory.EnvironmentFactory(
      recipes_path, hints_path, 5, max_steps=100, reuse_environments=False,
            visualise=visualise)

      env = env_sampler.sample_environment(task_name= 'make[bundle]')
      env.reset()
      #env.step(0)
      #env.step(0)
      #env.step(4)

    else:
      p = "gold"
      env_sampler = env_factory.EnvironmentFactory(
      recipes_path, hints_path, 6, max_steps=100, reuse_environments=False,
            visualise=visualise)

      env = env_sampler.sample_environment(task_name= 'make[goldarrow]')
      env.reset()
      env.step(1)
      env.step(4)
      env.step(1)
      env.step(4)
      env.step(1)
      env.step(1)
      env.step(4)
      
    r= solve(env, p, visualise=visualise)
    reward += r

  return reward
  
  
def collect(env, primitive):
{res}

print(evaluate())'''
# print(full_program_collect)
            # Create complete executable program
    full_program = f'''
import numpy as np
import time
import collections
import env_factory

def solve(env, item, visualise=False) -> float:
  """Runs the environment with a collect function that returns list of actions to take and returns total reward."""
  actions_to_take = craft(env, item)
  total_reward = 0.0

  for t in range(len(actions_to_take)):
    action = actions_to_take[t]
    reward, done, observations = env.step(action)
    total_reward += reward
    if done:
      break
#   print(item, total_reward, actions_to_take)
  return total_reward

def evaluate() -> float:
  """Evaluates a crafting policy on a sample task."""
  #max reward is 6 for this fucntion so any craft objet that can get when it is working properly
  visualise = False
  recipes_path = "craft/resources/recipes.yaml"
  hints_path = "craft/resources/hints.yaml"     
  reward = 0 
  for i in range(11):
    if(i == 0):
      item = "stick"
      env_sampler = env_factory.EnvironmentFactory(
      recipes_path, hints_path, 0, max_steps=100, reuse_environments=False,
            visualise=visualise)

      env = env_sampler.sample_environment(task_name= 'make[stick]')
      env.reset()
      env.step(1)
      env.step(4)
      reward += solve(env, item,  visualise=visualise) #should give +1
    
    elif(i==1):
      item = "stick"
      env_sampler = env_factory.EnvironmentFactory(
      recipes_path, hints_path, 0, max_steps=100, reuse_environments=False,
            visualise=visualise)

      env = env_sampler.sample_environment(task_name= 'make[stick]')
      env.reset()
      temp_reward = solve(env, item, visualise=visualise)  #should give 0 when it is working properly
      if temp_reward>0 :
        reward -= 0.3
      
    elif(i==2):
      item = "bridge"
      env_sampler = env_factory.EnvironmentFactory(
      recipes_path, hints_path, 1, max_steps=100, reuse_environments=False,
            visualise=visualise)

      env = env_sampler.sample_environment(task_name= 'make[bridge]')
      env.reset()
      env.step(1)
      env.step(4)
      reward += solve(env, item, visualise=visualise)  # 0 when working properly

    elif(i==3):
      item = "bridge"
      env_sampler = env_factory.EnvironmentFactory(
      recipes_path, hints_path, 1, max_steps=100, reuse_environments=False,
            visualise=visualise)

      env = env_sampler.sample_environment(task_name= 'make[bridge]')
      env.reset()
      temp_reward = solve(env, item, visualise=visualise) # 0 when working properly 
      if temp_reward>0 :
        reward -= 0.3

    elif(i==4):
      item = "plank"
      env_sampler = env_factory.EnvironmentFactory(
      recipes_path, hints_path, 2, max_steps=100, reuse_environments=False,
            visualise=visualise)

      env = env_sampler.sample_environment(task_name= 'make[plank]')
      env.reset()
      env.step(1)
      env.step(4)
      reward += solve(env, item, visualise=visualise) # +1 this does nnot work need to collect more before crafting

    elif(i==5):
      item = "cloth"
      env_sampler = env_factory.EnvironmentFactory(
      recipes_path, hints_path, 3, max_steps=100, reuse_environments=False,
            visualise=visualise)

      env = env_sampler.sample_environment(task_name= 'make[cloth]')
      env.reset()
      env.step(1)
      env.step(4)
      reward += solve(env, item, visualise=visualise)  #+1


    elif(i==6):
      item = "rope"
      env_sampler = env_factory.EnvironmentFactory(
      recipes_path, hints_path, 4, max_steps=100, reuse_environments=False,
            visualise=visualise)

      env = env_sampler.sample_environment(task_name= 'make[rope]')
      env.reset()
      env.step(0)
      env.step(0)
      env.step(4)
      reward += solve(env, item, visualise=visualise) #+1

    elif(i==7):
      item = "bundle"
      env_sampler = env_factory.EnvironmentFactory(
      recipes_path, hints_path, 5, max_steps=100, reuse_environments=False,
            visualise=visualise)

      env = env_sampler.sample_environment(task_name= 'make[bundle]')
      env.reset()
      env.step(0)
      env.step(0)
      env.step(4)
      env.step(0)
      env.step(4)
      reward += solve(env, item, visualise=visualise)  #+1

    elif(i==8):
      item = "bundle"
      env_sampler = env_factory.EnvironmentFactory(
      recipes_path, hints_path, 5, max_steps=100, reuse_environments=False,
            visualise=visualise)

      env = env_sampler.sample_environment(task_name= 'make[bundle]')
      env.reset()
      env.step(0)
      env.step(0)
      env.step(4)

      temp_reward = solve(env, item, visualise=visualise)
      if temp_reward>0 :
        reward -= 0.3

    elif(i==9):
      item = "goldarrow"
      env_sampler = env_factory.EnvironmentFactory(
      recipes_path, hints_path, 6, max_steps=100, reuse_environments=False,
            visualise=visualise)

      env = env_sampler.sample_environment(task_name= 'make[goldarrow]')
      env.reset()
      env.step(1)
      env.step(4)
      env.step(1)
      env.step(4)
      env.step(1)
      env.step(1)
      env.step(4)
      reward += solve(env, item, visualise=visualise)  # +1

    else:
      recipes_path_2 = "resources/recipes_for_synth.yaml"
      item = "arrow"
      env_sampler = env_factory.EnvironmentFactory(
            recipes_path_2, hints_path, 6, max_steps=100, 
            reuse_environments=False, visualise=False)
      env=env_sampler.sample_environment(task_name='make[arrow]')
      env.reset()
      # Actions to execute:
      env.step(0)
      env.step(2)
      env.step(2)
      env.step(4)
      env.step(0)
      env.step(0)
      env.step(0)
      env.step(0)
      env.step(0)
      env.step(0)
      env.step(2)
      env.step(4)
      env.step(2)
      env.step(2)
      env.step(2)
      env.step(2)
      env.step(2)
      env.step(2)
      env.step(2)
      env.step(4)
      env.step(1)
      env.step(1)
      env.step(4)
      reward+=solve(env, item, visualise=visualise) 
      
  return reward

def craft(env, item):
{res}

print(evaluate())
                        '''
            # print(full_program)
    with open(script_path, 'w') as f:
        f.write(full_program_collect.strip())

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
                # print("output ", output)
        try:
            return float(output), True
        except ValueError:
            return -1, True
    except subprocess.TimeoutExpired:
        return -1, False
    except subprocess.CalledProcessError as e:
        print(f"Process Error: Command failed with exit code {e.returncode}")
        print(f"Command: {e.cmd}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return -1, False
    finally:
        # Clean up the temporary file
        if os.path.exists(script_path):
            os.remove(script_path)

def response_gen(funcs, k):
    with open("prompt_specifications/specification_with_updated_nld.txt", "r") as f:
        prompt1 = f.read()

    funcs_text = "\n\n".join(
        [f"### Score: {score}\n```python\n{body}\n```" for score, body in funcs]
    )

    prompt = (
        prompt1
        + "\n\nHere are different implementations of `def collect(env, primitive):`\n"
        + funcs_text
        + "\n\nAnalyse the functions and give natural language feedback in bullet points."
    )

    print(prompt)
    client = genai.Client()

    response = client.models.generate_content(
                          model="gemini-2.5-pro", contents = prompt
                      )
    feedback = response.text

    correction_prompt = (
      prompt1
      + "\n\nFeedback:\n"
      + feedback
      + "\n\nHere are the candidate functions for `def collect(env, primitive):`\n"
      + funcs_text
      + "\n\nReturn a corrected and improved version of the function."
  )
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

  # build filename
    log_filename = f"results/explicit_feedback/feedback_sampling_{timestamp}_{k}.json"
    i = 0
    while i< 20:
        response = client.models.generate_content(
                              model="gemini-2.5-pro", contents = correction_prompt
                          )
        feedback = response.text
      # print("second generation\n", b)
        try:
            feedback = feedback[feedback.index("def collect(env, primitive):")+len("def collect(env, primitive):")+1:]
            feedback = feedback[:feedback.index("```")]
        except:
            continue
        i+=1
            
        log_entry = {
          "extracted_function_code": feedback,
          "evaluation_result": eval(feedback),
        }
        # print(eval_result)
        # Write to log file in JSON format
        with open(log_filename, "a") as log_file:
            log_file.write(json.dumps(log_entry, indent=2) + ", \n")
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logfile", type=str, required=True,
                        help="Path to the feedback_sampling.json log file")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of top functions to extract (ties included)")
    args = parser.parse_args()

    funcs = parse_log_file(args.logfile, k=args.k)

    if not funcs:
        print("No functions found.")
        return

    response_gen(funcs, args.k)

if __name__ == "__main__":
    main()
