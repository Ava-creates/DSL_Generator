import requests
import json
import os
import subprocess
stop_tokens = ["\ndef", "\nclass", "\n#", "\nimport"]

def eval(res):
            # with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = os.getcwd()
            script_path = os.path.join(temp_dir, 'generated_code.py')
            
            # Create complete executable program
            full_program = f'''
import numpy as np
import time

import env_factory

def solve(env, visualise=False) -> float:
  """Runs the environment with a craft function that returns list of actions to takr and returns total reward."""
  item = 14 
  actions_to_take = craft_func(env, 14)
  print("actions", actions_to_take)
  observations = env.reset()
  total_reward = 0.0

  for t in range(len(actions_to_take)):
    action = actions_to_take[t]
    reward, done, observations = env.step(action)
    print(reward)
    total_reward += reward
    # print(env._current_state.satisfies(None, 14))
    if reward:
      rewarding_frame = observations['image'].copy()
      rewarding_frame[:40] *= np.array([0, 1, 0])
    elif done:
      break

  return total_reward


def evaluate() -> float:
  """Evaluates a crafting policy on a sample task."""
  visualise = False
  recipes_path = "resources/recipes.yaml"
  hints_path = "resources/hints.yaml"

  env_sampler = env_factory.EnvironmentFactory(
      recipes_path, hints_path, max_steps=100, reuse_environments=False,
      visualise=visualise)

  env = env_sampler.sample_environment(task_name='make[stick]')
  return solve(env, visualise=visualise)

def craft_func(env, item_index):
{res}

print(evaluate())
                        '''
            # print(full_program)
            with open(script_path, 'w') as f:
                f.write(full_program.strip())

            try:
                # Convert input to string representatio
                
                # Execute in subprocess with timeout
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
                print("output ", output)
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
                
with open("prompt.txt", 'r') as f:
        prompt = f.read()

print(prompt)

api_url = "http://129.128.243.184:11434/api/generate"
headers = {"Content-Type": "application/json"}
payload = {
          "model": "qwen2.5-coder:32b", 
          "prompt": prompt, 
          "template": "{{.Prompt}}",
          "stream": False, 
          "options": {
            "num_ctx": 4096, 
            "stop": stop_tokens
          }
        }

log_file = os.path.join("results", f'meow_with_codebase_dsl_action_list.log')

for i in range(1000):
    res = requests.post(api_url, headers=headers, json=payload, timeout=300)
    res = res.json()["response"] 
    print(res)
    log_entry = {
        'function_body': res,
        'scores': eval(res)
    }
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + '\n')