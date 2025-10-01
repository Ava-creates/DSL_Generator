from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
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

  return actions_to_take


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
# Replace with your actual downloaded path
local_model_path = "/scratch/avani/qwen"  # from your snapshot_download

tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    device_map="auto",             # automatically selects GPUs if available
    torch_dtype=torch.float16,     # for large models like 32B
    trust_remote_code=True
)

model.eval()

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
for i in range(1000):

    try:
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=512,
                do_sample=True, 
                temperature=0.8,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        res = decoded[len(prompt):]

        for stop in stop_tokens:
            if stop in res:
                res = res.split(stop)[0]
                break

        log_entry = {
            'function_body': res,
            'scores': eval(res)
        }
        with open("meow_hf.log", "a") as f:
            f.write(json.dumps(log_entry) + '\n')
    except Exception as e:
        print(e)
