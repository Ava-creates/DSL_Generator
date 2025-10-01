from google import genai
import requests
with open("cfg/cfg.txt") as f:
    cfg = f.read()

# print(cfg)
with open("resources/recipes_for_synth.yaml") as f:
    recipes = f.read()

prompt = f"""
You are a Domain Specific Language (DSL) program generator for the Craft domain. 

## Context Free Grammar (CFG)
Here is the context-free grammar (CFG) that defines the DSL. Strictly follow this CFG when synthesising programs :

{cfg}

## Example Programs
Here are examples of programs written in this DSL:

COLLECT_FUNC(WOOD) ; MOVE_FUNC(RIGHT) ; CRAFT_FUNC(STICK)
COLLECT_FUNC(GRASS) ; MOVE_FUNC(RIGHT)

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

**make(goldarrow)**

## Output Format Instructions
Return ONLY the program string delimited by $ signs. Do not include any explanations, comments, or additional text outside the $ delimiters.

Example output format:
$CRAFT_FUNC(ARROW) ; COLLECT_FUNC(WOOD) ; MOVE_FUNC(RIGHT)$
"""

# with open("prompt_synth.txt", "w") as f:
#     f.write(prompt)


client = genai.Client()

for i in range(5):
# prompt = "You must act as a code completion model that is completing the last function. Please only return code that will fit in that function. Do not imports or add the function signature on the top. Return only the code that will be inside the function." + prompt
    response = client.models.generate_content(
                    model="gemini-2.5-pro", contents = prompt
                )
    b = response.text

    print(b)

# api_url = "http://129.128.243.184:11434/api/generate"
# headers = {"Content-Type": "application/json"}

        # while True:
# payload = {
#               "model": "qwen2.5-coder:32b", 
#               "prompt": prompt, 
#               "template": "{{.Prompt}}",
#               "stream": False, 
#               "options": {
#                 "num_ctx": 4096, 
#               }
#             }
# response = requests.post(api_url, headers=headers, json=payload, timeout=300)
# b = response.json()["response"]
# print(b)