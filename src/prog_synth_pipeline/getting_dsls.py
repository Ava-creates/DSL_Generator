import requests
import json
api_url = "http://129.128.243.184:11434/api/generate"
headers = {"Content-Type": "application/json"}

# prompt = """The Craft environment is a 2D world inspired by Minecraft, where an agent must complete hierarchical tasks with sparse rewards. The environment is represented by a 12×12 grid containing:

# Environment Elements->
# boundary: impassable edge
# water: can be crossed only by building a bridge
# stone: can be removed only using an axe
# workshop0, workshop1, workshop2, ..., workshopn: specialized crafting areas that are specified in the recipe file which is subject to change


# Primitive (Grabbable) Resources->
# wood
# iron
# grass
# rock
# gold
# gem


# Crafting System->
# Crafting occurs at specific workshops using the USE action for various items base don the recipe. Here are few examples of items:
# plank:
#   wood: 1
#   _at: workshop0

# and more items that one can get from the recipe file

# Already Implemented Agent Actions ->

# Movement
# The agent can move in four cardinal directions:
# UP, DOWN, LEFT, RIGHT
# Movement is blocked if the destination cell is occupied.

# USE Action
# Allows the agent to interact with the environment in the direction it is facing:
# Collects grabbable primitives (e.g., wood, rock)
# Use the workshop when all needed ingredients for crafting something are in the inventory
# Use the items in inventory to do stuff like building bridges, breaking rock, etc.

# Tasks to be performed (these are already given to the game)->
# make(item)
# get(item)

# Given the above specification, generate a CFG in Backus Normal Form(BNF) format that can be used to create a Domain Specific Language that will make playing games to achieve the specified task in the domain heirarichal, easy, and intuitive. The grammar can have terminal functions that will make use of the implementation of the craft enviornment in python. 
# Output the context free grammar(CFG) in the BNF format and two sample programs written in the CFG you generated. Output should be structured as a json.

# """"

prompt ="""
The environment is a 12×12 grid-based crafting world inspired by Minecraft. An agent moves around this 2D space to collect primitives, use tools, and craft complex items by combining materials according to recipes. The goal is to complete high-level tasks such as crafting items (e.g., goldarrow), which often require multiple intermediate steps, including overcoming obstacles and using specific workshops.

The environment contains:

Primitives: raw materials like wood, rock, or glue, which are used for crafting.

Tools: such as axes, bridges, or pickaxes, which allow the agent to overcome obstacles or harvest certain materials.

Obstacles: such as water, rocks, or trees that can block the agent’s path or access to resources.

Workshops: locations where crafting must occur, depending on the item being created (e.g., furnace, crafting table).

A cookbook: defines which primitives are needed to craft each item.

The agent’s behavior should reflect realistic and hierarchical planning. For example, to craft a goldarrow, the agent should:

Look up the recipe in the cookbook to determine which primitives are needed.

Plan to gather each primitive one at a time:

Find the location of the primitive.

Check whether there are obstacles near or between the agent and the primitive.

If an obstacle exists, determine which tool is needed to bypass or remove it.

If the agent does not already have the required tool, plan to get the tool first.

Once the path is clear, move to the primitive and collect it.

After collecting all required primitives:

Determine whether the item must be crafted at a specific workshop.

If so, find the location of the appropriate workshop and move there.

Craft the item.

The agent should reason about dependencies between actions (e.g., tools before primitives, location before movement), reuse tools when possible, and avoid unnecessary repetition. The resulting programs should make this logical structure explicit, breaking down high-level goals into meaningful, modular subtasks.

The grammar should reflect this structure and support reusable operations like:

Moving to a location

Locating resources

Gathering tools if needed

Navigating around obstacles

Using workshops

Performing crafting actions

Generate a general, reusable grammar that allows expressing such task-completion plans in this domain.
"""

payload = {
    "model": "qwen3:32b", 
    "prompt": prompt, 
    "stream": False, 
    "format": {
        "type": "object",
        "properties": {
            "cfg": {
                "type": "string"
            },
            "program1": {
                "type": "string"
            },
            "program2": {
                "type": "string"
            }
            },
        "required": [
            "cfg",
            "program1",
            "program2"
            ]
    }
        }
res = requests.post(api_url, headers=headers, json=payload, timeout=300)

res = res.json()["response"]
res = json.loads(res) 
# print(res)
# print(type(res))
print(res["cfg"])
print(res["program1"])
print(res["program2"])