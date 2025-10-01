from google import genai

recipes_path = "resources/recipes.yaml"

with open(recipes_path, "r") as f:
    recipes = f.read()

prompt = f'''
You are a context-free grammar (CFG) designer. Given the domain specified below in natural language, return a CFG in **BNF format** that models valid actions in this domain. Use **uppercase for all terminal function names and terminal symbols**. **Do not include assignments like X ::= "x"**—only include grammar rules. Terminal functions can take arguments in LPAR and RPAR. Also provide a short example program in the grammar.

Example format:
expr   ::= term PLUS expr | term
term   ::= factor TIMES term | factor
factor ::= NUMBER | LPAR expr RPAR


##Natural Language Description of Domain
Craft is a single-agent game in a pre-specified environment. 
The environment of craft is a grid world of size n * n. Each cell can be empty, contain an item, or part of natural terrain or functional structures. When the cell is nonempty, it is considered as blocked. A agent can move around the environment freely through empty cells. At each step, the agent can either move or perform a specific actions, such as collect or craft, towards the immediate cell that it is facing towards. 
At the beginning of each episode, the agent is placed at a starting cell and a distribution of items across the grid is initialized. The agent’s tasks involve either collecting primitives (raw resources) or crafting items. A item can only be crafted at the specific workshop mentioned in the recipes. 
The item to be craft are produced from primitives (or other crafted items) by following recipes. Each recipe specifies which items are required and at which workshop the crafting must occur. A primitive item might not need to be crafted but just collected. More complex items, such as axe, or flag, require intermediate items along with primitives. This all is specified in the recipe file of the environment. Please note a item can only be crafted at the specific workshop mentioned in the recipes. 
We can move in up, down ,left, and right directions.

Sometimes, primitive can be blocked by obstacles taht envioenment items that cannot be used as primitives for crafting stuff and are not boundary or workshop and needs the player to use a tool to pass the obstacle in order to reach and collect the primitive. The tool can be any crafted item.

###You can use the recipes file below to come up with the grammar. 

{recipes}

##Task
Return a CFG in BNF format that models valid actions in this domain followed by a dictionary of natural language description of the terminal functions. Also, return an example program written in the grammar.

'''
    
client = genai.Client()

# print(prompt)

response = client.models.generate_content(
                            model="gemini-2.5-pro", contents = prompt
                        )
b = response.text

print(b)