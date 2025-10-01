        prompt = f"""
    You are a Domain Specific Language (DSL) program generator for the Craft domain. 

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

    Also, when collecting gold you will need to craft a bridge and to get gem you will need an axe. 
    ## Task
    Generate a program that solves the following task :

    **{task}**

    ## Output Format Instructions
    Return ONLY the program string delimited by $ signs. Do not include any explanations, comments, or additional text outside the $ delimiters.

    Example output format:
    $program$
    """