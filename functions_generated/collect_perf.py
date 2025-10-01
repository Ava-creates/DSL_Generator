import numpy as np
import time
import collections
import env_factory
import craft
import env
def collect(env, primitive):
    """Returns a sequence of actions to collect only the specified primitive, using only the agent's current inventory. Do not pick up primitives that are not passed as the argument.

    This function computes a shortest path to reach and collect a given primitive (e.g., gold, gem, wood) in the environment.
    It accounts for obstacles and environmental constraints by allowing the agent to use tools already in inventory—
    for example:
        - Using a bridge to cross water in order to reach gold.
        - Using a pickaxe to mine gem.

    The function assumes the world is static except for changes resulting from tool use (e.g., placing a bridge).
    It does not perform crafting or attempt to acquire new items—only available tools in the inventory are used.

    Args:
        env (env.CraftLab): The CraftLab environment instance.
        primitive (str): The name of the primitive to collect.

    Returns:
        List[int]: A list of action indices the agent can execute to collect the primitive.
    """ 
    MAX_STEPS = 100
    UP, DOWN, LEFT, RIGHT, USE = 0, 1, 2, 3, 4

    action_list = []
    state = env._current_state
    target_index = state.world.cookbook.index[primitive]

    queue = collections.deque([(state.pos, 0, np.copy(state.inventory), [])])
    visited = set()
    # print(state.pos)
    while queue:
        pos, steps, inv, actions = queue.popleft()
        # print(pos, steps, actions)
        if steps >= MAX_STEPS:
            continue

        # Check if the position and inventory have been visited
        state_key = (tuple(pos), tuple(inv))
        if state_key in visited:
            continue
        visited.add(state_key)

        # Create a new state object for the current BFS step
        state = craft.CraftState(
            scenario=state.scenario,
            grid=np.copy(state.grid),
            pos=pos,
            dir=state.dir,
            inventory=np.copy(inv)
        )

        # # Check if the target primitive is next to the agent
        adjacent_cells = [
            (pos[0], pos[1] - 1),  # UP
            (pos[0], pos[1] + 1),  # DOWN
            (pos[0] - 1, pos[1]),  # LEFT
            (pos[0] + 1, pos[1])   # RIGHT
        ]


        for adj_pos in adjacent_cells:
            if 0 <= adj_pos[0] < state.grid.shape[0] and 0 <= adj_pos[1] < state.grid.shape[1]:
                cell_index = np.argmax(state.grid[adj_pos])

                if cell_index == target_index:
                    action_list = actions + [UP, USE] if adj_pos[1] < pos[1] else\
                                actions + [DOWN, USE] if adj_pos[1] > pos[1] else\
                                actions + [LEFT, USE] if adj_pos[0] < pos[0] else\
                                actions + [RIGHT, USE]
                    # print(action_list)
                    return action_list

        # Generate possible moves
        for i, new_pos in enumerate(adjacent_cells):
            if 0 <= new_pos[0] < state.grid.shape[0] and 0 <= new_pos[1] < state.grid.shape[1]:
                cell_index = np.argmax(state.grid[new_pos].any())
                if not cell_index:
                    queue.append((new_pos, steps + 1, inv, actions + [i]))

        # Check for tool usage
        # print(inv)
        inventory_items = np.where(inv > 0)[0]
        # print(inventory_items)
        for item_idx in inventory_items:

            tool_usage_conditions = {
                'gold': ('bridge', 'water'),
                'GEM': ('PICKAXE', 'ROCK'),
                'TREE': ('AXE', 'TREE'),
                'BOULDER': ('HAMMER', 'BOULDER'),
                'IRON_ORE': ('DRILL', 'IRON_ORE'),
                'BUSH': ('SHEARS', 'BUSH'),
                'STONE': ('HAMMER', 'STONE'),
            }

            if primitive in tool_usage_conditions:

                required_tool, target_resource = tool_usage_conditions[primitive]
                if item_idx == state.world.cookbook.index[required_tool]:
                    for dir_idx, adj_pos in enumerate(adjacent_cells):
                        if 0 <= adj_pos[0] < state.grid.shape[0] and 0 <= adj_pos[1] < state.grid.shape[1]:
                            cell_index = np.argmax(state.grid[adj_pos])
                            if cell_index == state.world.cookbook.index[target_resource]:
                                new_inv = inv.copy()
                                queue.append((adj_pos, steps + 2, new_inv, actions + [dir_idx, USE]))
    
    return []  # Return empty list and negative reward if target is unreachable

