import numpy as np
import time
import collections
import env_factory
import craft
import env
def collect(env, primitive):
    """
    Computes a shortest path to collect a specified primitive using a stateful Breadth-First Search.

    This function finds the shortest sequence of actions for the agent to move
    adjacent to a target primitive and collect it. The search accounts for the
    agent's current position, inventory, and the grid layout.

    The agent can use tools from its inventory to overcome obstacles (e.g., using a
    'bridge' to cross 'water'). The BFS simulates these actions by tracking changes
    to the grid and inventory, allowing it to discover paths through cleared obstacles.

    The state in the BFS queue consists of:
    - pos (tuple): The agent's (x, y) coordinates.
    - inventory (np.ndarray): The agent's current inventory.
    - grid (np.ndarray): The current state of the world grid for that search branch.
    - actions (list): The sequence of actions taken to reach this state.

    The search terminates upon finding a path that places the agent next to the
    target primitive, returning the full action sequence, including the final 'USE'
    action. If no path is found, it returns an empty list.

    Args:
        env (env.CraftLab): The CraftLab environment instance, providing access to the current state.
        primitive (str): The name of the primitive to collect (e.g., 'wood', 'gold').

    Returns:
        List[int]: A list of action indices to collect the primitive, or an empty
                   list if it's unreachable.
    """
    UP, DOWN, LEFT, RIGHT, USE = 0, 1, 2, 3, 4
    
    initial_state = env._current_state
    world = initial_state.world
    cookbook = world.cookbook
    grid_shape = initial_state.grid.shape

    try:
        target_idx = cookbook.index[primitive]
        water_idx = cookbook.index['water']
        bridge_idx = cookbook.index['bridge']
        stone_idx = cookbook.index['stone']
        axe_idx = cookbook.index['axe']
    except KeyError:
        # This occurs if a required item like 'water' or the primitive itself isn't in the cookbook.
        return []

    # Map obstacles to the tools required to clear them
    tool_for_obstacle = {
        water_idx: bridge_idx,
        stone_idx: axe_idx
    }

    # BFS queue stores tuples of: (position, inventory, grid, actions)
    queue = collections.deque([
        (
            initial_state.pos,
            initial_state.inventory.copy(),
            initial_state.grid.copy(),
            []
        )
    ])
    
    # Visited set prevents cycles and redundant computations.
    # The key includes position, inventory, and the grid state.
    visited = set()
    initial_state_key = (initial_state.pos, tuple(initial_state.inventory), initial_state.grid.tobytes())
    visited.add(initial_state_key)

    # Map action indices to their corresponding (dx, dy) deltas
    action_deltas = {
        UP:    (0, -1),
        DOWN:  (0, 1),
        LEFT:  (-1, 0),
        RIGHT: (1, 0),
    }
    
    while queue:
        pos, inventory, grid, actions = queue.popleft()

        if len(actions) > 200:  # Safety break to prevent searching infinitely on complex maps
            continue

        # Explore neighbors by trying each directional action from the current position
        for action, (dx, dy) in action_deltas.items():
            neighbor_pos = (pos[0] + dx, pos[1] + dy)

            # Check if the neighbor is within the grid bounds
            if not (0 <= neighbor_pos[0] < grid_shape[0] and 0 <= neighbor_pos[1] < grid_shape[1]):
                continue
            
            # Identify the content of the neighbor cell
            cell_idx = np.argmax(grid[neighbor_pos])

            # Case 1: Neighbor is the target primitive. We found a solution.
            if cell_idx == target_idx:
                # To collect, the agent must face the target and USE. The `action` will turn
                # the agent. The subsequent move will be blocked by the resource, but the
                # agent will be correctly oriented for the USE action.
                return actions + [action, USE]

            # Case 2: Neighbor is an empty, traversable cell.
            if cell_idx == 0:
                new_pos = neighbor_pos
                new_actions = actions + [action]
                
                # The grid and inventory don't change for a simple move.
                state_key = (new_pos, tuple(inventory), grid.tobytes())
                if state_key not in visited:
                    visited.add(state_key)
                    queue.append((new_pos, inventory, grid, new_actions))

            # Case 3: Neighbor is an obstacle that can be cleared with a tool.
            elif cell_idx in tool_for_obstacle:
                required_tool_idx = tool_for_obstacle[cell_idx]
                
                # Check if the agent has the necessary tool in its inventory.
                if inventory[required_tool_idx] > 0:
                    # Simulate using the tool: agent stays at `pos`, but inventory and grid change.
                    new_inventory = inventory.copy()
                    new_inventory[required_tool_idx] -= 1
                    
                    new_grid = grid.copy()
                    # Clear the obstacle cell, making it empty (represented by a zero vector).
                    new_grid[neighbor_pos] = 0.0

                    # The action sequence is to face the obstacle and use the tool.
                    new_actions = actions + [action, USE]
                    
                    # This new search state starts from the *same position* but with an updated world.
                    state_key = (pos, tuple(new_inventory), new_grid.tobytes())
                    if state_key not in visited:
                        visited.add(state_key)
                        queue.append((pos, new_inventory, new_grid, new_actions))

    # If the queue becomes empty, no path was found.
    return []
