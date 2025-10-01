import numpy as np
import collections
# def craft(env, item):
#     cookbook = env.world.cookbook
#     goal_index = cookbook.index[item]

#     if goal_index is None:
#         raise ValueError("Unknown item")

#     workshop_indices = env.world.workshop_indices

#     actions = []

#     # Find the closest workshop that can craft the desired item
#     closest_workshop_idx, min_distance = None, float('inf')
#     pos = np.array(env._current_state.pos)

#     for workshop_idx in workshop_indices:
#         # Calculate the mean position of all workshops of this type
#         workshop_pos_list = np.argwhere(env._current_state.grid[:, :, workshop_idx])
        
#         if len(workshop_pos_list) > 0:  # Check if there is any location for the workshop
#             workshop_pos_mean = workshop_pos_list.mean(axis=0)
#             distance = np.linalg.norm(pos - workshop_pos_mean, ord=2)
#             if distance < min_distance:
#                 closest_workshop_idx, min_distance = workshop_idx, distance

#     if closest_workshop_idx is None:
#         raise ValueError("No available workshop found")

#     # Calculate the closest position to move towards
#     target_positions = np.argwhere(env._current_state.grid[:, :, closest_workshop_idx])
#     nearest_target_pos = None
#     min_nearest_distance = float('inf')

#     for target_pos in target_positions:
#         distance = np.linalg.norm(pos - target_pos, ord=2)
#         if distance < min_nearest_distance:
#             nearest_target_pos = target_pos
#             min_nearest_distance = distance

#     # Move to the closest workshop position
#     while not np.array_equal(pos, nearest_target_pos):
#         dx, dy = nearest_target_pos - pos
        
#         # Determine direction to move in
#         if abs(dx) > abs(dy):  # Prioritize moving in x-direction first
#             actions.append(3 if dx > 0 else 2)
#             pos[0] += 1 if dx > 0 else -1
#         else:  # Then move in y-direction
#             actions.append(1 if dy > 0 else 0)
#             pos[1] += 1 if dy > 0 else -1

#     # Use the workshop to craft the item
#     actions.append(4)  # USE
#     # print(actions)
#     return actions

def craft(env, item):
    """
    Generates a sequence of actions to move to the correct workshop and craft an item.

    Args:
        env: The CraftLab environment instance.
        item (str): The name of the item to craft.

    Returns:
        list[int]: A list of action integers to execute.
    """
    world = env.world
    state = env._current_state
    cookbook = world.cookbook

    # 1. Find the recipe and required workshop for the item
    goal_idx = cookbook.index[item]
    if goal_idx is None:
        raise ValueError(f"Item '{item}' not found in cookbook.")

    recipe = cookbook.recipes.get(goal_idx)
    if not recipe or '_at' not in recipe:
        raise ValueError(f"No crafting recipe or workshop found for item '{item}'.")

    workshop_name = recipe['_at']
    workshop_idx = cookbook.index[workshop_name]

    # 2. Find all locations of the required workshop and their valid adjacent cells
    grid = state.grid
    (width, height, _) = grid.shape
    
    workshop_positions = np.argwhere(grid[:, :, workshop_idx])
    if workshop_positions.shape[0] == 0:
        raise ValueError(f"Required workshop '{workshop_name}' not found in the environment.")

    # A cell is walkable if it has no items/terrain on it
    walkable_mask = (np.sum(grid, axis=2) == 0)

    # Add the agent's current position as walkable for the start of the path
    start_pos = tuple(state.pos)
    walkable_mask[start_pos[0], start_pos[1]] = True

    # Identify all valid, walkable target cells adjacent to any required workshop
    target_cells = {}  # Map target cell -> corresponding workshop cell
    for pos in workshop_positions:
        wx, wy = pos
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            adj_pos = (wx + dx, wy + dy)
            if 0 <= adj_pos[0] < width and 0 <= adj_pos[1] < height:
                if walkable_mask[adj_pos[0], adj_pos[1]]:
                    target_cells[adj_pos] = tuple(pos)
    
    if not target_cells:
        raise RuntimeError(f"No accessible cells next to a '{workshop_name}'.")

    # 3. Pathfind from agent's position to the nearest target cell using BFS
    queue = collections.deque([(start_pos, [])])  # (current_pos, path_to_here)
    visited = {start_pos}
    
    path_to_target = None
    
    while queue:
        current_pos, path = queue.popleft()

        if current_pos in target_cells:
            path_to_target = path + [current_pos]
            break

        x, y = current_pos
        for dx, dy, action in [(0, -1, 0), (0, 1, 1), (-1, 0, 2), (1, 0, 3)]:  # UP, DOWN, LEFT, RIGHT
            next_pos = (x + dx, y + dy)
            if (
                0 <= next_pos[0] < width and
                0 <= next_pos[1] < height and
                walkable_mask[next_pos[0], next_pos[1]] and
                next_pos not in visited
            ):
                visited.add(next_pos)
                new_path = path + [current_pos]
                queue.append((next_pos, new_path))

    if path_to_target is None:
        raise RuntimeError(f"Cannot find a path to a '{workshop_name}'.")

    # 4. Convert the path of coordinates into a sequence of move actions
    actions = []
    # Path includes the start point, so iterate from the second element
    for i in range(1, len(path_to_target)):
        prev_pos = path_to_target[i - 1]
        curr_pos = path_to_target[i]
        dx = curr_pos[0] - prev_pos[0]
        dy = curr_pos[1] - prev_pos[1]

        if dy == -1:
            actions.append(0)  # UP
        elif dy == 1:
            actions.append(1)  # DOWN
        elif dx == -1:
            actions.append(2)  # LEFT
        elif dx == 1:
            actions.append(3)  # RIGHT

    # 5. Add final actions: turn to face the workshop, then use it
    final_pos = path_to_target[-1]
    workshop_pos = target_cells[final_pos]

    dx = workshop_pos[0] - final_pos[0]
    dy = workshop_pos[1] - final_pos[1]

    if dy == -1:
        actions.append(0)  # Face UP
    elif dy == 1:
        actions.append(1)  # Face DOWN
    elif dx == -1:
        actions.append(2)  # Face LEFT
    elif dx == 1:
        actions.append(3)  # Face RIGHT

    actions.append(4)  # USE

    return actions
