# -*- coding: utf-8 -*-
import numpy as np
import time
import collections
from craft import env_factory

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
    # print(item, total_reward, actions_to_take)
    return total_reward

def evaluate() -> float:
  """Evaluates a crafting policy on a sample task."""
  #max reward is 6 for this fucntion so any craft objet that can get when it is working properly
  visualise = False
  recipes_path = "resources/recipes.yaml"
  hints_path = "resources/hints.yaml"     
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
  

ACTION_MAP = {'DOWN': 0, 'UP': 1, 'LEFT': 2, 'RIGHT': 3, 'USE': 4}


def craft(env, item):
    """
    Generates a sequence of actions to navigate to the correct workshop,
    face it, and craft the specified item.

    This function implements a robust strategy by:
    1. Identifying the specific workshop required by the item's recipe.
    2. Finding all empty, accessible cells adjacent to that workshop type.
    3. Using Breadth-First Search (BFS) to find the shortest, obstacle-free
       path to one of these adjacent cells.
    4. Generating actions to turn the agent to face the workshop.
    5. Appending the 'USE' action to perform the craft.
    """
    cookbook = env.world.cookbook
    state = env._current_state

    # 1. Look up item and its recipe to find the required workshop
    goal_index = cookbook.index[item]
    if goal_index is None or goal_index not in cookbook.recipes:
        raise ValueError(f"Item '{item}' is not craftable or does not exist.")

    recipe = cookbook.recipes[goal_index]
    workshop_name = recipe['_at']
    workshop_index = cookbook.index[workshop_name]

    # 2. Get current environment state for pathfinding
    grid = state.grid
    # Use (y, x) convention for position, matching numpy's (row, col) indexing
    start_pos = tuple(state.pos)
    grid_height, grid_width = grid.shape[0], grid.shape[1]

    # A cell is considered blocked if it contains any object.
    blocked = grid.sum(axis=2) > 0

    # 3. Find all valid target positions (empty cells adjacent to the workshop)
    workshop_locations = np.argwhere(grid[:, :, workshop_index])
    if workshop_locations.shape[0] == 0:
        raise ValueError(f"Required workshop '{workshop_name}' not found in the environment.")

    valid_targets = set()
    for ws_pos in workshop_locations:
        y, x = ws_pos
        # Check 4 neighbors (UP, DOWN, LEFT, RIGHT)
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            adj_pos = (y + dy, x + dx)
            # Check if neighbor is within bounds and is not blocked
            if (0 <= adj_pos[0] < grid_height and
                0 <= adj_pos[1] < grid_width and
                not blocked[adj_pos[0], adj_pos[1]]):
                valid_targets.add(adj_pos)

    if not valid_targets:
        raise RuntimeError(f"No accessible cells next to workshop '{workshop_name}'.")

    # 4. Pathfind using BFS to the nearest valid target cell
    # Action mapping: 0:DOWN, 1:UP, 2:LEFT, 3:RIGHT
    # Corresponding (dy, dx) deltas for (y, x) coordinates:
    action_deltas = {
        0: (1, 0),   # DOWN (y increases)
        1: (-1, 0),  # UP (y decreases)
        2: (0, -1),  # LEFT (x decreases)
        3: (0, 1),   # RIGHT (x increases)
    }
    # Reverse mapping from delta to action for the orientation step
    delta_to_action = {v: k for k, v in action_deltas.items()}

    queue = collections.deque([(start_pos, [])])  # Stores (position, path_of_actions)
    visited = {start_pos}

    path_to_target = None
    final_pos = None

    # Handle the case where the agent is already at a target position
    if start_pos in valid_targets:
        path_to_target = []
        final_pos = start_pos
    else:
        while queue:
            current_pos, path = queue.popleft()

            for action, (dy, dx) in action_deltas.items():
                next_pos = (current_pos[0] + dy, current_pos[1] + dx)

                if not (0 <= next_pos[0] < grid_height and 0 <= next_pos[1] < grid_width):
                    continue
                if next_pos in visited:
                    continue
                
                if next_pos in valid_targets:
                    path_to_target = path + [action]
                    final_pos = next_pos
                    queue.clear()  # Path found, stop searching
                    break
                
                if not blocked[next_pos[0], next_pos[1]]:
                    visited.add(next_pos)
                    queue.append((next_pos, path + [action]))

    if path_to_target is None:
        raise RuntimeError(f"Could not find a path to an accessible spot near '{workshop_name}'.")
    # 5. Determine orientation and add actions to face the workshop
    # Find the specific workshop cell that `final_pos` is adjacent to
    target_workshop_pos = None
    for ws_pos in workshop_locations:
        ws_y, ws_x = ws_pos
        if abs(ws_y - final_pos[0]) + abs(ws_x - final_pos[1]) == 1:
            target_workshop_pos = (ws_y, ws_x)
            break
    # Determine the direction the agent needs to face
    dy = target_workshop_pos[0] - final_pos[0]
    dx = target_workshop_pos[1] - final_pos[1]
    target_action_for_facing = delta_to_action[(dy, dx)]
    # The agent's direction after the path is its last move action.
    # If the path is empty, the agent hasn't moved, so use its current direction.
    agent_dir_at_target = state.dir if not path_to_target else path_to_target[-1]
    final_actions = list(path_to_target)
    # If not already facing the workshop, perform a turn-in-place maneuver.
    # This involves moving to an adjacent cell and immediately moving back.
    if agent_dir_at_target != target_action_for_facing:
        opposites = {0: 1, 1: 0, 2: 3, 3: 2}  # DOWN/UP, LEFT/RIGHT
        turn_action_1 = opposites[target_action_for_facing]
        turn_action_2 = target_action_for_facing
        final_actions.extend([turn_action_1, turn_action_2])
    # 6. Add the USE action to perform the craft
    final_actions.append(4)  # USE
    return final_actions
    
def craft_perfect(env, item):
    """
    Generates a sequence of actions to move to the correct workshop,
    face it, and craft the specified item.
    """
    cookbook = env.world.cookbook
    state = env._current_state
    grid = state.grid
    start_pos = tuple(state.pos)
    # 1. Get Recipe & Required Workshop from the cookbook
    goal_index = cookbook.index[item]
    if goal_index is None:
        raise ValueError(f"Unknown item: {item}")
    recipe = cookbook.recipes.get(goal_index)
    if not recipe:
        raise ValueError(f"No recipe found for item: {item}")
    workshop_name = recipe.get('_at')
    if not workshop_name:
        raise ValueError(f"Recipe for {item} does not specify a workshop.")

    required_workshop_idx = cookbook.index[workshop_name]
    if required_workshop_idx is None:
        raise ValueError(f"Unknown workshop: {workshop_name}")

    # 2. Find all valid target cells (empty cells adjacent to the correct workshop)
    workshop_locations = np.argwhere(grid[:, :, required_workshop_idx])
    if workshop_locations.size == 0:
        raise ValueError(f"No '{workshop_name}' found on the grid.")

    # Map of {agent_target_pos: corresponding_workshop_pos}
    target_map = {}
    for wx, wy in workshop_locations:
        # Check 4-directional neighbors
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = wx + dx, wy + dy
            # Check if neighbor is valid and empty
            if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1] and not np.any(grid[nx, ny, :]):
                target_map[(nx, ny)] = (wx, wy)
    
    if not target_map:
        raise ValueError(f"No accessible cells next to any '{workshop_name}'.")
    
    # 3. Pathfinding using Breadth-First Search (BFS)
    # The BFS finds the shortest path of actions to the nearest valid target cell.
    
    # Action mapping: 0=DOWN, 1=UP, 2=LEFT, 3=RIGHT
    # Maps a delta (dx, dy) to a specific action
    action_map = {(0, -1): 0, (0, 1): 1, (-1, 0): 2, (1, 0): 3}
    
    # Queue stores tuples of (current_position, list_of_actions_to_get_here)
    queue = collections.deque([(start_pos, [])])
    visited = {start_pos}
    
    path_actions = []
    final_agent_pos = None
    final_workshop_pos = None

    # Handle edge case: agent is already next to the workshop
    if start_pos in target_map:
        path_actions = []
        final_agent_pos = start_pos
        final_workshop_pos = target_map[start_pos]
    else:
        found_path = False
        while queue:
            (cx, cy), current_actions = queue.popleft()

            # Explore neighbors
            for (dx, dy), action in action_map.items():
                nx, ny = cx + dx, cy + dy
                
                if (nx, ny) in visited:
                    continue
                
                # Check boundaries and if the cell is empty (not an obstacle)
                is_valid = (0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1])
                if is_valid and not np.any(grid[nx, ny, :]):
                    visited.add((nx, ny))
                    new_actions = current_actions + [action]
                    # If neighbor is a target, we found the shortest path
                    if (nx, ny) in target_map:
                        path_actions = new_actions
                        final_agent_pos = (nx, ny)
                        final_workshop_pos = target_map[(nx, ny)]
                        found_path = True
                        break 
                    queue.append(((nx, ny), new_actions))
            if found_path:
                break

    if final_agent_pos is None:
        raise RuntimeError(f"Could not find a path to a '{workshop_name}'.")

    # 4. Generate Final Turn and USE Actions
    actions = list(path_actions)
    
    agent_x, agent_y = final_agent_pos
    workshop_x, workshop_y = final_workshop_pos

    # Determine direction to face and the corresponding action.
    # This action "bumps" into the workshop, setting the agent's direction correctly.
    face_action = -1
    if workshop_y > agent_y: face_action = 1 # Face UP
    elif workshop_y < agent_y: face_action = 0 # Face DOWN
    elif workshop_x > agent_x: face_action = 3 # Face RIGHT
    elif workshop_x < agent_x: face_action = 2 # Face LEFT
    
    if face_action != -1:
        actions.append(face_action)
    
    # Finally, append the USE action to craft the item
    actions.append(4)

    return actions
print(evaluate()) 
