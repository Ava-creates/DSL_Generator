import numpy as np
import time
import collections
from craft import craft, env, env_factory

def solve(env, primitive, visualise=False) -> float:
  """Runs the environment with a collect function that returns list of actions to take and returns total reward."""
  actions_to_take = collect(env, primitive)
  total_reward = 0.0

  for t in range(len(actions_to_take)):
    action = actions_to_take[t]
    reward, done, observations = env.step(action)
    total_reward += reward
    if done:
      break

  if total_reward>0.5:
    return 0.2

  return total_reward


def evaluate() -> float:
  """Evaluates a crafting policy on a sample task."""
  visualise = False
  recipes_path = "craft/resources/recipes.yaml"
  hints_path = "craft/resources/hints.yaml"
  reward = 0 
  for i in range(10):
    if(i == 0):
      p = "wood"
      env_sampler = env_factory.EnvironmentFactory(
      recipes_path, hints_path, 0, max_steps=100, reuse_environments=False,
            visualise=visualise)

      env = env_sampler.sample_environment(task_name= 'make[stick]')
      env.reset()
    
    elif(i==1):
      p = "iron"
      env_sampler = env_factory.EnvironmentFactory(
      recipes_path, hints_path, 1, max_steps=100, reuse_environments=False,
            visualise=visualise)

      env = env_sampler.sample_environment(task_name= 'make[bridge]')
      env.reset()
      
    elif(i==2):
      p = "wood"
      env_sampler = env_factory.EnvironmentFactory(
      recipes_path, hints_path, 1, max_steps=100, reuse_environments=False,
            visualise=visualise)

      env = env_sampler.sample_environment(task_name= 'make[bridge]')
      env.reset()

    elif(i==3): #grass not present onthe grid should return empty list
      p = "grass"
      env_sampler = env_factory.EnvironmentFactory(
      recipes_path, hints_path, 1, max_steps=100, reuse_environments=False,
            visualise=visualise)

      env = env_sampler.sample_environment(task_name= 'make[bridge]')
      env.reset()

    elif(i==4):
      p = "wood"
      env_sampler = env_factory.EnvironmentFactory(
      recipes_path, hints_path, 2, max_steps=100, reuse_environments=False,
            visualise=visualise)

      env = env_sampler.sample_environment(task_name= 'make[plank]')
      env.reset()
      #env.step(1)
      #env.step(4)

    elif(i==5):
      p = "grass"
      env_sampler = env_factory.EnvironmentFactory(
      recipes_path, hints_path, 3, max_steps=100, reuse_environments=False,
            visualise=visualise)

      env = env_sampler.sample_environment(task_name= 'make[cloth]')
      env.reset()
      #env.step(1)
      #env.step(4)


    elif(i==6):
      p = "grass"
      env_sampler = env_factory.EnvironmentFactory(
      recipes_path, hints_path, 4, max_steps=100, reuse_environments=False,
            visualise=visualise)

      env = env_sampler.sample_environment(task_name= 'make[rope]')
      env.reset()
      #env.step(0)
      #env.step(0)
      #env.step(4)

    elif(i==7):
      p = "grass"
      env_sampler = env_factory.EnvironmentFactory(
      recipes_path, hints_path, 5, max_steps=100, reuse_environments=False,
            visualise=visualise)

      env = env_sampler.sample_environment(task_name= 'make[bundle]')
      env.reset()
      #env.step(0)
      #env.step(0)
      #env.step(4)
      #env.step(0)
      #env.step(4)

    elif(i==8):
      p = "wood"
      env_sampler = env_factory.EnvironmentFactory(
      recipes_path, hints_path, 5, max_steps=100, reuse_environments=False,
            visualise=visualise)

      env = env_sampler.sample_environment(task_name= 'make[bundle]')
      env.reset()
      #env.step(0)
      #env.step(0)
      #env.step(4)

    else:
      p = "gold"
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
      
    r= solve(env, p, visualise=visualise)
    reward += r

  return reward
  
  
def collect(env, primitive):
  """
  Generates a sequence of actions to navigate to and collect a specified primitive.

  This function uses a Breadth-First Search (BFS) algorithm to find the shortest
  path to an empty cell adjacent to the target primitive. It then adds the
  necessary actions to turn towards the primitive and collect it.

  Args:
    env: The CraftLab environment instance.
    primitive: The string name of the primitive to collect.

  Returns:
    A list of integer actions to be executed, or an empty list if the
    primitive cannot be reached or does not exist.
  """
  state = env._current_state
  grid = state.grid
  start_pos = state.pos  # Assumed to be (x, y) or (col, row)
  width, height, _ = grid.shape

  # Define movement vectors and their corresponding action constants from the craft module.
  # Assumes a coordinate system where pos=(x, y) with x for width and y for height.
  directions = {
      craft.UP:    (0, -1),  # Decrement y
      craft.DOWN:  (0, 1),   # Increment y
      craft.LEFT:  (-1, 0),  # Decrement x
      craft.RIGHT: (1, 0),   # Increment x
  }
  # Create a reverse mapping from vector to action for the final turn.
  action_for_vec = {v: k for k, v in directions.items()}

  # 1. Get the integer index for the primitive name.
  try:
    primitive_index = state.world.cookbook.index[primitive]
  except KeyError:
    # The requested primitive is not defined in the game's recipes.
    return []

  # 2. Find all locations of the primitive and identify adjacent, empty cells as goals.
  primitive_locations = np.argwhere(grid[:, :, primitive_index] > 0)
  if primitive_locations.size == 0:
    # The primitive does not exist on the current map.
    return []

  goal_positions = set()
  # This map stores which primitive location each goal corresponds to,
  # which is needed to determine the final turn direction.
  goal_to_primitive_map = {}

  for x, y in primitive_locations:
    # A goal is an empty cell *from which* the agent can collect the primitive.
    for action, (dx, dy) in directions.items():
      # The adjacent cell's coordinates.
      adj_x, adj_y = x + dx, y + dy
      if 0 <= adj_x < width and 0 <= adj_y < height:
        # Check if this adjacent cell is empty and thus traversable.
        if grid[adj_x, adj_y].sum() == 0:
          goal_pos = (adj_x, adj_y)
          goal_positions.add(goal_pos)
          # This goal allows collecting from primitive at (x, y).
          goal_to_primitive_map[goal_pos] = (x, y)

  if not goal_positions:
    # No accessible spots to collect the primitive from.
    return []

  # 3. BFS to find the shortest path to one of the goal positions.
  queue = collections.deque([(start_pos, [])])  # (position, path_of_actions)
  visited = {start_pos}

  found_path = None
  final_pos = None

  # Handle the edge case where the agent already starts at a goal position.
  if start_pos in goal_positions:
    found_path = []
    final_pos = start_pos
  else:
    while queue:
      (curr_x, curr_y), path = queue.popleft()

      for action, (dx, dy) in directions.items():
        next_x, next_y = curr_x + dx, curr_y + dy
        next_pos = (next_x, next_y)

        if not (0 <= next_x < width and 0 <= next_y < height) or next_pos in visited:
          continue

        # The agent can only move into completely empty cells.
        if grid[next_x, next_y].sum() == 0:
          visited.add(next_pos)
          new_path = path + [action]
          if next_pos in goal_positions:
            found_path = new_path
            final_pos = next_pos
            # Since BFS explores layer by layer, the first path found is the shortest.
            queue.clear()
            break
          queue.append((next_pos, new_path))

  # 4. If a path was found, construct the full action sequence.
  if found_path is None:
    # No path exists from the agent's position to any goal.
    return []

  # Determine the direction to face for collection.
  primitive_pos = goal_to_primitive_map[final_pos]
  vec_to_primitive = (primitive_pos[0] - final_pos[0], primitive_pos[1] - final_pos[1])
  turn_action = action_for_vec[vec_to_primitive]

  # The final sequence: move to the adjacent spot, turn to face, then collect.
  return found_path + [turn_action, craft.USE]


print(evaluate())