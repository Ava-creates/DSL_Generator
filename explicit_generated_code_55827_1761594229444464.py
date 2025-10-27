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
    # print(item, total_reward, actions_to_take)
   
  if total_reward>0.5:
    return [0.2, len(actions_to_take)]
  # print(primitive, total_reward, actions_to_take)
  return [total_reward, len(actions_to_take)]


def evaluate() -> float:
  """Evaluates a crafting policy on a sample task."""
  visualise = False
  recipes_path = "craft/resources/recipes.yaml"
  hints_path = "craft/resources/hints.yaml"
  reward = 0 
  actions = 0
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
    reward += r[0]
    actions += r[1]

  return [reward, actions]


def collect(env, primitive):
  import numpy as np
  from collections import deque

  # Get the index of the target primitive and the agent's current position
  primitive_index = env.world.cookbook.index[primitive]
  start_x, start_y = env._current_state.pos

  # Directions and their corresponding actions
  directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
  action_map = {(-1, 0): craft.LEFT, (1, 0): craft.RIGHT, 
                (0, -1): craft.DOWN, (0, 1): craft.UP}

  # Helper function to check if a cell is blocked by obstacles other than the primitive itself
  def is_blocked(x, y):
    return any(env._current_state.grid[x, y][i] > 0 for i in range(len(env._current_state.grid[x, y]))
               if i != primitive_index)

  # Helper function to check if a cell can be used with available tools
  def can_use_tools(x, y):
      tool_indices = [env.world.cookbook.index[tool] for tool in env.world.cookbook.index if tool.endswith("WORKSHOP")]
      return any(env._current_state.inventory[i] > 0 for i in tool_indices)

  # BFS to find the shortest path to a cell containing the primitive
  visited = set()
  queue = deque([(start_x, start_y, [])])  # (x, y, path)

  while queue:
    x, y, path = queue.popleft()

    if (x, y) in visited:
      continue

    visited.add((x, y))

    for dx, dy in directions:
      nx, ny = x + dx, y + dy
      
      # Check bounds
      if 0 <= nx < env._current_state.grid.shape[0] and 0 <= ny < env._current_state.grid.shape[1]:
        # If the cell contains the primitive, return the path to it
        if env._current_state.grid[nx, ny][primitive_index] > 0:
          move_action = action_map[(dx, dy)]
          use_action = craft.USE
          return path + [move_action, use_action]
        
        # If the cell is not blocked and hasn't been visited, add to queue
        if not is_blocked(nx, ny):
          move_action = action_map[(dx, dy)]
          queue.append((nx, ny, path + [move_action]))

  return []  # Return an empty list if no path found


print(evaluate())