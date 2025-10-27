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
  def _find_shortest_path(grid, start_pos, target_kind, inventory):
    width, height = grid.shape[:2]
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    queue = collections.deque([(start_pos[0], start_pos[1], [])])
    visited = set()
    
    while queue:
      x, y, path = queue.popleft()
      
      if grid[x, y].argmax() == target_kind and (x, y) != start_pos:
        return path
      
      for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in visited:
          cell_content_index = grid[nx, ny].argmax()
          # Allow movement on clear cells or through tools that are available in the inventory
          if grid[nx, ny].sum() == 1 or inventory[cell_content_index] > 0:  
            visited.add((nx, ny))
            queue.append((nx, ny, path + [(dx, dy)]))
    
    return None

  def _get_action_index_from_direction(direction):
    mapping = {(-1, 0): craft.LEFT, (1, 0): craft.RIGHT, (0, -1): craft.DOWN, (0, 1): craft.UP}
    return mapping.get(direction)

  current_state = env._current_state
  grid = current_state.grid
  pos = current_state.pos
  inventory = current_state.inventory
  
  primitive_index = current_state.world.cookbook.index.index(primitive)
  
  path_to_primitive = _find_shortest_path(grid, pos, primitive_index, inventory)
  
  if not path_to_primitive:
    return []
  
  actions = [_get_action_index_from_direction(step) for step in path_to_primitive]
  actions.append(craft.USE)  # Add action to collect the primitive
  
  return actions


print(evaluate())