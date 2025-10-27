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
  def get_primitive_index(primitive_name):
      return env._current_state.world.cookbook.index[primitive_name]

  actions = {
      "UP": 0,
      "DOWN": 1,
      "LEFT": 2,
      "RIGHT": 3,
      "USE": 4
  }

  def get_neighbors(pos):
      x, y = pos
      return [(x + dx, y + dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]]

  current_state = env._current_state

  queue = collections.deque([(current_state.pos, [])])
  visited = set([current_state.pos])

  target_primitive_index = get_primitive_index(primitive)

  while queue:
      pos, actions_taken = queue.popleft()

      if current_state.grid[pos[0], pos[1], target_primitive_index] > 0:
          return actions_taken + [actions["USE"]]

      neighbors = get_neighbors(pos)
      for neighbor in neighbors:
          nx, ny = neighbor

          # Ensure the neighbor is within bounds.
          if 0 <= nx < current_state.grid.shape[1] and 0 <= ny < current_state.grid.shape[0]:
              # Check if the cell is not blocked by non-grabbable entities.
              blocked = False
              for env_index in current_state.world.non_grabbable_indices:
                  if current_state.grid[ny, nx, env_index] > 0:
                      blocked = True

              if not blocked:
                  if neighbor not in visited:
                      visited.add(neighbor)

                      dx, dy = nx - pos[0], ny - pos[1]
                      if dx == 1:
                          action = actions["RIGHT"]
                      elif dx == -1:
                          action = actions["LEFT"]
                      elif dy == 1:
                          action = actions["DOWN"]
                      else:  # dy == -1
                          action = actions["UP"]

                      queue.append((neighbor, actions_taken + [action]))

              else:
                  # Check if any tool in inventory can be used to clear the path.
                  for tool_index in current_state.world.grabbable_indices:
                      if current_state.inventory[tool_index] > 0 and current_state.grid[ny, nx, tool_index] > 0:
                          return actions_taken + [actions["USE"]]

      # Check if there is any obstacle that needs to be cleared.
      for env_index in current_state.world.non_grabbable_indices:
          if current_state.grid[pos[0], pos[1], env_index] > 0:
              # Use the tool if available and applicable.
              for tool_index in current_state.world.grabbable_indices:
                  if current_state.inventory[tool_index] > 0:
                      return actions_taken + [actions["USE"]]

  return []


print(evaluate())