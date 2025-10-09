import numpy as np
import time
import collections
from craft import craft, env, env_factory

def solve(env, visualise=False) -> float:
  """Runs the environment with a collect function that returns list of actions to take and returns total reward."""
  actions_to_take = make_arrow(env)
  total_reward = 0.0

  for t in range(len(actions_to_take)):
    action = actions_to_take[t]
    reward, done, observations = env.step(action)
    total_reward += reward
    if done:
      break
  return total_reward

def evaluate() -> float:
  """Evaluates a collecting policy on a set of sample tasks."""
  #max reward is 4
  visualise = False
  recipes_path = "craft/resources/recipes.yaml"
  hints_path = "craft/resources/hints.yaml"
  reward = 0 

  env_sampler = env_factory.EnvironmentFactory(
  recipes_path, hints_path, 0, max_steps=100, reuse_environments=False,
            visualise=visualise)

  env = env_sampler.sample_environment(task_name= 'make[arrow]')
  reward = solve(env,  visualise=visualise)
  return reward


