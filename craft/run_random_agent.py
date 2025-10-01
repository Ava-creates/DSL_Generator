"""Try Craft environment with random agent."""

from __future__ import division
from __future__ import print_function

import numpy as np
import time

import env_factory


def run_loop(env, n_steps, visualise=False):
  possible_actions = env.action_specs()

  observations = env.reset()
  # if visualise:
  #   env.render_matplotlib(frame=observations['image'])
  #   time.sleep(20)  # Keep the image visible for 2 seconds
  # else:
  #   print("Initial observations:", observations)
  # print("VDFS \n", env.world.cookbook.index, "\n")
  # actions=[3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 4]
  actions =[3, 4, 1, 1, 2, 4, 0, 2, 2, 2, 2, 4, 2, 4, 0, 0, 0, 0, 3, 4 , 1, 1, 2, 2, 2, 4]
  # actions= [3,4,0,0,4,0,2,2,4,0,0,3,4,0,3,0,3,0,0,2,2,4]
  time.sleep(4)
  if False:
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
  a = {tuple(pos) for pos in np.argwhere(env._current_state.grid[:,:,4])}
  print(a)
  for t in range(len(actions)):
    # Random action
    # print("hehe")
    # action = np.random.choice(possible_actions.values())
    print(env._current_state.inventory)
    action = actions[t]
    # Step (this will plot if visualise is True)
    reward, done, observations = env.step(action)
    # print(reward)
    if visualise:
      env.render_matplotlib(frame=observations['image'])
    else:
      print("[{}] reward={} done={} \n observations: {}".format(
          t, reward, done, observations))
    time.sleep(1)    
    if reward:
      rewarding_frame = observations['image'].copy()
      rewarding_frame[:40] *= np.array([0, 1, 0])
      env.render_matplotlib(frame=rewarding_frame, delta_time=0.7)
      print("[{}] Got a rewaaaard! {:.1f}".format(t, reward))
    elif done:

      env.render_matplotlib(
          frame=np.zeros_like(observations['image']), delta_time=0.3)
      print("[{}] Finished with nothing... Reset".format(t))
    

def main():
  visualise = True
  recipes_path = "resources/recipes.yaml"
  hints_path = "resources/hints.yaml"
  # env_sampler = env_factory.EnvironmentFactory(
  #     recipes_path, hints_path, 6, max_steps=100, reuse_environments=False,
  #     visualise=visualise)
  recipes_path_2 = "resources/recipes_for_synth.yaml"
  item = "arrow"
  env_sampler = env_factory.EnvironmentFactory(
            recipes_path_2, hints_path, 6, max_steps=100, 
            reuse_environments=False, visualise=True)
  env=env_sampler.sample_environment(task_name='make[goldarrow]')
  # env = env_sampler.sample_environment(task_name='make[knife]')
  print("Environment: task {}: {}".format(env.task_name, env.task))
  run_loop(env, 100 * 3, visualise=visualise)


if __name__ == '__main__':
  main()
