"""Try Craft environment with random agent."""

from __future__ import division
from __future__ import print_function

import numpy as np
import time

from . import env_factory

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
  # actions =[3, 4, 1, 1, 2, 4, 0, 2, 2, 2, 2, 4, 2, 4, 0, 0, 0, 0, 3, 4 , 1, 1, 2, 2, 2, 4]
  # actions= [3,4,0,0,4,0,2,2,4,0,0,3,4,0,3,0,3,0,0,2,2,4]
  # actions = [0, 4, 0, 4, 2, 2, 2, 4, 1, 1, 1, 1, 4, 1, 1, 1, 3, 3, 4, 1, 3, 4, 0, 3, 4, 1, 1, 4, 2, 2, 4, 3, 3, 3, 3, 3, 3, 4, 0, 0, 2, 4, 0, 0, 0, 2, 4, 0, 2, 4, 3, 3, 4, 3, 4, 0, 2, 4, 0, 4, 0, 1, 2, 3, 2, 3, 0, 2, 1, 3, 2, 3, 0, 2, 4, 1, 1, 1, 1, 4, 1, 1, 1, 3, 3, 4, 1, 3, 4, 0, 3, 4, 1, 1, 4, 2, 2, 4, 3, 3, 3, 3, 3, 3, 4, 0, 0, 2, 4, 0, 0, 0, 2, 4, 0, 2, 4, 3, 3, 4, 0, 4, 0, 4, 2, 0, 1, 2, 0, 3, 3, 2, 1, 3, 2, 0, 2, 2, 2, 0, 2, 2, 0, 3, 0, 1, 0, 2, 2, 0, 1, 0, 1, 0, 1, 1, 2, 1, 2, 0, 0, 0, 0, 2, 3]
  actions = [0, 4, 0, 4, 2, 2, 2, 4, 1, 1, 1, 1, 4, 1, 1, 1, 3, 3, 4, 1, 3, 4, 0, 3, 4, 1, 1, 4, 2, 2, 4, 3, 3, 3, 3, 3, 3, 4, 0, 0, 2, 4, 0, 0, 0, 2, 4, 0, 2, 4, 3, 3, 4, 3, 4, 0, 2, 4, 0, 4, 2, 1, 0, 3, 3, 3, 2, 0, 2, 4, 1, 1, 1, 1, 4, 1, 1, 1, 3, 3, 4, 1, 3, 4, 0, 3, 4, 1, 1, 4, 2, 2, 4, 3, 3, 3, 3, 3, 3, 4, 0, 0, 2, 4, 0, 0, 0, 2, 4, 0, 2, 4, 3, 3, 4, 0, 4, 0, 4, 2, 1, 3, 3, 3, 0, 2, 3, 2, 2, 3, 1, 1, 1, 1, 0, 1, 3, 3, 1]
  time.sleep(4)
  # if False:
  #     env.step(0)
  #     env.step(2)
  #     env.step(2)
  #     env.step(4)
  #     env.step(0)
  #     env.step(0)
  #     env.step(0)
  #     env.step(0)
  #     env.step(0)
  #     env.step(0)
  #     env.step(2)
  #     env.step(4)
  #     env.step(2)
  #     env.step(2)
  #     env.step(2)
  #     env.step(2)
  #     env.step(2)
  #     env.step(2)
  #     env.step(2)
  #     env.step(4)
  #     env.step(1)
  #     env.step(1)
  #     env.step(4)  
  # a = {tuple(pos) for pos in np.argwhere(env._current_state.grid[:,:,4])}
  # print(a)
  total_reward = 0.0
  for t in range(len(actions)):
    action = actions[t]
    # Step (this will plot if visualise is True)
    reward, done, observations = env.step(action)
    total_reward += reward
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
      break
  print("Total reward:", total_reward)
    

def main():
  visualise = False
  recipes_path = "craft/resources/recipes.yaml"
  hints_path = "craft/resources/hints.yaml"
  task_name = "get[gem]"
  # env_sampler = env_factory.EnvironmentFactory(
  #     recipes_path, hints_path, 6, max_steps=100, reuse_environments=False,
  #     visualise=visualise)
  # recipes_path_2 = "craft/resources/recipes_for_synth.yaml"
  # item = "arrow"
  # env_sampler = env_factory.EnvironmentFactory(
  #           recipes_path_2, hints_path, 6, max_steps=100, 
  #           reuse_environments=False, visualise=True)
  # env=env_sampler.sample_environment(task_name='make[goldarrow]')

  env_sampler = env_factory.EnvironmentFactory(
    recipes_path,
    hints_path,
    7,
    max_steps=300,
    seed=0,
    reuse_environments=True,
    visualise=visualise,
  )

  env = env_sampler.sample_environment(task_name=task_name)
  env.reset()
  print("Environment: task {}: {}".format(env.task_name, env.task))
  try:
    import hashlib
    grid_md5 = hashlib.md5(env._current_state.grid.tobytes()).hexdigest()
    print(f"[{task_name}] grid_md5={grid_md5}")
    state = env._current_state
    pos = tuple(state.pos) if hasattr(state, "pos") else None
    direction = int(state.dir) if hasattr(state, "dir") else None
    inventory = getattr(state, "inventory", None)
    if inventory is not None:
      inv_nonzero = [
        (env.world.cookbook.index.get(i, str(i)), float(v))
        for i, v in enumerate(inventory) if v
      ]
    else:
      inv_nonzero = None
    print(f"[{task_name}] pos={pos} dir={direction} inv={inv_nonzero}")
  except Exception as e:
    print(f"[{task_name}]  Could not hash grid: {e}")
  
  run_loop(env, 100 * 3, visualise=visualise)


if __name__ == '__main__':
  main()
