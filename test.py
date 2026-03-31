import numpy as np
import time
import collections
from craft import craft, env, env_factory
import pandas as pd
import os
import json

def solve(env, primitive, visualise=False):
  """Runs the environment with a collect_primitive function that returns list of actions to take and returns total reward."""
  import copy

  def _to_list(x):
    try:
      if hasattr(x, "tolist"):
        return x.tolist()
      return list(x)
    except Exception:
      return x

  # Capture grid state before function execution (with agent position)
  grid_before_cells = []
  g = env._current_state.grid
  cookbook = env.world.cookbook
  for y in range(g.shape[1]):
    row = []
    for x in range(g.shape[0]):
      cell = g[x, y]
      indices = [i for i, v in enumerate(cell) if v]
      if indices:
        row.append(str(cookbook.index.get(indices[0])).strip().lower())
      else:
        row.append("")
    grid_before_cells.append(row)
  agent_pos = env._current_state.pos
  grid_before = f"Grid shape: {env._current_state.grid.shape if hasattr(env._current_state.grid, 'shape') else 'N/A'}\nAgent position: {agent_pos}"

  # Capture state before (for pass_check)
  state = env._current_state
  pos_before = _to_list(state.pos)
  inventory_before = _to_list(state.inventory.copy() if hasattr(state.inventory, 'copy') else state.inventory)
  dir_before = state.dir

  # Execute function to get actions using a deepcopy (function call uses the copied env)
  env_for_func = copy.deepcopy(env)
  actions_to_take = collect_primitive(env_for_func, primitive)
  if actions_to_take is None:
    actions_to_take = []

  total_reward = 0.0
  actions_count = 0
  for action in actions_to_take:
    reward, done, observations = env.step(action)
    total_reward += reward
    actions_count += 1
    if done:
      break

  # Capture grid state after function execution (with agent position)
  grid_after_cells = []
  g = env._current_state.grid
  cookbook = env.world.cookbook
  for y in range(g.shape[1]):
    row = []
    for x in range(g.shape[0]):
      cell = g[x, y]
      indices = [i for i, v in enumerate(cell) if v]
      if indices:
        row.append(str(cookbook.index.get(indices[0])).strip().lower())
      else:
        row.append("")
    grid_after_cells.append(row)
  agent_pos = env._current_state.pos

  # Capture state after (for pass_check)
  state = env._current_state
  pos_after = _to_list(state.pos) 
  inventory_after = _to_list(state.inventory.copy())
  dir_after = state.dir

  def _pairs_to_dict(value):
    if isinstance(value, dict):
      return value
    if isinstance(value, list) and all(isinstance(pair, (list, tuple)) and len(pair) == 2 for pair in value):
      return {str(k): v for k, v in value}
    return value

  def _inventory_list_to_dict(inv):
    if not isinstance(inv, list):
      return inv
    cb = env.world.cookbook if hasattr(env, "world") else None
    idx = cb.index if cb and hasattr(cb, "index") else None
    if idx is not None and hasattr(idx, "get"):
      return {str(idx.get(i)): v for i, v in enumerate(inv) if v}
    return inv

  class _InvList(list):
    def get(self, key, default=0):
      return default

  pass_check = None
  spec = getattr(env.scenario, 'spec', None) if hasattr(env, "scenario") else None
  if spec is not None:
    pass_check = spec.get('pass_check')
  if isinstance(pass_check, str):
    pass_check = pass_check.replace('\\"', '"')
    pass_check = pass_check.replace('\"', '"')


  inventory_before = _inventory_list_to_dict(_pairs_to_dict(inventory_before))
  inventory_after = _inventory_list_to_dict(_pairs_to_dict(inventory_after))
  if isinstance(inventory_before, list):
    inventory_before = _InvList(inventory_before)
  if isinstance(inventory_after, list):
    inventory_after = _InvList(inventory_after)
  total_reward = 0
  if pass_check:
    passed = eval(pass_check, {}, {
      'pos_before': pos_before,
      'pos_after': pos_after,
      'inventory_before': inventory_before,
      'inventory_after': inventory_after,
      'grid_before_cells': grid_before_cells,
      'grid_after_cells': grid_after_cells,
      'dir_before': dir_before,
      'dir_after': dir_after,
    })
  else:
    passed = False
  test_type = spec.get('test_type', 'positive') if spec is not None else 'positive'
  if passed:
    total_reward += 100.0 if test_type == 'positive' else 1.0
  grid_before = None
  grid_after = None

  return [total_reward, actions_count, grid_before, dir_after]
def evaluate():
  """Evaluates COLLECT_PRIMITIVE behavior across multiple test cases."""
  visualise = False
  total_reward = 0.0
  total_actions = 0
  grid_before = None
  grid_after = None
  _grid_spec_paths = ['experiments/experiment_20260329_183119_29942/grids/collect_primitive_dsl1_case0.json', 'experiments/experiment_20260329_183119_29942/grids/collect_primitive_dsl1_case1.json', 'experiments/experiment_20260329_183119_29942/grids/collect_primitive_dsl1_case10.json', 'experiments/experiment_20260329_183119_29942/grids/collect_primitive_dsl1_case11.json', 'experiments/experiment_20260329_183119_29942/grids/collect_primitive_dsl1_case12.json', 'experiments/experiment_20260329_183119_29942/grids/collect_primitive_dsl1_case13.json', 'experiments/experiment_20260329_183119_29942/grids/collect_primitive_dsl1_case14.json', 'experiments/experiment_20260329_183119_29942/grids/collect_primitive_dsl1_case2.json', 'experiments/experiment_20260329_183119_29942/grids/collect_primitive_dsl1_case3.json', 'experiments/experiment_20260329_183119_29942/grids/collect_primitive_dsl1_case4.json', 'experiments/experiment_20260329_183119_29942/grids/collect_primitive_dsl1_case5.json', 'experiments/experiment_20260329_183119_29942/grids/collect_primitive_dsl1_case6.json', 'experiments/experiment_20260329_183119_29942/grids/collect_primitive_dsl1_case7.json', 'experiments/experiment_20260329_183119_29942/grids/collect_primitive_dsl1_case8.json', 'experiments/experiment_20260329_183119_29942/grids/collect_primitive_dsl1_case9.json']
  ans = [0]*len(_grid_spec_paths)  #tracking what testcases pass or failed - 0 for failed 1 for passed
  i = 0 
  for grid_spec_path in _grid_spec_paths:

    recipes_path = "craft/resources/recipes.yaml"
    hints_path = "craft/resources/hints.yaml"
    grid_spec = None
    with open(grid_spec_path, "r", encoding="utf-8") as f:
      grid_spec = json.load(f)
    task_name = grid_spec.get("task_name")
    custom_grid_path = grid_spec_path if os.path.exists(grid_spec_path) else None
    env_sampler = env_factory.EnvironmentFactory(
        recipes_path, hints_path, 7, max_steps=400, reuse_environments=False,
              visualise=visualise, custom_grid_path=custom_grid_path)
    env = env_sampler.sample_environment(task_name=task_name)
    # Attach grid_spec to scenario so pass_check is available in solve()
    try:
      if grid_spec is not None and hasattr(env, "scenario"):
        env.scenario.spec = grid_spec
    except Exception:
      pass
  
    env.reset()
    # arg_values = grid_spec.get("arg_values", {}) if isinstance(grid_spec, dict) else {}
    arg_values = grid_spec["arg_values"] if isinstance(grid_spec, dict) else {}
    primitive = arg_values["primitive"]
    if isinstance(primitive, str):
      primitive = primitive.lower()    
    result = solve(env, primitive, visualise=visualise)
    if isinstance(result, list) and len(result) >= 2:
      total_reward += result[0] if result[0] is not None else 0.0
      if result[0] is not None:
        ans[i] = 1 if result[0] > 0 else 0
      i += 1
      total_actions += result[1] if result[1] is not None else 0
      if len(result) >= 3 and result[2] is not None:
        grid_before = result[2]
      if len(result) >= 4 and result[3] is not None:
        grid_after = result[3]
  # Return as list: [total_reward, actions_count, grid_before, grid_after]
  return [total_reward, total_actions, ans, grid_after]

def collect_primitive(env, primitive):
  """
  Picks up one instance of the given primitive that is currently reachable.
  
  Args:
      env: The current environment instance.
        primitive (str): Function-specific argument(s).
    
      Returns: List[int]: A sequence of raw integer action codes accepted by env.step().

  """
  return []

print(evaluate())