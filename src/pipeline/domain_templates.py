#!/usr/bin/env python3
"""
Domain-specific templates for generated solve/evaluate functions.
"""


def _safe_name(func_name: str) -> str:
    return func_name.lower().replace("-", "_")


def craft_solve_template_basic(func_name: str, func_params: str, func_call_args: str) -> str:
    """Craft solve() template aligned to current smth.py solve (no prints)."""
    safe_name = _safe_name(func_name)
    return f'''def solve({func_params}, visualise=False):
  """Runs the environment with a {safe_name} function that returns list of actions to take and returns total reward."""
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
  grid_before = f"Grid shape: {{env._current_state.grid.shape if hasattr(env._current_state.grid, 'shape') else 'N/A'}}\\nAgent position: {{agent_pos}}"

  # Capture state before (for pass_check)
  state = env._current_state
  pos_before = _to_list(state.pos)
  inventory_before = _to_list(state.inventory.copy() if hasattr(state.inventory, 'copy') else state.inventory)
  dir_before = state.dir

  # Execute function to get actions using a deepcopy (function call uses the copied env)
  env_for_func = copy.deepcopy(env)
  actions_to_take = {safe_name}({func_call_args.replace("env", "env_for_func", 1)})
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
      return {{str(k): v for k, v in value}}
    return value

  def _inventory_list_to_dict(inv):
    if not isinstance(inv, list):
      return inv
    cb = env.world.cookbook if hasattr(env, "world") else None
    idx = cb.index if cb and hasattr(cb, "index") else None
    if idx is not None and hasattr(idx, "get"):
      return {{str(idx.get(i)): v for i, v in enumerate(inv) if v}}
    return inv

  class _InvList(list):
    def get(self, key, default=0):
      return default

  pass_check = None
  spec = getattr(env.scenario, 'spec', None) if hasattr(env, "scenario") else None
  if spec is not None:
    pass_check = spec.get('pass_check')
  if isinstance(pass_check, str):
    pass_check = pass_check.replace('\\\\"', '"')
    pass_check = pass_check.replace('\\"', '"')


  inventory_before = _inventory_list_to_dict(_pairs_to_dict(inventory_before))
  inventory_after = _inventory_list_to_dict(_pairs_to_dict(inventory_after))
  if isinstance(inventory_before, list):
    inventory_before = _InvList(inventory_before)
  if isinstance(inventory_after, list):
    inventory_after = _InvList(inventory_after)
  total_reward = 0
  if pass_check:
    passed = eval(pass_check, {{}}, {{
      'pos_before': pos_before,
      'pos_after': pos_after,
      'inventory_before': inventory_before,
      'inventory_after': inventory_after,
      'grid_before_cells': grid_before_cells,
      'grid_after_cells': grid_after_cells,
      'dir_before': dir_before,
      'dir_after': dir_after,
    }})
  else:
    passed = False
  test_type = spec.get('test_type', 'positive') if spec is not None else 'positive'
  if passed:
    total_reward += 100.0 if test_type == 'positive' else 1.0
  grid_before = None
  grid_after = None

  return [total_reward, actions_count, grid_before, dir_after]'''


def craft_evaluate_template(
    display_name: str,
    env_setup: str,
    args_definitions: str,
    func_call_args: str,
    grid_spec_paths_var: str | None = None,
    ) -> str:
    """Craft evaluate() template for running solve() in sample environment(s)."""
    def _indent(block: str, spaces: int) -> str:
        prefix = " " * spaces
        return "\n".join(prefix + line if line.strip() else line for line in block.splitlines())

    if grid_spec_paths_var:
        env_setup_in_loop = _indent(env_setup, 2)
        args_def_in_loop = _indent(args_definitions, 2) if args_definitions else ""
        return f'''@funsearch.run
def evaluate():
  """Evaluates {display_name} behavior across multiple test cases."""
  visualise = False
  total_reward = 0.0
  total_actions = 0
  grid_before = None
  grid_after = None
  _grid_spec_paths = {grid_spec_paths_var}
  ans = [0]*len(_grid_spec_paths)  #tracking what testcases pass or failed - 0 for failed 1 for passed
  i = 0 
  for grid_spec_path in _grid_spec_paths:
{env_setup_in_loop}
    env.reset()
    # arg_values = grid_spec.get("arg_values", {{}}) if isinstance(grid_spec, dict) else {{}}
{args_def_in_loop}    
    result = solve({func_call_args}, visualise=visualise)
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
'''

    return f'''@funsearch.run
def evaluate():
  """Evaluates {display_name} behavior in a sample environment."""
  visualise = False
  {env_setup}
  env.reset()
  arg_values = grid_spec.get("arg_values", {{}}) if isinstance(grid_spec, dict) else {{}}
{args_definitions}  
  result = solve({func_call_args}, visualise=visualise)
  # Return as list: [total_reward, actions_count, grid_before, grid_after]
  return result
'''


def craft_env_setup(
    recipes_path: str,
    hints_path: str,
    task_name: str,
    grid_spec_path: str,
) -> str:
    """Craft env setup block with grid/task handling."""
    return f"""
  recipes_path = "{recipes_path}"
  hints_path = "{hints_path}"
  task_name = "{task_name}"
  grid_spec = None
  if os.path.exists(grid_spec_path):
    try:
      with open(grid_spec_path, "r", encoding="utf-8") as f:
        grid_spec = json.load(f)
      task_name = grid_spec.get("task_name", task_name) or task_name
    except Exception:
      pass
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
  """


def craft_env_setup_from_var(
    recipes_path: str,
    hints_path: str,
    task_name: str,
    grid_spec_path_var: str,
) -> str:
    """Craft env setup block using a grid_spec_path variable."""
    return f"""
  import os
  import json
  recipes_path = "{recipes_path}"
  hints_path = "{hints_path}"
  grid_spec_path = {grid_spec_path_var}
  task_name = "{task_name}"
  grid_spec = None
  if grid_spec_path and os.path.exists(grid_spec_path):
    try:
      with open(grid_spec_path, "r", encoding="utf-8") as f:
        grid_spec = json.load(f)
      task_name = grid_spec.get("task_name", task_name) or task_name
    except Exception:
      pass
  custom_grid_path = grid_spec_path if grid_spec_path and os.path.exists(grid_spec_path) else None
  env_sampler = env_factory.EnvironmentFactory(
      recipes_path, hints_path, 7, max_steps=300, reuse_environments=False,
            visualise=visualise, custom_grid_path=custom_grid_path)
  env = env_sampler.sample_environment(task_name=task_name)
  # Attach grid_spec to scenario so pass_check is available in solve()
  try:
    if grid_spec is not None and hasattr(env, "scenario"):
      env.scenario.spec = grid_spec
  except Exception:
    pass
  """
