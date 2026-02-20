#!/usr/bin/env python3
"""
Domain-specific templates for generated solve/evaluate functions.
"""


def _safe_name(func_name: str) -> str:
    return func_name.lower().replace("-", "_")


def craft_solve_template_for_prompt(func_name: str, args: str) -> str:
    """Craft solve() template with reward-logic insertion marker."""
    if args:
        func_params = f"env, {args}, visualise=False"
        func_call_args = f"env, {args}"
    else:
        func_params = "env, visualise=False"
        func_call_args = "env"

    safe_name = _safe_name(func_name)

    return f'''def solve({func_params}):
  """Runs the environment with a {safe_name} function that returns list of actions to take and returns total reward."""
  # Capture grid state before function execution (with agent position)
  grid_before = None
  grid_before_cells = None
  def _to_list(x):
    try:
      if hasattr(x, "tolist"):
        return x.tolist()
      return list(x)
    except Exception:
      return x
  try:
    if hasattr(env, '_current_state') and hasattr(env._current_state, 'grid'):
      try:
        from test import grid_to_markdown
        # Get agent position for grid representation - ensure it's a tuple
        agent_pos = None
        if hasattr(env._current_state, 'pos'):
          pos = env._current_state.pos
          # Convert to tuple if it's a numpy array or list
          if hasattr(pos, '__iter__') and not isinstance(pos, str):
            agent_pos = tuple(pos) if len(pos) == 2 else None
          elif isinstance(pos, tuple):
            agent_pos = pos
        grid_before = grid_to_markdown(env._current_state.grid, env.world.cookbook, agent_pos)
        # Also build a 2D grid of item names for pass_check.
        try:
          g = env._current_state.grid
          cookbook = env.world.cookbook
          grid_before_cells = []
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
        except Exception:
          grid_before_cells = None
      except (ImportError, AttributeError) as e:
        agent_pos = None
        if hasattr(env._current_state, 'pos'):
          pos = env._current_state.pos
          if hasattr(pos, '__iter__') and not isinstance(pos, str):
            agent_pos = tuple(pos) if len(pos) == 2 else None
        grid_before = f"Grid shape: {{env._current_state.grid.shape if hasattr(env._current_state.grid, 'shape') else 'N/A'}}\\nAgent position: {{agent_pos}}"
        # Fallback: try to build grid cells even if markdown failed.
        try:
          g = env._current_state.grid
          cookbook = env.world.cookbook
          grid_before_cells = []
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
        except Exception:
          grid_before_cells = None
  except Exception as e:
    pass
  
  # Capture state before (for reward computation)
  state_before = {{}}
  if hasattr(env, '_current_state'):
    state = env._current_state
    if hasattr(state, 'pos'):
      if hasattr(state.pos, '__iter__') and not isinstance(state.pos, str):
        try:
          state_before['pos'] = _to_list(state.pos)
        except Exception:
          state_before['pos'] = state.pos
      else:
        state_before['pos'] = state.pos
    if hasattr(state, 'inventory'):
      inv = state.inventory.copy() if hasattr(state.inventory, 'copy') else state.inventory
      state_before['inventory'] = _to_list(inv)
    if hasattr(state, 'dir'):
      state_before['dir'] = state.dir
    # Also store individual variables for convenience
    pos_before = state_before.get('pos')
    inventory_before = state_before.get('inventory')
    dir_before = state_before.get('dir')
  else:
    pos_before = None
    inventory_before = None
    dir_before = None
  
  # Call function to get actions
  actions_to_take = {safe_name}({func_call_args})
  if actions_to_take is None:
    actions_to_take = []
  
  # Execute actions and accumulate environment rewards
  actions_count = 0
  total_reward = 0.0
  for action in actions_to_take:
    reward, done, obs = env.step(action)
    total_reward += reward
    actions_count += 1
    if done:
      break
  
  # Capture state after (for reward computation)
  state_after = {{}}
  if hasattr(env, '_current_state'):
    state = env._current_state
    if hasattr(state, 'pos'):
      if hasattr(state.pos, '__iter__') and not isinstance(state.pos, str):
        try:
          state_after['pos'] = _to_list(state.pos)
        except Exception:
          state_after['pos'] = state.pos
      else:
        state_after['pos'] = state.pos
    if hasattr(state, 'inventory'):
      inv = state.inventory.copy() if hasattr(state.inventory, 'copy') else state.inventory
      state_after['inventory'] = _to_list(inv)
    if hasattr(state, 'dir'):
      state_after['dir'] = state.dir
    # Also store individual variables for convenience
    pos_after = state_after.get('pos')
    inventory_after = state_after.get('inventory')
    dir_after = state_after.get('dir')
  else:
    pos_after = None
    inventory_after = None
    dir_after = None
  
  # Capture grid state after function execution (with agent position)
  grid_after = None
  grid_after_cells = None
  try:
    if hasattr(env, '_current_state') and hasattr(env._current_state, 'grid'):
      try:
        from test import grid_to_markdown
        # Get agent position for grid representation - ensure it's a tuple
        agent_pos = None
        if hasattr(env._current_state, 'pos'):
          pos = env._current_state.pos
          # Convert to tuple if it's a numpy array or list
          if hasattr(pos, '__iter__') and not isinstance(pos, str):
            agent_pos = tuple(pos) if len(pos) == 2 else None
          elif isinstance(pos, tuple):
            agent_pos = pos
        grid_after = grid_to_markdown(env._current_state.grid, env.world.cookbook, agent_pos)
        # Also build a 2D grid of item names for pass_check.
        try:
          g = env._current_state.grid
          cookbook = env.world.cookbook
          grid_after_cells = []
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
        except Exception:
          grid_after_cells = None
      except (ImportError, AttributeError) as e:
        agent_pos = None
        if hasattr(env._current_state, 'pos'):
          pos = env._current_state.pos
          if hasattr(pos, '__iter__') and not isinstance(pos, str):
            agent_pos = tuple(pos) if len(pos) == 2 else None
        grid_after = f"Grid shape: {{env._current_state.grid.shape if hasattr(env._current_state.grid, 'shape') else 'N/A'}}\\nAgent position: {{agent_pos}}"
        # Fallback: try to build grid cells even if markdown failed.
        try:
          g = env._current_state.grid
          cookbook = env.world.cookbook
          grid_after_cells = []
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
        except Exception:
          grid_after_cells = None
  except Exception as e:
    pass

  # Compute reward using pass_check only; no fallback logic
  new_reward = 0.0
  pass_check = None
  arg_values = {{}}
  if hasattr(env, 'scenario') and hasattr(env.scenario, 'spec'):
    spec = env.scenario.spec
    if isinstance(spec, dict):
      pass_check = spec.get('pass_check')
      arg_values = spec.get('arg_values', {{}}) if isinstance(spec, dict) else {{}}
  if isinstance(pass_check, str) and pass_check.strip():
    pass_check = pass_check.replace('\\\\"', '"')
    try:
      passed = bool(eval(pass_check, {{}}, {{
          'env': env,
          'pos_before': pos_before,
          'pos_after': pos_after,
          'inventory_before': inventory_before,
          'inventory_after': inventory_after,
          'dir_before': dir_before,
          'dir_after': dir_after,
          'grid_before': grid_before,
          'grid_after': grid_after,
          'grid_before_cells': grid_before_cells,
          'grid_after_cells': grid_after_cells,
      }}))
    except Exception:
      passed = False
    new_reward = 1.0 if passed else 0.0
  total_reward += new_reward

  # Return [total_reward, actions_count, grid_before, grid_after]
  return [total_reward, actions_count, grid_before, grid_after]'''


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
      'actions_count': actions_count,
      'dir_before': dir_before,
      'dir_after': dir_after,
    }})
  else:
    passed = False
  total_reward += 1.0 if passed else 0.0
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
  ans =[0]*10  #tracking what testcases pass or failed - 0 for failed 1 for passed
  i = 0 
  for grid_spec_path in {grid_spec_paths_var}:
{env_setup_in_loop}
    env.reset()
    arg_values = grid_spec.get("arg_values", {{}}) if isinstance(grid_spec, dict) else {{}}
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
