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
      state_before['pos'] = tuple(state.pos) if hasattr(state.pos, '__iter__') and not isinstance(state.pos, str) else state.pos
    if hasattr(state, 'inventory'):
      state_before['inventory'] = state.inventory.copy() if hasattr(state.inventory, 'copy') else state.inventory
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
      state_after['pos'] = tuple(state.pos) if hasattr(state.pos, '__iter__') and not isinstance(state.pos, str) else state.pos
    if hasattr(state, 'inventory'):
      state_after['inventory'] = state.inventory.copy() if hasattr(state.inventory, 'copy') else state.inventory
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

  # Compute additional reward based on whether function worked correctly (LLM-generated logic)
  new_reward = 0.0
  try:
    # If grid spec provides a pass_check, use it for reward.
    pass_check = None
    if hasattr(env, 'scenario') and hasattr(env.scenario, 'spec'):
      spec = env.scenario.spec
      if isinstance(spec, dict):
        pass_check = spec.get('pass_check')
    if isinstance(pass_check, str) and pass_check.strip():
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
      new_reward = 5.0 if passed else 0.0
      total_reward += new_reward
      return [total_reward, actions_count, grid_before, grid_after]
    # <--- YOUR new_reward LOGIC GOES HERE --->
    # Generate code that updates new_reward based on whether the function worked correctly
    # Use pos_before/after, inventory_before/after, dir_before/after to check if function achieved its intended effect
    pass
  except Exception as e:
    pass
  total_reward += new_reward

  # Return [total_reward, actions_count, grid_before, grid_after]
  return [total_reward, actions_count, grid_before, grid_after]'''


def craft_solve_template_basic(func_name: str, func_params: str, func_call_args: str) -> str:
    """Craft solve() template without reward-logic block."""
    safe_name = _safe_name(func_name)
    return f'''def solve({func_params}, visualise=False):
  """Runs the environment with a {safe_name} function that returns list of actions to take and returns total reward."""
  # Capture grid state before function execution (with agent position)
  grid_before = None
  grid_before_cells = None
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
  
  # Execute function to get actions using a deepcopy
  import copy
  env_for_func = copy.deepcopy(env)
  actions_to_take = {safe_name}({func_call_args.replace("env", "env_for_func", 1)})
  # Ensure actions_to_take is a list (handle None case)
  if actions_to_take is None:
    actions_to_take = []
  total_reward = 0.0
  actions_count = len(actions_to_take)

  # Execute actions
  for t in range(len(actions_to_take)):
    action = actions_to_take[t]
    reward, done, observations = env.step(action)
    total_reward += reward
    if done:
      break

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

  # Optional: use pass_check from grid spec to compute reward
  try:
    pass_check = None
    if hasattr(env, 'scenario') and hasattr(env.scenario, 'spec'):
      spec = env.scenario.spec
      if isinstance(spec, dict):
        pass_check = spec.get('pass_check')
    if isinstance(pass_check, str) and pass_check.strip():
      try:
        passed = bool(eval(pass_check, {{}}, {{
            'env': env,
            'grid_before': grid_before,
            'grid_after': grid_after,
            'grid_before_cells': grid_before_cells,
            'grid_after_cells': grid_after_cells,
            'actions_count': actions_count,
        }}))
      except Exception:
        passed = False
      total_reward += 5.0 if passed else 0.0
  except Exception:
    pass

  # Return [total_reward, actions_count, grid_before, grid_after]
  return [total_reward, actions_count, grid_before, grid_after]'''


def craft_evaluate_template(
    display_name: str,
    env_setup: str,
    args_definitions: str,
    func_call_args: str,
    ) -> str:
    """Craft evaluate() template for running solve() in a sample environment."""
    return f'''@funsearch.run
def evaluate():
  """Evaluates {display_name} behavior in a sample environment."""
  visualise = False
  {env_setup}
  env.reset()
{args_definitions}  result = solve({func_call_args}, visualise=visualise)
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
  import os
  import json
  recipes_path = "{recipes_path}"
  hints_path = "{hints_path}"
  grid_spec_path = r"{grid_spec_path}"
  task_name = "{task_name}"
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
  """
