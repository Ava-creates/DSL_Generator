"""Crafter-specific FunSearch templates.

These mirror the shape of :mod:`domains.craft.templates` but drop grid/cookbook
concepts and snapshot state from Crafter's ``info`` dict. Test cases are
``{task_name, scenario, max_steps, pass_check, init_actions?}`` JSON specs;
the scenario deterministically overrides the post-reset world.
"""

from __future__ import annotations


def _safe_name(func_name: str) -> str:
    return func_name.lower().replace("-", "_")


def crafter_solve_template_basic(
    func_name: str, func_params: str, func_call_args: str
) -> str:
    """Crafter solve() template. Snapshots inventory/achievements via info."""
    safe_name = _safe_name(func_name)
    return f'''def solve({func_params}, visualise=False):
  """Runs the Crafter env with a {safe_name} function that returns list of action ints and returns total reward."""
  import copy
  from domains.crafter.observations import local_grid_cells

  def _info_snapshot(e, radius=4):
    info = getattr(e, "info", {{}}) or {{}}
    inv = dict(info.get("inventory", {{}}))
    ach = dict(info.get("achievements", {{}}))
    pos = info.get("player_pos")
    if pos is not None:
      try:
        pos = list(pos)
      except Exception:
        pos = list(tuple(pos))
    grid = local_grid_cells(e, radius=radius)
    return inv, ach, pos, grid

  inventory_before, achievements_before, pos_before, grid_before_info = _info_snapshot(env)
  grid_before_cells = grid_before_info.get("cells")
  facing_before = grid_before_info.get("facing")

  env_for_func = copy.deepcopy(env)
  actions_to_take = {safe_name}({func_call_args.replace("env", "env_for_func", 1)})
  if actions_to_take is None:
    raise RuntimeError(
      "{safe_name} returned None; terminal functions must return a list of action codes "
      "(use [] when no steps are required, never implicit None)."
    )

  total_reward = 0.0
  actions_count = 0
  for action in actions_to_take:
    reward, done, observations = env.step(action)
    total_reward += reward
    actions_count += 1
    if done:
      break

  inventory_after, achievements_after, pos_after, grid_after_info = _info_snapshot(env)
  grid_after_cells = grid_after_info.get("cells")
  facing_after = grid_after_info.get("facing")

  pass_check = None
  spec = getattr(env, "scenario_spec", None)
  if spec is None:
    spec = getattr(env, "_test_case_spec", None)
  if isinstance(spec, dict):
    pass_check = spec.get("pass_check")
  if isinstance(pass_check, str):
    pass_check = pass_check.replace(\'\\\\"\', \'"\').replace(\'\\"\', \'"\')

  class _CountDict(dict):
    def __getitem__(self, key):
      return self.get(key, 0)

  inventory_before = _CountDict(inventory_before)
  inventory_after = _CountDict(inventory_after)
  achievements_before = _CountDict(achievements_before)
  achievements_after = _CountDict(achievements_after)

  context = {{
    "inventory_before": inventory_before,
    "inventory_after": inventory_after,
    "achievements_before": achievements_before,
    "achievements_after": achievements_after,
    "pos_before": pos_before,
    "pos_after": pos_after,
    "grid_before_cells": grid_before_cells,
    "grid_after_cells": grid_after_cells,
    "facing_before": facing_before,
    "facing_after": facing_after,
    "total_reward": total_reward,
    "actions_count": actions_count,
  }}

  passed = eval(pass_check, context, context) if pass_check else False
  test_type = spec.get("test_type", "positive") if isinstance(spec, dict) else "positive"

  total_reward = 0.0
  if passed:
    total_reward += 100.0 if test_type == "positive" else 1.0

  return [total_reward, actions_count, None, None]'''


def crafter_solve_template_task_env_basic(
    func_name: str, func_params: str, func_call_args: str
) -> str:
    """Minimal Crafter solve() template for task-env baseline runs."""
    safe_name = _safe_name(func_name)
    return f'''def solve({func_params}, visualise=False):
  """Runs the Crafter env with a {safe_name} function that returns list of action ints and returns total reward."""
  import copy

  env_for_func = copy.deepcopy(env)
  actions_to_take = {safe_name}({func_call_args.replace("env", "env_for_func", 1)})
  if actions_to_take is None:
    raise RuntimeError(
      "{safe_name} returned None; terminal functions must return a list of action codes "
      "(use [] when no steps are required, never implicit None)."
    )

  total_reward = 0.0
  actions_count = 0
  for action in actions_to_take:
    reward, done, observations = env.step(action)
    total_reward += reward
    actions_count += 1
    if done:
      break

  return [total_reward, actions_count, "", ""]'''


def crafter_evaluate_template(
    display_name: str,
    env_setup: str,
    args_definitions: str,
    func_call_args: str,
    grid_spec_paths_var: str | None = None,
) -> str:
    """Crafter evaluate() template mirroring craft_evaluate_template."""

    def _indent(block: str, spaces: int) -> str:
        prefix = " " * spaces
        return "\n".join(prefix + line if line.strip() else line for line in block.splitlines())

    if grid_spec_paths_var:
        env_setup_in_loop = _indent(env_setup, 2)
        args_def_in_loop = _indent(args_definitions, 2) if args_definitions else ""
        return f'''@funsearch.run
def evaluate():
  """Evaluates {display_name} behavior across multiple Crafter test cases."""
  visualise = False
  total_reward = 0.0
  total_actions = 0
  _grid_spec_paths = {grid_spec_paths_var}
  ans = [0]*len(_grid_spec_paths)
  i = 0
  for grid_spec_path in _grid_spec_paths:
{env_setup_in_loop}
{args_def_in_loop}
    result = solve({func_call_args}, visualise=visualise)
    if isinstance(result, list) and len(result) >= 2:
      total_reward += result[0] if result[0] is not None else 0.0
      if result[0] is not None:
        ans[i] = 1 if result[0] > 0 else 0
      i += 1
      total_actions += result[1] if result[1] is not None else 0
  return [total_reward, total_actions, ans, None]
'''

    return f'''@funsearch.run
def evaluate():
  """Evaluates {display_name} behavior in a sample Crafter environment."""
  visualise = False
  {env_setup}
  arg_values = grid_spec.get("arg_values", {{}}) if isinstance(grid_spec, dict) else {{}}
{args_definitions}
  result = solve({func_call_args}, visualise=visualise)
  return result
'''


def crafter_env_setup(task_name: str, grid_spec_path: str) -> str:
    """Crafter env-setup block. Loads the scenario-based spec and builds env."""
    return """
  from domains.crafter.env_wrapper import CrafterEnvWrapper
  grid_spec = None
  with open(grid_spec_path, "r", encoding="utf-8") as f:
    grid_spec = json.load(f)
  task_name = grid_spec.get("task_name")
  max_steps = int(grid_spec.get("max_steps", 400))
  scenario = grid_spec.get("scenario")
  if not isinstance(scenario, dict) or not scenario:
    raise ValueError(f"Crafter test case must contain a non-empty 'scenario': {grid_spec_path}")
  env = CrafterEnvWrapper(task=task_name, max_steps=max_steps, seed=0)
  env.reset(scenario=scenario)
  env._test_case_spec = grid_spec
  for _act in grid_spec.get("init_actions", []) or []:
    env.step(_act)
  """


def crafter_baseline_evaluate_template(
    display_name: str,
    func_call_args: str,
    task_name: str,
    max_steps: int = 400,
) -> str:
    """Crafter evaluate() template for baseline (seed-sweep) runs."""
    return f'''@funsearch.run
def evaluate():
  """Evaluates {display_name} behavior on sampled Crafter seeds."""
  from domains.crafter.env_wrapper import CrafterEnvWrapper
  visualise = False
  seeds = range(2, 20, 2)
  total_reward = 0
  total_actions = 0
  for seed in seeds:
    env = CrafterEnvWrapper(task="{task_name}", max_steps={int(max_steps)}, seed=seed)
    env.reset()
    r, a, _, _ = solve({func_call_args}, visualise=visualise)
    total_reward += r
    total_actions += a
  return [total_reward, total_actions, "", ""]
'''
