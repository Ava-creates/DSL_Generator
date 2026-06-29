import argparse
import json
import os
import sys
import time
import collections

import numpy as np
import pandas as pd
from craft import craft, env, env_factory

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
  sys.path.insert(0, PROJECT_ROOT)

from src.pipeline.cfg_evaluator import CFGEvaluator
from src.utils.test import grid_to_markdown

def solve(env, item, workshop, visualise=False):
  """Runs the environment with a craft function that returns list of actions to take and returns total reward."""
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
  print( "cookbook.index", env.world.cookbook.index)
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
  actions_to_take = craft(env_for_func, item, workshop)
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
#   print("pass_check", pass_check)
  if isinstance(pass_check, str):
    pass_check = pass_check.replace('\\"', '"')

  inventory_before = _inventory_list_to_dict(_pairs_to_dict(inventory_before))
  inventory_after = _inventory_list_to_dict(_pairs_to_dict(inventory_after))
  if isinstance(inventory_before, list):
    inventory_before = _InvList(inventory_before)
  if isinstance(inventory_after, list):
    inventory_after = _InvList(inventory_after)
  print(inventory_before)
  print(inventory_after)  
  print(  pos_after)
  total_reward = 0.0
  print("grid_before_cells", grid_before_cells)
  if pass_check:
    passed = eval(pass_check, {}, {
      'pos_before': pos_before,
      'pos_after': pos_after,
      'inventory_before': inventory_before,
      'inventory_after': inventory_after,
      'grid_before_cells': grid_before_cells,
      'grid_after_cells': grid_after_cells,
      'actions_count': actions_count,
      'dir_before': dir_before,
      'dir_after': dir_after,
    })
  else:
    passed = False
  total_reward += 1.0 if passed else 0.0
  grid_before = None
  grid_after = None

  return [total_reward, actions_count, grid_before, dir_after]

def solve_old(env, item, workshop, visualise=False):
  """Runs the environment with a craft function that returns list of actions to take and returns total reward."""
  def _to_list(x):
    try:
      if hasattr(x, "tolist"):
        return x.tolist()
      return list(x)
    except Exception:
      return x
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
        grid_before = f"Grid shape: {env._current_state.grid.shape if hasattr(env._current_state.grid, 'shape') else 'N/A'}\nAgent position: {agent_pos}"
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
  
  # Capture state before (for pass_check)
  pos_before = None
  inventory_before = None
  dir_before = None

  if hasattr(env, '_current_state'):
      state = env._current_state
      if hasattr(state, 'pos'):
        pos_before = state.pos
        if hasattr(pos_before, '__iter__') and not isinstance(pos_before, str):
          try:
            pos_before = _to_list(pos_before)
          except Exception:
            pass
      if hasattr(state, 'inventory'):
        inv = state.inventory.copy() if hasattr(state.inventory, 'copy') else state.inventory
        inventory_before = _to_list(inv)
      if hasattr(state, 'dir'):
        dir_before = state.dir
  
  # Execute function to get actions using a deepcopy
  import copy
  env_for_func = copy.deepcopy(env)
  actions_to_take = craft(env_for_func, item, workshop)
  # Ensure actions_to_take is a list (handle None case)
  if actions_to_take is None:
    actions_to_take = []
  actions_to_take = [3, 3, 0, 4]
  total_reward = 0.0
  actions_count = 4

  # Execute actions
  for t in range(len(actions_to_take)):
    action = actions_to_take[t]
    reward, done, observations = env.step(action)
    total_reward += reward
    actions_count += 1
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
        grid_after = f"Grid shape: {env._current_state.grid.shape if hasattr(env._current_state.grid, 'shape') else 'N/A'}\nAgent position: {agent_pos}"
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

  # Capture state after (for pass_check)
  pos_after = None
  inventory_after = None
  dir_after = None
  
  if hasattr(env, '_current_state'):
      state = env._current_state
      if hasattr(state, 'pos'):
        pos_after = state.pos
        if hasattr(pos_after, '__iter__') and not isinstance(pos_after, str):
          try:
            pos_after = _to_list(pos_after)
          except Exception:
            pass
      if hasattr(state, 'inventory'):
        inv = state.inventory.copy() if hasattr(state.inventory, 'copy') else state.inventory
        inventory_after = _to_list(inv)
      if hasattr(state, 'dir'):
        dir_after = state.dir

 
  # Debug: print types of pass_check inputs
  def _type_name(x):
    try:
      return type(x).__name__
    except Exception:
      return "unknown"
  try:
    print("[check.py types] pos_before:", _type_name(pos_before), "pos_after:", _type_name(pos_after))
    print("[check.py types] inventory_before:", _type_name(inventory_before), "inventory_after:", _type_name(inventory_after))
    print("[check.py types] dir_before:", _type_name(dir_before), "dir_after:", _type_name(dir_after))
    print("[check.py types] grid_before_cells:", _type_name(grid_before_cells), "grid_after_cells:", _type_name(grid_after_cells))
    print("[check.py types] actions_count:", _type_name(actions_count))
  except Exception:
    pass

  pass_check = None
  if hasattr(env, 'scenario') and hasattr(env.scenario, 'spec'):
    spec = env.scenario.spec
    if isinstance(spec, dict):
      pass_check = spec.get('pass_check')
  passed = False
  if isinstance(pass_check, str) and pass_check.strip():
        class _InvList(list):
          def get(self, key, default=0):
            return default
        if isinstance(inventory_before, list) and not hasattr(inventory_before, "get"):
          inventory_before = _InvList(inventory_before)
        if isinstance(inventory_after, list) and not hasattr(inventory_after, "get"):
          inventory_after = _InvList(inventory_after)
        passed = bool(eval(pass_check, {}, {
            'env': env,
            'pos_before': pos_before,
            'pos_after': pos_after,
            'inventory_before': inventory_before,
            'inventory_after': inventory_after,
            'grid_before_cells': grid_before_cells,
            'grid_after_cells': grid_after_cells,
            'actions_count': actions_count,
            'dir_before': dir_before,
            'dir_after': dir_after,
        }))
  total_reward += 1.0 if passed else 0.0

  # Return [total_reward, actions_count, grid_before, grid_after]
  return [total_reward, actions_count, grid_before, grid_after]

def evaluate():
  """Evaluates CRAFT behavior across multiple test cases."""
  visualise = False
  total_reward = 0.0
  total_actions = 0
  grid_before = None
  grid_after = None
  for grid_spec_path in ['experiments/experiment_success/grids/craft_dsl0_case0.json', 'experiments/experiment_success/grids/craft_dsl0_case1.json', 'experiments/experiment_success/grids/craft_dsl0_case2.json', 'experiments/experiment_success/grids/craft_dsl0_case3.json', 'experiments/experiment_success/grids/craft_dsl0_case4.json', 'experiments/experiment_success/grids/craft_dsl0_case5.json', 'experiments/experiment_success/grids/craft_dsl0_case6.json']:

    recipes_path = "craft/resources/recipes.yaml"
    hints_path = "craft/resources/hints.yaml"
    # task_name = "make[plank]"
    with open(grid_spec_path, "r", encoding="utf-8") as f:
          grid_spec = json.load(f)
    # print(grid_spec)
    task_name = grid_spec.get("task_name")

    custom_grid_path = grid_spec_path if os.path.exists(grid_spec_path) else None
    env_sampler = env_factory.EnvironmentFactory(
        recipes_path, hints_path, 7, max_steps=300, reuse_environments=False,
              visualise=visualise, custom_grid_path=custom_grid_path)
    env = env_sampler.sample_environment(task_name=task_name)
    # Attach grid_spec to scenario so pass_check is available in solve()

    env.scenario.spec = grid_spec

  
    env.reset()
    arg_values = grid_spec.get("arg_values", {}) if isinstance(grid_spec, dict) else {}
    item = arg_values.get("item", "wood")  # Item 'wood' is present on the grid
    workshop = arg_values.get("workshop", "WORKSHOP0")  # Argument value from CFG  
    print(item, workshop)  
    result = solve(env, item, workshop, visualise=visualise)
    print(result)
    if isinstance(result, list) and len(result) >= 2:
      total_reward += result[0] if result[0] is not None else 0.0
      total_actions += result[1] if result[1] is not None else 0

  # Return as list: [total_reward, actions_count, grid_before, grid_after]
  return [total_reward, total_actions, grid_before, grid_after]

def craft(env, item, workshop):
  # Retrieve current state, position, and grid
  state = env._current_state
  pos = np.array(state.pos)          # (x, y) coordinates
  grid = state.grid

  # Locate the nearest workshop cell (any workshop type)
  target_loc = None
  min_dist = None
  for wk_idx in env.world.workshop_indices:
      locations = np.argwhere(grid[:, :, wk_idx] > 0)
      for loc in locations:
          dist = np.abs(loc - pos).sum()
          if (min_dist is None) or (dist < min_dist):
              min_dist = dist
              target_loc = loc

  # If no workshop is found, return an empty plan
  if target_loc is None:
      return []

  # Map direction strings to action codes from the environment
  act_spec = env.action_specs()
  move_actions = {
      "UP": act_spec.get("UP", 1),
      "DOWN": act_spec.get("DOWN", 0),
      "LEFT": act_spec.get("LEFT", 2),
      "RIGHT": act_spec.get("RIGHT", 3),
  }
  use_action = act_spec.get("USE", 4)

  # Build a Manhattan path to the workshop cell
  actions = []
  diff = target_loc - pos  # [dx, dy]

  # Move vertically first (UP/DOWN)
  if diff[1] > 0:
      actions += [move_actions["UP"]] * int(diff[1])
  elif diff[1] < 0:
      actions += [move_actions["DOWN"]] * int(-diff[1])

  # Then move horizontally (RIGHT/LEFT)
  if diff[0] > 0:
      actions += [move_actions["RIGHT"]] * int(diff[0])
  elif diff[0] < 0:
      actions += [move_actions["LEFT"]] * int(-diff[0])

  # Use the workshop (craft/interact)
  actions.append(use_action)

  return actions


def _grid_cells(env):
  g = env._current_state.grid
  cookbook = env.world.cookbook
  rows = []
  for y in range(g.shape[1]):
    row = []
    for x in range(g.shape[0]):
      cell = g[x, y]
      indices = [i for i, v in enumerate(cell) if v]
      if indices:
        row.append(str(cookbook.index.get(indices[0])).strip().lower())
      else:
        row.append("")
    rows.append(row)
  return rows


def _find_cells(grid_cells, names):
  hits = []
  name_set = {n.lower() for n in names}
  for y, row in enumerate(grid_cells):
    for x, cell in enumerate(row):
      if cell in name_set:
        hits.append((x, y, cell))
  return hits


def _format_inventory(env):
  state = env._current_state
  index = env.world.cookbook.index
  items = []
  for idx, count in enumerate(state.inventory):
    if not count:
      continue
    name = index.get(idx)
    items.append(f"{name}={int(count)}")
  return items


def _load_dsl_evaluator(experiment_dir, cfg_path):
  with open(cfg_path, "r", encoding="utf-8") as f:
    cfg_text = json.load(f)["cfg"]
  return CFGEvaluator(
      cfg=cfg_text,
      final_functions_dir=os.path.join(experiment_dir, "final_functions"),
  )


def _print_grid_state(env, title):
  print(f"\n--- {title} ---")
  print(f"pos={tuple(env._current_state.pos)} dir={env._current_state.dir} inventory={_format_inventory(env)}")
  print(grid_to_markdown(
      env._current_state.grid,
      env.world.cookbook,
      tuple(env._current_state.pos),
      include_indices=True,
  ))


def run_dsl_program_trace(label, program, task, seed, experiment_dir, cfg_path, max_steps=400):
  import copy

  print("\n" + "=" * 80)
  print(f"CASE: {label}")
  print(f"task={task} seed={seed}")
  print(f"program: {program}")
  print("=" * 80)

  evaluator = _load_dsl_evaluator(experiment_dir, cfg_path)
  dsl = evaluator.dsl_evaluator
  sampler = env_factory.EnvironmentFactory(
      "craft/resources/recipes.yaml",
      "craft/resources/hints.yaml",
      7,
      max_steps=max_steps,
      seed=int(seed),
      reuse_environments=False,
      visualise=False,
  )
  env = sampler.sample_environment(task_name=task)
  env.reset()

  _print_grid_state(env, "Grid (start)")

  tokens = dsl.tokenize_program(program)
  print(f"\nTokens ({len(tokens)}): {tokens}")

  total_reward = 0.0
  step_count = 0
  inventory_trace = []
  last_inventory = _format_inventory(env)

  for idx, token in enumerate(tokens, start=1):
    print("\n" + "-" * 80)
    print(f"TOKEN {idx}/{len(tokens)}: {token}")
    func_call = dsl.extract_function_call(token)
    if not func_call:
      print("  (not a function call, skipping)")
      continue

    func_name, args = func_call
    safe_name = dsl._sanitize_function_name(func_name)
    if safe_name not in dsl.function_implementations:
      print(f"  MISSING IMPLEMENTATION: {func_name} (sanitized: {safe_name})")
      continue

    func = dsl.function_implementations[safe_name]
    normalized_args = [a.lower() if isinstance(a, str) else a for a in args]
    env_for_func = copy.deepcopy(env)
    if normalized_args:
      actions = func(env_for_func, *normalized_args)
    else:
      actions = func(env_for_func)

    actions = actions if isinstance(actions, list) else ([actions] if actions is not None else [])
    print(f"  planned_actions={len(actions)}")
    if actions:
      print(f"  first_actions={actions[:20]}{'...' if len(actions) > 20 else ''}")

    token_reward = 0.0
    for action in actions:
      if step_count >= max_steps:
        print("  hit max_steps, stopping")
        break
      reward, done, _ = env.step(action)
      token_reward += float(reward) if reward is not None else 0.0
      total_reward += float(reward) if reward is not None else 0.0
      step_count += 1
      if done:
        print(f"  episode done after action {action}")
        break

    inv_items = _format_inventory(env)
    changed = inv_items != last_inventory
    if changed:
      inventory_trace.append({"token": token, "inventory": inv_items})
      last_inventory = inv_items

    print(f"  token_reward={token_reward} total_reward={total_reward} steps_so_far={step_count}")
    print(f"  inventory={inv_items} changed={changed}")
    _print_grid_state(env, f"Grid after {token}")

    if step_count >= max_steps:
      break

  goal_name, goal_arg = env.task.goal
  goal_satisfied = env._current_state.satisfies(goal_name, goal_arg)
  result = {
      "success": total_reward >= 10,
      "total_reward": total_reward,
      "steps": step_count,
      "inventory_trace": inventory_trace,
      "goal_satisfied": goal_satisfied,
  }

  print("\n" + "=" * 80)
  print("SUMMARY")
  print("=" * 80)
  print(json.dumps(result, indent=2))
  return result


def run_gem_program_check(label, program, seed, experiment_dir, cfg_path, max_steps=400):
  return run_dsl_program_trace(
      label=label,
      program=program,
      task="get[gem]",
      seed=seed,
      experiment_dir=experiment_dir,
      cfg_path=cfg_path,
      max_steps=max_steps,
  )


def run_gem_compare(experiment_dir, cfg_path, seed):
  success_program = (
      "MOVE_TO_PRIMITIVE(WOOD); GATHER(WOOD); MOVE_TO_PRIMITIVE(IRON); GATHER(IRON); "
      "MOVE_TO_WORKSHOP_FOR(AXE); CRAFT(AXE); MOVE_TO(CELL_A); APPLY_TOOL(AXE,STONE); "
      "MOVE_TO(CELL_B); FACE_TOWARD(GEM); GATHER(GEM)"
  )
  fail_program = "MOVE_TO_PRIMITIVE(GEM); FACE_TOWARD(GEM); GATHER(GEM)"

  ok = run_gem_program_check("synthesis_success", success_program, seed, experiment_dir, cfg_path)
  bad = run_gem_program_check("synthesis_fail_direct_gather", fail_program, seed, experiment_dir, cfg_path)

  print("\n" + "=" * 80)
  print("SUMMARY")
  print("=" * 80)
  print(f"seed={seed}")
  print(f"synthesis_success: evaluator_success={ok.get('success')} reward={ok.get('total_reward')} steps={ok.get('steps')}")
  print(f"synthesis_fail_direct_gather: evaluator_success={bad.get('success')} reward={bad.get('total_reward')} steps={bad.get('steps')}")
  print("Note: gem is always behind stone on get[gem] grids (make_cave=True).")
  print("Evaluator marks success when total_reward>=10 (goal satisfied at least once), not only when gem remains in inventory.")


def main():
  parser = argparse.ArgumentParser(description="CRAFT terminal/grid checker")
  parser.add_argument(
      "--mode",
      choices=["craft_cases", "gem_compare", "dsl_trace"],
      default="dsl_trace",
      help="dsl_trace: per-token grids; gem_compare: get[gem] success vs fail; craft_cases: legacy grid specs",
  )
  parser.add_argument(
      "--experiment_dir",
      default="experiments_archive/experiment_20260409_122603_25492",
      help="Archive experiment with dsl2 cfg + final_functions",
  )
  parser.add_argument(
      "--cfg_path",
      default=None,
      help="CFG json path (default: <experiment_dir>/cfg/cfg_output_2.json)",
  )
  parser.add_argument("--task", default="get[gem]", help="Craft task name, e.g. make[clothbundle]")
  parser.add_argument("--seed", type=int, default=0, help="Environment seed")
  parser.add_argument("--program", default=None, help="DSL program string to run")
  parser.add_argument("--label", default="custom", help="Label for this run")
  parser.add_argument("--max_steps", type=int, default=400, help="Max environment steps")
  args = parser.parse_args()

  experiment_dir = os.path.join(PROJECT_ROOT, args.experiment_dir)
  cfg_path = args.cfg_path or os.path.join(experiment_dir, "cfg", "cfg_output_2.json")

  if args.mode == "craft_cases":
    print(evaluate())
    return

  if args.mode == "dsl_trace":
    if not args.program:
      raise SystemExit("--program is required for --mode dsl_trace")
    run_dsl_program_trace(
        args.label,
        args.program,
        args.task,
        args.seed,
        experiment_dir,
        cfg_path,
        max_steps=args.max_steps,
    )
    return

  if args.program:
    run_gem_program_check(args.label, args.program, args.seed, experiment_dir, cfg_path)
    return

  run_gem_compare(experiment_dir, cfg_path, args.seed)


if __name__ == "__main__":
  main()