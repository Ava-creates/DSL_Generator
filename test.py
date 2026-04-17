import copy
import json
import importlib.util
import os
import argparse
import re
from typing import Callable, Dict, List, Optional

from craft import env_factory


def _json_default(obj):
  """Serialize numpy scalars and other numeric-like objects to native Python types."""
  if hasattr(obj, "item"):
    try:
      return obj.item()
    except Exception:
      pass
  raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


FINAL_FUNCTIONS_DIR = (
  "experiments/baseline_20260405_155319_4617430/final_functions"
)

# Task -> (filename, function_name)
TASK_TO_FUNCTION: Dict[str, tuple[str, str]] = {
  "get[gem]": ("get_gem_func0.py", "get_gem"),
  "get[gold]": ("get_gold_func0.py", "get_gold"),
  "get[grass]": ("get_grass_func0.py", "get_grass"),
  "get[iron]": ("get_iron_func0.py", "get_iron"),
  "get[wood]": ("get_wood_func0.py", "get_wood"),
  "make[axe]": ("make_axe_func0.py", "make_axe"),
  "make[bed]": ("make_bed_func0.py", "make_bed"),
  "make[bridge]": ("make_bridge_func0.py", "make_bridge"),
  "make[bundle]": ("make_bundle_func0.py", "make_bundle"),
  "make[cloth]": ("make_cloth_func0.py", "make_cloth"),
  "make[flag]": ("make_flag_func0.py", "make_flag"),
  "make[goldarrow]": ("make_golden_arrow_func0.py", "make_golden_arrow"),
  "make[ladder]": ("make_ladder_func0.py", "make_ladder"),
  "make[plank]": ("make_plank_func0.py", "make_plank"),
  "make[rope]": ("make_rope_func0.py", "make_rope"),
  "make[shears]": ("make_shears_func0.py", "make_shears"),
  "make[stick]": ("make_stick_func0.py", "make_stick"),
}

FINAL_TASKS: List[str] = list(TASK_TO_FUNCTION.keys())


def _load_function(module_path: str, function_name: str) -> Callable:
  spec = importlib.util.spec_from_file_location(function_name, module_path)
  if spec is None or spec.loader is None:
    raise ImportError(f"Could not import module from {module_path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  fn = getattr(module, function_name, None)
  if fn is None or not callable(fn):
    raise AttributeError(f"Function {function_name} not found in {module_path}")
  return fn


def _resolve_round_filename(filename: str, func_evolution_round: int) -> str:
  """Swap `_funcN.py` suffix to the requested function round."""
  return re.sub(r"_func\d+\.py$", f"_func{int(func_evolution_round)}.py", filename)


def _legacy_dsl_filename(filename: str) -> str:
  """Map dsl-free baseline filename to legacy `_dsl0_funcN.py` format."""
  return re.sub(r"_func(\d+)\.py$", r"_dsl0_func\1.py", filename)


def load_final_functions(
  final_functions_dir: str = FINAL_FUNCTIONS_DIR,
  func_evolution_round: int = 0,
  tasks: Optional[List[str]] = None,
  strict: bool = False,
) -> Dict[str, Callable]:
  loaded: Dict[str, Callable] = {}
  selected_tasks = tasks or list(TASK_TO_FUNCTION.keys())
  for task in selected_tasks:
    if task not in TASK_TO_FUNCTION:
      if strict:
        raise KeyError(f"No final function mapping for task: {task}")
      print(f"[warn] Skipping unknown task mapping: {task}")
      continue

    filename, function_name = TASK_TO_FUNCTION[task]
    resolved_filename = _resolve_round_filename(filename, func_evolution_round)
    module_path = os.path.join(final_functions_dir, resolved_filename)
    if not os.path.exists(module_path):
      legacy_filename = _legacy_dsl_filename(resolved_filename)
      legacy_path = os.path.join(final_functions_dir, legacy_filename)
      if os.path.exists(legacy_path):
        module_path = legacy_path
      else:
        if strict:
          raise FileNotFoundError(
            f"Missing final function file: {module_path} (and legacy fallback {legacy_path})"
          )
        print(
          f"[warn] Missing final function file for task {task}: {module_path} "
          f"(legacy fallback: {legacy_path})"
        )
        continue
    try:
      loaded[task] = _load_function(module_path, function_name)
    except Exception as exc:
      if strict:
        raise
      print(f"[warn] Skipping task {task}; failed to load {module_path}: {exc}")
  return loaded


def solve(env, policy_fn: Callable, visualise: bool = False) -> List[object]:
  """Run one env episode by calling policy_fn(env_copy) to get action list."""
  _ = visualise
  env_for_policy = copy.deepcopy(env)
  actions = policy_fn(env_for_policy)

  if actions is None:
    raise RuntimeError("Final function returned None; expected list[int].")

  total_reward = 0.0
  actions_count = 0
  done = False

  for action in actions:
    reward, done, _ = env.step(action)
    total_reward += reward
    actions_count += 1
    if done:
      break

  return [total_reward, actions_count, done]


def evaluate_final(
  tasks: Optional[List[str]] = None,
  final_functions_dir: str = FINAL_FUNCTIONS_DIR,
  func_evolution_round: int = 0,
  recipes_path: str = "craft/resources/recipes.yaml",
  hints_path: str = "craft/resources/hints.yaml",
  max_steps: int = 400,
  seed: int = 13,
  visualise: bool = False,
  strict_tasks: bool = False,
):
  """
  Evaluate final functions by sampling one env per task.

  No testcase-grid files are used.
  """
  requested_tasks = tasks or FINAL_TASKS
  function_map = load_final_functions(
    final_functions_dir,
    func_evolution_round=func_evolution_round,
    tasks=requested_tasks,
    strict=strict_tasks,
  )
  task_list = [task for task in requested_tasks if task in function_map]
  if not task_list:
    raise RuntimeError(
      f"No runnable tasks found in {final_functions_dir}. "
      "Use --strict_tasks for hard failure on missing files."
    )

  total_reward = 0.0
  total_actions = 0
  solved_tasks = 0
  per_task = []
  s = []

  for task in task_list:
    if task not in function_map:
      raise KeyError(f"No final function mapping for task: {task}")

    sampler = env_factory.EnvironmentFactory(
      recipes_path,
      hints_path,
      7,
      max_steps=max_steps,
      seed=seed,
      reuse_environments=False,
      visualise=visualise,
    )

    env = sampler.sample_environment(task_name=task)
    env.reset()

    reward, actions_count, done = solve(env, function_map[task], visualise=visualise)

    total_reward += reward
    total_actions += actions_count
    solved_tasks += int(bool(done))
    if done:
        s.append(task)

    per_task.append(
      {
        "task": task,
        "reward": reward,
        "actions": actions_count,
        "done": bool(done),
      }
    )
  print(solved_tasks)
  print(s)
  print(total_reward)
  return {
    "tasks": task_list,
    "num_tasks": len(task_list),
    "solved_tasks": solved_tasks,
    "total_reward": total_reward,
    "total_actions": total_actions,
    "per_task": per_task,
  }


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Evaluate baseline final functions")
  parser.add_argument(
    "--final_functions_dir",
    type=str,
    default=FINAL_FUNCTIONS_DIR,
    help="Directory containing final function files",
  )
  parser.add_argument(
    "--func_evolution_round",
    type=int,
    default=0,
    help="Function round suffix to load (e.g., 1 loads *_func1.py)",
  )
  parser.add_argument(
    "--output_json",
    type=str,
    default=None,
    help="Optional path to write evaluation JSON",
  )
  parser.add_argument(
    "--seed",
    type=int,
    default=0,
    help="Random seed for environment generation",
  )
  parser.add_argument(
    "--require_all_solved",
    action="store_true",
    help="Exit non-zero unless all mapped tasks are solved",
  )
  parser.add_argument(
    "--tasks",
    type=str,
    default=None,
    help="Comma-separated tasks to evaluate, e.g. get[gem],make[axe]",
  )
  parser.add_argument(
    "--strict_tasks",
    action="store_true",
    help="Fail if any requested task mapping or file is missing",
  )
  return parser.parse_args()


if __name__ == "__main__":
  args = _parse_args()
  selected_tasks = None
  if args.tasks:
    selected_tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

  result = evaluate_final(
    tasks=selected_tasks,
    final_functions_dir=args.final_functions_dir,
    func_evolution_round=args.func_evolution_round,
    seed=args.seed,
    strict_tasks=args.strict_tasks,
  )
  result["all_solved"] = bool(result.get("solved_tasks", 0) == result.get("num_tasks", 0))

  if args.output_json:
    out_dir = os.path.dirname(args.output_json)
    if out_dir:
      os.makedirs(out_dir, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
      json.dump(result, f, indent=2, default=_json_default)

  print(result)
  if args.require_all_solved and not result["all_solved"]:
    raise SystemExit(1)
