#!/usr/bin/env python3

import argparse
import json
import os
import sys
from typing import List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from craft import env_factory
from src.utils.test import grid_to_markdown

from src.pipeline.dsl_evaluator import DSLEvaluator, load_function_implementations


def _load_cfg(experiment_dir: str, cfg_version: int) -> str:
    cfg_path = os.path.join(
        experiment_dir,
        "cfg",
        f"cfg_output_{cfg_version}.json",
    )
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cfg = data.get("cfg")
    if not cfg:
        raise ValueError(f"No 'cfg' key found in {cfg_path}")
    return cfg


def _build_env(task: str):
    recipes_path = "craft/resources/recipes.yaml"
    hints_path = "craft/resources/hints.yaml"
    sampler = env_factory.EnvironmentFactory(
        recipes_path,
        hints_path,
        7,
        max_steps=300,
        reuse_environments=False,
        visualise=False,
    )
    env = sampler.sample_environment(task_name=task)
    if hasattr(env, "reset"):
        env.reset()
    return env


def _reset_env(env):
    if hasattr(env, "reset"):
        env.reset()


def _step_env(env, action):
    return env.step(action)


def _format_inventory(env) -> List[str]:
    state = getattr(env, "_current_state", None)
    if state is None:
        return []
    inventory = getattr(state, "inventory", None)
    if inventory is None:
        return []
    world = getattr(env, "world", None)
    if world is None:
        return []
    cookbook = getattr(world, "cookbook", None)
    if cookbook is None:
        return []
    index = getattr(cookbook, "index", None)
    items = []
    for idx, count in enumerate(inventory):
        if not count:
            continue
        name = str(idx)
        if index is not None:
            try:
                resolved = index.get(idx)
                if resolved is not None:
                    name = str(resolved)
            except Exception:
                # Fall back to numeric index if anything goes wrong
                name = str(idx)
        items.append(f"{name}={float(count)}")
    return items


def run_program_with_inventory(
    experiment_dir: str,
    task: str,
    cfg_version: int,
    program: str,
) -> None:
    cfg = _load_cfg(experiment_dir, cfg_version)

    final_functions_dir = os.path.join(experiment_dir, "final_functions")
    # Load implementations for this CFG round (cfg_output_N.json -> *_dslN_func0.py).
    import tempfile
    import shutil

    dsl_tag = f"_dsl{cfg_version}_func0"
    temp_dir = tempfile.mkdtemp(prefix="inv_functions_")
    for filename in os.listdir(final_functions_dir):
        if not filename.endswith(".py"):
            continue
        if dsl_tag not in filename:
            continue
        src = os.path.join(final_functions_dir, filename)
        dst = os.path.join(temp_dir, filename)
        shutil.copy2(src, dst)

    implementations = load_function_implementations(temp_dir)

    evaluator = DSLEvaluator(
        cfg=cfg,
        function_implementations=implementations,
        env_factory=None,
        env_reset=_reset_env,
        env_step=_step_env,
    )

    env = _build_env(task)

    tokens = evaluator.tokenize_program(program)

    total_reward = 0.0
    done = False
    steps = 0
    actions_taken: List[int] = []

    def _get_grid_md(env):
        state = getattr(env, "_current_state", None)
        world = getattr(env, "world", None)
        if state is not None and world is not None and hasattr(world, "cookbook"):
            return grid_to_markdown(state.grid, world.cookbook, getattr(state, "pos", None))
        return None

    safe_task = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in task)
    log_path = os.path.join(experiment_dir, f"craft_grids_{safe_task}.md")

    # Truncate/create fresh log file
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"# Program trace: {task}\n\n")
        f.write(f"**Program:** `{program}`\n\n")
        f.write("---\n\n")

    print(f"Task: {task}")
    print(f"Program:\n  {program}")
    print(f"\nLog: {log_path}")
    print("\nInitial inventory: " + (", ".join(_format_inventory(env)) or "<empty>"))
    print("=" * 80)

    global_action_num = 0

    for idx, token in enumerate(tokens, start=1):
        if done:
            break

        func_call = evaluator.extract_function_call(token)
        if not func_call:
            continue

        func_name, args = func_call
        normalized_args = [
            arg.lower() if isinstance(arg, str) else arg for arg in args
        ]

        safe_name = evaluator._sanitize_function_name(func_name)
        impl = implementations.get(safe_name)
        if impl is None:
            impl = implementations.get(func_name.lower())
        if impl is None:
            impl = implementations.get(func_name)
        if impl is None:
            raise KeyError(
                f"No implementation found for function {func_name} (sanitized: {safe_name})"
            )

        # Capture grid before calling function
        grid_before_call = _get_grid_md(env)

        # Call function — capture which actions it returns
        if normalized_args:
            actions = impl(env, *normalized_args)
        else:
            actions = impl(env)

        if not isinstance(actions, list):
            actions = [actions]

        header = f"## Token {idx}: `{token}`  →  actions from `{func_name}`: {actions}"
        print(header)

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(header + "\n\n")
            if grid_before_call:
                f.write("### Grid before function call\n\n")
                f.write(grid_before_call + "\n\n")

        # Execute each low-level action, logging grid after each one
        for local_step, action in enumerate(actions, start=1):
            if done:
                break

            global_action_num += 1
            grid_pre = _get_grid_md(env)

            reward, done, _obs = _step_env(env, action)
            if reward is not None:
                total_reward += float(reward)
            steps += 1
            actions_taken.append(action)

            grid_post = _get_grid_md(env)
            inv_items = _format_inventory(env)
            inv_str = ", ".join(inv_items) if inv_items else "<empty>"

            step_line = (
                f"  action {global_action_num} ({local_step}/{len(actions)} from {func_name}): "
                f"code={action}  reward={reward}  done={done}  inv=[{inv_str}]"
            )
            print(step_line)

            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"### Action {global_action_num} ({local_step}/{len(actions)}) — code `{action}` from `{func_name}`\n\n")
                f.write(f"reward={reward}  done={done}  inventory: {inv_str}\n\n")
                if grid_pre:
                    f.write("**Grid before:**\n\n" + grid_pre + "\n\n")
                if grid_post:
                    f.write("**Grid after:**\n\n" + grid_post + "\n\n")
                f.write("---\n\n")

        print("-" * 80)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("====\n\n")

    summary = f"Finished. total_reward={total_reward}, steps={steps}, done={done}"
    print(summary)
    print("Actions taken:")
    print(f"  {actions_taken}")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"## Summary\n\n{summary}\n\nActions: {actions_taken}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a single DSL program for a Craft task and print the "
            "agent's inventory after each PICKUP and CRAFT."
        )
    )
    parser.add_argument(
        "--experiment_dir",
        type=str,
        required=True,
        help="Path to the experiment directory",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task name, e.g. make[goldarrow]",
    )
    parser.add_argument(
        "--cfg_version",
        type=int,
        default=0,
        help="CFG round index: loads cfg/cfg_output_<n>.json and final_functions/*_dsl<n>_func0.py. Default: 0",
    )
    parser.add_argument(
        "--program",
        type=str,
        required=True,
        help="DSL program string to execute",
    )

    args = parser.parse_args()

    run_program_with_inventory(
        experiment_dir=args.experiment_dir,
        task=args.task,
        cfg_version=args.cfg_version,
        program=args.program,
    )


if __name__ == "__main__":
    main()

