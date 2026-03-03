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
    # Only load implementations for DSL 0, func 0 to mirror the
    # configuration used during evaluation runs.
    import tempfile
    import shutil

    temp_dir = tempfile.mkdtemp(prefix="inv_functions_")
    for filename in os.listdir(final_functions_dir):
        if not filename.endswith(".py"):
            continue
        if "_dsl0_func0" not in filename:
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

    print(f"Task: {task}")
    print(f"Program:\n  {program}")
    print("\nInitial inventory:")
    print("  " + ", ".join(_format_inventory(env)) or "  <empty>")
    print("-" * 80)

    # Prepare optional grid log file for CRAFT steps
    safe_task = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in task)
    craft_log_path = os.path.join(experiment_dir, f"craft_grids_{safe_task}.md")

    for idx, token in enumerate(tokens, start=1):
        if done:
            break

        func_call = evaluator.extract_function_call(token)
        if not func_call:
            continue

        func_name, args = func_call
        upper_name = func_name.upper()
        normalized_args = [
            arg.lower() if isinstance(arg, str) else arg for arg in args
        ]

        # Capture grid before for CRAFT only
        grid_md_before = None
        if upper_name == "CRAFT":
            state_before = getattr(env, "_current_state", None)
            world = getattr(env, "world", None)
            if state_before is not None and world is not None and hasattr(world, "cookbook"):
                grid_md_before = grid_to_markdown(
                    state_before.grid,
                    world.cookbook,
                    getattr(state_before, "pos", None),
                )

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

        if normalized_args:
            actions = impl(env, *normalized_args)
        else:
            actions = impl(env)

        if not isinstance(actions, list):
            actions = [actions]

        # Execute low-level actions
        if upper_name == "CRAFT":
            # For CRAFT, log grid before/after each low-level step to file
            for local_step, action in enumerate(actions, start=1):
                if done:
                    break

                # Grid before this low-level step
                state_before_step = getattr(env, "_current_state", None)
                world = getattr(env, "world", None)
                grid_before_step = None
                grid_after_step = None
                if state_before_step is not None and world is not None and hasattr(world, "cookbook"):
                    grid_before_step = grid_to_markdown(
                        state_before_step.grid,
                        world.cookbook,
                        getattr(state_before_step, "pos", None),
                    )

                reward, done, _obs = _step_env(env, action)
                if reward is not None:
                    total_reward += float(reward)
                steps += 1
                actions_taken.append(action)

                # Grid after this low-level step
                state_after_step = getattr(env, "_current_state", None)
                if state_after_step is not None and world is not None and hasattr(world, "cookbook"):
                    grid_after_step = grid_to_markdown(
                        state_after_step.grid,
                        world.cookbook,
                        getattr(state_after_step, "pos", None),
                    )

                # Append per-step grids to markdown file
                with open(craft_log_path, "a", encoding="utf-8") as f:
                    f.write(f"## CRAFT step {idx}.{local_step} (action={action})\n\n")
                    if grid_before_step is not None:
                        f.write("### Grid before low-level step\n\n")
                        f.write(grid_before_step + "\n\n")
                    if grid_after_step is not None:
                        f.write("### Grid after low-level step\n\n")
                        f.write(grid_after_step + "\n\n")
                    f.write("---\n\n")

            # After all low-level steps, also log summary inventory and final grids once
            grid_md_after = None
            state_after = getattr(env, "_current_state", None)
            world = getattr(env, "world", None)
            if state_after is not None and world is not None and hasattr(world, "cookbook"):
                grid_md_after = grid_to_markdown(
                    state_after.grid,
                    world.cookbook,
                    getattr(state_after, "pos", None),
                )

            inv_items = _format_inventory(env)

            print(f"Step {idx}: {token}")
            if grid_md_before is not None:
                print("  Grid before craft (markdown):")
                print(grid_md_before)
            if grid_md_after is not None:
                print("  Grid after craft (markdown):")
                print(grid_md_after)
            print("  Inventory: " + (", ".join(inv_items) if inv_items else "<empty>"))
            print("-" * 80)

            with open(craft_log_path, "a", encoding="utf-8") as f:
                f.write(f"## Step {idx}: {token} (summary)\n\n")
                if grid_md_before is not None:
                    f.write("### Grid before craft (summary)\n\n")
                    f.write(grid_md_before + "\n\n")
                if grid_md_after is not None:
                    f.write("### Grid after craft (summary)\n\n")
                    f.write(grid_md_after + "\n\n")
                f.write("### Inventory after craft\n\n")
                f.write((", ".join(inv_items) if inv_items else "<empty>") + "\n\n")
                f.write("====\n\n")
        else:
            # Non-CRAFT terminals: just execute actions normally
            for action in actions:
                if done:
                    break
                reward, done, _obs = _step_env(env, action)
                if reward is not None:
                    total_reward += float(reward)
                steps += 1
                actions_taken.append(action)

    print(f"Finished. total_reward={total_reward}, steps={steps}, done={done}")
    print("Actions taken:")
    print(f"  {actions_taken}")


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
        help="CFG version index (uses cfg_output_<n>.json). Default: 0",
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

