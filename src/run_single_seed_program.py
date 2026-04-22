#!/usr/bin/env python3
"""
Run one DSL program on one task + seed using an experiment's CFG/functions.
"""

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from craft import env_factory
from src.pipeline.cfg_evaluator import CFGEvaluator
from src.utils.test import grid_to_markdown


def _load_cfg_text(cfg_path: Path) -> str:
    cfg_payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg_text = cfg_payload["cfg"]
    return str(cfg_text)


def _format_inventory(env) -> list[str]:
    state = env._current_state
    inventory = state.inventory
    index = env.world.cookbook.index
    items: list[str] = []
    for idx, count in enumerate(inventory):
        if not count:
            continue
        name_obj = None
        if hasattr(index, "get"):
            name_obj = index.get(idx)
        if name_obj is None and hasattr(index, "reverse_contents"):
            reverse_contents = index.reverse_contents
            if hasattr(reverse_contents, "get"):
                name_obj = reverse_contents.get(idx)
        name = str(name_obj) if name_obj is not None else str(idx)
        items.append(f"{name}={int(count)}")
    return items


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one program for one task+seed using experiment outputs."
    )
    parser.add_argument("--task", required=True, help="Task token, e.g. make[stick]")
    parser.add_argument("--seed", required=True, type=int, help="Seed integer, e.g. 10")
    parser.add_argument("--program", required=True, help="DSL program string")
    parser.add_argument(
        "--experiment_dir",
        required=True,
        help="Experiment directory containing cfg/cfg_output.json and final_functions/",
    )
    parser.add_argument(
        "--cfg_path",
        default=None,
        help="Optional explicit path to cfg JSON (defaults to <experiment_dir>/cfg/cfg_output.json)",
    )
    parser.add_argument(
        "--final_functions_dir",
        default=None,
        help="Optional explicit functions dir (defaults to <experiment_dir>/final_functions)",
    )
    parser.add_argument(
        "--recipes_path",
        default="craft/resources/recipes.yaml",
        help="Recipes yaml path",
    )
    parser.add_argument(
        "--hints_path",
        default="craft/resources/hints.yaml",
        help="Hints yaml path",
    )
    parser.add_argument("--max_steps", default=400, type=int, help="Max env steps")
    parser.add_argument(
        "--timeout",
        default=60.0,
        type=float,
        help="Wall-clock timeout seconds inside CFGEvaluator",
    )
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir).expanduser().resolve()
    cfg_path = (
        Path(args.cfg_path).expanduser().resolve()
        if args.cfg_path
        else experiment_dir / "cfg" / "cfg_output.json"
    )
    final_functions_dir = (
        Path(args.final_functions_dir).expanduser().resolve()
        if args.final_functions_dir
        else experiment_dir / "final_functions"
    )

    if not cfg_path.exists():
        raise FileNotFoundError(f"CFG file not found: {cfg_path}")
    if not final_functions_dir.exists():
        raise FileNotFoundError(f"Functions dir not found: {final_functions_dir}")

    cfg_text = _load_cfg_text(cfg_path)
    evaluator = CFGEvaluator(cfg=cfg_text, final_functions_dir=str(final_functions_dir))

    sampler = env_factory.EnvironmentFactory(
        args.recipes_path,
        args.hints_path,
        7,
        max_steps=int(args.max_steps),
        seed=int(args.seed),
        reuse_environments=False,
        visualise=False,
    )
    env = sampler.sample_environment(task_name=args.task)
    env.reset()

    start_pos = tuple(env._current_state.pos)
    start_dir = int(env._current_state.dir)
    start_inv = _format_inventory(env)
    grid_markdown = grid_to_markdown(
        env._current_state.grid,
        env.world.cookbook,
        tuple(env._current_state.pos),
        include_indices=True,
    )

    result = evaluator.evaluate_program(
        program=args.program,
        env=env,
        max_steps=int(args.max_steps),
        timeout=float(args.timeout),
    )
    end_pos = tuple(env._current_state.pos)
    end_dir = int(env._current_state.dir)
    end_inv = _format_inventory(env)
    end_grid_markdown = grid_to_markdown(
        env._current_state.grid,
        env.world.cookbook,
        tuple(env._current_state.pos),
        include_indices=True,
    )

    print("=== Run Inputs ===")
    print(f"experiment_dir: {experiment_dir}")
    print(f"cfg_path: {cfg_path}")
    print(f"final_functions_dir: {final_functions_dir}")
    print(f"task: {args.task}")
    print(f"seed: {args.seed}")
    print(f"program: {args.program}")
    print(f"start_pos: {start_pos}")
    print(f"start_dir: {start_dir}")
    print(f"start_inventory: {start_inv}")
    print("\n=== Grid Markdown (Start) ===")
    print(grid_markdown)
    print(f"\nend_pos: {end_pos}")
    print(f"end_dir: {end_dir}")
    print(f"end_inventory: {end_inv}")
    print("\n=== Grid Markdown (End) ===")
    print(end_grid_markdown)

    print("\n=== Result ===")
    print(json.dumps(result, indent=2))

    return 0 if result.get("success", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
