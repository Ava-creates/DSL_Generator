#!/usr/bin/env python3
"""Diagnose grid specs vs FunSearch evaluate() for a primitive function.

Checks:
  1. arg_values keys vs function parameter names (common cause of runtime_failure)
  2. Whether pass_check passes when a reference implementation runs the grid

Example:
  python scripts/investigate_grid_pass_check.py \
    --experiment experiments/pipeline_hf_20260611_151047_run2_2104814 \
    --func move --dsl-round 1
"""

from __future__ import annotations

import argparse
import ast
import copy
import glob
import importlib.util
import json
import os
import re
import sys


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_func_init_params(experiment_dir: str, func: str, dsl_round: int) -> list[str]:
    path = os.path.join(
        experiment_dir,
        "functions_generated",
        f"{func.lower()}_dsl{dsl_round}_func0_func_init.py",
    )
    if not os.path.isfile(path):
        raise FileNotFoundError(f"func_init not found: {path}")
    tree = ast.parse(open(path, encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == func.lower():
            return [a.arg for a in node.args.args if a.arg != "env"]
    raise ValueError(f"No def {func.lower()}(...) in {path}")


def _load_reference_move(repo_root: str, experiment_dir: str):
    path = os.path.join(experiment_dir, "final_functions/move_dsl0_func0.py")
    if not os.path.isfile(path):
        return None
    spec = importlib.util.spec_from_file_location("move_dsl0_ref", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.move


def _to_list(x):
    if hasattr(x, "tolist"):
        return x.tolist()
    return list(x)


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


def _inventory_list_to_dict(inv, env):
    if not isinstance(inv, list):
        return inv
    cb = env.world.cookbook
    idx = cb.index if cb and hasattr(cb, "index") else None
    if idx is not None and hasattr(idx, "get"):
        return {str(idx.get(i)): v for i, v in enumerate(inv) if v}
    return inv


class _InvList(list):
    def get(self, key, default=0):
        return default


def run_pass_check(grid_path: str, move_fn, param_names: list[str]) -> dict:
    from craft import env_factory

    recipes_path = os.path.join(_repo_root(), "craft/resources/recipes.yaml")
    hints_path = os.path.join(_repo_root(), "craft/resources/hints.yaml")

    with open(grid_path, encoding="utf-8") as f:
        grid_spec = json.load(f)

    arg_values = grid_spec.get("arg_values") or {}
    missing = [p for p in param_names if p not in arg_values]
    alias_used = None
    eval_arg_values = dict(arg_values)
    if missing and param_names == ["integer"] and "steps" in arg_values:
        alias_used = "steps->integer"
        eval_arg_values["integer"] = eval_arg_values["steps"]

    task_name = grid_spec.get("task_name", "get[wood]")
    env_sampler = env_factory.EnvironmentFactory(
        recipes_path,
        hints_path,
        7,
        max_steps=300,
        reuse_environments=False,
        visualise=False,
        custom_grid_path=grid_path,
    )
    env = env_sampler.sample_environment(task_name=task_name)
    env.scenario.spec = grid_spec
    env.reset()

    runtime_error = None
    passed = None
    pos_before = pos_after = None
    dir_before = dir_after = None

    if move_fn is None:
        runtime_error = "no reference move implementation loaded"
    else:
        state = env._current_state
        pos_before = _to_list(state.pos)
        dir_before = state.dir
        grid_before_cells = _grid_cells(env)
        inventory_before = _inventory_list_to_dict(
            _to_list(state.inventory.copy()), env
        )

        call_args = {}
        for p in param_names:
            if p not in eval_arg_values:
                runtime_error = (
                    f"KeyError: arg_values missing '{p}' (have {list(arg_values)})"
                )
                break
            call_args[p] = eval_arg_values[p]
        else:
            env_for_func = copy.deepcopy(env)
            steps_val = call_args.get("integer", call_args.get("steps"))
            actions = move_fn(env_for_func, steps_val)
            for action in actions or []:
                env.step(action)

            state = env._current_state
            pos_after = _to_list(state.pos)
            dir_after = state.dir
            grid_after_cells = _grid_cells(env)
            inventory_after = _inventory_list_to_dict(
                _to_list(state.inventory.copy()), env
            )

            if isinstance(inventory_before, list):
                inventory_before = _InvList(inventory_before)
            if isinstance(inventory_after, list):
                inventory_after = _InvList(inventory_after)

            pass_check = grid_spec.get("pass_check", "")
            ctx = {
                "pos_before": pos_before,
                "pos_after": pos_after,
                "inventory_before": inventory_before,
                "inventory_after": inventory_after,
                "grid_before_cells": grid_before_cells,
                "grid_after_cells": grid_after_cells,
                "dir_before": dir_before,
                "dir_after": dir_after,
            }
            passed = bool(eval(pass_check, ctx, ctx)) if pass_check else False

    return {
        "file": os.path.basename(grid_path),
        "arg_values": grid_spec.get("arg_values"),
        "param_names": param_names,
        "missing_params": missing,
        "alias_used": alias_used,
        "runtime_error": runtime_error,
        "pos_before": pos_before,
        "pos_after": pos_after,
        "dir_before": dir_before,
        "dir_after": dir_after,
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--func", default="move")
    parser.add_argument("--dsl-round", type=int, default=1)
    args = parser.parse_args()

    repo = _repo_root()
    if repo not in sys.path:
        sys.path.insert(0, repo)

    exp = args.experiment
    if not os.path.isabs(exp):
        exp = os.path.join(repo, exp)

    param_names = _load_func_init_params(exp, args.func, args.dsl_round)
    pattern = os.path.join(exp, "grids", f"{args.func.lower()}_dsl{args.dsl_round}_case*.json")
    grid_paths = sorted(glob.glob(pattern), key=lambda p: int(re.search(r"case(\d+)", p).group(1)))

    print(f"Function params (from func_init): {param_names}")
    print(f"Grids: {len(grid_paths)} files\n")

    move_fn = _load_reference_move(repo, exp) if args.func.lower() == "move" else None
    if move_fn:
        print("Reference: final_functions/move_dsl0_func0.move\n")

    n_key_mismatch = n_pass = n_fail = n_runtime = 0

    for path in grid_paths:
        r = run_pass_check(path, move_fn, param_names)
        if r["missing_params"] and not r["alias_used"]:
            n_key_mismatch += 1
            status = f"KEY MISMATCH missing={r['missing_params']} in {r['arg_values']}"
        elif r["runtime_error"]:
            n_runtime += 1
            status = f"RUNTIME: {r['runtime_error']}"
        elif r["passed"]:
            n_pass += 1
            status = "pass_check OK"
        else:
            n_fail += 1
            status = (
                f"pass_check FAIL pos {r['pos_before']}->{r['pos_after']} "
                f"dir {r['dir_before']}->{r['dir_after']}"
            )
        if r["alias_used"]:
            status += f" (aliased {r['alias_used']})"
        print(f"{r['file']}: {status}")

    print(
        f"\nSummary: key_mismatch={n_key_mismatch}, pass_check_pass={n_pass}, "
        f"pass_check_fail={n_fail}, runtime={n_runtime}"
    )

    log_hint = os.path.join(
        repo,
        "scripts/log",
        os.path.basename(exp),
        "stage_implement_cfg_pipeline_hf_20260611_impl_MOVE_5372231.out",
    )
    if os.path.isfile(log_hint):
        print(f"\nSLURM log: {log_hint}")
        print("  sed -n '780,820p' <log>   # first KeyError")
        print("  rg 'runtime_failure|\\[init check\\]' <log>")

    if n_key_mismatch:
        print(
            "\nFunSearch hits runtime_failure before pass_check runs. "
            "Grid regen cannot fix arg-name mismatch."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
