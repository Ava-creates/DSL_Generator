#!/usr/bin/env python3
"""Align grid arg_values keys to CFG-derived parameter names (domain-agnostic).

When a grid has the wrong key name but a single argument, renames the lone key
to the name the CFG expects. Reports grids that cannot be aligned automatically.

Example:
  python scripts/align_grid_arg_values.py --experiment experiments/pipeline_hf_... --func MOVE --dsl-round 1 --fix
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from src.pipeline.cfg_to_funsearch_pipeline import extract_function_args


def _load_cfg(experiment_dir: str, dsl_round: int) -> str:
    for name in (f"cfg_output_{dsl_round}.json", "cfg_output.json"):
        path = os.path.join(experiment_dir, "cfg", name)
        if os.path.isfile(path):
            return json.load(open(path, encoding="utf-8")).get("cfg", "")
    raise FileNotFoundError(f"No cfg for dsl_round={dsl_round} under {experiment_dir}/cfg")


def _expected_arg_names(func: str, cfg: str) -> list[str]:
    raw = extract_function_args(func, cfg)
    if raw == "arg":
        return []
    return [a.strip() for a in raw.split(",") if a.strip()]


def _align_arg_values(arg_values: dict, expected: list[str], fix: bool) -> tuple[dict, str | None]:
    if not expected:
        return arg_values, None
    if not isinstance(arg_values, dict):
        return arg_values, "arg_values is not an object"
    if set(arg_values.keys()) == set(expected):
        return arg_values, None
    if len(expected) == 1 and len(arg_values) == 1:
        only_key = next(iter(arg_values))
        target = expected[0]
        if only_key != target:
            if not fix:
                return arg_values, f"wrong key {only_key!r}, expected {target!r}"
            return {target: arg_values[only_key]}, f"renamed {only_key!r} -> {target!r}"
    return arg_values, f"keys {sorted(arg_values)} != expected {expected}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", required=True)
    p.add_argument("--func", required=True)
    p.add_argument("--dsl-round", type=int, default=1)
    p.add_argument("--fix", action="store_true")
    args = p.parse_args()

    exp = args.experiment
    if not os.path.isabs(exp):
        exp = os.path.join(_REPO, exp)
    cfg = _load_cfg(exp, args.dsl_round)
    expected = _expected_arg_names(args.func, cfg)
    func = args.func.lower()
    pattern = os.path.join(exp, "grids", f"{func}_dsl{args.dsl_round}_case*.json")
    paths = sorted(glob.glob(pattern), key=lambda x: int(re.search(r"case(\d+)", x).group(1)))

    print(f"CFG expects arg_values keys: {expected}")
    print(f"Grids: {len(paths)}")

    n_ok = n_fixed = n_fail = 0
    for path in paths:
        data = json.load(open(path, encoding="utf-8"))
        aligned, note = _align_arg_values(data.get("arg_values", {}), expected, args.fix)
        if note and note.startswith("renamed"):
            data["arg_values"] = aligned
            if args.fix:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                    f.write("\n")
            print(f"{os.path.basename(path)}: FIXED ({note})")
            n_fixed += 1
        elif note:
            print(f"{os.path.basename(path)}: FAIL ({note})")
            n_fail += 1
        else:
            print(f"{os.path.basename(path)}: ok")
            n_ok += 1

    print(f"Summary: ok={n_ok}, fixed={n_fixed}, fail={n_fail}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
