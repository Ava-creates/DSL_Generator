#!/usr/bin/env python3
"""Bootstrap terminal-function ablation experiments from existing FunSearch runs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from src.utils.experiment_paths import build_default_experiment_dir


def _copytree(src: str, dst: str) -> None:
    if not os.path.isdir(src):
        return
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser(description="Clone CFG artifacts for terminal-function ablation")
    parser.add_argument("--source", required=True, help="Existing experiment dir")
    parser.add_argument("--mode", choices=["llm_best_of_n", "llm_chained"], required=True)
    parser.add_argument("--run_index", type=int, default=1)
    parser.add_argument("--dest", default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    source = os.path.abspath(args.source)
    if not os.path.isdir(source):
        print(f"Source not found: {source}", file=sys.stderr)
        return 1

    prefix = {
        "llm_best_of_n": "pipeline_hf_llm_bon",
        "llm_chained": "pipeline_hf_llm_chained",
    }[args.mode]

    dest = args.dest or build_default_experiment_dir(job_prefix=prefix, run_index=args.run_index)
    dest = os.path.abspath(dest)

    print(f"Source: {source}")
    print(f"Dest:   {dest}")
    print(f"Mode:   {args.mode}")
    if args.dry_run:
        return 0

    os.makedirs(dest, exist_ok=True)
    for sub in ("cfg", "grids", "function_specific_prompts", "functions_generated"):
        _copytree(os.path.join(source, sub), os.path.join(dest, sub))
    if os.path.isdir(os.path.join(source, "status")):
        _copytree(os.path.join(source, "status"), os.path.join(dest, "status"))

    meta = {
        "ablation_mode": args.mode,
        "cloned_from": source,
        "run_index": args.run_index,
        "terminal_function_mode": args.mode,
    }
    with open(os.path.join(dest, "ablation_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    cfg_main = os.path.join(dest, "cfg", "cfg_output.json")
    if os.path.isfile(cfg_main):
        with open(cfg_main, encoding="utf-8") as f:
            cfg_data = json.load(f)
        with open(os.path.join(dest, "cfg", "cfg_output_0.json"), "w", encoding="utf-8") as f:
            json.dump(cfg_data, f, indent=2)

    print("\nNext steps:")
    print(f"  export EXPERIMENT_DIR={dest}")
    print("  export SKIP_CFG_GENERATION=true")
    print(f"  bash scripts/submit_with_config.sh config/experiment_config_hf_{args.mode}.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
