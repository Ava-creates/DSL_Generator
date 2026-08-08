#!/usr/bin/env python3
"""
Evaluate DSL expressiveness by running handwritten programs on failed tasks.

Workflow:
  1. summarize  — pick best DSL per run, list synthesis-failed tasks
  2. init       — optional template for failed tasks (you can add any tasks later)
  3. run        — evaluate programs on every test seed (--task/--program or programs.json)
  4. run-manifest — evaluate programs listed in handwritten_expressiveness_manifest.json
  5. run-batch  — run all experiments that have a filled programs.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.cfg_evaluator import CFGEvaluator
from src.pipeline.cfg_to_funsearch_pipeline import sanitize_function_name
from src.pipeline.integrated_pipeline import DEFAULT_TEST_SEEDS, load_final_functions

HANDWRITTEN_DIRNAME = "handwritten_expressiveness"
PROGRAMS_FILENAME = "programs.json"
RESULTS_FILENAME = "handwritten_seed_outcomes.jsonl"
SUMMARY_FILENAME = "handwritten_summary.json"


def _safe_task_token(task: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(task)).strip("_")
    return token or "task"


def resolve_cfg_for_dsl_round(experiment_dir: str, dsl_round: int) -> str:
    """Return the CFG JSON for a historical DSL round (not cfg_output.json)."""
    cfg_dir = os.path.join(experiment_dir, "cfg")
    versioned = os.path.join(cfg_dir, f"cfg_output_{dsl_round}.json")
    if os.path.isfile(versioned):
        return versioned
    raise FileNotFoundError(
        f"No CFG for dsl_round={dsl_round} under {cfg_dir} "
        f"(expected cfg_output_{dsl_round}.json)"
    )


def load_cfg_payload(cfg_path: str) -> dict[str, Any]:
    with open(cfg_path, encoding="utf-8") as fh:
        return json.load(fh)


def seed_outcomes_for_dsl(experiment_dir: str, dsl_round: int) -> dict[str, list[dict]]:
    base = os.path.join(
        experiment_dir,
        "results_tracking",
        f"dsl{dsl_round}",
        "tasks",
    )
    by_task: dict[str, list[dict]] = defaultdict(list)
    if not os.path.isdir(base):
        return by_task
    for task_dir in glob.glob(os.path.join(base, "*")):
        path = os.path.join(task_dir, "program_synthesis_seed_outcomes.jsonl")
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                by_task[entry["task"]].append(entry)
    return by_task


def score_dsl_round(by_task: dict[str, list[dict]]) -> tuple[int, int, int, int]:
    if not by_task:
        return 0, 0, 0, 0
    tasks_total = len(by_task)
    tasks_solved = sum(1 for entries in by_task.values() if any(e.get("solved") for e in entries))
    seeds_solved = sum(1 for entries in by_task.values() for e in entries if e.get("solved"))
    seeds_total = sum(len(entries) for entries in by_task.values())
    return tasks_solved, seeds_solved, tasks_total, seeds_total


def discover_dsl_rounds(experiment_dir: str) -> list[int]:
    rounds: set[int] = set()
    tracking = os.path.join(experiment_dir, "results_tracking")
    if os.path.isdir(tracking):
        for name in os.listdir(tracking):
            match = re.fullmatch(r"dsl(\d+)", name)
            if match and os.path.isdir(os.path.join(tracking, name)):
                rounds.add(int(match.group(1)))
    cfg_dir = os.path.join(experiment_dir, "cfg")
    if os.path.isdir(cfg_dir):
        for name in os.listdir(cfg_dir):
            match = re.fullmatch(r"cfg_output_(\d+)\.json", name)
            if match:
                rounds.add(int(match.group(1)))
    return sorted(rounds)


def pick_best_dsl_round(experiment_dir: str) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for dsl_round in discover_dsl_rounds(experiment_dir):
        by_task = seed_outcomes_for_dsl(experiment_dir, dsl_round)
        if not by_task:
            continue
        tasks_solved, seeds_solved, tasks_total, seeds_total = score_dsl_round(by_task)
        candidate = {
            "dsl_round": dsl_round,
            "tasks_solved": tasks_solved,
            "tasks_total": tasks_total,
            "seeds_solved": seeds_solved,
            "seeds_total": seeds_total,
            "by_task": by_task,
        }
        if best is None:
            best = candidate
            continue
        if (tasks_solved, seeds_solved) > (best["tasks_solved"], best["seeds_solved"]):
            best = candidate
    return best


def failed_tasks_for_dsl(by_task: dict[str, list[dict]]) -> list[str]:
    failed = []
    for task, entries in sorted(by_task.items()):
        if not any(e.get("solved") for e in entries):
            failed.append(task)
    return failed


def handwritten_dir(experiment_dir: str, dsl_round: int) -> str:
    return os.path.join(experiment_dir, HANDWRITTEN_DIRNAME, f"dsl{dsl_round}")


def programs_path(experiment_dir: str, dsl_round: int) -> str:
    return os.path.join(handwritten_dir(experiment_dir, dsl_round), PROGRAMS_FILENAME)


def _build_evaluator(
    experiment_dir: str,
    dsl_round: int,
) -> tuple[CFGEvaluator, str]:
    cfg_path = resolve_cfg_for_dsl_round(experiment_dir, dsl_round)
    cfg_payload = load_cfg_payload(cfg_path)
    cfg_text = str(cfg_payload["cfg"])
    terminals = cfg_payload.get("terminals", {})

    final_functions = load_final_functions(
        experiment_dir,
        terminals=terminals,
        dsl_round=dsl_round,
    )
    if not final_functions:
        raise FileNotFoundError(
            f"No final functions for dsl{dsl_round} in {experiment_dir}/final_functions"
        )

    temp_func_dir = os.path.join(
        handwritten_dir(experiment_dir, dsl_round),
        "_eval_functions",
    )
    os.makedirs(temp_func_dir, exist_ok=True)
    for func_name, func_code in final_functions.items():
        safe_name = sanitize_function_name(func_name)
        with open(os.path.join(temp_func_dir, f"{safe_name}.py"), "w", encoding="utf-8") as fh:
            fh.write(func_code.strip() + "\n")

    overrides_dir = os.path.join(handwritten_dir(experiment_dir, dsl_round), "overrides")
    if os.path.isdir(overrides_dir):
        for name in sorted(os.listdir(overrides_dir)):
            if name.endswith(".py"):
                shutil.copy2(
                    os.path.join(overrides_dir, name),
                    os.path.join(temp_func_dir, name),
                )

    evaluator = CFGEvaluator(cfg=cfg_text, final_functions_dir=temp_func_dir)
    return evaluator, cfg_path


def evaluate_handwritten_programs(
    experiment_dir: str,
    dsl_round: int,
    programs: dict[str, str],
    test_seeds: list[int],
    *,
    programs_by_seed: dict[str, dict[int, str]] | None = None,
    max_steps: int = 400,
    timeout: float = 180.0,
    recipes_path: str = "craft/resources/recipes.yaml",
    hints_path: str = "craft/resources/hints.yaml",
) -> tuple[list[dict], dict[str, Any]]:
    from craft import env_factory

    programs_by_seed = programs_by_seed or {}
    evaluator, cfg_path = _build_evaluator(
        experiment_dir, dsl_round
    )
    out_dir = handwritten_dir(experiment_dir, dsl_round)
    os.makedirs(out_dir, exist_ok=True)

    tasks = sorted(set(programs) | set(programs_by_seed))
    outcomes: list[dict] = []
    per_task: dict[str, dict[str, Any]] = {}

    for task in tasks:
        default_program = programs.get(task, "").strip()
        seed_programs = programs_by_seed.get(task, {})
        task_seeds: dict[str, str] = {}
        seed_program_used: dict[str, str] = {}
        solved_any = False
        solved_count = 0
        evaluated_count = 0

        for seed in test_seeds:
            program = seed_programs.get(int(seed), default_program).strip()
            if not program:
                continue
            evaluated_count += 1
            sampler = env_factory.EnvironmentFactory(
                recipes_path,
                hints_path,
                7,
                max_steps=max_steps,
                seed=int(seed),
                reuse_environments=False,
                visualise=False,
            )
            env = sampler.sample_environment(task_name=task)
            env.reset()

            result = evaluator.evaluate_program(
                program=program,
                env=env,
                max_steps=max_steps,
                timeout=timeout,
            )
            solved = bool(result.get("success", False))
            entry = {
                "task": task,
                "seed": int(seed),
                "solved": solved,
                "program": program,
                "program_source": "per_seed" if int(seed) in seed_programs else "shared",
                "total_reward": result.get("total_reward", 0.0),
                "steps_taken": result.get("steps_taken", result.get("steps", 0)),
                "failure_reason": result.get("failure_reason") or result.get("error"),
                "dsl_round": dsl_round,
                "cfg_path": cfg_path,
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            outcomes.append(entry)
            task_seeds[str(seed)] = "success" if solved else "failure"
            seed_program_used[str(seed)] = program
            if solved:
                solved_any = True
                solved_count += 1

        if evaluated_count == 0:
            continue

        per_task[task] = {
            "default_program": default_program or None,
            "programs_by_seed": {str(k): v for k, v in seed_programs.items()},
            "solved": solved_any,
            "seeds_solved": solved_count,
            "seeds_total": evaluated_count,
            "seed_results": task_seeds,
            "seed_programs": seed_program_used,
        }

    summary = {
        "experiment_dir": os.path.abspath(experiment_dir),
        "dsl_round": dsl_round,
        "cfg_path": cfg_path,
        "test_seeds": test_seeds,
        "tasks_evaluated": len(per_task),
        "tasks_solved": sum(1 for t in per_task.values() if t["solved"]),
        "seeds_solved": sum(t["seeds_solved"] for t in per_task.values()),
        "seeds_total": sum(t["seeds_total"] for t in per_task.values()),
        "per_task": per_task,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return outcomes, summary


def cmd_summarize(args: argparse.Namespace) -> int:
    experiment_dirs = [os.path.abspath(p) for p in args.experiment_dirs]
    rows = []
    for experiment_dir in experiment_dirs:
        best = pick_best_dsl_round(experiment_dir)
        if best is None:
            print(f"{os.path.basename(experiment_dir)}: no synthesis seed outcomes found")
            continue
        failed = failed_tasks_for_dsl(best["by_task"])
        rows.append((experiment_dir, best, failed))
        print(f"\n=== {os.path.basename(experiment_dir)} ===")
        print(
            f"Best DSL: dsl{best['dsl_round']} — "
            f"tasks {best['tasks_solved']}/{best['tasks_total']}, "
            f"seeds {best['seeds_solved']}/{best['seeds_total']}"
        )
        print(f"Failed tasks ({len(failed)}):")
        for task in failed:
            cfg_path = resolve_cfg_for_dsl_round(experiment_dir, best["dsl_round"])
            cfg_payload = load_cfg_payload(cfg_path)
            example = cfg_payload.get("example", "")
            print(f"  - {task}")
            if args.verbose and example:
                print(f"      example program: {example}")
        print(f"Handwritten template: {programs_path(experiment_dir, best['dsl_round'])}")

    if args.write_manifest and rows:
        manifest_path = os.path.abspath(args.write_manifest)
        existing_runs: list[dict[str, Any]] = []
        if os.path.isfile(manifest_path):
            existing_runs = _load_manifest(manifest_path).get("runs", [])
        manifest = {
            "_instructions": (
                "Use 'programs' for one program tested on every seed (works when the DSL has "
                "grid-adaptive terminals like PICKUP_NEAREST / NAVIGATE_TO). "
                "Use 'programs_by_seed' when the CFG is low-level (MOVE/TURN/FACE only) and "
                "each seed needs a different program: "
                "{ \"get[gem]\": { \"0\": \"...\", \"5\": \"...\" } }. "
                "Per-seed entries override 'programs' for that seed. Empty strings are skipped. "
                "Then run: python src/evaluate_expressiveness.py run-manifest --manifest <this file>"
            ),
            "runs": _merge_manifest_runs(existing_runs, rows),
        }
        os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"\nWrote manifest: {manifest_path}")
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    experiment_dir = os.path.abspath(args.experiment_dir)
    dsl_round = args.dsl_round
    if dsl_round is None:
        best = pick_best_dsl_round(experiment_dir)
        if best is None:
            print("No synthesis outcomes; pass --dsl_round explicitly.", file=sys.stderr)
            return 1
        dsl_round = best["dsl_round"]
        failed = failed_tasks_for_dsl(best["by_task"])
    else:
        by_task = seed_outcomes_for_dsl(experiment_dir, dsl_round)
        if not by_task:
            print(f"No synthesis outcomes for dsl{dsl_round}.", file=sys.stderr)
            return 1
        failed = failed_tasks_for_dsl(by_task)

    out_path = programs_path(experiment_dir, dsl_round)
    if os.path.isfile(out_path) and not args.force:
        print(f"Already exists (use --force to overwrite): {out_path}")
        return 0

    cfg_path = resolve_cfg_for_dsl_round(experiment_dir, dsl_round)
    cfg_payload = load_cfg_payload(cfg_path)

    payload = {
        "dsl_round": dsl_round,
        "cfg_path": cfg_path,
        "example_program": cfg_payload.get("example", ""),
        "terminals": cfg_payload.get("terminals", {}),
        "failed_tasks": failed,
        "programs": {task: "" for task in failed},
        "programs_by_seed": {task: {} for task in failed},
        "_instructions": (
            "Use 'programs' for one program on all seeds (grid-adaptive DSL). "
            "Use 'programs_by_seed' for low-level DSLs: {task: {seed: program}}. "
            "Or use the top-level handwritten_expressiveness_manifest.json."
        ),
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    print(f"Wrote template for {len(failed)} failed tasks: {out_path}")
    return 0


def _non_empty_programs(programs_raw: dict[str, Any]) -> dict[str, str]:
    return {k: str(v).strip() for k, v in programs_raw.items() if str(v).strip()}


def _normalize_programs_by_seed(
    raw: dict[str, Any],
) -> dict[str, dict[int, str]]:
    """task -> seed(int) -> program."""
    out: dict[str, dict[int, str]] = {}
    for task, seed_map in raw.items():
        if not isinstance(seed_map, dict):
            continue
        per_seed: dict[int, str] = {}
        for seed_key, program in seed_map.items():
            text = str(program).strip()
            if text:
                per_seed[int(seed_key)] = text
        if per_seed:
            out[str(task)] = per_seed
    return out


def _load_programs_file(path: str) -> tuple[int, dict[str, str], dict[str, dict[int, str]]]:
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    dsl_round = int(payload["dsl_round"])
    programs = _non_empty_programs(payload.get("programs", {}))
    programs_by_seed = _normalize_programs_by_seed(payload.get("programs_by_seed", {}))
    if not programs and not programs_by_seed:
        raise ValueError(f"No non-empty programs in {path}")
    return dsl_round, programs, programs_by_seed


def _programs_from_manifest_run(
    run: dict[str, Any],
) -> tuple[dict[str, str], dict[str, dict[int, str]]]:
    return (
        _non_empty_programs(run.get("programs", {})),
        _normalize_programs_by_seed(run.get("programs_by_seed", {})),
    )


def _dsl_round_from_manifest_run(run: dict[str, Any]) -> int:
    if "dsl_round" in run:
        return int(run["dsl_round"])
    if "best_dsl_round" in run:
        return int(run["best_dsl_round"])
    raise ValueError(
        f"Run entry missing dsl_round / best_dsl_round: {run.get('experiment_dir')}"
    )


def _load_manifest(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _merge_manifest_runs(
    existing_runs: list[dict[str, Any]],
    new_rows: list[tuple[str, dict[str, Any], list[str]]],
) -> list[dict[str, Any]]:
    """Preserve existing ``programs`` when re-summarizing."""
    by_dir = {os.path.abspath(r["experiment_dir"]): r for r in existing_runs}
    merged: list[dict[str, Any]] = []
    for exp, best, failed in new_rows:
        exp_abs = os.path.abspath(exp)
        prev = by_dir.get(exp_abs, {})
        programs = dict(prev.get("programs", {}))
        programs_by_seed = {
            task: dict(seeds)
            for task, seeds in prev.get("programs_by_seed", {}).items()
            if isinstance(seeds, dict)
        }
        for task in failed:
            programs.setdefault(task, "")
            programs_by_seed.setdefault(task, {})
        merged.append(
            {
                "experiment_dir": exp_abs,
                "best_dsl_round": best["dsl_round"],
                "synthesis_tasks_solved": best["tasks_solved"],
                "synthesis_tasks_total": best["tasks_total"],
                "synthesis_seeds_solved": best["seeds_solved"],
                "synthesis_seeds_total": best["seeds_total"],
                "failed_tasks": failed,
                "programs": programs,
                "programs_by_seed": programs_by_seed,
            }
        )
    return merged


def _resolve_dsl_round(experiment_dir: str, dsl_round: int | None) -> int:
    if dsl_round is not None:
        return int(dsl_round)
    best = pick_best_dsl_round(experiment_dir)
    if best is None:
        raise ValueError(
            "Could not infer dsl_round (no synthesis outcomes). Pass --dsl_round."
        )
    return int(best["dsl_round"])


def _run_one_experiment(
    experiment_dir: str,
    dsl_round: int,
    programs: dict[str, str],
    programs_by_seed: dict[str, dict[int, str]],
    args: argparse.Namespace,
) -> int:
    test_seeds = [int(s) for s in (args.test_seeds or DEFAULT_TEST_SEEDS)]
    n_tasks = len(set(programs) | set(programs_by_seed))
    print(f"Evaluating {n_tasks} task(s) on seeds {test_seeds}")
    if programs_by_seed:
        print(f"  per-seed programs: {sum(len(v) for v in programs_by_seed.values())} entries")
    print(f"Experiment: {experiment_dir}")
    print(f"DSL round: {dsl_round}")

    outcomes, summary = evaluate_handwritten_programs(
        experiment_dir,
        dsl_round,
        programs,
        test_seeds,
        programs_by_seed=programs_by_seed,
        max_steps=args.max_steps,
        timeout=args.timeout,
        recipes_path=args.recipes_path,
        hints_path=args.hints_path,
    )

    out_dir = handwritten_dir(experiment_dir, dsl_round)
    outcomes_path = os.path.join(out_dir, RESULTS_FILENAME)
    summary_path = os.path.join(out_dir, SUMMARY_FILENAME)
    with open(outcomes_path, "w", encoding="utf-8") as fh:
        for entry in outcomes:
            fh.write(json.dumps(entry) + "\n")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")

    print(f"\nHandwritten expressiveness: tasks {summary['tasks_solved']}/{summary['tasks_evaluated']}, "
          f"seeds {summary['seeds_solved']}/{summary['seeds_total']}")
    print(f"Wrote: {outcomes_path}")
    print(f"Wrote: {summary_path}")
    for task, info in summary["per_task"].items():
        status = "PASS" if info["solved"] else "FAIL"
        print(f"  {status} {task}: {info['seeds_solved']}/{info['seeds_total']} seeds")
    return 0 if summary["tasks_solved"] == summary["tasks_evaluated"] else 1


def cmd_run(args: argparse.Namespace) -> int:
    experiment_dir = os.path.abspath(args.experiment_dir)
    programs: dict[str, str] = {}
    programs_by_seed: dict[str, dict[int, str]] = {}
    dsl_round: int | None = args.dsl_round

    if args.task or args.program:
        if not args.task or not args.program:
            print("Pass both --task and --program together.", file=sys.stderr)
            return 1
        programs = {args.task: args.program.strip()}
        dsl_round = _resolve_dsl_round(experiment_dir, dsl_round)

    programs_file = args.programs_file
    if programs_file:
        programs_file = os.path.abspath(programs_file)
        if not os.path.isfile(programs_file):
            print(f"Missing programs file: {programs_file}", file=sys.stderr)
            return 1
        file_dsl_round, file_programs, file_by_seed = _load_programs_file(programs_file)
        dsl_round = dsl_round if dsl_round is not None else file_dsl_round
        programs = {**file_programs, **programs}
        for task, seed_map in file_by_seed.items():
            programs_by_seed.setdefault(task, {}).update(seed_map)

    elif not programs and not programs_by_seed:
        best = pick_best_dsl_round(experiment_dir)
        if best is None:
            print("No synthesis outcomes. Use --task/--program or --programs_file.", file=sys.stderr)
            return 1
        programs_file = programs_path(experiment_dir, best["dsl_round"])
        if not os.path.isfile(programs_file):
            print(f"Missing programs file: {programs_file}", file=sys.stderr)
            print("Use --task/--program, --manifest, or pass --programs_file.", file=sys.stderr)
            return 1
        dsl_round, programs, programs_by_seed = _load_programs_file(programs_file)

    if not programs and not programs_by_seed:
        print("No programs to evaluate.", file=sys.stderr)
        return 1
    if dsl_round is None:
        dsl_round = _resolve_dsl_round(experiment_dir, None)

    return _run_one_experiment(
        experiment_dir, dsl_round, programs, programs_by_seed, args
    )


def cmd_run_manifest(args: argparse.Namespace) -> int:
    manifest_path = os.path.abspath(args.manifest)
    if not os.path.isfile(manifest_path):
        print(f"Manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    manifest = _load_manifest(manifest_path)
    runs = manifest.get("runs", [])
    if not runs:
        print("Manifest has no runs.", file=sys.stderr)
        return 1

    only = set(args.only) if args.only else None
    exit_code = 0
    ran = 0
    for idx, run in enumerate(runs):
        experiment_dir = os.path.abspath(run["experiment_dir"])
        label = os.path.basename(experiment_dir)
        if only is not None and label not in only and str(idx) not in only:
            continue

        programs, programs_by_seed = _programs_from_manifest_run(run)
        if not programs and not programs_by_seed:
            print(f"SKIP (no programs filled): {label}")
            continue

        dsl_round = _dsl_round_from_manifest_run(run)
        print(f"\n{'=' * 72}\n{label}\n{'=' * 72}")
        code = _run_one_experiment(
            experiment_dir, dsl_round, programs, programs_by_seed, args
        )
        ran += 1
        if code != 0:
            exit_code = code

    if ran == 0:
        print("No runs evaluated. Fill 'programs' in the manifest (non-empty strings).", file=sys.stderr)
        return 1
    return exit_code


def cmd_run_batch(args: argparse.Namespace) -> int:
    pattern = args.glob_pattern
    dirs = sorted(glob.glob(pattern))
    if not dirs:
        print(f"No directories matched: {pattern}", file=sys.stderr)
        return 1
    exit_code = 0
    for experiment_dir in dirs:
        best = pick_best_dsl_round(experiment_dir)
        if best is None:
            continue
        path = programs_path(experiment_dir, best["dsl_round"])
        if not os.path.isfile(path):
            print(f"SKIP (no programs.json): {experiment_dir}")
            continue
        try:
            dsl_round, programs, programs_by_seed = _load_programs_file(path)
        except ValueError as exc:
            print(f"SKIP ({exc}): {path}")
            continue
        print(f"\n{'=' * 72}\n{os.path.basename(experiment_dir)}\n{'=' * 72}")
        ns = argparse.Namespace(
            experiment_dir=experiment_dir,
            programs_file=path,
            task=None,
            program=None,
            dsl_round=dsl_round,
            test_seeds=args.test_seeds,
            max_steps=args.max_steps,
            timeout=args.timeout,
            recipes_path=args.recipes_path,
            hints_path=args.hints_path,
        )
        code = _run_one_experiment(
            experiment_dir, dsl_round, programs, programs_by_seed, ns
        )
        if code != 0:
            exit_code = code
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_sum = sub.add_parser("summarize", help="Show best DSL and failed tasks per run")
    p_sum.add_argument(
        "experiment_dirs",
        nargs="+",
        help="Experiment directories (e.g. experiments/pipeline_hf_*_run1_*)",
    )
    p_sum.add_argument("--verbose", action="store_true", help="Print CFG example programs")
    p_sum.add_argument(
        "--write-manifest",
        metavar="PATH",
        help="Write JSON manifest of best DSL + failed tasks",
    )

    p_init = sub.add_parser("init", help="Create programs.json template for failed tasks")
    p_init.add_argument("--experiment_dir", required=True)
    p_init.add_argument(
        "--dsl_round",
        type=int,
        default=None,
        help="DSL round (default: best by synthesis results)",
    )
    p_init.add_argument("--force", action="store_true", help="Overwrite existing template")

    p_run = sub.add_parser("run", help="Evaluate handwritten programs on all seeds")
    p_run.add_argument("--experiment_dir", required=True)
    p_run.add_argument("--programs_file", default=None, help="JSON with any task->program pairs")
    p_run.add_argument(
        "--task",
        default=None,
        help="Single task to test (with --program); use programs.json for many tasks",
    )
    p_run.add_argument(
        "--program",
        default=None,
        help="DSL program for --task; evaluated on every test seed",
    )
    p_run.add_argument(
        "--dsl_round",
        type=int,
        default=None,
        help="DSL round (default: from programs.json or best synthesis DSL)",
    )
    p_run.add_argument(
        "--test_seeds",
        type=int,
        nargs="+",
        default=None,
        help=f"Test seeds (default: {DEFAULT_TEST_SEEDS})",
    )
    p_run.add_argument("--max_steps", type=int, default=400)
    p_run.add_argument("--timeout", type=float, default=180.0)
    p_run.add_argument("--recipes_path", default="craft/resources/recipes.yaml")
    p_run.add_argument("--hints_path", default="craft/resources/hints.yaml")

    p_batch = sub.add_parser("run-batch", help="Run all experiments with filled programs.json")
    p_batch.add_argument(
        "--glob-pattern",
        default=str(PROJECT_ROOT / "experiments" / "pipeline_hf_*"),
        help="Glob of experiment directories",
    )
    p_batch.add_argument("--test_seeds", type=int, nargs="+", default=None)
    p_batch.add_argument("--max_steps", type=int, default=400)
    p_batch.add_argument("--timeout", type=float, default=180.0)
    p_batch.add_argument("--recipes_path", default="craft/resources/recipes.yaml")
    p_batch.add_argument("--hints_path", default="craft/resources/hints.yaml")

    p_manifest = sub.add_parser(
        "run-manifest",
        help="Evaluate programs from handwritten_expressiveness_manifest.json",
    )
    p_manifest.add_argument(
        "--manifest",
        default=str(PROJECT_ROOT / "experiments" / "handwritten_expressiveness_manifest.json"),
        help="Manifest JSON with per-run 'programs' map",
    )
    p_manifest.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Run subset by experiment dir basename or 0-based index (e.g. pipeline_hf_..._run2_2104814 2)",
    )
    p_manifest.add_argument("--test_seeds", type=int, nargs="+", default=None)
    p_manifest.add_argument("--max_steps", type=int, default=400)
    p_manifest.add_argument("--timeout", type=float, default=180.0)
    p_manifest.add_argument("--recipes_path", default="craft/resources/recipes.yaml")
    p_manifest.add_argument("--hints_path", default="craft/resources/hints.yaml")

    args = parser.parse_args()
    if args.command == "summarize":
        return cmd_summarize(args)
    if args.command == "init":
        return cmd_init(args)
    if args.command == "run":
        return cmd_run(args)
    if args.command == "run-manifest":
        return cmd_run_manifest(args)
    if args.command == "run-batch":
        return cmd_run_batch(args)
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
