#!/usr/bin/env python3
"""Evaluate unique DSL1 synth programs on all test seeds (cross-seed coverage).

Reproducibility:
- PYTHONHASHSEED=0 (re-execs if unset)
- reuse_environments=True (match synthesis grids)
- program uniqueness matches appendix norm_prog (semicolon whitespace)
- existing cross_seed_dsl*_coverage.json is reused unless --force
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Reproducibility: hash seed must be set before the interpreter starts.
import os
import re

if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable, *sys.argv])

from craft import env_factory
from src.evaluate_expressiveness import _build_evaluator, seed_outcomes_for_dsl
from src.pipeline.integrated_pipeline import DEFAULT_TEST_SEEDS

TASKS = [
    "get[gem]",
    "get[iron]",
    "get[wood]",
    "get[grass]",
    "get[gold]",
    "make[plank]",
    "make[stick]",
    "make[cloth]",
    "make[rope]",
    "make[bridge]",
    "make[bundle]",
    "make[flag]",
    "make[bed]",
    "make[axe]",
    "make[shears]",
    "make[ladder]",
    "make[goldarrow]",
    "make[goldhammer]",
    "make[clothbundle]",
    "make[clothbundleextra]",
]


def _cache_key(task: str, program: str, seed: int) -> str:
    return json.dumps({"task": task, "program": program, "seed": int(seed)}, sort_keys=True)


def load_cache(path: Path) -> dict[str, bool]:
    if not path.is_file():
        return {}
    cache: dict[str, bool] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        cache[_cache_key(row["task"], row["program"], row["seed"])] = bool(row["solved"])
    return cache


def norm_prog(prog: str) -> str:
    """Match appendix uniqueness: collapse whitespace around semicolons."""
    return re.sub(r"\s*;\s*", ";", str(prog).strip())


def unique_programs_by_task(experiment_dir: str, dsl_round: int) -> dict[str, dict[str, list[int]]]:
    by_task = seed_outcomes_for_dsl(experiment_dir, dsl_round)
    out: dict[str, dict[str, list[int]]] = {}
    for task in TASKS:
        prog_to_seeds: dict[str, list[int]] = defaultdict(list)
        canonical: dict[str, str] = {}
        for entry in by_task.get(task, []):
            if not entry.get("solved"):
                continue
            program = entry.get("solved_program")
            if not program:
                continue
            key = norm_prog(program)
            # Keep one surface form per normalized key (first seen).
            canonical.setdefault(key, str(program).strip())
            prog_to_seeds[key].append(int(entry["seed"]))
        out[task] = {
            canonical[key]: sorted(set(seeds)) for key, seeds in prog_to_seeds.items()
        }
    return out


def evaluate_cross_seed(
    experiment_dir: str,
    dsl_round: int,
    unique: dict[str, dict[str, list[int]]],
    cache_path: Path,
    *,
    max_steps: int = 400,
    timeout: float = 300.0,
    recipes_path: str = "craft/resources/recipes.yaml",
    hints_path: str = "craft/resources/hints.yaml",
) -> dict[str, dict[str, list[int]]]:
    cache = load_cache(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    evaluator, _cfg_path = _build_evaluator(experiment_dir, dsl_round)
    seeds = list(DEFAULT_TEST_SEEDS)
    solved_by_prog: dict[str, dict[str, list[int]]] = {task: {} for task in TASKS}

    # Class-level guard (not instance monkeypatch): instance wrappers break
    # DSLEvaluator's deepcopy+terminal-step pattern (closure steps the original
    # env). Class methods bind correctly on deepcopies. Matches synthesis
    # success semantics while failing fast after max_steps instead of spinning.
    from craft.env import CraftLab
    _orig_craft_step = CraftLab.step

    def _craft_step_fail_after_budget(self, action, num_steps=1):
        if getattr(self, "steps", 0) >= getattr(self, "max_steps", max_steps):
            raise RuntimeError(
                f"ran out of steps (exceeded max_steps={self.max_steps})"
            )
        return _orig_craft_step(self, action, num_steps=num_steps)

    CraftLab.step = _craft_step_fail_after_budget

    pending = []
    for task, prog_map in unique.items():
        for program in prog_map:
            pending.append((task, program))
    total_evals = len(pending) * len(seeds)
    done = 0

    with cache_path.open("a", encoding="utf-8") as cache_fh:
        for task, program in pending:
            cross_solved: list[int] = []
            for seed in seeds:
                key = _cache_key(task, program, seed)
                if key in cache:
                    solved = cache[key]
                else:
                    sampler = env_factory.EnvironmentFactory(
                        recipes_path,
                        hints_path,
                        7,
                        max_steps=max_steps,
                        seed=int(seed),
                        reuse_environments=True,
                        visualise=False,
                    )
                    env = sampler.sample_environment(task_name=task)
                    env.reset()
                    # Do NOT monkeypatch env.step: DSLEvaluator deepcopy()s the env and
                    # terminal fns step the copy; a wrapped step closes over the original
                    # env and breaks ENSURE_INVENTORY-style programs. Match synthesis:
                    # evaluate_program + wall-clock timeout only.
                    result = evaluator.evaluate_program(
                        program=program,
                        env=env,
                        max_steps=max_steps,
                        timeout=timeout,
                    )
                    # Protocol: only goal success counts. Timeout, out-of-steps,
                    # parse/exec errors, and missing reward are all failures.
                    err = (result.get("error") or "")
                    timed_out = "timed out" in err.lower() or "timeout" in err.lower()
                    solved = bool(result.get("success", False)) and not timed_out
                    cache[key] = solved
                    cache_fh.write(
                        json.dumps(
                            {
                                "task": task,
                                "program": program,
                                "seed": int(seed),
                                "solved": solved,
                            }
                        )
                        + "\n"
                    )
                    cache_fh.flush()
                done += 1
                if solved:
                    cross_solved.append(int(seed))
                print(
                    f"[{done}/{total_evals}] {task} seed={seed} solved={solved}",
                    flush=True,
                )
            solved_by_prog[task][program] = cross_solved
    CraftLab.step = _orig_craft_step
    return solved_by_prog


def coverage_task_payload(
    task: str,
    prog_to_synth: dict[str, list[int]],
    prog_to_cross: dict[str, list[int]],
) -> dict:
    programs = []
    for program, synth_seeds in prog_to_synth.items():
        cross = prog_to_cross.get(program, [])
        programs.append(
            {
                "program": program,
                "synth_seeds": synth_seeds,
                "cross_seed_solved": cross,
                "coverage": len(cross),
            }
        )
    programs.sort(key=lambda row: (-row["coverage"], min(row["synth_seeds"]) if row["synth_seeds"] else 10**9))
    best = programs[0] if programs else None
    synth_solved = sum(len(seeds) for seeds in prog_to_synth.values())
    payload = {
        "seeds_solved_in_synth": synth_solved,
        "unique_programs": len(programs),
        "any_program_solves_all_10": bool(best and best["coverage"] == 10),
        "best_coverage": int(best["coverage"]) if best else 0,
        "programs": programs,
    }
    if best:
        payload["best_program"] = best["program"]
    return payload


def programs_task_payload(
    task: str,
    by_task_entries: list[dict],
    seeds: list[int],
) -> dict:
    program_by_seed: dict[str, str] = {}
    seeds_failed: list[int] = []
    for seed in seeds:
        match = None
        for entry in by_task_entries:
            if int(entry["seed"]) == int(seed):
                match = entry
                break
        if match and match.get("solved") and match.get("solved_program"):
            program_by_seed[str(int(seed))] = str(match["solved_program"])
        else:
            seeds_failed.append(int(seed))
    freq: dict[str, int] = {}
    for program in program_by_seed.values():
        freq[program] = freq.get(program, 0) + 1
    shared = {prog: n for prog, n in freq.items() if n >= 2}
    return {
        "seeds_solved": len(program_by_seed),
        "seeds_failed": seeds_failed,
        "unique_programs": len(freq),
        "program_by_seed": program_by_seed,
        "program_freq": freq,
        "shared_programs": shared,
    }


def protocol_task_row(coverage_task: dict) -> dict:
    programs = coverage_task["programs"]
    coverages = [row["coverage"] for row in programs]
    mean_g = (sum(coverages) / len(coverages)) if coverages else 0.0
    return {
        "synth": coverage_task["seeds_solved_in_synth"],
        "max_g": coverage_task["best_coverage"],
        "mean_g": mean_g,
        "uniq": coverage_task["unique_programs"],
    }


def md_escape_prog(program: str) -> str:
    return f"`{program}`"


def render_run_md(label: str, rel_path: str, programs_run: dict, seeds: list[int]) -> str:
    lines = [
        f"## {label}",
        "",
        f"Path: `{rel_path}`",
        "",
        "| Task | seeds | unique | reused? | programs (freq) |",
        "|------|------:|-------:|---------|-----------------|",
    ]
    tasks = programs_run["tasks"]
    for task in TASKS:
        rec = tasks[task]
        n = rec["seeds_solved"]
        uniq = rec["unique_programs"]
        if uniq == 0:
            reused = "—"
            freq_cell = "—"
        else:
            reused = "yes" if rec["shared_programs"] else "no"
            items = sorted(rec["program_freq"].items(), key=lambda kv: (-kv[1], kv[0]))
            freq_cell = "<br>".join(f"{md_escape_prog(p)} ×{c}" for p, c in items)
        lines.append(f"| `{task}` | {n} | {uniq} | {reused} | {freq_cell} |")

    multi = [task for task in TASKS if tasks[task]["unique_programs"] > 1]
    if multi:
        lines.append("")
        lines.append(f"### {label} per-seed programs (tasks with >1 unique program)")
        for task in multi:
            rec = tasks[task]
            lines.append("")
            lines.append(f"#### `{task}`")
            lines.append("")
            for seed in seeds:
                prog = rec["program_by_seed"].get(str(seed))
                if prog:
                    lines.append(f"- seed {seed}: {md_escape_prog(prog)}")
                else:
                    lines.append(f"- seed {seed}: *(unsolved)*")
    lines.append("")
    return "\n".join(lines)


def insert_or_replace_run(runs: list[dict], new_run: dict, after_label: str) -> list[dict]:
    for i, run in enumerate(runs):
        if run.get("label") == new_run["label"]:
            runs[i] = new_run
            return runs
    for i, run in enumerate(runs):
        if run.get("label") == after_label:
            runs.insert(i + 1, new_run)
            return runs
    runs.append(new_run)
    return runs


def insert_protocol_run(runs: list[dict], new_run: dict, after_thesis_run: int) -> list[dict]:
    for i, run in enumerate(runs):
        if run.get("label") == new_run["label"]:
            runs[i] = new_run
            return runs
    for i, run in enumerate(runs):
        if run.get("thesis_run") == after_thesis_run:
            runs.insert(i + 1, new_run)
            return runs
    runs.append(new_run)
    return runs


def update_reports(
    *,
    label: str,
    rel_path: str,
    cfg_path: str,
    thesis_run: int,
    coverage_run: dict,
    programs_run: dict,
    per_task: dict,
    seeds: list[int],
) -> None:
    reports = PROJECT_ROOT / "reports"

    coverage_path = reports / "dsl1_cross_seed_coverage.json"
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    coverage["runs"] = insert_or_replace_run(coverage["runs"], coverage_run, "api_run12")
    coverage_path.write_text(json.dumps(coverage, indent=2) + "\n", encoding="utf-8")

    protocol_path = reports / "dsl1_cross_seed_protocol_table.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    sum_max_g = sum(row["max_g"] for row in per_task.values())
    sum_mean_g = sum(row["mean_g"] for row in per_task.values())
    protocol_run = {
        "thesis_run": thesis_run,
        "label": label,
        "seeds_solved": programs_run["summary"]["seeds_solved"],
        "sum_max_g": sum_max_g,
        "sum_mean_g": round(sum_mean_g, 4),
        "tasks_with_max10": sum(1 for row in per_task.values() if row["max_g"] == 10),
    }
    protocol["runs"] = insert_protocol_run(protocol["runs"], protocol_run, 12)
    protocol[f"{label}_per_task"] = per_task
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")

    programs_path = reports / "dsl1_cross_seed_programs.json"
    programs = json.loads(programs_path.read_text(encoding="utf-8"))
    programs["runs"] = insert_or_replace_run(programs["runs"], programs_run, "api_run12")
    programs_path.write_text(json.dumps(programs, indent=2) + "\n", encoding="utf-8")

    md_path = reports / "dsl1_cross_seed_programs.md"
    md = md_path.read_text(encoding="utf-8")
    md = md.replace(
        "API run 13 has no `results_tracking/dsl1/tasks` (DSL1 synth not complete).\n\n",
        "",
    )
    s = programs_run["summary"]
    new_row = (
        f"| {label} | {s['tasks_with_any_solve']}/20 | {s['seeds_solved']}/200 | "
        f"{s['unique_program_instances_sum']} | {s['tasks_with_a_program_reused_across_seeds']} | "
        f"{s['tasks_all_10_seeds_same_program']} | {s['tasks_every_solved_seed_distinct_program']} |"
    )
    old_row = "| api_run13 | — | — | — | — | — | no dsl1/tasks |"
    if old_row not in md:
        raise ValueError("Could not find api_run13 placeholder row in programs.md")
    md = md.replace(old_row, new_row)
    section = render_run_md(label, rel_path, programs_run, seeds)
    marker = "## api_run14\n"
    if marker not in md:
        raise ValueError("Could not find ## api_run14 section in programs.md")
    label_heading = f"## {label}\n"
    pre, post = md.split(marker, 1)
    if label_heading in pre and pre.rfind(label_heading) > pre.find("## api_run12\n"):
        start = pre.rfind(label_heading)
        md = pre[:start] + section + "\n" + marker + post
    else:
        md = pre + section + "\n" + marker + post
    md_path.write_text(md, encoding="utf-8")

    print("PROTOCOL", json.dumps(protocol_run, indent=2))
    print("PER_TASK")
    for task, row in per_task.items():
        print(
            f"  {task:24} synth={row['synth']:2} max_g={row['max_g']:2} "
            f"mean_g={row['mean_g']:.4f} uniq={row['uniq']}"
        )
    print(
        f"sum_max_g={sum_max_g} sum_mean_g={sum_mean_g:.4f} "
        f"tasks_with_max10={protocol_run['tasks_with_max10']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--thesis_run", required=True, type=int)
    parser.add_argument("--dsl_round", default=1, type=int)
    parser.add_argument("--update-reports", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run eval even if cross_seed_dsl*_coverage.json already exists.",
    )
    args = parser.parse_args()

    experiment_dir = str(Path(args.experiment_dir).expanduser().resolve())
    rel_path = str(Path(experiment_dir).relative_to(PROJECT_ROOT))
    cfg_path = str(Path(experiment_dir) / "cfg" / f"cfg_output_{args.dsl_round}.json")
    seeds = list(DEFAULT_TEST_SEEDS)

    coverage_out = Path(experiment_dir) / f"cross_seed_dsl{args.dsl_round}_coverage.json"
    programs_out = Path(experiment_dir) / f"cross_seed_dsl{args.dsl_round}_programs.json"
    if coverage_out.is_file() and programs_out.is_file() and not args.force:
        print(
            "Reusing existing artifacts (reproducible):\n"
            f"  {coverage_out}\n  {programs_out}\n"
            "Pass --force to re-evaluate.",
            flush=True,
        )
        coverage_run = json.loads(coverage_out.read_text(encoding="utf-8"))
        programs_run = json.loads(programs_out.read_text(encoding="utf-8"))
        per_task = {
            task: protocol_task_row(coverage_run["tasks"][task]) for task in TASKS
        }
        if args.update_reports:
            update_reports(
                label=args.label,
                rel_path=rel_path,
                cfg_path=cfg_path,
                thesis_run=args.thesis_run,
                coverage_run=coverage_run,
                programs_run=programs_run,
                per_task=per_task,
                seeds=seeds,
            )
        sum_max_g = sum(row["max_g"] for row in per_task.values())
        sum_mean_g = sum(row["mean_g"] for row in per_task.values())
        print(
            f"sum_max_g={sum_max_g} sum_mean_g={sum_mean_g:.4f} "
            f"tasks_with_max10={sum(1 for row in per_task.values() if row['max_g'] == 10)}"
        )
        return 0

    unique = unique_programs_by_task(experiment_dir, args.dsl_round)
    n_unique = sum(len(v) for v in unique.values())
    print(f"unique programs: {n_unique}  evals: {n_unique * len(seeds)}", flush=True)

    cache_path = Path(experiment_dir) / f"cross_seed_dsl{args.dsl_round}_eval_cache.jsonl"
    cross = evaluate_cross_seed(experiment_dir, args.dsl_round, unique, cache_path)

    coverage_tasks = {}
    per_task = {}
    n_evals = 0
    for task in TASKS:
        coverage_tasks[task] = coverage_task_payload(task, unique[task], cross[task])
        per_task[task] = protocol_task_row(coverage_tasks[task])
        n_evals += len(unique[task]) * len(seeds)

    coverage_run = {
        "label": args.label,
        "path": rel_path,
        "cfg": cfg_path,
        "tasks": coverage_tasks,
        "summary": {
            "tasks_with_synth_solve": sum(
                1 for t in TASKS if coverage_tasks[t]["seeds_solved_in_synth"] > 0
            ),
            "tasks_with_a_program_that_solves_all_10_seeds": sum(
                1 for t in TASKS if coverage_tasks[t]["any_program_solves_all_10"]
            ),
            "evals": n_evals,
        },
    }

    by_task = seed_outcomes_for_dsl(experiment_dir, args.dsl_round)
    program_tasks = {
        task: programs_task_payload(task, by_task.get(task, []), seeds) for task in TASKS
    }
    programs_run = {
        "label": args.label,
        "path": rel_path,
        "complete": True,
        "tasks": program_tasks,
        "summary": {
            "tasks_with_any_solve": sum(1 for t in TASKS if program_tasks[t]["seeds_solved"] > 0),
            "seeds_solved": sum(program_tasks[t]["seeds_solved"] for t in TASKS),
            "unique_program_instances_sum": sum(program_tasks[t]["unique_programs"] for t in TASKS),
            "tasks_with_a_program_reused_across_seeds": sum(
                1 for t in TASKS if program_tasks[t]["shared_programs"]
            ),
            "tasks_all_10_seeds_same_program": sum(
                1
                for t in TASKS
                if program_tasks[t]["unique_programs"] == 1
                and program_tasks[t]["seeds_solved"] == 10
            ),
            "tasks_every_solved_seed_distinct_program": sum(
                1
                for t in TASKS
                if program_tasks[t]["seeds_solved"] > 0
                and program_tasks[t]["unique_programs"] == program_tasks[t]["seeds_solved"]
            ),
        },
    }

    out_dir = Path(experiment_dir)
    (out_dir / f"cross_seed_dsl{args.dsl_round}_coverage.json").write_text(
        json.dumps(coverage_run, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / f"cross_seed_dsl{args.dsl_round}_programs.json").write_text(
        json.dumps(programs_run, indent=2) + "\n", encoding="utf-8"
    )

    if args.update_reports:
        update_reports(
            label=args.label,
            rel_path=rel_path,
            cfg_path=cfg_path,
            thesis_run=args.thesis_run,
            coverage_run=coverage_run,
            programs_run=programs_run,
            per_task=per_task,
            seeds=seeds,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
