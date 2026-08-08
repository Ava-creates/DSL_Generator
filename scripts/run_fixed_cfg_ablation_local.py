#!/usr/bin/env python3
"""Fixed-CFG terminal-function ablation (local, no SLURM, no DSL evolution).

Reuse a completed pipeline CFG (e.g. HF run 4 DSL~1), regenerate terminal
functions under funsearch / llm_best_of_n / llm_chained + explicit feedback,
then run program synthesis on all tasks x seeds only.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from src.utils.experiment_paths import build_default_experiment_dir
from src.utils.pipeline_state import update_state

DEFAULT_TASKS = [
    "get[gem]", "get[iron]", "get[wood]", "get[grass]", "get[gold]",
    "make[plank]", "make[stick]", "make[cloth]", "make[rope]", "make[bridge]",
    "make[bundle]", "make[flag]", "make[bed]", "make[axe]", "make[shears]",
    "make[ladder]", "make[goldarrow]", "make[goldhammer]",
    "make[clothbundle]", "make[clothbundleextra]",
]

MODE_PREFIX = {
    "funsearch": "ablation_fixed_cfg_fs",
    "llm_best_of_n": "ablation_fixed_cfg_bon",
    "llm_chained": "ablation_fixed_cfg_chained",
}


def _rewrite_paths(obj, src: str, dst: str):
    if isinstance(obj, str):
        return obj.replace(src, dst)
    if isinstance(obj, dict):
        return {k: _rewrite_paths(v, src, dst) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_rewrite_paths(v, src, dst) for v in obj]
    return obj


def _copytree(src: str, dst: str) -> None:
    if not os.path.isdir(src):
        return
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def bootstrap_experiment(*, source: str, dest: str, dsl_round: int, mode: str) -> list[str]:
    source = os.path.abspath(source)
    dest = os.path.abspath(dest)
    os.makedirs(dest, exist_ok=True)

    cfg_src = os.path.join(source, "cfg", f"cfg_output_{dsl_round}.json")
    if not os.path.isfile(cfg_src):
        raise FileNotFoundError(f"Missing {cfg_src}")

    cfg_dir = os.path.join(dest, "cfg")
    os.makedirs(cfg_dir, exist_ok=True)
    shutil.copy2(cfg_src, os.path.join(cfg_dir, f"cfg_output_{dsl_round}.json"))
    shutil.copy2(cfg_src, os.path.join(cfg_dir, "cfg_output.json"))

    with open(cfg_src, encoding="utf-8") as f:
        cfg_data = json.load(f)
    terminals = list(cfg_data.get("terminals", {}).keys())
    if not terminals:
        raise ValueError(f"No terminals in {cfg_src}")

    for sub in ("function_specific_prompts", "functions_generated", "grids"):
        _copytree(os.path.join(source, sub), os.path.join(dest, sub))

    status_src = os.path.join(source, "status", f"dsl{dsl_round}", "file_generation", "status")
    if os.path.isfile(status_src):
        with open(status_src, encoding="utf-8") as f:
            status = json.load(f)
        status = _rewrite_paths(status, source, dest)
        status_dir = os.path.join(dest, "status", f"dsl{dsl_round}", "file_generation")
        os.makedirs(status_dir, exist_ok=True)
        with open(os.path.join(status_dir, "status"), "w", encoding="utf-8") as f:
            json.dump(status, f, indent=2)

    meta = {
        "ablation_type": "fixed_cfg_terminal_function",
        "cloned_from": source,
        "dsl_round": dsl_round,
        "terminal_function_mode": mode,
        "terminals": terminals,
    }
    with open(os.path.join(dest, "ablation_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    update_state(
        dest,
        dsl_round=dsl_round,
        func_evolution_round=0,
        phase="fixed_cfg_ablation",
        function_implementation_total=len(terminals),
        function_implementation_remaining=len(terminals),
        test_tasks_submitted=0,
        pipeline_model_type=os.environ.get("MODEL_TYPE", ""),
        tasks=DEFAULT_TASKS,
    )
    return terminals


def copy_funsearch_final_functions(*, source: str, dest: str, dsl_round: int) -> None:
    src_ff = os.path.join(source, "final_functions")
    dst_ff = os.path.join(dest, "final_functions")
    os.makedirs(dst_ff, exist_ok=True)
    pattern = re.compile(rf"_dsl{dsl_round}(?:_|\.|$)")
    for name in os.listdir(src_ff):
        if pattern.search(name):
            shutil.copy2(os.path.join(src_ff, name), os.path.join(dst_ff, name))


def run_implement_local(
    *,
    experiment_dir: str,
    spec_file: str,
    terminals: list[str],
    mode: str,
    dsl_round: int,
    model_type: str,
    total_samples: int,
    num_ef_iterations: int,
    openai_compat_key_file: str | None,
) -> int:
    script = os.path.join(_REPO, "src", "pipeline", "stages", "stage_implement_cfg_single.py")
    ok = 0
    for func_name in terminals:
        cmd = [
            sys.executable, script,
            "--experiment_dir", experiment_dir,
            "--spec_file", spec_file,
            "--function_name", func_name,
            "--model_type", model_type,
            "--dsl_round", str(dsl_round),
            "--func_evolution_round", "0",
            "--total_samples", str(total_samples),
            "--num_iterations", str(num_ef_iterations),
            "--terminal_function_mode", mode,
        ]
        if openai_compat_key_file:
            cmd.extend(["--openai_compat_key_file", openai_compat_key_file])
        print(f"\n[implement] {func_name} mode={mode}")
        if subprocess.run(cmd, cwd=_REPO).returncode == 0:
            ok += 1
        else:
            print(f"[implement] FAILED {func_name}", file=sys.stderr)
    print(f"\n[implement] {ok}/{len(terminals)} functions OK")
    return 0 if ok else 1


def run_test_tasks_local(
    *,
    experiment_dir: str,
    tasks: list[str],
    dsl_round: int,
    model_type: str,
    max_attempts: int,
    openai_compat_key_file: str | None,
) -> int:
    script = os.path.join(_REPO, "src", "pipeline", "stages", "stage_test_tasks.py")
    cmd = [
        sys.executable, script,
        "--experiment_dir", experiment_dir,
        "--tasks", *tasks,
        "--dsl_round", str(dsl_round),
        "--func_evolution_round", "0",
        "--max_attempts", str(max_attempts),
        "--model_type", model_type,
    ]
    if openai_compat_key_file:
        cmd.extend(["--openai_compat_key_file", openai_compat_key_file])
    print(f"\n[test_tasks] {len(tasks)} tasks, dsl_round={dsl_round}")
    return subprocess.run(cmd, cwd=_REPO).returncode


def score_experiment(experiment_dir: str, dsl_round: int) -> None:
    from src.evaluate_expressiveness import score_dsl_round

    by_task: dict[str, list] = {}
    for pat in (
        f"{experiment_dir}/results_tracking/dsl{dsl_round}/tasks/*/program_synthesis_seed_outcomes.jsonl",
        f"{experiment_dir}/results_tracking/dsl{dsl_round}/func0/tasks/*/program_synthesis_seed_outcomes.jsonl",
    ):
        for fp in glob.glob(pat):
            with open(fp, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    e = json.loads(line)
                    by_task.setdefault(e["task"], []).append(e)
    ts, ss, tt, st = score_dsl_round(by_task)
    print(f"  tasks solved: {ts}/{tt or 20}")
    print(f"  seeds solved: {ss}/{st or 200}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fixed-CFG terminal-function ablation (local)")
    parser.add_argument(
        "--source",
        default="experiments/pipeline_hf_20260611_151047_run4_2104814",
    )
    parser.add_argument("--dsl-round", type=int, default=1)
    parser.add_argument(
        "--modes", nargs="+",
        default=["llm_chained", "llm_best_of_n"],
        choices=["funsearch", "llm_best_of_n", "llm_chained"],
    )
    parser.add_argument("--model-type", default=os.environ.get("MODEL_TYPE", "openai_compat"))
    parser.add_argument("--spec-file", default="prompt_specifications/spec_template.txt")
    parser.add_argument("--total-samples", type=int, default=100)
    parser.add_argument("--num-ef-iterations", type=int, default=30)
    parser.add_argument("--max-attempts", type=int, default=30)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--openai-compat-key-file", default=None)
    parser.add_argument("--skip-implement", action="store_true")
    parser.add_argument("--skip-test-tasks", action="store_true")
    parser.add_argument(
        "--copy-funsearch-from-source", action="store_true",
        help="funsearch mode: reuse final_functions from source (19/20 reference arm)",
    )
    parser.add_argument("--dest-root", default="experiments")
    args = parser.parse_args()

    tasks = args.tasks or DEFAULT_TASKS
    source = os.path.abspath(args.source)

    if args.model_type == "openai_compat":
        os.environ.setdefault("MODEL_TYPE", "openai_compat")
        from src.utils.openai_compat_cold_start import maybe_cold_start_openai_compat
        maybe_cold_start_openai_compat(key_file=args.openai_compat_key_file)

    for i, mode in enumerate(args.modes, start=1):
        prefix = MODE_PREFIX[mode]
        dest = build_default_experiment_dir(job_prefix=prefix, run_index=i, base_root=args.dest_root)
        print(f"\n{'='*72}\nMode: {mode}\nDest: {dest}\n{'='*72}")

        terminals = bootstrap_experiment(source=source, dest=dest, dsl_round=args.dsl_round, mode=mode)

        if not args.skip_implement:
            if mode == "funsearch" and args.copy_funsearch_from_source:
                print("[implement] Copying FunSearch final_functions from source")
                copy_funsearch_final_functions(source=source, dest=dest, dsl_round=args.dsl_round)
            else:
                run_implement_local(
                    experiment_dir=dest,
                    spec_file=args.spec_file,
                    terminals=terminals,
                    mode=mode,
                    dsl_round=args.dsl_round,
                    model_type=args.model_type,
                    total_samples=args.total_samples,
                    num_ef_iterations=args.num_ef_iterations,
                    openai_compat_key_file=args.openai_compat_key_file,
                )

        if not args.skip_test_tasks:
            run_test_tasks_local(
                experiment_dir=dest,
                tasks=tasks,
                dsl_round=args.dsl_round,
                model_type=args.model_type,
                max_attempts=args.max_attempts,
                openai_compat_key_file=args.openai_compat_key_file,
            )

        print(f"\n[results] {dest}")
        score_experiment(dest, args.dsl_round)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
