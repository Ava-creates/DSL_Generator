#!/usr/bin/env python3
"""Fixed-CFG terminal-function ablation (local, no SLURM, no DSL evolution).

Fix one CFG (e.g. HF run 4 DSL~1). FunSearch arm uses existing source results
only — not regenerated. Regenerate terminal functions for llm_chained and
llm_best_of_n + explicit feedback, then program synthesis (no evolution loop).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from src.utils.config_loader import export_config_to_env, load_config
from src.utils.experiment_paths import build_default_experiment_dir
from src.utils.pipeline_state import update_state

DEFAULT_TEST_SEEDS = list(range(0, 50, 5))
_TEXT_SUFFIXES = (".txt", ".py", ".json", ".yaml", ".yml")

DEFAULT_TASKS = [
    "get[gem]", "get[iron]", "get[wood]", "get[grass]", "get[gold]",
    "make[plank]", "make[stick]", "make[cloth]", "make[rope]", "make[bridge]",
    "make[bundle]", "make[flag]", "make[bed]", "make[axe]", "make[shears]",
    "make[ladder]", "make[goldarrow]", "make[goldhammer]",
    "make[clothbundle]", "make[clothbundleextra]",
]

ABLATION_MODES = ("llm_chained", "llm_best_of_n")


def _require_funsearch_submodule() -> None:
    impl = os.path.join(_REPO, "funsearch", "implementation", "funsearch.py")
    if not os.path.isfile(impl):
        print(
            "ERROR: funsearch submodule missing. Run:\n"
            "  git submodule update --init funsearch",
            file=sys.stderr,
        )
        raise SystemExit(1)


MODE_PREFIX = {
    "llm_best_of_n": "ablation_fixed_cfg_bon",
    "llm_chained": "ablation_fixed_cfg_chained",
}


def _experiment_path_aliases(path: str) -> tuple[str, str]:
    abs_path = os.path.abspath(path)
    rel_path = os.path.relpath(abs_path, _REPO)
    return abs_path, rel_path


def _replace_experiment_path(text: str, src: str, dst: str) -> str:
    src_abs, src_rel = _experiment_path_aliases(src)
    dst_abs, dst_rel = _experiment_path_aliases(dst)
    for old, new in ((src_abs, dst_abs), (src_rel, dst_rel)):
        if old and old in text:
            text = text.replace(old, new)
    return text


def _rewrite_paths(obj, src: str, dst: str):
    if isinstance(obj, str):
        return _replace_experiment_path(obj, src, dst)
    if isinstance(obj, dict):
        return {k: _rewrite_paths(v, src, dst) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_rewrite_paths(v, src, dst) for v in obj]
    return obj


def _rewrite_experiment_paths_in_tree(root: str, src: str, dst: str) -> int:
    """Rewrite embedded experiment paths (e.g. _grid_spec_paths) after copying artifacts."""
    if not os.path.isdir(root):
        return 0
    updated = 0
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if not name.endswith(_TEXT_SUFFIXES):
                continue
            path = os.path.join(dirpath, name)
            with open(path, encoding="utf-8") as f:
                content = f.read()
            new_content = _replace_experiment_path(content, src, dst)
            if new_content != content:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(new_content)
                updated += 1
    return updated


def load_test_seeds_from_source(source: str, dsl_round: int) -> list[int]:
    """Match program-synthesis seeds used in the reference run (for fair scoring)."""
    seeds: set[int] = set()
    for pat in (
        f"{source}/results_tracking/dsl{dsl_round}/tasks/*/program_synthesis_seed_outcomes.jsonl",
        f"{source}/results_tracking/dsl{dsl_round}/func0/tasks/*/program_synthesis_seed_outcomes.jsonl",
    ):
        for fp in glob.glob(pat):
            with open(fp, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    seeds.add(int(json.loads(line)["seed"]))
    if seeds:
        return sorted(seeds)
    return list(DEFAULT_TEST_SEEDS)


def summarize_dsl_grids(experiment_dir: str, dsl_round: int) -> tuple[int, list[str]]:
    grid_dir = os.path.join(experiment_dir, "grids")
    if not os.path.isdir(grid_dir):
        return 0, []
    marker = f"_dsl{dsl_round}_"
    files = sorted(f for f in os.listdir(grid_dir) if f.endswith(".json") and marker in f)
    prefixes = sorted({f.split(marker, 1)[0] for f in files})
    return len(files), prefixes


def configure_run4_grid_env(*, experiment_dir: str, grid_regeneration_attempts: int) -> None:
    """Evaluate terminal functions on the copied run-4 DSL grid specs (no regen)."""
    os.environ["USE_EXISTING_GRID_SPECS"] = "1"
    os.environ["GRID_REGENERATION_ATTEMPTS"] = str(grid_regeneration_attempts)
    os.environ["GRID_SPEC_DIR"] = os.path.join(experiment_dir, "grids")


def _copytree(src: str, dst: str) -> None:
    if not os.path.isdir(src):
        return
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _score_dsl_round(by_task: dict[str, list]) -> tuple[int, int, int, int]:
    if not by_task:
        return 0, 0, 0, 0
    tasks_total = len(by_task)
    tasks_solved = sum(1 for entries in by_task.values() if any(e.get("solved") for e in entries))
    seeds_solved = sum(1 for entries in by_task.values() for e in entries if e.get("solved"))
    seeds_total = sum(len(entries) for entries in by_task.values())
    return tasks_solved, seeds_solved, tasks_total, seeds_total


def score_experiment(experiment_dir: str, dsl_round: int) -> tuple[int, int, int, int]:
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
    return _score_dsl_round(by_task)


def print_funsearch_reference(source: str, dsl_round: int) -> None:
    """Print existing FunSearch results from source (not regenerated)."""
    ts, ss, tt, st = score_experiment(source, dsl_round)
    print(f"\n{'='*72}")
    print("Reference arm: funsearch (from --source, NOT regenerated)")
    print(f"  source: {source}")
    print(f"  tasks solved: {ts}/{tt or 20}")
    print(f"  seeds solved: {ss}/{st or 200}")
    print(f"{'='*72}")


def bootstrap_experiment(*, source: str, dest: str, dsl_round: int, mode: str) -> list[str]:
    source = os.path.abspath(source)
    dest = os.path.abspath(dest)
    os.makedirs(dest, exist_ok=True)

    cfg_src = os.path.join(source, "cfg", f"cfg_output_{dsl_round}.json")
    if not os.path.isfile(cfg_src):
        rel = os.path.relpath(cfg_src, _REPO)
        raise FileNotFoundError(
            f"Missing reference run assets: {rel}\n\n"
            "Export from Vulcan, copy to laptop, unpack at repo root:\n"
            "  bash scripts/export_fixed_cfg_ablation_assets.sh \
"
            "    experiments/pipeline_hf_20260611_151047_run4_2104814 1 run4_dsl1.tar.gz\n"
            "  scp vulcan:~/DSL_Generator/run4_dsl1.tar.gz ~/Desktop/DSL_Generator/\n"
            "  cd ~/Desktop/DSL_Generator && tar -xzf run4_dsl1.tar.gz\n"
            f"Then verify: test -f {rel}"
        )

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

    rewritten = 0
    for sub in ("function_specific_prompts", "functions_generated"):
        rewritten += _rewrite_experiment_paths_in_tree(os.path.join(dest, sub), source, dest)

    grid_count, grid_prefixes = summarize_dsl_grids(dest, dsl_round)
    if grid_count == 0:
        raise FileNotFoundError(
            f"No DSL {dsl_round} grid specs under {dest}/grids — export run-4 assets first "
            f"(scripts/export_fixed_cfg_ablation_assets.sh)."
        )
    print(
        f"[bootstrap] Reused {grid_count} dsl{dsl_round} grid specs "
        f"({len(grid_prefixes)} terminal families); rewrote paths in {rewritten} files"
    )

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
        "grid_spec_count": grid_count,
        "grid_spec_prefixes": grid_prefixes,
        "use_existing_grid_specs": True,
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
    grid_regeneration_attempts: int,
    openai_compat_key_file: str | None,
) -> int:
    configure_run4_grid_env(
        experiment_dir=experiment_dir,
        grid_regeneration_attempts=grid_regeneration_attempts,
    )
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
            "--grid_regeneration_attempts", str(grid_regeneration_attempts),
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
    test_seeds: list[int],
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
        "--test_seeds", *[str(s) for s in test_seeds],
        "--model_type", model_type,
    ]
    if openai_compat_key_file:
        cmd.extend(["--openai_compat_key_file", openai_compat_key_file])
    print(f"\n[test_tasks] {len(tasks)} tasks, dsl_round={dsl_round}, seeds={test_seeds}")
    return subprocess.run(cmd, cwd=_REPO).returncode


def main() -> int:
    _require_funsearch_submodule()
    parser = argparse.ArgumentParser(description="Fixed-CFG terminal-function ablation (local)")
    parser.add_argument(
        "--source",
        default="experiments/pipeline_hf_20260611_151047_run4_2104814",
        help="Reference pipeline run (FunSearch scores read from here; CFG cloned from here)",
    )
    parser.add_argument("--dsl-round", type=int, default=1)
    parser.add_argument(
        "--modes", nargs="+", default=list(ABLATION_MODES), choices=ABLATION_MODES,
        help="Arms to regenerate (funsearch is reference-only from --source)",
    )
    parser.add_argument("--model-type", default=os.environ.get("MODEL_TYPE", "openai_compat"))
    parser.add_argument("--spec-file", default="prompt_specifications/spec_template.txt")
    parser.add_argument("--total-samples", type=int, default=500, help="Match FunSearch total_samples (HF run 4 uses 500)")
    parser.add_argument("--num-ef-iterations", type=int, default=30)
    parser.add_argument("--max-attempts", type=int, default=30)
    parser.add_argument(
        "--grid-regeneration-attempts",
        type=int,
        default=0,
        help="Grid regen during terminal-function eval (0 = reuse run-4 dsl grids only)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML config (sets USE_EXISTING_GRID_SPECS, etc.)",
    )
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--openai-compat-key-file", default=None)
    parser.add_argument("--skip-implement", action="store_true")
    parser.add_argument("--skip-test-tasks", action="store_true")
    parser.add_argument("--dest-root", default="experiments")
    args = parser.parse_args()

    if args.config:
        os.environ["EXPERIMENT_CONFIG"] = args.config
        export_config_to_env(load_config(args.config))
        cfg_tasks = load_config(args.config).get("tasks")
        if not args.tasks and cfg_tasks:
            args.tasks = cfg_tasks

    cfg = load_config(args.config)
    grid_regeneration_attempts = int(
        args.grid_regeneration_attempts
        if args.grid_regeneration_attempts is not None
        else cfg.get("grid_regeneration_attempts", 0)
    )

    tasks = args.tasks or DEFAULT_TASKS
    source = os.path.abspath(args.source)
    test_seeds = load_test_seeds_from_source(source, args.dsl_round)

    print_funsearch_reference(source, args.dsl_round)
    print(
        f"[config] Terminal-function grids: reuse run-4 dsl{args.dsl_round} specs "
        f"(USE_EXISTING_GRID_SPECS=1, grid_regeneration_attempts={grid_regeneration_attempts})"
    )
    print(f"[config] Program-synthesis seeds (from source): {test_seeds}")

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
            run_implement_local(
                experiment_dir=dest,
                spec_file=args.spec_file,
                terminals=terminals,
                mode=mode,
                dsl_round=args.dsl_round,
                model_type=args.model_type,
                total_samples=args.total_samples,
                num_ef_iterations=args.num_ef_iterations,
                grid_regeneration_attempts=grid_regeneration_attempts,
                openai_compat_key_file=args.openai_compat_key_file,
            )

        if not args.skip_test_tasks:
            run_test_tasks_local(
                experiment_dir=dest,
                tasks=tasks,
                dsl_round=args.dsl_round,
                model_type=args.model_type,
                max_attempts=args.max_attempts,
                test_seeds=test_seeds,
                openai_compat_key_file=args.openai_compat_key_file,
            )

        print(f"\n[results] {dest}")
        ts, ss, tt, st = score_experiment(dest, args.dsl_round)
        print(f"  tasks solved: {ts}/{tt or 20}")
        print(f"  seeds solved: {ss}/{st or 200}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
