#!/usr/bin/env python3
"""
Aggregate per-task Stage 5 outputs into a consolidated test_tasks status,
then generate plots and update pipeline state once.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from typing import Dict, List, Tuple

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _project_root)

from src.utils.pipeline_state import (
    mark_dsl_evolution_submitted,
    read_state,
    resolve_model_type_for_chained_jobs,
    update_state,
)
from src.utils.per_task_test_paths import (
    legacy_per_task_test_results_tracking_dir,
    program_synthesis_task_shard_dir,
    refactor_per_task_results_tracking_dir,
)
from src.utils.results_tracker import ResultsTracker
from src.utils.status_manager import read_status, write_status


def _safe_task_token(task: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(task)).strip("_")
    return token or "task"


def _parse_tasks(raw_tasks: List[str]) -> List[str]:
    tasks = raw_tasks
    if len(tasks) == 1 and tasks[0].endswith('.json') and os.path.exists(tasks[0]):
        with open(tasks[0], 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        tasks = cfg.get('tasks', [])
    elif len(tasks) == 1 and tasks[0].startswith('['):
        try:
            tasks = json.loads(tasks[0])
        except Exception:
            pass
    elif len(tasks) == 1 and ' ' in tasks[0]:
        tasks = tasks[0].split()
    return list(dict.fromkeys(tasks))


def _status_file_for_task(experiment_dir: str, dsl_round: int, task: str) -> str:
    return os.path.join(
        experiment_dir,
        'status',
        f'dsl{dsl_round}',
        'test_tasks_tasks',
        f'{_safe_task_token(task)}.json',
    )


def _collect_task_statuses(
    experiment_dir: str,
    dsl_round: int,
    func_evolution_round: int,
    tasks: List[str],
) -> Tuple[Dict[str, bool], List[str]]:
    task_results: Dict[str, bool] = {}
    missing: List[str] = []

    for task in tasks:
        status_file = _status_file_for_task(experiment_dir, dsl_round, task)
        if not os.path.exists(status_file):
            missing.append(task)
            continue

        try:
            with open(status_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            missing.append(task)
            continue

        status = data.get('status', '')
        status_dsl = int(data.get('dsl_round', -1))
        status_func = data.get('func_evolution_round')
        status_func = 0 if status_func is None else int(status_func)

        if status != 'completed' or status_dsl != int(dsl_round) or status_func != int(func_evolution_round):
            missing.append(task)
            continue

        task_success = bool(data.get('success', False))
        task_results[task] = task_success

    return task_results, missing


def _read_json_if_exists(path: str, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return default


def _per_task_shard_results_dir(
    experiment_dir: str,
    dsl_round: int,
    func_evolution_round: int,
    task: str,
) -> str:
    """Resolve shard directory: new dsl/func/prog_synthoutput layout, then older paths."""
    token = _safe_task_token(task)
    candidates = [
        program_synthesis_task_shard_dir(
            experiment_dir,
            dsl_round=dsl_round,
            func_evolution_round=func_evolution_round,
            task_token=token,
        ),
        refactor_per_task_results_tracking_dir(
            experiment_dir,
            dsl_round=dsl_round,
            func_evolution_round=func_evolution_round,
            task_token=token,
        ),
        legacy_per_task_test_results_tracking_dir(
            experiment_dir,
            dsl_round=dsl_round,
            func_evolution_round=func_evolution_round,
            task_token=token,
        ),
    ]
    marker = "synthesis_results.json"
    for d in candidates:
        if os.path.isfile(os.path.join(d, marker)):
            return d
    return candidates[0]


def _dsl_evolution_already_recorded_success(experiment_dir: str, dsl_round_from: int) -> bool:
    st = read_status(experiment_dir, int(dsl_round_from), "evolve_dsl")
    if not isinstance(st, dict):
        return False
    return st.get("status") == "completed" and st.get("evolved") is True


def _submit_dsl_evolution_job(args, failing_tasks: List[str]) -> bool:
    if not mark_dsl_evolution_submitted(args.experiment_dir):
        print("[Chaining] DSL evolution already marked submitted; not submitting duplicate job.")
        return True

    resolved_model_type = resolve_model_type_for_chained_jobs(args.experiment_dir, args.model_type)
    experiment_name = os.path.basename(os.path.normpath(args.experiment_dir))
    log_dir = os.path.join(_project_root, "scripts", "log", experiment_name)
    os.makedirs(log_dir, exist_ok=True)

    using_api = resolved_model_type == "openai_compat"
    cpus = os.environ.get("DSL_EVOLUTION_CPUS", "4" if using_api else "32")
    mem = os.environ.get("DSL_EVOLUTION_MEM", "32G" if using_api else "256G")
    walltime = os.environ.get("DSL_EVOLUTION_TIME", "10:00:00")
    gres = os.environ.get("DSL_EVOLUTION_GRES", "" if using_api else "gpu:4").strip()
    account = os.environ.get("SLURM_ACCOUNT", os.environ.get("SBATCH_ACCOUNT", "aip-lelis")).strip()

    env_pairs = [
        f"EXPERIMENT_DIR={args.experiment_dir}",
        f"FAILING_TASKS={' '.join(failing_tasks)}",
        f"RECIPES_PATH={args.recipes_path}",
        f"MAX_RETRIES={int(args.dsl_max_retries)}",
        f"DSL_VERSION={int(args.dsl_round)}",
        f"MAX_FAILED_PROGRAMS={int(args.max_failed_programs)}",
        f"MODEL_TYPE={resolved_model_type}",
    ]
    if args.openai_compat_key_file:
        env_pairs.append(f"OPENAI_COMPAT_KEY_FILE={args.openai_compat_key_file}")
    experiment_config = os.environ.get("EXPERIMENT_CONFIG", "").strip()
    if experiment_config:
        env_pairs.append(f"EXPERIMENT_CONFIG={experiment_config}")
    export_str = ",".join(["ALL", *env_pairs])

    slurm_script = os.path.join(_project_root, "scripts", "stages", "stage_evolve_dsl.slurm")
    submit_cmd = [
        "sbatch",
        "--parsable",
        "--job-name",
        f"{experiment_name[:20]}_evolve_dsl",
        "--output",
        os.path.join(log_dir, "stage_evolve_dsl_%j.out"),
        "--error",
        os.path.join(log_dir, "stage_evolve_dsl_%j.err"),
        "--time",
        walltime,
        "--cpus-per-task",
        cpus,
        "--mem",
        mem,
        "--export",
        export_str,
    ]
    if gres:
        submit_cmd.extend(["--gres", gres])
    if account:
        submit_cmd.extend(["--account", account])
    submit_cmd.append(slurm_script)

    result = subprocess.run(
        submit_cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        print(f"[Chaining] Submitted DSL evolution job: {result.stdout.strip()}")
        return True

    update_state(args.experiment_dir, dsl_evolution_submitted=0)
    print(f"[Chaining] Failed to submit DSL evolution job: {result.stderr.strip()}", file=sys.stderr)
    return False


def _merge_parallel_results(
    experiment_dir: str,
    dsl_round: int,
    func_evolution_round: int,
    tasks: List[str],
) -> None:
    func_round = int(func_evolution_round)

    incoming_results = []
    incoming_seed_outcomes = []
    for task in tasks:
        task_dir = _per_task_shard_results_dir(
            experiment_dir, dsl_round, func_round, task
        )
        results_path = os.path.join(task_dir, 'synthesis_results.json')
        seed_outcomes_path = os.path.join(task_dir, 'program_synthesis_seed_outcomes.jsonl')

        task_results = _read_json_if_exists(results_path, [])
        if isinstance(task_results, list):
            incoming_results.extend(task_results)
        if os.path.exists(seed_outcomes_path):
            try:
                with open(seed_outcomes_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        incoming_seed_outcomes.append(json.loads(line))
            except Exception:
                pass

    tracker = ResultsTracker(experiment_dir)

    # Keep only the latest parallel task run outputs for this round.
    seen = set()

    def _key(entry: dict):
        return (
            entry.get('task'),
            entry.get('cfg_version'),
            entry.get('func_evolution_round'),
            entry.get('seed'),
            entry.get('attempt_in_seed'),
            entry.get('timestamp'),
            entry.get('program'),
        )

    merged = []
    total_steps = 0
    for item in incoming_results:
        if not isinstance(item, dict):
            continue
        key = _key(item)
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
        total_steps += int(item.get('steps', 0) or 0)

    tracker.results = merged
    tracker.interactions = dict(tracker.interactions) if isinstance(tracker.interactions, dict) else {}
    for k in ('funsearch', 'explicit_feedback', 'program_synthesis'):
        tracker.interactions.setdefault(k, 0)
    tracker.interactions['program_synthesis'] = int(total_steps)
    tracker._save_results()
    tracker._save_interactions()

    # Overwrite consolidated seed outcomes with latest per-task outputs.
    seed_outcome_path = os.path.join(experiment_dir, 'results_tracking', 'program_synthesis_seed_outcomes.jsonl')
    os.makedirs(os.path.dirname(seed_outcome_path), exist_ok=True)
    with open(seed_outcome_path, 'w', encoding='utf-8') as f:
        for entry in incoming_seed_outcomes:
            f.write(json.dumps(entry) + '\n')

    print(f"[Aggregation] Wrote {len(merged)} synthesis entries from latest parallel task runs")


def main() -> int:
    parser = argparse.ArgumentParser(description='Aggregate Stage 5 per-task test jobs')
    parser.add_argument('--experiment_dir', type=str, required=True, help='Experiment directory')
    parser.add_argument('--tasks', type=str, nargs='+', required=True, help='List of tasks to aggregate')
    parser.add_argument('--dsl_round', type=int, default=0, help='DSL round')
    parser.add_argument('--func_evolution_round', type=int, default=0, help='Function evolution round')
    parser.add_argument('--recipes_path', type=str, default='craft/resources/recipes.yaml', help='Path to recipes YAML for chained DSL evolution')
    parser.add_argument('--dsl_max_retries', type=int, default=int(os.environ.get('DSL_EVOLUTION_MAX_RETRIES', '10')), help='Maximum retries for chained DSL evolution')
    parser.add_argument('--max_failed_programs', type=int, default=int(os.environ.get('MAX_FAILED_PROGRAMS_FOR_DSL', '30')), help='Maximum failed programs per task for DSL evolution context')
    parser.add_argument('--model_type', type=str, default=os.environ.get('MODEL_TYPE', '').strip() or None, choices=['huggingface', 'ollama', 'gemini', 'openai_compat'], help='LLM backend for chained DSL evolution')
    parser.add_argument('--openai_compat_key_file', type=str, default=os.environ.get('OPENAI_COMPAT_KEY_FILE', '').strip() or None, help='Key file for OpenAI-compatible API when chaining DSL evolution')
    args = parser.parse_args()

    tasks = _parse_tasks(args.tasks)
    if not tasks:
        print('No tasks provided for aggregation', file=sys.stderr)
        return 1

    task_results, missing = _collect_task_statuses(
        args.experiment_dir,
        int(args.dsl_round),
        int(args.func_evolution_round),
        tasks,
    )

    if missing:
        print(f"Waiting for per-task statuses; missing/incomplete: {missing}")
        return 1

    _merge_parallel_results(
        args.experiment_dir,
        int(args.dsl_round),
        int(args.func_evolution_round),
        tasks,
    )

    all_solved = all(task_results.values())
    failing_tasks = [task for task, ok in task_results.items() if not ok]

    stage_status = {
        'stage': 'test_tasks',
        'status': 'completed',
        'mode': 'aggregated_from_single_task_jobs',
        'dsl_round': int(args.dsl_round),
        'func_evolution_round': int(args.func_evolution_round),
        'task_results': task_results,
        'all_solved': all_solved,
        'failing_tasks': failing_tasks,
    }
    write_status(args.experiment_dir, int(args.dsl_round), 'test_tasks', stage_status)

    tracker = ResultsTracker(args.experiment_dir)
    print('[Generating Plots] Creating plots from aggregated results...')
    try:
        if tracker.results:
            tracker.plot_reward_vs_interactions(
                dsl_round=int(args.dsl_round),
                func_evolution_round=int(args.func_evolution_round),
            )
            tracker.plot_all_tasks_combined(
                dsl_round=int(args.dsl_round),
                func_evolution_round=int(args.func_evolution_round),
            )
            print('Plots generated successfully')
        else:
            print('No results found for plotting')
    except Exception as e:
        print(f'Warning: Could not generate plots: {e}')

    state = read_state(args.experiment_dir)
    function_impl_total = state.get('function_implementation_total', 0)
    max_function_evolutions = int(state.get('max_function_evolutions', 1) or 0)
    dsl_evolutions_remaining = int(state.get('dsl_evolutions_remaining', 0) or 0)

    if all_solved:
        update_state(
            args.experiment_dir,
            test_tasks_submitted=0,
            function_implementation_remaining=0,
            phase='complete',
            test_tasks_aggregate_submitted=0,
        )
        print('ALL TASKS SOLVED! Pipeline complete.')
    else:
        print(f"{len(failing_tasks)}/{len(tasks)} tasks failed: {failing_tasks}")
        # With unified_pipeline semantics, zero function-evolution rounds means: skip
        # straight to DSL evolution on failure. Do not re-arm all per-function work.
        if max_function_evolutions == 0 and dsl_evolutions_remaining > 0:
            if _dsl_evolution_already_recorded_success(args.experiment_dir, int(args.dsl_round)):
                print(
                    '[Chaining] evolve_dsl status already completed for this dsl_round; '
                    'not updating pipeline_state (idempotent re-aggregate).'
                )
                return 0
            # test_tasks_submitted=0: aggregate finished this test wave; do not mimic the
            # function-evolution branch (which uses test_tasks_submitted=1 to arm per-function work).
            update_state(
                args.experiment_dir,
                test_tasks_submitted=0,
                function_implementation_remaining=0,
                phase='dsl_evolution',
                test_tasks_aggregate_submitted=0,
            )
            print(
                'max_function_evolutions=0: prepared state for DSL / CFG evolution '
                f'(dsl_evolutions_remaining={dsl_evolutions_remaining}).'
            )
            if not _submit_dsl_evolution_job(args, failing_tasks):
                return 1
        elif max_function_evolutions == 0 and dsl_evolutions_remaining <= 0:
            update_state(
                args.experiment_dir,
                test_tasks_submitted=0,
                function_implementation_remaining=0,
                phase='complete',
                test_tasks_aggregate_submitted=0,
            )
            print(
                'max_function_evolutions=0 and dsl_evolutions_remaining exhausted; '
                'not scheduling further DSL evolution.'
            )
        else:
            update_state(
                args.experiment_dir,
                test_tasks_submitted=1,
                function_implementation_remaining=function_impl_total,
                test_tasks_aggregate_submitted=0,
            )
            print('Prepared state for function evolution chaining.')

    return 0


if __name__ == '__main__':
    sys.exit(main())
