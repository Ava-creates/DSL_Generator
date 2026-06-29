"""Paths for parallel per-task program synthesis (Stage 5) shards.

Layout (current)::

    <experiment>/results_tracking/dsl{n}/func{m}/prog_synthoutput/<task_token>/
        synthesis_results.json
        interactions.json
        program_synthesis_seed_outcomes.jsonl
        program_synthesis_seed_outcomes_by_task/...

Older layouts are still resolved during aggregation (see ``stage_test_tasks_aggregate``).
"""

from __future__ import annotations

import os


def program_synthesis_task_shard_dir(
    experiment_dir: str,
    *,
    dsl_round: int,
    func_evolution_round: int,
    task_token: str,
    tasks_subdir: str = "tasks",
) -> str:
    """Directory for one parallel test_tasks job's synthesis artifacts (flat, no inner ``results_tracking``)."""
    return os.path.join(
        experiment_dir,
        "results_tracking",
        f"dsl{int(dsl_round)}",
        f"func{int(func_evolution_round)}",
        tasks_subdir,
        task_token,
    )


def refactor_per_task_results_tracking_dir(
    experiment_dir: str,
    *,
    dsl_round: int,
    func_evolution_round: int,
    task_token: str,
) -> str:
    """Intermediate refactor path: ``results_tracking/per_task/test_tasks/.../results_tracking``."""
    return os.path.join(
        experiment_dir,
        "results_tracking",
        "per_task",
        "test_tasks",
        f"dsl{int(dsl_round)}",
        f"func{int(func_evolution_round)}",
        task_token,
        "results_tracking",
    )


def legacy_per_task_test_results_tracking_dir(
    experiment_dir: str,
    *,
    dsl_round: int,
    func_evolution_round: int,
    task_token: str,
) -> str:
    """Original layout: ``task_runs/test_tasks/.../results_tracking``."""
    return os.path.join(
        experiment_dir,
        "task_runs",
        "test_tasks",
        f"dsl{int(dsl_round)}",
        f"func{int(func_evolution_round)}",
        task_token,
        "results_tracking",
    )
