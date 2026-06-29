"""Load formatted failed synthesis attempts from persisted program-synthesis logs."""

from __future__ import annotations

import json
import os
from typing import Dict, List


def _func_evolution_round(result: dict) -> int:
    """Return func_evolution_round as int; JSON null and missing keys become 0."""
    value = result.get("func_evolution_round")
    if value is None:
        return 0
    return int(value)


def extract_failed_programs_from_synthesis_results(
    experiment_dir: str,
    failing_tasks: List[str],
    dsl_version: int = 0,
    max_programs_per_task: int = 30,
) -> Dict[str, List[str]]:
    """Extract failed programs for failing tasks from ``synthesis_results.json``.

    Uses only results from:
    - the current DSL version being evolved (cfg_version == dsl_version)
    - the latest function evolution round observed for that DSL version

    Caps programs per task to ``max_programs_per_task`` (most recent unique entries).

    Expected path: ``<experiment_dir>/results_tracking/synthesis_results.json``.
    """
    synthesis_results_path = os.path.join(experiment_dir, "results_tracking", "synthesis_results.json")

    if not os.path.exists(synthesis_results_path):
        return {}

    with open(synthesis_results_path, "r", encoding="utf-8") as f:
        synthesis_results = json.load(f)

    if not isinstance(synthesis_results, list):
        return {}

    source_cfg_version = dsl_version

    relevant_results = [
        r for r in synthesis_results
        if isinstance(r, dict)
        and r.get("cfg_version", 0) == source_cfg_version
        and r.get("task") in failing_tasks
    ]

    if not relevant_results:
        return {}

    latest_func_round = max(_func_evolution_round(r) for r in relevant_results)

    latest_round_results = [
        r for r in relevant_results
        if _func_evolution_round(r) == latest_func_round
    ]

    failed_programs_by_task: Dict[str, List[str]] = {}

    for task in failing_tasks:
        failed_programs = []
        seen_program_keys = set()
        duplicate_skips = 0

        task_failed_results = [
            r for r in latest_round_results
            if r.get("task") == task and not r.get("success", False)
        ]

        for result in reversed(task_failed_results):
            program = result.get("program", "")
            program_key = " ".join(str(program).split()).strip().lower()
            if program_key in seen_program_keys:
                duplicate_skips += 1
                continue
            seen_program_keys.add(program_key)

            failure_reason = result.get("failure_reason", "Unknown")

            inventory_before = result.get("inventory_before", {})
            inventory_after = result.get("inventory_after", {})
            inventory_trace = result.get("inventory_trace", [])

            lines = [f"Program:\n{program}"]

            if inventory_trace:
                lines.append(
                    "Inventory changes during the program (whole inventory after the function where the change happened):"
                )
                for entry in inventory_trace:
                    token = entry.get("token", "?")
                    inv = entry.get("inventory", [])
                    inv_str = ", ".join(inv) if inv else "<empty>"
                    lines.append(f"  {token} -> {inv_str}")
            elif inventory_before or inventory_after:
                lines.append(f"Inventory before: {inventory_before}")
                lines.append(f"Inventory after: {inventory_after}")

            if failure_reason and str(failure_reason).strip().lower() != "unknown":
                lines.append(f"Failure: {failure_reason}")
            failed_programs.append("\n".join(lines))

            if max_programs_per_task > 0 and len(failed_programs) >= max_programs_per_task:
                break

        failed_programs.reverse()

        if failed_programs:
            failed_programs_by_task[task] = failed_programs
            print(
                f"[synthesis log] {len(failed_programs)} failed programs for task {task!r} "
                f"(cap={max_programs_per_task}, cfg_version={source_cfg_version}, "
                f"func_evolution_round={latest_func_round}, duplicates_skipped={duplicate_skips})"
            )

    return failed_programs_by_task
