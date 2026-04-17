#!/usr/bin/env python3
"""Generate plotting artifacts for one experiment or all experiments.

This script backfills missing plotting outputs from existing artifacts:
- FunSearch logs under results/funsearch/*.log
- Explicit feedback logs under explicit_feedback/dsl*/feedback_*.json

Outputs are written under results_tracking/:
- funsearch/
- explicit_feedback/dsl*/
- baseline/dsl*/
"""

import argparse
import os
import re
from typing import Dict, Optional, Tuple

from src.utils.results_tracker import (
    plot_funsearch_reward_vs_interactions,
    plot_explicit_feedback_reward_vs_interactions,
    plot_baseline_reward_vs_interactions,
    best_funsearch_reward,
)


def _extract_safe_name_from_funsearch_log(filename: str) -> str:
    """Best-effort extraction of safe function name from FunSearch log filename."""
    # Typical pattern:
    # huggingface_q2.5_<safe_name>_dsl<d>_func<f>.txt_...log
    match = re.search(r"q2\.5_(.+?)_dsl\d+_func\d+\.txt", filename)
    if match:
        return match.group(1)

    # Fallback: try between q2.5_ and _dsl
    if "q2.5_" in filename and "_dsl" in filename:
        return filename.split("q2.5_", 1)[1].split("_dsl", 1)[0]

    # Last fallback: stem without extension
    return os.path.splitext(filename)[0]


def _extract_dsl_folder_from_funsearch_log(filename: str) -> str:
    """Extract dsl folder name from log file, defaulting to dsl0."""
    match = re.search(r"_dsl(\d+)_func\d+\.txt", filename)
    if match:
        return f"dsl{match.group(1)}"
    return "dsl0"


def _extract_context_from_funsearch_log(filename: str) -> Optional[Tuple[str, int, int]]:
    """Extract (safe_name, dsl_round, func_round) from a FunSearch log filename."""
    match = re.search(r"q2\.5_(.+?)_dsl(\d+)_func(\d+)\.txt", filename)
    if not match:
        return None
    return match.group(1), int(match.group(2)), int(match.group(3))


def _extract_safe_name_from_feedback(filename: str) -> Optional[str]:
    """Extract safe function name from feedback_<safe>_dsl<d>_func<f>.json."""
    match = re.match(r"feedback_(.+?)_dsl\d+_func\d+\.json$", filename)
    if match:
        return match.group(1)
    return None


def _extract_context_from_feedback(filename: str) -> Optional[Tuple[str, int, int]]:
    """Extract (safe_name, dsl_round, func_round) from feedback filename."""
    match = re.match(r"feedback_(.+?)_dsl(\d+)_func(\d+)\.json$", filename)
    if not match:
        return None
    return match.group(1), int(match.group(2)), int(match.group(3))


def _generate_for_experiment(experiment_dir: str) -> Dict[str, int]:
    counts = {
        "funsearch": 0,
        "explicit": 0,
        "baseline": 0,
    }

    # 1) FunSearch plots from logs
    funsearch_logs_dir = os.path.join(experiment_dir, "results", "funsearch")
    if os.path.isdir(funsearch_logs_dir):
        for name in sorted(os.listdir(funsearch_logs_dir)):
            if not name.endswith(".log"):
                continue
            log_path = os.path.join(funsearch_logs_dir, name)
            safe_name = _extract_safe_name_from_funsearch_log(name)
            dsl_folder = _extract_dsl_folder_from_funsearch_log(name)
            funsearch_out_dir = os.path.join(experiment_dir, "results_tracking", "funsearch", dsl_folder)
            os.makedirs(funsearch_out_dir, exist_ok=True)
            out = plot_funsearch_reward_vs_interactions(
                log_file=log_path,
                output_dir=funsearch_out_dir,
                function_name=safe_name,
            )
            if out:
                counts["funsearch"] += 1

    # 2) Explicit and baseline plots from feedback JSONs
    explicit_root = os.path.join(experiment_dir, "explicit_feedback")
    if os.path.isdir(explicit_root):
        for dsl_folder in sorted(os.listdir(explicit_root)):
            dsl_path = os.path.join(explicit_root, dsl_folder)
            if not os.path.isdir(dsl_path):
                continue

            explicit_out_dir = os.path.join(experiment_dir, "results_tracking", "explicit_feedback", dsl_folder)
            baseline_out_dir = os.path.join(experiment_dir, "results_tracking", "baseline", dsl_folder)
            os.makedirs(explicit_out_dir, exist_ok=True)
            os.makedirs(baseline_out_dir, exist_ok=True)

            for name in sorted(os.listdir(dsl_path)):
                if not (name.startswith("feedback_") and name.endswith(".json")):
                    continue

                feedback_ctx = _extract_context_from_feedback(name)
                if feedback_ctx:
                    safe_name, feedback_dsl_round, feedback_func_round = feedback_ctx
                else:
                    safe_name = _extract_safe_name_from_feedback(name)
                    feedback_dsl_round = None
                    feedback_func_round = None

                if not safe_name:
                    continue

                feedback_path = os.path.join(dsl_path, name)

                # Try to find matching FunSearch log for this function.
                funsearch_log = None
                if os.path.isdir(funsearch_logs_dir):
                    for log_name in sorted(os.listdir(funsearch_logs_dir)):
                        if not log_name.endswith(".log"):
                            continue
                        log_ctx = _extract_context_from_funsearch_log(log_name)
                        if log_ctx:
                            log_safe_name, log_dsl_round, log_func_round = log_ctx
                            same_context = (
                                feedback_dsl_round is not None
                                and feedback_func_round is not None
                                and log_dsl_round == feedback_dsl_round
                                and log_func_round == feedback_func_round
                            )
                            if log_safe_name == safe_name and same_context:
                                funsearch_log = os.path.join(funsearch_logs_dir, log_name)
                                break
                            continue

                        # Backward-compatible fallback for older non-versioned names.
                        log_safe_name = _extract_safe_name_from_funsearch_log(log_name)
                        if log_safe_name == safe_name:
                            funsearch_log = os.path.join(funsearch_logs_dir, log_name)
                            break

                if not funsearch_log:
                    raise RuntimeError(
                        "Missing matching FunSearch log for explicit-feedback seed anchor: "
                        f"function={safe_name}, dsl_round={feedback_dsl_round}, func_round={feedback_func_round}, "
                        f"feedback_file={feedback_path}"
                    )

                seed_reward = best_funsearch_reward(funsearch_log)
                if seed_reward is None:
                    raise RuntimeError(
                        "Unable to compute FunSearch seed reward from matching log: "
                        f"function={safe_name}, funsearch_log={funsearch_log}"
                    )

                exp_plot = plot_explicit_feedback_reward_vs_interactions(
                    feedback_file=feedback_path,
                    output_dir=explicit_out_dir,
                    function_name=safe_name,
                    seed_reward=seed_reward,
                )
                if exp_plot:
                    counts["explicit"] += 1

                if funsearch_log and os.path.exists(funsearch_log):
                    baseline_plot = plot_baseline_reward_vs_interactions(
                        funsearch_log_file=funsearch_log,
                        explicit_feedback_file=feedback_path,
                        output_dir=baseline_out_dir,
                        function_name=safe_name,
                    )
                    if baseline_plot:
                        counts["baseline"] += 1

    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate FunSearch/Explicit/Baseline plots")
    parser.add_argument("experiment_dir", nargs="?", default=None, help="Path to experiment directory")
    parser.add_argument("--dsl_round", type=int, default=None, help="Unused (kept for stage compatibility)")
    parser.add_argument("--all", action="store_true", help="Process all experiments under experiments/")
    args = parser.parse_args()

    experiments = []
    if args.all:
        root = "experiments"
        if not os.path.isdir(root):
            print("No experiments directory found.")
            return 1
        for name in sorted(os.listdir(root)):
            path = os.path.join(root, name)
            if os.path.isdir(path):
                experiments.append(path)
    else:
        if not args.experiment_dir:
            print("Provide experiment_dir or use --all")
            return 1
        experiments = [args.experiment_dir]

    total = {"funsearch": 0, "explicit": 0, "baseline": 0}
    for exp in experiments:
        counts = _generate_for_experiment(exp)
        total["funsearch"] += counts["funsearch"]
        total["explicit"] += counts["explicit"]
        total["baseline"] += counts["baseline"]
        print(f"[{exp}] funsearch={counts['funsearch']} explicit={counts['explicit']} baseline={counts['baseline']}")

    print(
        "Generated plots summary: "
        f"funsearch={total['funsearch']} explicit={total['explicit']} baseline={total['baseline']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
