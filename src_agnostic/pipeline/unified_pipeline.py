#!/usr/bin/env python3
"""Domain-agnostic unified pipeline.

Mirrors :mod:`src.pipeline.unified_pipeline` but takes a ``--domain`` flag so
the same orchestration works for any registered
:class:`~domains.base.DomainAdapter`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

EXIT_CODE_DSL_EVOLVED = 100
EXIT_CODE_SUCCESS = 0
EXIT_CODE_FAILURE = 1

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from domains.base import DomainAdapter
from domains.registry import get_adapter, list_domains

from src_agnostic.pipeline.cfg_to_funsearch_pipeline import get_cfg, implement_cfg
from src_agnostic.pipeline.integrated_pipeline import (
    evolve_dsl,
    evolve_functions_with_failing_tasks,
    run_failure_analysis_for_dsl_evolution,
    test_cfg_on_tasks,
)
from src.utils.results_tracker import ResultsTracker

try:
    from vllm import LLM as vLLM
except ImportError:  # pragma: no cover - vLLM optional at import time
    vLLM = None


def _save_checkpoint(experiment_dir: str, data: dict) -> str:
    path = os.path.join(experiment_dir, "checkpoint.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def run_unified_pipeline(
    experiment_dir: str,
    spec_file: str,
    tasks: List[str],
    *,
    adapter: DomainAdapter,
    max_dsl_evolutions: int = 2,
    max_function_evolutions: int = 3,
    skip_cfg_generation: bool = False,
    cfg_output_file: Optional[str] = None,
    max_cfg_retries: int = 10,
    codebase_path: Optional[str] = None,
    max_attempts: int = 1,
    model_type: str = "huggingface",
    shared_vllm=None,
    resume_from_checkpoint: bool = False,
    local: bool = False,
) -> int:
    """Run the unified CFG / FunSearch / evolution loop for ``adapter``."""
    print(f"\n{'='*80}")
    print(f"UNIFIED PIPELINE (domain='{adapter.spec.name}')")
    print(f"{'='*80}")
    print(f"Experiment directory: {experiment_dir}")
    print(f"Tasks to solve: {tasks}")
    print(f"Max DSL evolutions: {max_dsl_evolutions}")
    print(f"Max function evolutions: {max_function_evolutions}")

    start_dsl_round = 0
    checkpoint_type = None
    resume_func_round = 0
    resume_failing_tasks: List[str] = []
    if resume_from_checkpoint:
        checkpoint_path = os.path.join(experiment_dir, "checkpoint.json")
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, "r") as f:
                checkpoint = json.load(f)
            if checkpoint.get("domain_name") and checkpoint["domain_name"] != adapter.spec.name:
                print(
                    f" ERROR: checkpoint was produced for domain "
                    f"'{checkpoint['domain_name']}' but resume requested for '{adapter.spec.name}'."
                )
                return EXIT_CODE_FAILURE
            start_dsl_round = checkpoint.get("dsl_round", 0)
            checkpoint_type = checkpoint.get("checkpoint_type", "dsl_evolution")
            max_dsl_evolutions = checkpoint.get("max_dsl_evolutions", max_dsl_evolutions)
            max_function_evolutions = checkpoint.get("max_function_evolutions", max_function_evolutions)
            spec_file = checkpoint.get("spec_file", spec_file)
            tasks = checkpoint.get("tasks", tasks)
            model_type = checkpoint.get("model_type", model_type)
            skip_cfg_generation = checkpoint.get("skip_cfg_generation", skip_cfg_generation)
            cfg_output_file = checkpoint.get("cfg_output_file", cfg_output_file)
            print(f"Resuming from checkpoint: DSL round {start_dsl_round + 1}/{max_dsl_evolutions}")
            if checkpoint_type == "function_evolution":
                resume_func_round = checkpoint.get("func_round", 0)
                resume_failing_tasks = checkpoint.get("failing_tasks", [])
                print(f"  Checkpoint type: Function Evolution (round {resume_func_round + 1})")
                print(f"  Failing tasks: {resume_failing_tasks}")
            else:
                print("  Checkpoint type: DSL Evolution")
        else:
            print("   Checkpoint file not found, starting from beginning")

    if shared_vllm is None:
        if model_type == "openai_compat":
            from src.pipeline.explicit_feedback_generation import OpenAICompatLLMWrapper

            key_file = os.environ.get("OPENAI_COMPAT_KEY_FILE", "").strip() or None
            print("\n[Setup] Initializing OpenAI-compat LLM wrapper...")
            shared_vllm = OpenAICompatLLMWrapper(key_file)
            print(" OpenAI-compat LLM wrapper created")
        elif vLLM is not None:
            print("\n[Setup] Initializing shared vLLM instance...")
            shared_vllm = vLLM(model="/scratch/avani/gpt", tensor_parallel_size=4)
            print(" Shared vLLM instance created")
        else:
            raise RuntimeError(
                "No LLM backend available: vLLM not importable and "
                "model_type is not 'openai_compat'."
            )

    os.makedirs(experiment_dir, exist_ok=True)
    for subdir in (
        "function_specific_prompts",
        "functions_generated",
        os.path.join("results", "funsearch"),
        "cfg",
        "final_functions",
        "domain_assets",
    ):
        os.makedirs(os.path.join(experiment_dir, subdir), exist_ok=True)

    adapter.write_nld_file(
        os.path.join(experiment_dir, "domain_assets", f"nld_{adapter.spec.name}.txt")
    )
    adapter.write_domain_context_file(
        os.path.join(experiment_dir, "domain_assets", f"domain_context_{adapter.spec.name}.txt")
    )

    results_tracker = ResultsTracker(experiment_dir)

    cfg, terminals, example, success = get_cfg(
        adapter=adapter,
        experiment_dir=experiment_dir,
        skip_cfg_generation=skip_cfg_generation,
        cfg_output_file=cfg_output_file,
        max_cfg_retries=max_cfg_retries,
        shared_vllm=shared_vllm,
    )
    if not success or not cfg or not terminals:
        print(" Failed to get valid CFG. Cannot proceed.")
        return EXIT_CODE_FAILURE

    print(f"\n Got CFG with {len(terminals)} terminal functions")

    for dsl_round in range(start_dsl_round, max_dsl_evolutions):
        print(f"\n{'='*80}")
        print(f"DSL Evolution Round {dsl_round + 1}/{max_dsl_evolutions}")
        print(f"{'='*80}")

        _save_checkpoint(
            experiment_dir,
            {
                "domain_name": adapter.spec.name,
                "dsl_round": dsl_round,
                "func_round": 0,
                "max_dsl_evolutions": max_dsl_evolutions,
                "max_function_evolutions": max_function_evolutions,
                "spec_file": spec_file,
                "tasks": tasks,
                "model_type": model_type,
                "skip_cfg_generation": True,
                "cfg_output_file": os.path.join(experiment_dir, "cfg", "cfg_output.json"),
                "failing_tasks": [],
                "checkpoint_type": "dsl_evolution",
            },
        )

        skip_implementation = False
        if (
            resume_from_checkpoint
            and checkpoint_type == "function_evolution"
            and resume_failing_tasks
            and dsl_round == start_dsl_round
        ):
            print("\n[Resuming] Skipping CFG implementation - resuming from function evolution")
            failing_tasks = resume_failing_tasks
            all_solved = False
            skip_implementation = True
            resume_from_checkpoint = False

        if not skip_implementation:
            if results_tracker is not None:
                results_tracker.current_evolution_interactions = {
                    "funsearch": 0,
                    "explicit_feedback": 0,
                    "program_synthesis": 0,
                }

            print("\n[Step 2a] Implementing CFG...")
            implementation_success, _final_functions = implement_cfg(
                cfg=cfg,
                terminals=terminals,
                example=example,
                spec_file=spec_file,
                experiment_dir=experiment_dir,
                adapter=adapter,
                model_type=model_type,
                shared_vllm=shared_vllm,
                results_tracker=results_tracker,
                dsl_round=dsl_round,
                func_evolution_round=None,
                codebase_path=codebase_path,
            )
            if not implementation_success:
                print(" CFG implementation failed. Stopping pipeline.")
                return EXIT_CODE_FAILURE

            print("\n[Step 2b] Testing CFG on tasks...")
            task_results = test_cfg_on_tasks(
                experiment_dir=experiment_dir,
                tasks=tasks,
                cfg=cfg,
                terminals=terminals,
                adapter=adapter,
                max_attempts=max_attempts,
                shared_vllm=shared_vllm,
                results_tracker=results_tracker,
                cfg_version=dsl_round,
                func_evolution_round=None,
                model_type=model_type,
            )

            all_solved = all(task_results.values())
            failing_tasks = [t for t, ok in task_results.items() if not ok]

            if results_tracker is not None:
                rewards_per_task = {}
                for task in tasks:
                    evolution_results = [
                        r
                        for r in results_tracker.get_task_results(task)
                        if r.get("func_evolution_round") is None and r.get("cfg_version") == dsl_round
                    ]
                    rewards_per_task[task] = max((r["reward"] for r in evolution_results), default=0.0)
                results_tracker.save_evolution_metrics(
                    dsl_round=dsl_round,
                    func_evolution_round=None,
                    steps_in_evolution=results_tracker.current_evolution_interactions.copy(),
                    rewards_per_task=rewards_per_task,
                )

            results_tracker.plot_reward_vs_interactions(dsl_round=dsl_round, func_evolution_round=None)
            results_tracker.plot_all_tasks_combined(dsl_round=dsl_round, func_evolution_round=None)
            results_tracker.plot_tasks_separately_from_metrics(dsl_round=dsl_round, func_evolution_round=None)

            if all_solved:
                print(f"\n{'='*80}\n ALL TASKS SOLVED!\n{'='*80}")
                checkpoint_path = os.path.join(experiment_dir, "checkpoint.json")
                if os.path.exists(checkpoint_path):
                    os.remove(checkpoint_path)
                return EXIT_CODE_SUCCESS

            print(f"\n   {len(failing_tasks)}/{len(tasks)} tasks failed: {failing_tasks}")

        start_func_round = 0
        if resume_from_checkpoint:
            checkpoint_path = os.path.join(experiment_dir, "checkpoint.json")
            if os.path.exists(checkpoint_path):
                with open(checkpoint_path, "r") as f:
                    checkpoint = json.load(f)
                start_func_round = checkpoint.get("func_round", 0)

        for func_round in range(start_func_round, max_function_evolutions):
            print(f"\n  Function Evolution Round {func_round + 1}/{max_function_evolutions}")

            if results_tracker is not None:
                results_tracker.current_evolution_interactions = {
                    "funsearch": 0,
                    "explicit_feedback": 0,
                    "program_synthesis": 0,
                }

            _save_checkpoint(
                experiment_dir,
                {
                    "domain_name": adapter.spec.name,
                    "dsl_round": dsl_round,
                    "func_round": func_round,
                    "max_dsl_evolutions": max_dsl_evolutions,
                    "max_function_evolutions": max_function_evolutions,
                    "spec_file": spec_file,
                    "tasks": tasks,
                    "model_type": model_type,
                    "skip_cfg_generation": True,
                    "cfg_output_file": os.path.join(experiment_dir, "cfg", "cfg_output.json"),
                    "checkpoint_type": "function_evolution",
                    "failing_tasks": failing_tasks,
                },
            )

            specification = ""
            if os.path.exists(spec_file):
                with open(spec_file, "r") as f:
                    specification = f.read()

            evolved = evolve_functions_with_failing_tasks(
                experiment_dir=experiment_dir,
                failing_tasks=failing_tasks,
                terminals=terminals,
                specification=specification,
                adapter=adapter,
                spec_file=spec_file,
                cfg=cfg,
                shared_vllm=shared_vllm,
                dsl_round=dsl_round,
                func_evolution_round=func_round,
            )
            if not evolved:
                print("   Function evolution failed or produced no results")
                break

            task_results = test_cfg_on_tasks(
                experiment_dir=experiment_dir,
                tasks=failing_tasks,
                cfg=cfg,
                terminals=terminals,
                adapter=adapter,
                max_attempts=max_attempts,
                shared_vllm=shared_vllm,
                results_tracker=results_tracker,
                cfg_version=dsl_round,
                func_evolution_round=func_round,
                model_type=model_type,
            )

            all_solved = all(task_results.values())
            failing_tasks = [t for t, ok in task_results.items() if not ok]

            if results_tracker is not None:
                rewards_per_task = {}
                for task in failing_tasks:
                    evolution_results = [
                        r
                        for r in results_tracker.get_task_results(task)
                        if r.get("func_evolution_round") == func_round and r.get("cfg_version") == dsl_round
                    ]
                    rewards_per_task[task] = max((r["reward"] for r in evolution_results), default=0.0)
                results_tracker.save_evolution_metrics(
                    dsl_round=dsl_round,
                    func_evolution_round=func_round,
                    steps_in_evolution=results_tracker.current_evolution_interactions.copy(),
                    rewards_per_task=rewards_per_task,
                )

            results_tracker.plot_reward_vs_interactions(dsl_round=dsl_round, func_evolution_round=func_round)
            results_tracker.plot_all_tasks_combined(dsl_round=dsl_round, func_evolution_round=func_round)
            results_tracker.plot_tasks_separately_from_metrics(dsl_round=dsl_round, func_evolution_round=func_round)

            if all_solved:
                print(f"\n  All tasks solved after function evolution round {func_round + 1}!")
                return EXIT_CODE_SUCCESS

        if failing_tasks:
            print("\n  DSL Evolution (tasks still failing)")
            max_dsl_retries = 10
            dsl_success = False
            new_cfg = cfg
            new_terminals = terminals

            failure_analysis_cached = run_failure_analysis_for_dsl_evolution(
                experiment_dir=experiment_dir,
                failing_tasks=failing_tasks,
                cfg=cfg,
                terminals=terminals,
                failed_programs_by_task=None,
                adapter=adapter,
                shared_vllm=shared_vllm,
                model_type=model_type,
            )

            for dsl_attempt in range(1, max_dsl_retries + 1):
                new_cfg, new_terminals, attempt_success = evolve_dsl(
                    experiment_dir=experiment_dir,
                    failing_tasks=failing_tasks,
                    cfg=cfg,
                    terminals=terminals,
                    failure_analysis=failure_analysis_cached,
                    adapter=adapter,
                    shared_vllm=shared_vllm,
                    model_type=model_type,
                )
                if attempt_success and new_cfg != cfg:
                    dsl_success = True
                    break

            if dsl_success and new_cfg != cfg:
                cfg = new_cfg
                terminals = new_terminals
                cfg_path = os.path.join(experiment_dir, "cfg", "cfg_output.json")
                if os.path.exists(cfg_path):
                    with open(cfg_path, "r") as f:
                        cfg_data = json.load(f)
                        example = cfg_data.get("example", None)
                _save_checkpoint(
                    experiment_dir,
                    {
                        "domain_name": adapter.spec.name,
                        "dsl_round": dsl_round + 1,
                        "func_round": 0,
                        "max_dsl_evolutions": max_dsl_evolutions,
                        "max_function_evolutions": max_function_evolutions,
                        "spec_file": spec_file,
                        "tasks": tasks,
                        "model_type": model_type,
                        "skip_cfg_generation": True,
                        "cfg_output_file": cfg_path,
                        "checkpoint_type": "dsl_evolution",
                        "failing_tasks": [],
                    },
                )
                if dsl_round < max_dsl_evolutions - 1 and not local:
                    print(f"\n  DSL EVOLVED — exit code {EXIT_CODE_DSL_EVOLVED} for resubmission")
                    return EXIT_CODE_DSL_EVOLVED
            else:
                print(f"\n   DSL evolution failed after {max_dsl_retries} attempts")
        else:
            return EXIT_CODE_SUCCESS

    print(f"\n{'='*80}\n PIPELINE COMPLETED WITHOUT SOLVING ALL TASKS\n{'='*80}")
    return EXIT_CODE_FAILURE


def _load_tasks_argument(tasks_args: List[str]) -> List[str]:
    if len(tasks_args) == 1 and tasks_args[0].endswith(".json"):
        tasks_file = tasks_args[0]
        if not os.path.exists(tasks_file):
            raise FileNotFoundError(f"Tasks file not found: {tasks_file}")
        with open(tasks_file, "r") as f:
            cfg = json.load(f)
            return cfg.get("tasks", [])
    return tasks_args


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Domain-agnostic unified pipeline: Get CFG, implement, test, evolve."
    )
    parser.add_argument("--domain", type=str, required=True, choices=list_domains(), help="Domain plugin name")
    parser.add_argument("--experiment_dir", type=str, required=True)
    parser.add_argument("--spec_file", type=str, required=True)
    parser.add_argument("--tasks", type=str, nargs="+", required=True)
    parser.add_argument("--max_dsl_evolutions", type=int, default=3)
    parser.add_argument("--max_function_evolutions", type=int, default=3)
    parser.add_argument("--skip_cfg_generation", action="store_true")
    parser.add_argument("--cfg_output_file", type=str, default=None)
    parser.add_argument("--max_cfg_retries", type=int, default=10)
    parser.add_argument("--codebase_path", type=str, default=None)
    parser.add_argument("--max_attempts", type=int, default=1)
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["huggingface", "ollama", "gemini", "openai_compat"],
        default="huggingface",
    )
    parser.add_argument("--resume_from_checkpoint", action="store_true")
    parser.add_argument("--local", action="store_true")
    # Back-compat: accept the craft-only flags but use them only when the
    # selected adapter recognizes them.
    parser.add_argument("--recipes_path", type=str, default=None)
    parser.add_argument("--hints_path", type=str, default=None)
    parser.add_argument("--nld_path", type=str, default=None)

    args = parser.parse_args()
    tasks = _load_tasks_argument(args.tasks)

    adapter_kwargs = {}
    if args.domain == "craft":
        if args.recipes_path:
            adapter_kwargs["recipes_path"] = args.recipes_path
        if args.hints_path:
            adapter_kwargs["hints_path"] = args.hints_path
        if args.nld_path:
            adapter_kwargs["nld_path"] = args.nld_path
    elif args.domain == "crafter":
        if args.nld_path:
            adapter_kwargs["nld_path"] = args.nld_path

    adapter = get_adapter(args.domain, **adapter_kwargs)
    return run_unified_pipeline(
        experiment_dir=args.experiment_dir,
        spec_file=args.spec_file,
        tasks=tasks,
        adapter=adapter,
        max_dsl_evolutions=args.max_dsl_evolutions,
        max_function_evolutions=args.max_function_evolutions,
        skip_cfg_generation=args.skip_cfg_generation,
        cfg_output_file=args.cfg_output_file,
        max_cfg_retries=args.max_cfg_retries,
        codebase_path=args.codebase_path,
        max_attempts=args.max_attempts,
        model_type=args.model_type,
        resume_from_checkpoint=args.resume_from_checkpoint,
        local=args.local,
    )


if __name__ == "__main__":
    sys.exit(main())
