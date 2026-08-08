"""LLM-only terminal function generation for ablation vs FunSearch.

Modes:
- llm_best_of_n: identical prompt on every sample; no cross-sample memory.
- llm_chained: each iteration appends the two most recent function bodies and their eval scores.

Both modes evaluate candidates with the same FunSearch Evaluator, write JSONL logs
compatible with explicit_feedback_generation.parse_log_file, and are followed by
the same explicit-feedback stage.
"""

from __future__ import annotations

import dataclasses
import os
from datetime import datetime
from typing import Any, Optional, Sequence, Tuple

from funsearch.implementation import code_manipulation, config as config_lib, evaluator, programs_database, sampler
from funsearch.implementation.funsearch import (
    FunSearch,
    _extract_function_names,
    _format_init_check_failure_for_grid_prompt,
)
from src.pipeline.explicit_feedback_generation import get_end_score
from src.utils.config_loader import funsearch_grid_regen_kwargs_from_config, load_config


def _model_label(model_type: str) -> str:
    return "vllm" if model_type in ("huggingface", "vllm") else model_type


def _build_log_path(
    *,
    results_dir: str,
    model_type: str,
    func_file: str,
    func_init_file: str,
    spec_file: str,
    mode: str,
) -> str:
    os.makedirs(results_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_func = os.path.basename(func_file).replace("/", ":").replace("\\", ":")
    safe_init = os.path.basename(func_init_file).replace("/", ":").replace("\\", ":")
    safe_spec = os.path.basename(spec_file or "specification").replace("/", "").replace("\\", "")
    label = _model_label(model_type)
    return os.path.join(
        results_dir,
        f"{label}_{mode}_{safe_func}_{safe_init}_{safe_spec}_{stamp}.log",
    )


def _prepare_specification(
    specification: str,
    func_file: str,
    func_init_file: str,
    *,
    model_type: str,
    shared_vllm=None,
) -> Tuple[str, str, str, code_manipulation.Program]:
    funsearch = FunSearch(model_type=model_type, shared_vllm=shared_vllm)
    spec = funsearch._replace_function_in_specification(specification, func_file, func_init_file)
    function_to_evolve, function_to_run = _extract_function_names(spec)
    template = code_manipulation.text_to_program(spec)
    return spec, function_to_evolve, function_to_run, template


def _build_evolve_only_llm_prompt(
    template: code_manipulation.Program,
    function_to_evolve: str,
) -> str:
    """Sampler prompt: preface + evolve stub only (no solve/evaluate harness).

    FunSearch keeps the full program for evaluation but sends the LLM only the
    function header it must complete. Match that split for llm_best_of_n/chained.
    """
    evolve_fn = template.get_function(function_to_evolve)
    header = dataclasses.replace(evolve_fn, body="")
    prompt_program = dataclasses.replace(template, functions=[header])
    return str(prompt_program)


def _create_evaluator(
    *,
    specification: str,
    func_file: str,
    func_init_file: str,
    spec_file: str,
    inputs: Sequence[Any],
    experiment_dir: str,
    model_type: str,
    shared_vllm=None,
    results_tracker=None,
    log_file: str,
    grid_regeneration_attempts: int,
    grid_lookup_experiment_dir: Optional[str] = None,
) -> Tuple[evaluator.Evaluator, programs_database.ProgramsDatabase, str, code_manipulation.Program, FunSearch]:
    spec, function_to_evolve, function_to_run, template = _prepare_specification(
        specification, func_file, func_init_file, model_type=model_type, shared_vllm=shared_vllm
    )
    config = config_lib.Config(
        **funsearch_grid_regen_kwargs_from_config(),
        programs_database=config_lib.ProgramsDatabaseConfig(),
        grid_regeneration_attempts=grid_regeneration_attempts,
    )
    database = programs_database.ProgramsDatabase(
        config.programs_database, template, function_to_evolve
    )
    funsearch = FunSearch(model_type=model_type, shared_vllm=shared_vllm)
    if results_tracker is not None:
        funsearch.results_tracker = results_tracker

    ev = evaluator.Evaluator(
        database,
        template,
        _model_label(model_type),
        function_to_evolve,
        function_to_run,
        inputs,
        func_init_file,
        spec,
        function_name=func_file,
        experiment_dir=experiment_dir,
        shared_vllm=shared_vllm,
        vllm_lock=funsearch.vllm_lock,
        results_tracker=results_tracker,
        log_file=log_file,
    )

    initial = template.get_function(function_to_evolve).body
    check = ev.analyse(initial, island_id=None, version_generated=None)
    if check == -1:
        regen_attempts = max(0, int(grid_regeneration_attempts))
        print(
            f"[llm_terminal] Initial implementation failed pass_check; "
            f"regenerating grids (up to {regen_attempts} attempts)"
        )
        regenerated_any = False
        for _ in range(regen_attempts):
            init_note = _format_init_check_failure_for_grid_prompt(ev.get_last_evaluation_record())
            regenerated = funsearch._regenerate_grids_if_needed(
                spec,
                spec_file,
                function_to_evolve,
                config=config,
                experiment_dir=grid_lookup_experiment_dir or experiment_dir,
                init_check_failure=init_note,
            )
            if regenerated:
                regenerated_any = True
                check = ev.analyse(initial, island_id=None, version_generated=None)
                if check != -1:
                    break
        if check == -1:
            if regenerated_any:
                print(
                    "[llm_terminal] Grid regeneration exhausted; "
                    "continuing with failing init (will still sample)."
                )
            else:
                print("[llm_terminal] Grid regeneration unavailable; continuing with failing init.")

    return ev, database, function_to_evolve, template, funsearch


def _draw_body(
    llm: sampler.LLM,
    prompt: str,
    function_to_evolve: str,
) -> str:
    return llm.draw_samples(prompt, function_to_evolve)[0]


def run_llm_best_of_n(
    *,
    specification: str,
    inputs: Sequence[Any],
    func_file: str,
    func_init_file: str,
    spec_file: str,
    experiment_dir: str,
    model_type: str = "huggingface",
    shared_vllm=None,
    results_tracker=None,
    num_samples: int = 1000,
    grid_regeneration_attempts: Optional[int] = None,
    grid_lookup_experiment_dir: Optional[str] = None,
) -> str:
    """Generate num_samples independent LLM candidates with the same prompt."""
    if grid_regeneration_attempts is None:
        grid_regeneration_attempts = int(load_config().get("grid_regeneration_attempts", 5))

    results_dir = os.path.join(experiment_dir, "results", "llm_best_of_n")
    log_file = _build_log_path(
        results_dir=results_dir,
        model_type=model_type,
        func_file=func_file,
        func_init_file=func_init_file,
        spec_file=spec_file,
        mode="best_of_n",
    )

    ev, _database, function_to_evolve, template, funsearch = _create_evaluator(
        specification=specification,
        func_file=func_file,
        func_init_file=func_init_file,
        spec_file=spec_file,
        inputs=inputs,
        experiment_dir=results_dir,
        model_type=model_type,
        shared_vllm=shared_vllm,
        results_tracker=results_tracker,
        log_file=log_file,
        grid_regeneration_attempts=grid_regeneration_attempts,
        grid_lookup_experiment_dir=grid_lookup_experiment_dir,
    )

    llm = sampler.LLM(
        1,
        model_type=model_type,
        function_name=function_to_evolve,
        shared_vllm=shared_vllm,
        vllm_lock=funsearch.vllm_lock,
    )

    # Same prompt every iteration: domain context + evolve stub (no solve/evaluate harness).
    base_prompt = _build_evolve_only_llm_prompt(template, function_to_evolve)

    print(f"[llm_best_of_n] Generating {num_samples} independent samples for {function_to_evolve}")
    for i in range(num_samples):
        sample = _draw_body(llm, base_prompt, function_to_evolve)
        score = ev.analyse(sample, island_id=0, version_generated=1)
        if (i + 1) % 50 == 0 or i == 0:
            print(f"[llm_best_of_n] sample {i + 1}/{num_samples} score={score}")

    print(f"[llm_best_of_n] Wrote log: {log_file}")
    return log_file


def _format_chained_history(history: list[tuple[float, str]]) -> str:
    """Format up to the two most recent (score, body) pairs for the chained prompt."""
    if not history:
        return ""
    blocks = [
        f"### Score: {score}\n```python\n{body}\n```"
        for score, body in history[-2:]
    ]
    return (
        "\n\nPrevious implementations and their evaluation scores (most recent last):\n"
        + "\n\n".join(blocks)
    )


def run_llm_chained(
    *,
    specification: str,
    inputs: Sequence[Any],
    func_file: str,
    func_init_file: str,
    spec_file: str,
    experiment_dir: str,
    model_type: str = "huggingface",
    shared_vllm=None,
    results_tracker=None,
    num_iterations: int = 1000,
    grid_regeneration_attempts: Optional[int] = None,
    grid_lookup_experiment_dir: Optional[str] = None,
) -> str:
    """Iteratively generate candidates; each step adds the two most recent bodies + scores.

    No reflection or NL feedback in the loop. Explicit feedback runs separately afterward.
    """
    if grid_regeneration_attempts is None:
        grid_regeneration_attempts = int(load_config().get("grid_regeneration_attempts", 5))

    results_dir = os.path.join(experiment_dir, "results", "llm_chained")
    log_file = _build_log_path(
        results_dir=results_dir,
        model_type=model_type,
        func_file=func_file,
        func_init_file=func_init_file,
        spec_file=spec_file,
        mode="chained",
    )

    ev, _database, function_to_evolve, template, funsearch = _create_evaluator(
        specification=specification,
        func_file=func_file,
        func_init_file=func_init_file,
        spec_file=spec_file,
        inputs=inputs,
        experiment_dir=results_dir,
        model_type=model_type,
        shared_vllm=shared_vllm,
        results_tracker=results_tracker,
        log_file=log_file,
        grid_regeneration_attempts=grid_regeneration_attempts,
        grid_lookup_experiment_dir=grid_lookup_experiment_dir,
    )

    llm = sampler.LLM(
        1,
        model_type=model_type,
        function_name=function_to_evolve,
        shared_vllm=shared_vllm,
        vllm_lock=funsearch.vllm_lock,
    )

    base_prompt = _build_evolve_only_llm_prompt(template, function_to_evolve)

    history: list[tuple[float, str]] = []

    print(f"[llm_chained] Starting chained generation for {num_iterations} iterations (2-body history)")
    for i in range(num_iterations):
        prompt = base_prompt + _format_chained_history(history)
        sample = _draw_body(llm, prompt, function_to_evolve)
        score = ev.analyse(sample, island_id=0, version_generated=i + 1)
        history.append((float(score) if score is not None else 0.0, sample))
        history = history[-2:]
        if (i + 1) % 50 == 0 or i == 0:
            print(f"[llm_chained] iter {i + 1}/{num_iterations} score={score}")

    print(f"[llm_chained] Wrote log: {log_file}")
    return log_file


def find_llm_log_file(
    func_name: str,
    experiment_dir: str,
    mode: str,
    dsl_round: Optional[int] = None,
    func_evolution_round: Optional[int] = None,
) -> Optional[str]:
    """Find the newest LLM terminal-function log for a function."""
    del func_evolution_round
    from src.pipeline.cfg_to_funsearch_pipeline import sanitize_function_name

    safe_name = sanitize_function_name(func_name)
    subdir = "llm_best_of_n" if mode == "llm_best_of_n" else "llm_chained"
    results_dir = os.path.join(experiment_dir, "results", subdir)
    if not os.path.isdir(results_dir):
        return None

    version_bits = []
    if dsl_round is not None:
        version_bits.append(f"dsl{dsl_round}")

    candidates = []
    for name in os.listdir(results_dir):
        if not name.endswith(".log"):
            continue
        if safe_name.lower() not in name.lower():
            continue
        if version_bits and not all(bit in name for bit in version_bits):
            continue
        path = os.path.join(results_dir, name)
        candidates.append(path)

    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def top_score_from_log(log_file: str) -> Optional[float]:
    best = None
    with open(log_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = __import__("json").loads(line)
            end = get_end_score(record.get("scores") or {})
            if end is None:
                continue
            if best is None or end > best:
                best = end
    return best
