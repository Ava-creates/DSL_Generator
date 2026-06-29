"""Domain-aware wrapper around :mod:`src.pipeline.integrated_pipeline`.

The upstream module hard-codes ``craft.env_factory.EnvironmentFactory`` in
:func:`synthesize_and_test_programs`. This wrapper swaps that call for the
domain adapter's :meth:`env_factory` while keeping the rest of the upstream
logic (prompt construction, parallelism, checkpointing) intact.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from domains.base import DomainAdapter

import src.pipeline.integrated_pipeline as _upstream


class _AdapterEnvFactoryShim:
    """Shim that mimics ``craft.env_factory`` module surface."""

    def __init__(self, adapter: DomainAdapter) -> None:
        self._adapter = adapter

    def EnvironmentFactory(
        self,
        recipes_path: Any = None,
        hints_path: Any = None,
        env_type: Any = 7,
        *,
        max_steps: int = 400,
        seed: int = 0,
        reuse_environments: bool = False,
        visualise: bool = False,
        custom_grid_path: Optional[str] = None,
    ):
        return self._adapter.env_factory(
            max_steps=max_steps,
            seed=seed,
            reuse_environments=reuse_environments,
            visualise=visualise,
            custom_test_case_path=custom_grid_path,
        )


class with_adapter:
    """Rebind craft env_factory on :mod:`src.pipeline.integrated_pipeline`."""

    def __init__(self, adapter: DomainAdapter) -> None:
        self.adapter = adapter
        self._prev_env_factory = None

    def __enter__(self) -> DomainAdapter:
        self._prev_env_factory = _upstream.env_factory
        _upstream.env_factory = _AdapterEnvFactoryShim(self.adapter)
        return self.adapter

    def __exit__(self, exc_type, exc, tb) -> None:
        _upstream.env_factory = self._prev_env_factory


# ---------------------------------------------------------------------------
# Public wrappers (thin passthroughs that keep the adapter bound).
# ---------------------------------------------------------------------------

def test_cfg_on_tasks(
    experiment_dir: str,
    tasks: List[str],
    cfg: str,
    terminals: Dict[str, str],
    *,
    adapter: DomainAdapter,
    max_attempts: int = 1,
    shared_vllm=None,
    results_tracker=None,
    cfg_version: Optional[int] = None,
    func_evolution_round: Optional[int] = None,
    synthesis_prompt_path: Optional[str] = None,
    test_seeds: Optional[List[int]] = None,
    seed_outcome_log_path: Optional[str] = None,
    model_type: str = "huggingface",
    include_final_functions_in_prompt: bool = False,
    openai_compat_key_file: Optional[str] = None,
) -> Dict[str, bool]:
    recipes_path = adapter.default_recipes_path or ""
    hints_path = adapter.default_hints_path or ""
    with with_adapter(adapter):
        return _upstream.test_cfg_on_tasks(
            experiment_dir=experiment_dir,
            tasks=tasks,
            cfg=cfg,
            terminals=terminals,
            recipes_path=recipes_path,
            hints_path=hints_path,
            max_attempts=max_attempts,
            shared_vllm=shared_vllm,
            results_tracker=results_tracker,
            cfg_version=cfg_version,
            func_evolution_round=func_evolution_round,
            synthesis_prompt_path=synthesis_prompt_path,
            test_seeds=test_seeds,
            seed_outcome_log_path=seed_outcome_log_path,
            model_type=model_type,
            include_final_functions_in_prompt=include_final_functions_in_prompt,
            openai_compat_key_file=openai_compat_key_file,
        )


def run_failure_analysis_for_dsl_evolution(
    experiment_dir: str,
    failing_tasks: List[str],
    cfg: str,
    terminals: Dict[str, str],
    failed_programs_by_task: Optional[Dict[str, List[str]]],
    *,
    adapter: DomainAdapter,
    shared_vllm=None,
    model_type: str = "huggingface",
    openai_compat_key_file: Optional[str] = None,
) -> str:
    with with_adapter(adapter):
        return _upstream.run_failure_analysis_for_dsl_evolution(
            experiment_dir=experiment_dir,
            failing_tasks=failing_tasks,
            cfg=cfg,
            terminals=terminals,
            failed_programs_by_task=failed_programs_by_task,
            shared_vllm=shared_vllm,
            model_type=model_type,
            openai_compat_key_file=openai_compat_key_file,
        )


def evolve_dsl(
    experiment_dir: str,
    failing_tasks: List[str],
    cfg: str,
    terminals: Dict[str, str],
    failure_analysis: str,
    *,
    adapter: DomainAdapter,
    shared_vllm=None,
    new_dsl_round: Optional[int] = None,
    recipes_text: Optional[str] = None,
    model_type: str = "huggingface",
    openai_compat_key_file: Optional[str] = None,
):
    recipes_blob = recipes_text if recipes_text is not None else adapter.domain_text_for_prompt()
    with with_adapter(adapter):
        return _upstream.evolve_dsl(
            experiment_dir=experiment_dir,
            failing_tasks=failing_tasks,
            cfg=cfg,
            recipes=recipes_blob,
            terminals=terminals,
            failure_analysis=failure_analysis,
            shared_vllm=shared_vllm,
            new_dsl_round=new_dsl_round,
            model_type=model_type,
            openai_compat_key_file=openai_compat_key_file,
        )


def evolve_functions_with_failing_tasks(
    experiment_dir: str,
    failing_tasks: List[str],
    terminals: Dict[str, str],
    specification: str,
    *,
    adapter: DomainAdapter,
    spec_file: str = "",
    cfg: str = "",
    shared_vllm=None,
    dsl_round: Optional[int] = None,
    func_evolution_round: Optional[int] = None,
    total_samples: int = 1000,
) -> bool:
    from src_agnostic.pipeline.cfg_to_funsearch_pipeline import (
        with_adapter as cfg_with_adapter,
    )

    with with_adapter(adapter), cfg_with_adapter(adapter, shared_vllm=shared_vllm):
        return _upstream.evolve_functions_with_failing_tasks(
            experiment_dir=experiment_dir,
            failing_tasks=failing_tasks,
            terminals=terminals,
            specification=specification,
            spec_file=spec_file,
            cfg=cfg,
            shared_vllm=shared_vllm,
            dsl_round=dsl_round,
            func_evolution_round=func_evolution_round,
            total_samples=total_samples,
        )


def evolve_dsl_and_restart(
    experiment_dir: str,
    failing_tasks: List[str],
    cfg: str,
    terminals: Dict[str, str],
    spec_file: str,
    *,
    adapter: DomainAdapter,
    shared_vllm=None,
    model_type: str = "huggingface",
    max_retries: int = 10,
    failed_programs_by_task: Optional[Dict[str, List[str]]] = None,
):
    recipes_blob = adapter.domain_text_for_prompt()

    from src_agnostic.pipeline.cfg_to_funsearch_pipeline import (
        with_adapter as cfg_with_adapter,
    )

    with with_adapter(adapter), cfg_with_adapter(adapter, shared_vllm=shared_vllm):
        return _upstream.evolve_dsl_and_restart(
            experiment_dir=experiment_dir,
            failing_tasks=failing_tasks,
            cfg=cfg,
            recipes=recipes_blob,
            spec_file=spec_file,
            terminals=terminals,
            shared_vllm=shared_vllm,
            model_type=model_type,
            max_retries=max_retries,
            failed_programs_by_task=failed_programs_by_task,
        )


# Re-export helpers unchanged.
from src.pipeline.integrated_pipeline import (  # noqa: F401
    check_final_functions_exist,
    ensure_terminals_match_cfg,
    extract_and_save_cfg,
    synthesize_and_test_programs,
)
