"""CrafterAdapter wraps ``danijar/crafter`` as a domain plugin."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import yaml

from domains.base import DomainAdapter, DomainSpec, EnvFactoryLike, EnvLike
from domains.crafter.env_wrapper import CrafterEnvFactory, CrafterEnvWrapper
from domains.crafter.observations import (
    grid_to_markdown,
    local_grid_cells,
    semantic_id_to_name,
)
from domains.crafter.templates import (
    crafter_baseline_evaluate_template,
    crafter_env_setup,
    crafter_evaluate_template,
    crafter_solve_template_basic,
    crafter_solve_template_task_env_basic,
)
from domains.crafter.test_case_generation import ensure_function_test_case


_SPEC_YAML_PATH = os.path.join(os.path.dirname(__file__), "spec.yaml")
_DEFAULT_NLD_PATH = "prompt_specifications/nld_crafter.txt"


def _load_file(path: str) -> str:
    if not path or not os.path.isfile(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class CrafterAdapter(DomainAdapter):
    """Domain adapter for the Crafter survival env."""

    test_case_prompt_path: str = "prompt_specifications/crafter_testcase_prompt.txt"

    def __init__(
        self,
        *,
        spec_path: str = _SPEC_YAML_PATH,
        nld_path: str = _DEFAULT_NLD_PATH,
    ) -> None:
        self._spec_path = spec_path
        self._nld_path = nld_path

        raw = _load_yaml(spec_path)
        nld_text = _load_file(nld_path) or raw.get("description", "")

        self.spec = DomainSpec(
            name=raw.get("name", "crafter"),
            nld=nld_text,
            native_actions=list(raw.get("native_actions", []) or []),
            entities=list(raw.get("entities", []) or []),
            tasks=list(raw.get("tasks", []) or []),
            recipes=dict(raw.get("recipes", {}) or {}) or None,
            extra_context=dict(raw.get("extra_context", {}) or {}),
        )

    # ------------------------------------------------------------------
    # Legacy-compatible paths
    # ------------------------------------------------------------------

    @property
    def default_recipes_path(self) -> Optional[str]:
        return None

    @property
    def default_hints_path(self) -> Optional[str]:
        return None

    @property
    def nld_path(self) -> str:
        return self._nld_path

    # ------------------------------------------------------------------
    # Env lifecycle
    # ------------------------------------------------------------------

    def env_factory(
        self,
        *,
        max_steps: int = 400,
        seed: int = 0,
        reuse_environments: bool = False,
        visualise: bool = False,
        custom_test_case_path: Optional[str] = None,
    ) -> EnvFactoryLike:
        return CrafterEnvFactory(
            max_steps=max_steps,
            seed=seed,
            reuse_environments=reuse_environments,
            visualise=visualise,
            custom_test_case_path=custom_test_case_path,
        )

    def build_env(
        self,
        *,
        task: str,
        seed: int = 0,
        max_steps: int = 400,
        test_case: Optional[Dict[str, Any]] = None,
        visualise: bool = False,
    ) -> EnvLike:
        tc_max_steps = max_steps
        init_actions: List[int] = []
        scenario: Optional[Dict[str, Any]] = None
        if isinstance(test_case, dict):
            tc_max_steps = int(test_case.get("max_steps", max_steps))
            init_actions = list(test_case.get("init_actions", []) or [])
            scenario = test_case.get("scenario")
        env = CrafterEnvWrapper(
            task=task,
            max_steps=tc_max_steps,
            seed=seed,
        )
        env.reset(scenario=scenario)
        if isinstance(test_case, dict):
            env._test_case_spec = test_case  # type: ignore[attr-defined]
        for act in init_actions:
            env.step(act)
        return env

    def task_succeeded(self, env: EnvLike, task: str) -> bool:
        info = getattr(env, "info", {}) or {}
        achievements = info.get("achievements", {}) or {}
        return int(achievements.get(task, 0)) >= 1

    def snapshot_state(
        self,
        env: EnvLike,
        *,
        grid_radius: int = 4,
        include_full_semantic: bool = False,
    ) -> Dict[str, Any]:
        info = getattr(env, "info", {}) or {}
        inventory = dict(info.get("inventory", {}) or {})
        achievements = dict(info.get("achievements", {}) or {})
        pos = info.get("player_pos")
        if pos is not None:
            pos = [int(v) for v in pos]

        grid = local_grid_cells(env, radius=grid_radius)

        snapshot: Dict[str, Any] = {
            "inventory": inventory,
            "achievements": achievements,
            "player_pos": pos,
            "facing": grid.get("facing"),
            "grid_cells": grid.get("cells"),
            "grid_origin": grid.get("origin"),
            "grid_player_local": grid.get("player_local"),
            "grid_radius": grid_radius,
        }
        if include_full_semantic:
            semantic = info.get("semantic")
            if semantic is not None:
                snapshot["semantic"] = (
                    semantic.tolist() if hasattr(semantic, "tolist") else semantic
                )
                snapshot["semantic_id_to_name"] = semantic_id_to_name(env)
        return snapshot

    def render_state_markdown(self, env: EnvLike, *, grid_radius: int = 4) -> str:
        info = getattr(env, "info", {}) or {}
        lines: List[str] = ["### Crafter state"]
        pos = info.get("player_pos")
        if pos is not None:
            lines.append(f"- player_pos: {[int(v) for v in pos]}")
        grid = local_grid_cells(env, radius=grid_radius)
        lines.append(f"- facing: {grid.get('facing')}")
        inv = dict(info.get("inventory", {}) or {})
        if inv:
            lines.append("- inventory:")
            for k, v in sorted(inv.items()):
                if v:
                    lines.append(f"  - {k}: {v}")
        ach = {k: v for k, v in (info.get("achievements", {}) or {}).items() if v}
        if ach:
            lines.append("- achievements unlocked:")
            for k in sorted(ach):
                lines.append(f"  - {k}")
        lines.append("")
        lines.append("```")
        lines.append(grid_to_markdown(grid))
        lines.append("```")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Templates
    # ------------------------------------------------------------------

    def solve_template(self, *, func_name: str, func_params: str, func_call_args: str) -> str:
        return crafter_solve_template_basic(
            func_name=func_name,
            func_params=func_params,
            func_call_args=func_call_args,
        )

    def solve_template_task_env(
        self, *, func_name: str, func_params: str, func_call_args: str
    ) -> str:
        return crafter_solve_template_task_env_basic(
            func_name=func_name,
            func_params=func_params,
            func_call_args=func_call_args,
        )

    def evaluate_template(
        self,
        *,
        display_name: str,
        env_setup: str,
        args_definitions: str,
        func_call_args: str,
        test_case_paths_var: Optional[str] = None,
    ) -> str:
        return crafter_evaluate_template(
            display_name=display_name,
            env_setup=env_setup,
            args_definitions=args_definitions,
            func_call_args=func_call_args,
            grid_spec_paths_var=test_case_paths_var,
        )

    def env_setup_code(self, *, task_name: str, test_case_path: str) -> str:
        return crafter_env_setup(task_name=task_name, grid_spec_path=test_case_path)

    def baseline_evaluate_template(
        self,
        *,
        display_name: str,
        func_call_args: str,
        task_name: str,
        max_steps: int = 400,
    ) -> str:
        return crafter_baseline_evaluate_template(
            display_name=display_name,
            func_call_args=func_call_args,
            task_name=task_name,
            max_steps=max_steps,
        )

    # ------------------------------------------------------------------
    # Test-case generation
    # ------------------------------------------------------------------

    def generate_test_case(
        self,
        *,
        func_name: str,
        description: str,
        func_args: str,
        output_path: str,
        shared_vllm: Any = None,
        prompt_path: str = "prompt_specifications/crafter_testcase_prompt.txt",
        existing_cases: Optional[List[Dict[str, Any]]] = None,
        attempts: int = 5,
        require_test_type: bool = True,
        skip_positive_grids: bool = False,
        positive_grids: int = 10,
        negative_grids: int = 4,
        edge_grids: int = 1,
        cfg_text: str = "",
        codebase_text: str = "",
        default_task_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        return ensure_function_test_case(
            func_name=func_name,
            description=description,
            output_path=output_path,
            valid_tasks=self.spec.tasks,
            valid_actions=self.spec.native_actions,
            env_description=self.spec.nld,
            domain_text=self.domain_text_for_prompt(),
            func_args=func_args,
            default_task_name=default_task_name,
            shared_vllm=shared_vllm,
            prompt_path=prompt_path,
            attempts=attempts,
            existing_cases=existing_cases,
            codebase_text=codebase_text,
            require_test_type=require_test_type,
            skip_positive_grids=skip_positive_grids,
            positive_grids=positive_grids,
            negative_grids=negative_grids,
            edge_grids=edge_grids,
        )
