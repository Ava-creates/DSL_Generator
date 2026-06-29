"""CraftAdapter wraps the existing ``craft/`` package as a domain plugin."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import yaml

from domains.base import DomainAdapter, DomainSpec, EnvFactoryLike, EnvLike
from domains.craft.templates import (
    craft_baseline_evaluate_template,
    craft_env_setup,
    craft_evaluate_template,
    craft_solve_template_basic,
    craft_solve_template_task_env_basic,
)


_DEFAULT_RECIPES_PATH = "craft/resources/recipes.yaml"
_DEFAULT_HINTS_PATH = "craft/resources/hints.yaml"
_DEFAULT_NLD_PATH = "prompt_specifications/nld.txt"


def _load_file(path: str) -> str:
    if not path or not os.path.isfile(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _load_recipes(recipes_path: str) -> Dict[str, Any]:
    if not os.path.isfile(recipes_path):
        return {}
    with open(recipes_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _craft_native_actions() -> List[Dict[str, Any]]:
    """Native integer actions exposed by ``craft.env.CraftLab.action_specs``."""
    return [
        {"name": "DOWN", "code": 0, "description": "Move one cell down (south)."},
        {"name": "UP", "code": 1, "description": "Move one cell up (north)."},
        {"name": "LEFT", "code": 2, "description": "Move one cell left (west)."},
        {"name": "RIGHT", "code": 3, "description": "Move one cell right (east)."},
        {
            "name": "USE",
            "code": 4,
            "description": (
                "Interact with the cell the agent is facing: pick up a primitive, "
                "use a workshop to craft, or apply a tool against an obstacle."
            ),
        },
    ]


class CraftAdapter(DomainAdapter):
    """Domain adapter for the existing symbolic Craft environment."""

    def __init__(
        self,
        recipes_path: str = _DEFAULT_RECIPES_PATH,
        hints_path: str = _DEFAULT_HINTS_PATH,
        nld_path: str = _DEFAULT_NLD_PATH,
        env_type: int = 7,
    ) -> None:
        self._recipes_path = recipes_path
        self._hints_path = hints_path
        self._nld_path = nld_path
        self._env_type = env_type

        recipes_yaml = _load_recipes(recipes_path)
        hints_yaml = {}
        if os.path.isfile(hints_path):
            with open(hints_path, "r", encoding="utf-8") as f:
                hints_yaml = yaml.safe_load(f) or {}

        primitives: List[str] = list(recipes_yaml.get("primitives", []) or [])
        environment_objs: List[str] = list(recipes_yaml.get("environment", []) or [])
        recipes: Dict[str, Any] = dict(recipes_yaml.get("recipes", {}) or {})
        entities = sorted(set(primitives) | set(environment_objs) | set(recipes.keys()))

        tasks = sorted(hints_yaml.keys()) if hints_yaml else [
            f"get[{p}]" for p in primitives
        ] + [f"make[{r}]" for r in recipes]

        self.spec = DomainSpec(
            name="craft",
            nld=_load_file(nld_path),
            native_actions=_craft_native_actions(),
            entities=entities,
            tasks=tasks,
            recipes=recipes,
            extra_context={
                "primitives": primitives,
                "environment": environment_objs,
                "task_format": (
                    "Tasks use the form get[<primitive>] or make[<item>]. "
                    "get[X] is solved when X is in the agent's inventory; "
                    "make[X] additionally requires crafting at the correct workshop."
                ),
            },
        )

    # ------------------------------------------------------------------
    # Legacy-compatible paths (so existing helpers that take recipes_path
    # keep working without change).
    # ------------------------------------------------------------------

    @property
    def default_recipes_path(self) -> Optional[str]:
        return self._recipes_path

    @property
    def default_hints_path(self) -> Optional[str]:
        return self._hints_path

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
        from craft import env_factory as _env_factory

        return _env_factory.EnvironmentFactory(
            self._recipes_path,
            self._hints_path,
            self._env_type,
            max_steps=max_steps,
            seed=seed,
            reuse_environments=reuse_environments,
            visualise=visualise,
            custom_grid_path=custom_test_case_path,
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
        test_case_path = test_case.get("_path") if isinstance(test_case, dict) else None
        sampler = self.env_factory(
            max_steps=max_steps,
            seed=seed,
            reuse_environments=False,
            visualise=visualise,
            custom_test_case_path=test_case_path,
        )
        env = sampler.sample_environment(task_name=task)
        env.reset()
        if isinstance(test_case, dict) and hasattr(env, "scenario"):
            env.scenario.spec = test_case
        return env

    def task_succeeded(self, env: EnvLike, task: str) -> bool:
        goal_name, goal_arg = env.task.goal
        return bool(env._current_state.satisfies(goal_name, goal_arg))

    def snapshot_state(self, env: EnvLike) -> Dict[str, Any]:
        state = env._current_state
        cookbook = env.world.cookbook
        inv = state.inventory
        inventory_dict = {
            str(cookbook.index.get(i)): float(v)
            for i, v in enumerate(inv)
            if v
        }
        return {
            "pos": [int(v) for v in state.pos],
            "dir": int(state.dir),
            "inventory": inventory_dict,
            "grid_shape": list(state.grid.shape),
        }

    def render_state_markdown(self, env: EnvLike) -> str:
        from src.utils.test import grid_to_markdown

        state = env._current_state
        return grid_to_markdown(
            state.grid,
            env.world.cookbook,
            state.pos,
            include_indices=True,
        )

    # ------------------------------------------------------------------
    # Templates
    # ------------------------------------------------------------------

    def solve_template(self, *, func_name: str, func_params: str, func_call_args: str) -> str:
        return craft_solve_template_basic(
            func_name=func_name,
            func_params=func_params,
            func_call_args=func_call_args,
        )

    def solve_template_task_env(
        self, *, func_name: str, func_params: str, func_call_args: str
    ) -> str:
        return craft_solve_template_task_env_basic(
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
        return craft_evaluate_template(
            display_name=display_name,
            env_setup=env_setup,
            args_definitions=args_definitions,
            func_call_args=func_call_args,
            grid_spec_paths_var=test_case_paths_var,
        )

    def env_setup_code(self, *, task_name: str, test_case_path: str) -> str:
        return craft_env_setup(
            recipes_path=self._recipes_path,
            hints_path=self._hints_path,
            task_name=task_name,
            grid_spec_path=test_case_path,
        )

    def baseline_evaluate_template(
        self,
        *,
        display_name: str,
        func_call_args: str,
        task_name: str,
        max_steps: int = 400,
    ) -> str:
        return craft_baseline_evaluate_template(
            display_name=display_name,
            func_call_args=func_call_args,
            task_name=task_name,
            recipes_path=self._recipes_path,
            hints_path=self._hints_path,
            max_steps=max_steps,
        )

    # ------------------------------------------------------------------
    # Test-case generation - delegate to existing grid_generation module
    # ------------------------------------------------------------------

    def generate_test_case(
        self,
        *,
        func_name: str,
        description: str,
        func_args: str,
        output_path: str,
        shared_vllm: Any = None,
        prompt_path: str = "prompt_specifications/grid_prompt.txt",
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
        from src.pipeline.grid_generation import ensure_function_grid_spec

        return ensure_function_grid_spec(
            func_name=func_name,
            description=description,
            recipes_path=self._recipes_path,
            output_path=output_path,
            shared_vllm=shared_vllm,
            default_task_name=default_task_name,
            prompt_path=prompt_path,
            func_args=func_args or "None",
            env_description=self.spec.nld,
            recipes_text=self.domain_text_for_prompt(),
            attempts=attempts,
            existing_cases=existing_cases,
            cfg_text=cfg_text,
            codebase_text=codebase_text,
            require_test_type=require_test_type,
            skip_positive_grids=skip_positive_grids,
            positive_grids=positive_grids,
            negative_grids=negative_grids,
            edge_grids=edge_grids,
        )

    def domain_text_for_prompt(self) -> str:
        """Return the raw ``recipes.yaml`` text to maintain parity with src/."""
        raw = _load_file(self._recipes_path)
        if raw:
            return raw
        return super().domain_text_for_prompt()
