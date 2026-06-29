"""Domain abstraction for the DSL Generator pipeline.

Each supported environment (Craft, Crafter, ...) provides a
:class:`DomainAdapter` implementation that encapsulates everything the
pipeline needs to know that is domain-specific:

* environment construction and rollout bookkeeping
* the natural-language + structured description that goes into prompts
* FunSearch solve/evaluate/env-setup templates
* per-terminal test-case generation and pass-check evaluation
* state snapshots used in synthesis prompts

The rest of the pipeline under ``src_agnostic/`` consumes a ``DomainAdapter``
and never imports a concrete domain package directly.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple


@dataclass
class DomainSpec:
    """Serializable description of a domain.

    This is the single source of truth that replaces the combination of
    ``recipes.yaml``, ``hints.yaml``, and ``nld.txt`` used in the original
    Craft-only pipeline.
    """

    name: str
    nld: str
    native_actions: List[Dict[str, Any]] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    tasks: List[str] = field(default_factory=list)
    recipes: Optional[Dict[str, Any]] = None
    extra_context: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_recipes(self) -> bool:
        return bool(self.recipes)


class EnvLike(Protocol):
    """Minimal interface we expect any domain env to expose.

    Concrete envs may expose additional domain-specific attributes
    (e.g. ``_current_state`` for Craft). Templates produced by the domain
    adapter may use those attributes freely; only the generic pipeline code
    is restricted to this protocol.
    """

    def reset(self, seed: int = 0) -> Any: ...
    def step(self, action: Any, num_steps: int = 1) -> Tuple[float, bool, Any]: ...


class EnvFactoryLike(Protocol):
    """Interface mimicking ``craft.env_factory.EnvironmentFactory``.

    Providing an env-factory shim on each adapter lets us reuse large
    sections of the existing pipeline code with minimal rewrites.
    """

    def sample_environment(self, task_name: Optional[str] = None) -> EnvLike: ...


class DomainAdapter(ABC):
    """Abstract interface every domain plugin must implement."""

    #: Populated by the subclass ``__init__``.
    spec: DomainSpec

    # ------------------------------------------------------------------
    # Environment lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def env_factory(
        self,
        *,
        max_steps: int = 400,
        seed: int = 0,
        reuse_environments: bool = False,
        visualise: bool = False,
        custom_test_case_path: Optional[str] = None,
    ) -> EnvFactoryLike:
        """Return an :class:`EnvFactoryLike` configured for this domain.

        ``custom_test_case_path`` is an optional path to a per-terminal test
        case spec produced by :meth:`generate_test_case`. Domains that do not
        use file-backed test cases may ignore it.
        """

    @abstractmethod
    def build_env(
        self,
        *,
        task: str,
        seed: int = 0,
        max_steps: int = 400,
        test_case: Optional[Dict[str, Any]] = None,
        visualise: bool = False,
    ) -> EnvLike:
        """One-shot helper: construct an env for ``task`` ready for rollout."""

    @abstractmethod
    def task_succeeded(self, env: EnvLike, task: str) -> bool:
        """Return whether ``task`` has been achieved in the current env state."""

    @abstractmethod
    def snapshot_state(self, env: EnvLike) -> Dict[str, Any]:
        """Return a JSON-serializable state snapshot for synthesis prompts."""

    @abstractmethod
    def render_state_markdown(self, env: EnvLike) -> str:
        """Human-readable state rendering for prompt injection."""

    # ------------------------------------------------------------------
    # FunSearch terminal templates
    # ------------------------------------------------------------------

    @abstractmethod
    def solve_template(
        self,
        *,
        func_name: str,
        func_params: str,
        func_call_args: str,
    ) -> str:
        """Return the ``solve(...)`` template as a Python source string."""

    @abstractmethod
    def evaluate_template(
        self,
        *,
        display_name: str,
        env_setup: str,
        args_definitions: str,
        func_call_args: str,
        test_case_paths_var: Optional[str] = None,
    ) -> str:
        """Return the ``evaluate()`` template as a Python source string."""

    @abstractmethod
    def env_setup_code(
        self,
        *,
        task_name: str,
        test_case_path: str,
    ) -> str:
        """Python snippet that binds ``env`` (and optionally ``grid_spec``) for evaluate()."""

    def baseline_evaluate_template(
        self,
        *,
        display_name: str,
        func_call_args: str,
        task_name: str,
        max_steps: int = 400,
    ) -> str:
        """Optional baseline template (task-env, no test case grids).

        Default implementation raises ``NotImplementedError``; override to
        support baseline runs for this domain.
        """
        raise NotImplementedError(
            f"Domain '{self.spec.name}' does not provide a baseline evaluate template."
        )

    # ------------------------------------------------------------------
    # Per-terminal test-case generation
    # ------------------------------------------------------------------

    @abstractmethod
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
        """Produce a per-terminal test case JSON and write it to ``output_path``.

        Returns the parsed spec dict on success, ``None`` if generation
        failed or was skipped.
        """

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------

    def domain_text_for_prompt(self) -> str:
        """Structured domain context injected where Craft used recipes.yaml text."""
        parts: List[str] = []
        if self.spec.native_actions:
            parts.append("Native actions available in the environment:")
            for act in self.spec.native_actions:
                name = act.get("name", "?")
                desc = act.get("description", "")
                args = act.get("args")
                sig = f"{name}({', '.join(args)})" if args else name
                parts.append(f"  - {sig}: {desc}" if desc else f"  - {sig}")
            parts.append("")
        if self.spec.entities:
            parts.append("Entities / objects in this world:")
            parts.append("  " + ", ".join(self.spec.entities))
            parts.append("")
        if self.spec.tasks:
            parts.append("Tasks / goals an agent can be asked to accomplish:")
            parts.append("  " + ", ".join(self.spec.tasks))
            parts.append("")
        if self.spec.recipes:
            parts.append("Crafting recipes:")
            for item, recipe in self.spec.recipes.items():
                parts.append(f"  {item}: {recipe}")
            parts.append("")
        for key, val in self.spec.extra_context.items():
            parts.append(f"{key}:")
            parts.append(f"  {val}")
            parts.append("")
        return "\n".join(parts).rstrip() or "(no additional domain context)"

    def write_domain_context_file(self, path: str) -> str:
        """Persist :meth:`domain_text_for_prompt` to disk and return the path.

        Some stages of the pipeline still expect a ``recipes_path`` on disk.
        Callers can use this helper to materialize a faithful replacement.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.domain_text_for_prompt())
        return path

    def write_nld_file(self, path: str) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.spec.nld)
        return path

    # ------------------------------------------------------------------
    # Legacy compatibility helpers
    # ------------------------------------------------------------------

    #: Domain-specific prompt template used by ``generate_test_case``.
    #: Override on subclasses when the domain ships its own prompt file.
    #: Defaults to Craft's ``grid_prompt.txt`` for backward compatibility.
    test_case_prompt_path: str = "prompt_specifications/grid_prompt.txt"

    @property
    def default_recipes_path(self) -> Optional[str]:
        """Path used when a stage still needs a ``recipes_path`` argument.

        Implementations that have a native YAML file may return it directly;
        others should return the materialized :meth:`write_domain_context_file`
        path.
        """
        return None

    @property
    def default_hints_path(self) -> Optional[str]:
        return None
