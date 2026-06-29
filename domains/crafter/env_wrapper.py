"""Wrapper around ``crafter`` exposing a ``(reward, done, obs)`` contract.

The existing pipeline's solve/evaluate templates call ``env.step(action)`` and
unpack three values. Crafter's gym env returns five, so we wrap it and keep
the latest ``info`` dict available as ``env.info`` for snapshot helpers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import crafter  # type: ignore
except ImportError:  # pragma: no cover - optional at import time
    crafter = None


class CrafterEnvWrapper:
    """Adapter around ``crafter.Env`` / ``gym.make('CrafterReward-v1')``."""

    def __init__(
        self,
        *,
        task: Optional[str] = None,
        max_steps: int = 400,
        area: Tuple[int, int] = (64, 64),
        view: Tuple[int, int] = (9, 9),
        length: Optional[int] = None,
        seed: Optional[int] = None,
        reward: bool = True,
    ) -> None:
        if crafter is None:
            raise ImportError(
                "The 'crafter' package is required for CrafterAdapter. "
                "Install it with `pip install crafter`."
            )
        self.task = task
        self.max_steps = max_steps
        self._env = crafter.Env(
            area=area,
            view=view,
            length=length or max_steps,
            seed=seed,
            reward=reward,
        )
        self._seed = seed
        self._done = False
        self._steps = 0
        self.info: Dict[str, Any] = {
            "inventory": {},
            "achievements": {k: 0 for k in self._default_achievements()},
        }
        self._latest_obs = None
        # Replicate the craft env.task.goal shape for code that inspects it.
        self.task_spec: Dict[str, Any] = {"task_name": task}

    # ------------------------------------------------------------------
    # Gym-like surface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        *,
        scenario: Optional[Dict[str, Any]] = None,
    ) -> Any:
        if seed is not None:
            self._seed = seed
            self._env._seed = seed  # crafter stores seed as attribute
        obs = self._env.reset()
        self._done = False
        self._steps = 0
        self._latest_obs = obs
        self._refresh_info()
        if scenario:
            from domains.crafter.scenario import apply_scenario

            apply_scenario(self, scenario)
            self._refresh_info()
        return obs

    def apply_scenario(self, scenario: Dict[str, Any]) -> None:
        """Mutate the env to match ``scenario``. Call right after ``reset()``."""
        from domains.crafter.scenario import apply_scenario

        apply_scenario(self, scenario)
        self._refresh_info()

    def _refresh_info(self) -> None:
        """Rebuild ``info`` from the current world (used after scenario injection)."""
        inner = self._env
        sem = inner._sem_view() if hasattr(inner, "_sem_view") else None
        self.info = {
            "inventory": dict(inner._player.inventory),
            "achievements": dict(inner._player.achievements),
            "player_pos": np.asarray(inner._player.pos),
            "semantic": sem,
            "discount": 1.0,
            "reward": 0.0,
        }

    def step(self, action: Any, num_steps: int = 1) -> Tuple[float, bool, Any]:
        if isinstance(action, str):
            action = self._action_name_to_int(action)
        reward = 0.0
        done = False
        obs = self._latest_obs
        for _ in range(max(1, int(num_steps))):
            if self._done:
                break
            obs, r, done, info = self._env.step(int(action))
            reward += float(r)
            self.info = dict(info)
            self._steps += 1
            if done or self._steps >= self.max_steps:
                self._done = True
                break
        self._latest_obs = obs
        return reward, self._done, obs

    # ------------------------------------------------------------------
    # Helpers exposed to adapter and templates
    # ------------------------------------------------------------------

    @property
    def inventory(self) -> Dict[str, int]:
        return dict(self.info.get("inventory", {}))

    @property
    def achievements(self) -> Dict[str, int]:
        return dict(self.info.get("achievements", {}))

    @property
    def player_pos(self) -> Optional[List[int]]:
        pos = self.info.get("player_pos")
        if pos is None:
            return None
        return list(pos)

    @staticmethod
    def _default_achievements() -> List[str]:
        return [
            "collect_coal", "collect_diamond", "collect_drink", "collect_iron",
            "collect_plant", "collect_sapling", "collect_stone", "collect_wood",
            "defeat_skeleton", "defeat_zombie", "eat_cow", "eat_plant",
            "make_iron_pickaxe", "make_iron_sword", "make_stone_pickaxe",
            "make_stone_sword", "make_wood_pickaxe", "make_wood_sword",
            "place_furnace", "place_plant", "place_stone", "place_table",
            "wake_up",
        ]

    @staticmethod
    def _action_name_to_int(name: str) -> int:
        mapping = {
            "noop": 0, "move_left": 1, "move_right": 2, "move_up": 3,
            "move_down": 4, "do": 5, "sleep": 6, "place_stone": 7,
            "place_table": 8, "place_furnace": 9, "place_plant": 10,
            "make_wood_pickaxe": 11, "make_stone_pickaxe": 12,
            "make_iron_pickaxe": 13, "make_wood_sword": 14,
            "make_stone_sword": 15, "make_iron_sword": 16,
        }
        key = name.lower()
        if key not in mapping:
            raise ValueError(f"Unknown Crafter action name: {name}")
        return mapping[key]

    # Mimic craft.env.CraftLab's ``task.goal`` duck-typing.
    class _TaskShim:
        def __init__(self, name: Optional[str]) -> None:
            self.goal = ("achievement", name or "")

    @property
    def task_obj(self) -> "CrafterEnvWrapper._TaskShim":
        return CrafterEnvWrapper._TaskShim(self.task)

    # Upstream code sometimes reads env.task.goal
    def __getattr__(self, name: str) -> Any:  # pragma: no cover - fall-through
        if name == "task":
            return self.task_obj
        raise AttributeError(name)


class CrafterEnvFactory:
    """``EnvironmentFactory``-shaped object for the agnostic shim path."""

    def __init__(
        self,
        *,
        max_steps: int = 400,
        seed: int = 0,
        reuse_environments: bool = False,
        visualise: bool = False,
        custom_test_case_path: Optional[str] = None,
    ) -> None:
        self._max_steps = max_steps
        self._seed = seed
        self._reuse = reuse_environments
        self._visualise = visualise
        self._custom_test_case_path = custom_test_case_path
        self._cache: Optional[CrafterEnvWrapper] = None

    def sample_environment(self, task_name: Optional[str] = None) -> CrafterEnvWrapper:
        if self._reuse and self._cache is not None and self._cache.task == task_name:
            return self._cache
        env = CrafterEnvWrapper(
            task=task_name,
            max_steps=self._max_steps,
            seed=self._seed,
        )
        self._cache = env
        return env
