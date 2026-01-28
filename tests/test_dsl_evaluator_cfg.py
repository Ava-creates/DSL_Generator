import json
from pathlib import Path

import importlib.util
import pytest

from craft import env_factory
from src.pipeline.dsl_evaluator import DSLEvaluator


def _load_cfg():
    cfg_path = Path("experiments/experiment_20260113_153016/cfg/cfg_output_0.json")
    data = json.loads(cfg_path.read_text())
    return data["cfg"]


def test_dsl_evaluator_with_cfg_output_0():
    cfg = _load_cfg()

    def _load_apply_tool_impl():
        module_path = Path(
            "experiments/experiment_20260113_153016/final_functions/apply_tool_dsl0_func1.py"
        )
        spec = importlib.util.spec_from_file_location("apply_tool_dsl0_func1", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.apply_tool

    def move(_env, direction):
        assert direction in {"NORTH", "SOUTH", "EAST", "WEST"}
        return [0, 0, 0]

    def turn(_env, direction):
        assert direction in {"NORTH", "SOUTH", "EAST", "WEST"}
        return [1]

    def collect(_env, item):
        assert item in {"WOOD", "IRON", "GRASS", "ROCK", "GOLD", "GEM"}
        return [2, 2]

    def craft(_env, item, workshop):
        assert workshop.startswith("WORKSHOP")
        return [3]

    apply_tool = _load_apply_tool_impl()

    def identify_tool(_env, tool, obstacle):
        assert obstacle in {"WATER", "STONE", "BOUNDARY"}
        return [1]

    def set_goal(_env, item):
        return [2]

    evaluator = DSLEvaluator(
        cfg=cfg,
        function_implementations={
            "move": move,
            "turn": turn,
            "collect": collect,
            "craft": craft,
            "apply_tool": apply_tool,
            "identify_tool": identify_tool,
            "set_goal": set_goal,
        },
    )

    program = (
        "SET_GOAL(AXE); "  #1
        "MOVE(NORTH); " #3
        "TURN(EAST); "#1
        "COLLECT(WOOD); "#2
        "CRAFT(AXE,WORKSHOP0); "#1
        "IDENTIFY_TOOL(KNIFE,WATER); "#1
        "APPLY_TOOL(KNIFE,WATER)" #1
    )

    env_sampler = env_factory.EnvironmentFactory(
        "craft/resources/recipes.yaml",
        "craft/resources/hints.yaml",
        7,
        max_steps=300,
        reuse_environments=False,
        visualise=False,
    )
    env = env_sampler.sample_environment(task_name="make[goldarrow]")

    result = evaluator.evaluate_program(program, env=env, max_steps=300)

    assert result["steps"] == len(result["actions_taken"])
    assert result["steps"] > 0
