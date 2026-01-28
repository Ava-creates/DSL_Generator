import pytest

from src.pipeline.dsl_evaluator import DSLEvaluator


class DummyEnv:
    def __init__(self, task_name="make[knife]"):
        self.task_name = task_name
        self.reset_called = False
        self.steps = []

    def reset(self):
        self.reset_called = True

    def step(self, action):
        # Record action and return fixed reward without terminating the task.
        self.steps.append(action)
        return 5.0, False, {}


def _build_cfg():
    # Minimal CFG for a single MOVE() statement.
    return """
program ::= statement
statement ::= MOVE LPAR RPAR
LPAR ::= '('
RPAR ::= ')'
""".strip()


def test_dsl_evaluator_success_is_reward_based():
    cfg = _build_cfg()

    def move(_env):
        # Two actions -> total reward 10 in DummyEnv.
        return [1, 2]

    evaluator = DSLEvaluator(cfg=cfg, function_implementations={"move": move})
    env = DummyEnv(task_name="make[knife]")

    result = evaluator.evaluate_program("MOVE()", env=env, max_steps=10)

    assert env.reset_called is True
    assert result["steps"] == 2
    assert result["total_reward"] == pytest.approx(10.0)
    # Success is based purely on reward threshold (>= 10), not task completion.
    assert result["success"] is True


def test_dsl_evaluator_does_not_check_task_completion():
    cfg = _build_cfg()

    def move(_env):
        # One action gives reward 5, below success threshold.
        return [42]

    evaluator = DSLEvaluator(cfg=cfg, function_implementations={"move": move})
    env = DummyEnv(task_name="make[goldarrow]")

    result = evaluator.evaluate_program("MOVE()", env=env, max_steps=10)

    assert result["total_reward"] == pytest.approx(5.0)
    assert result["success"] is False
