#!/usr/bin/env python3
"""
Quick helper to exercise ensure_function_grid_spec with vLLM.
Uses the same vLLM setup path as test_evaluation_generation.py.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.grid_generation import ensure_function_grid_spec


def main():
    try:
        from vllm import LLM as vLLM
    except Exception as e:
        print(f" vLLM import failed: {e}")
        return

    # Mirror the model path used elsewhere; adjust if needed for your node.
    model_path = "/scratch/avani/gpt"
    try:
        shared_vllm = vLLM(model=model_path, tensor_parallel_size=4)
    except Exception as e:
        print(f" Could not create vLLM instance: {e}")
        return

    output_path = "experiments/experiment_test/grids/turn_dsl0_case0.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    spec = ensure_function_grid_spec(
        func_name="USE_TOOL",
        description="Turn the agent to face a direction.",
        recipes_path="craft/resources/recipes.yaml",
        output_path=output_path,
        shared_vllm=shared_vllm,
        default_task_name=None,
        prompt_path="prompt_specifications/grid_prompt.txt",
        func_args="direction:int",
        env_description=open("prompt_specifications/nld.txt").read(),
        recipes_text=open("craft/resources/recipes.yaml").read(),
        attempts=5,
    )

    print(f" Generated spec at {output_path}")
    print(spec)


if __name__ == "__main__":
    main()
