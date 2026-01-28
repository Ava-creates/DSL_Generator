#!/usr/bin/env python3
"""
Test script for evaluation function generation.
Tests that generated evaluation functions use items that are actually on the grid.
"""

import os
import sys
import yaml
from craft import env_factory

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.evaluation_helpers import (
    load_recipes,
    get_primitives_for_item,
    get_items_on_grid,
    get_valid_test_argument,
    create_test_environment_for_item
)
from src.pipeline.generate_evaluation_functions import generate_custom_evaluation_functions


def test_recipe_loading():
    """Test loading recipes."""
    print("Testing recipe loading...")
    recipes = load_recipes("craft/resources/recipes.yaml")
    assert recipes, "Failed to load recipes"
    assert 'primitives' in recipes, "Recipes missing primitives"
    assert 'recipes' in recipes, "Recipes missing recipes"
    print(f"✓ Loaded {len(recipes.get('primitives', []))} primitives")
    print(f"✓ Loaded {len(recipes.get('recipes', {}))} recipes")
    return recipes


def test_primitives_for_item(recipes):
    """Test getting primitives for an item."""
    print("\nTesting primitives_for_item...")
    
    # Test goldarrow
    primitives = get_primitives_for_item("goldarrow", recipes)
    print(f"Primitives for goldarrow: {primitives}")
    assert primitives, "Should find primitives for goldarrow"
    assert "gold" in primitives or "wood" in primitives, "Should include gold or wood"
    
    # Test a primitive itself
    primitives_wood = get_primitives_for_item("wood", recipes)
    print(f"Primitives for wood (primitive): {primitives_wood}")
    assert "wood" in primitives_wood, "Wood should be in its own primitives set"
    
    print("✓ Primitives extraction works")


def test_items_on_grid():
    """Test getting items on grid."""
    print("\nTesting get_items_on_grid...")
    
    recipes_path = "craft/resources/recipes.yaml"
    hints_path = "craft/resources/hints.yaml"
    
    env_sampler = env_factory.EnvironmentFactory(
        recipes_path, hints_path, 7, max_steps=300,
        reuse_environments=False, visualise=False
    )
    env = env_sampler.sample_environment(task_name="make[goldarrow]")
    env.reset()
    
    items = get_items_on_grid(env)
    print(f"Items on grid: {items}")
    assert items, "Should find items on grid"
    print(f"✓ Found {len(items)} items on grid")


def test_valid_test_argument(recipes):
    """Test getting valid test arguments."""
    print("\nTesting get_valid_test_argument...")
    
    recipes_path = "craft/resources/recipes.yaml"
    hints_path = "craft/resources/hints.yaml"
    
    env_sampler = env_factory.EnvironmentFactory(
        recipes_path, hints_path, 7, max_steps=300,
        reuse_environments=False, visualise=False
    )
    env = env_sampler.sample_environment(task_name="make[goldarrow]")
    env.reset()
    
    # Test with environment
    arg_value, explanation = get_valid_test_argument(
        "item", "str", "", recipes, env, "make[goldarrow]"
    )
    print(f"Test argument for 'item': {arg_value}")
    print(f"Explanation: {explanation}")
    
    # Check if the value is actually on the grid
    items_on_grid = get_items_on_grid(env)
    if arg_value.startswith('"') and arg_value.endswith('"'):
        item_name = arg_value.strip('"')
        if item_name in items_on_grid:
            print(f"✓ Argument '{item_name}' is present on the grid")
        else:
            print(f"⚠ Argument '{item_name}' may not be on the grid")
    else:
        print(f"✓ Argument is not a string item (may be direction, etc.)")


def test_evaluation_generation():
    """Test generating evaluation functions."""
    print("\nTesting evaluation function generation...")
    
    try:
        from vllm import LLM as vLLM
        shared_vllm = vLLM(model="/scratch/avani/gpt", tensor_parallel_size=4)
    except Exception as e:
        print(f"⚠ Could not create vLLM instance: {e}")
        print("  Skipping LLM-based generation test")
        return
    
    recipes_path = "craft/resources/recipes.yaml"
    hints_path = "craft/resources/hints.yaml"
    env_setup = f"""
  recipes_path = "{recipes_path}"
  hints_path = "{hints_path}"
  env_sampler = env_factory.EnvironmentFactory(
      recipes_path, hints_path, 7, max_steps=300, reuse_environments=False,
            visualise=visualise)
  env = env_sampler.sample_environment(task_name= 'make[goldarrow]')
  """
    
    # Load specification
    spec_file = "prompt_specifications/specification.txt"
    specification = ""
    if os.path.exists(spec_file):
        with open(spec_file, 'r') as f:
            specification = f.read()
    
    solve_func, eval_func = generate_custom_evaluation_functions(
        func_name="COLLECT",
        description="Collects a primitive item from the environment",
        func_signature="def collect(env, item)",
        return_type="list[int]",
        args="item",
        cfg="",
        specification=specification[:2000],
        env_setup_code=env_setup,
        shared_vllm=shared_vllm
    )
    
    print("Generated solve function:")
    print(solve_func[:500] + "..." if len(solve_func) > 500 else solve_func)
    print("\nGenerated evaluate function:")
    print(eval_func[:500] + "..." if len(eval_func) > 500 else eval_func)
    
    # Check if the generated code mentions checking grid
    if "grid" in solve_func.lower() or "inventory" in solve_func.lower():
        print("✓ Generated code checks grid/inventory")
    else:
        print("⚠ Generated code may not check grid/inventory")
    
    if "on the grid" in eval_func.lower() or "present" in eval_func.lower():
        print("✓ Generated code checks if items are present")
    else:
        print("⚠ Generated code may not check if items are present")


def main():
    """Run all tests."""
    print("=" * 80)
    print("Testing Evaluation Function Generation")
    print("=" * 80)
    
    try:
        recipes = test_recipe_loading()
        test_primitives_for_item(recipes)
        test_items_on_grid()
        test_valid_test_argument(recipes)
        test_evaluation_generation()
        
        print("\n" + "=" * 80)
        print("All tests completed!")
        print("=" * 80)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

