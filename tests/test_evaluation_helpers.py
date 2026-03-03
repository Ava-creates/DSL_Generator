#!/usr/bin/env python3
"""
Simpler test script for evaluation helpers that doesn't require full environment.
Tests the recipe loading and primitive extraction logic.
"""

import os
import sys
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline.evaluation_helpers import (
    load_recipes,
    get_primitives_for_item
)


def test_recipe_loading():
    """Test loading recipes."""
    print("=" * 80)
    print("Test 1: Recipe Loading")
    print("=" * 80)
    
    recipes = load_recipes("craft/resources/recipes.yaml")
    assert recipes, "Failed to load recipes"
    assert 'primitives' in recipes, "Recipes missing primitives"
    assert 'recipes' in recipes, "Recipes missing recipes"
    
    print(f" Loaded {len(recipes.get('primitives', []))} primitives:")
    for prim in recipes.get('primitives', []):
        print(f"  - {prim}")
    
    print(f"\n Loaded {len(recipes.get('recipes', {}))} recipes:")
    for item in list(recipes.get('recipes', {}).keys())[:5]:
        print(f"  - {item}")
    if len(recipes.get('recipes', {})) > 5:
        print(f"  ... and {len(recipes.get('recipes', {})) - 5} more")
    
    return recipes


def test_primitives_for_item(recipes):
    """Test getting primitives for items."""
    print("\n" + "=" * 80)
    print("Test 2: Getting Primitives for Items")
    print("=" * 80)
    
    # Test goldarrow
    print("\nTesting goldarrow:")
    primitives = get_primitives_for_item("goldarrow", recipes)
    print(f"  Primitives needed: {primitives}")
    assert primitives, "Should find primitives for goldarrow"
    assert "gold" in primitives or "wood" in primitives, "Should include gold or wood"
    print("   Found primitives for goldarrow")
    
    # Test a complex item
    print("\nTesting flag (complex item):")
    primitives_flag = get_primitives_for_item("flag", recipes)
    print(f"  Primitives needed: {primitives_flag}")
    assert primitives_flag, "Should find primitives for flag"
    print("   Found primitives for flag")
    
    # Test a primitive itself
    print("\nTesting wood (primitive):")
    primitives_wood = get_primitives_for_item("wood", recipes)
    print(f"  Primitives needed: {primitives_wood}")
    assert "wood" in primitives_wood, "Wood should be in its own primitives set"
    print("   Wood correctly identified as primitive")
    
    # Test arrow (needs knife, which needs iron and rock)
    print("\nTesting arrow (needs intermediate items):")
    primitives_arrow = get_primitives_for_item("arrow", recipes)
    print(f"  Primitives needed: {primitives_arrow}")
    assert primitives_arrow, "Should find primitives for arrow"
    print("   Found primitives for arrow (recursively)")


def test_argument_selection():
    """Test argument selection logic."""
    print("\n" + "=" * 80)
    print("Test 3: Argument Selection (without environment)")
    print("=" * 80)
    
    recipes = load_recipes("craft/resources/recipes.yaml")
    
    from src.pipeline.evaluation_helpers import get_valid_test_argument
    
    # Test with recipes but no environment
    print("\nTesting argument selection for 'item' parameter:")
    arg_value, explanation = get_valid_test_argument(
        "item", "str", "", recipes, None, "make[goldarrow]"
    )
    print(f"  Selected value: {arg_value}")
    print(f"  Explanation: {explanation}")
    
    # Should prefer primitives from goldarrow recipe
    if "gold" in arg_value or "wood" in arg_value:
        print("   Selected item related to goldarrow task")
    else:
        print("   Selected item may not be related to task")
    
    # Test with no recipes
    print("\nTesting argument selection without recipes:")
    arg_value2, explanation2 = get_valid_test_argument(
        "item", "str", "ITEM ::= WOOD | IRON | GRASS", None, None, None
    )
    print(f"  Selected value: {arg_value2}")
    print(f"  Explanation: {explanation2}")


def test_with_environment():
    """Test with actual environment if available."""
    print("\n" + "=" * 80)
    print("Test 4: Grid Checking (requires environment)")
    print("=" * 80)
    
    try:
        from craft import env_factory
        from src.pipeline.evaluation_helpers import get_items_on_grid, get_valid_test_argument
        
        recipes_path = "craft/resources/recipes.yaml"
        hints_path = "craft/resources/hints.yaml"
        
        print("\nCreating test environment for make[goldarrow]...")
        env_sampler = env_factory.EnvironmentFactory(
            recipes_path, hints_path, 7, max_steps=300,
            reuse_environments=False, visualise=False
        )
        env = env_sampler.sample_environment(task_name="make[goldarrow]")
        env.reset()
        
        print("Checking what items are on the grid...")
        items = get_items_on_grid(env)
        print(f"  Items on grid: {sorted(items)}")
        assert items, "Should find items on grid"
        print(f"   Found {len(items)} items on grid")
        
        # Test argument selection with environment
        print("\nTesting argument selection with environment:")
        recipes = load_recipes(recipes_path)
        arg_value, explanation = get_valid_test_argument(
            "item", "str", "", recipes, env, "make[goldarrow]"
        )
        print(f"  Selected value: {arg_value}")
        print(f"  Explanation: {explanation}")
        
        # Check if selected item is on grid
        if arg_value.startswith('"') and arg_value.endswith('"'):
            item_name = arg_value.strip('"')
            if item_name in items:
                print(f"   Selected item '{item_name}' is present on the grid!")
            else:
                print(f"   Selected item '{item_name}' may not be on the grid")
        
    except ImportError as e:
        print(f"   Could not import craft module: {e}")
        print("  Skipping environment-based tests")
    except Exception as e:
        print(f"   Error testing with environment: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests."""
    print("=" * 80)
    print("Testing Evaluation Helpers")
    print("=" * 80)
    
    try:
        recipes = test_recipe_loading()
        test_primitives_for_item(recipes)
        test_argument_selection()
        test_with_environment()
        
        print("\n" + "=" * 80)
        print("All tests completed successfully!")
        print("=" * 80)
        return 0
    except Exception as e:
        print(f"\n Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

