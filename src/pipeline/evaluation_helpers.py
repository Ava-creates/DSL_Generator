#!/usr/bin/env python3
"""
Helper functions for generating evaluation function arguments that are actually present on the grid.
"""

import yaml
from typing import Dict, Optional, Set, Tuple

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None


def load_recipes(recipes_path: str) -> Dict:
    """Load recipes from YAML file.
    
    Args:
        recipes_path: Path to recipes YAML file
        
    Returns:
        Dictionary with 'primitives', 'recipes', and 'environment' keys
    """
    try:
        with open(recipes_path, 'r') as f:
            recipes = yaml.load(f, Loader=yaml.FullLoader)
        return recipes
    except Exception as e:
        print(f"Warning: Could not load recipes from {recipes_path}: {e}")
        return {}


def get_primitives_for_item(item_name: str, recipes: Dict) -> Set[str]:
    """Get all primitives needed to craft an item (recursively).
    
    Args:
        item_name: Name of the item (e.g., "goldarrow")
        recipes: Recipes dictionary from YAML
        
    Returns:
        Set of primitive names needed for this item
    """
    if not recipes or 'recipes' not in recipes:
        return set()
    
    primitives_set = set(recipes.get('primitives', []))
    recipes_dict = recipes.get('recipes', {})
    
    if item_name not in recipes_dict:
        # Item might be a primitive itself
        if item_name in primitives_set:
            return {item_name}
        return set()
    
    needed_primitives = set()
    recipe = recipes_dict[item_name]
    
    for ingredient, count in recipe.items():
        if ingredient.startswith('_'):  # Skip special keys like _at
            continue
        if ingredient in primitives_set:
            needed_primitives.add(ingredient)
        elif ingredient in recipes_dict:
            # Recursively get primitives for this ingredient
            sub_primitives = get_primitives_for_item(ingredient, recipes)
            needed_primitives.update(sub_primitives)
    
    return needed_primitives


def get_items_on_grid(env) -> Set[str]:
    """Get names of items/primitives that are actually present on the grid.
    
    Args:
        env: Environment instance with _current_state.grid
        
    Returns:
        Set of item/primitive names present on the grid
    """
    items_on_grid = set()
    
    if not NUMPY_AVAILABLE:
        return items_on_grid
    
    try:
        if not hasattr(env, '_current_state') or not hasattr(env._current_state, 'grid'):
            return items_on_grid
        
        grid = env._current_state.grid
        if not hasattr(env, 'world') or not hasattr(env.world, 'cookbook'):
            return items_on_grid
        
        cookbook = env.world.cookbook
        # Grid shape is (W, H, n_kinds)
        # Check each kind to see if it's present on the grid
        for kind_idx in range(grid.shape[2]):
            # Check if any cell has this kind
            if np.any(grid[:, :, kind_idx] > 0):
                # Get the name for this index
                try:
                    item_name = cookbook.index.get(kind_idx)
                    if item_name:
                        items_on_grid.add(item_name)
                except (KeyError, AttributeError):
                    pass
    
    except Exception as e:
        print(f"Warning: Could not get items on grid: {e}")
    
    return items_on_grid


def get_valid_test_argument(
    arg_name: str,
    arg_type: str,
    cfg: str,
    recipes: Optional[Dict] = None,
    env: Optional[object] = None,
    task_name: Optional[str] = None
) -> Tuple[Optional[str], str]:
    """Get a valid test argument value.
    
    Selection strategy is domain-agnostic:
    1) Prefer CFG leaf values for this argument symbol.
    2) If possible, prefer CFG values that are also present on the current grid.
    3) Fall back to type-based defaults.
    
    Args:
        arg_name: Name of the argument (e.g., "item", "direction")
        arg_type: Type of argument ("str", "int", "float")
        cfg: CFG string for context
        recipes: Optional recipes dictionary
        env: Optional environment instance to check grid
        task_name: Optional task name (e.g., "make[goldarrow]") to get related items
        
    Returns:
        Tuple of (argument_value, explanation)
        - argument_value: The value to use (quoted if string, or None if not found)
        - explanation: Explanation of how the value was chosen
    """
    def _format_value(value: str) -> str:
        if arg_type == "str":
            return f'"{value}"'
        return value

    # Expand allowed leaf values for this argument from CFG (domain-agnostic).
    allowed_values: set[str] = set()
    if cfg:
        try:
            from src.pipeline.cfg_symbol_utils import build_cfg_rule_map, expand_symbol_to_terminals
            rule_map = build_cfg_rule_map(cfg)
            symbol = arg_name.strip().upper()
            if symbol in rule_map:
                allowed_values = expand_symbol_to_terminals(symbol, rule_map)
        except Exception:
            allowed_values = set()

    # If we have recipes and task_name, collect task-related candidates.
    candidate_items: set[str] = set()
    
    if recipes and task_name:
        # Extract item name from task (e.g., "make[goldarrow]" -> "goldarrow")
        import re
        task_match = re.search(r'\[([^\]]+)\]', task_name)
        if task_match:
            task_item = task_match.group(1)
            # Get primitives needed for this item
            primitives = get_primitives_for_item(task_item, recipes)
            candidate_items.update(primitives)
            # Also add the item itself if it's a primitive
            if task_item in recipes.get('primitives', []):
                candidate_items.add(task_item)
    
    # If we have an environment, check what's actually on the grid
    items_on_grid: set[str] = set()
    if env:
        items_on_grid = get_items_on_grid(env)

    allowed_upper = {v.upper() for v in allowed_values}
    allowed_lower_to_value = {v.lower(): v for v in allowed_values}
    task_candidates_lower = {str(v).strip().lower() for v in candidate_items if str(v).strip()}
    grid_items_lower = {str(v).strip().lower() for v in items_on_grid if str(v).strip()}

    # Prefer task-relevant values that are both allowed by CFG and on-grid.
    if allowed_values and task_candidates_lower and grid_items_lower:
        overlap = sorted(task_candidates_lower.intersection(grid_items_lower))
        for candidate_lower in overlap:
            candidate_upper = candidate_lower.upper()
            if candidate_upper in allowed_upper:
                chosen = allowed_lower_to_value.get(candidate_lower, candidate_upper)
                return _format_value(chosen), f"# CFG value '{chosen}' is task-relevant and present on grid"

    # Otherwise prefer allowed CFG values that are present on-grid.
    if allowed_values and grid_items_lower:
        for candidate_lower in sorted(grid_items_lower):
            candidate_upper = candidate_lower.upper()
            if candidate_upper in allowed_upper:
                chosen = allowed_lower_to_value.get(candidate_lower, candidate_upper)
                return _format_value(chosen), f"# CFG value '{chosen}' is present on the grid"

    # Fall back to any allowed CFG leaf value.
    if allowed_values:
        chosen = sorted(allowed_values)[0]
        return _format_value(chosen), "# Argument value from CFG leaf set"
    
    # Last resort: type-based defaults
    if arg_type == "str":
        return '"test_input"', "# Default test value"
    elif arg_type == "int":
        return "0", "# Default integer value"
    elif arg_type == "float":
        return "0.0", "# Default float value"
    else:
        return '"test_input"', "# Default test value"


def create_test_environment_for_item(
    item_name: str,
    recipes_path: str,
    hints_path: str,
    env_factory_module,
    task_name: Optional[str] = None
) -> Optional[object]:
    """Create a test environment for a specific task.
    
    Args:
        item_name: Name of item for fallback task construction
        recipes_path: Path to recipes YAML
        hints_path: Path to hints YAML
        env_factory_module: Environment factory module (e.g., env_factory)
        task_name: Optional explicit task name (e.g., "get[grass]" or "make[goldarrow]")
        
    Returns:
        Environment instance or None if creation failed
    """
    try:
        effective_task_name = task_name or f"make[{item_name}]"
        env_sampler = env_factory_module.EnvironmentFactory(
            recipes_path, hints_path, 7, max_steps=400,
            reuse_environments=False, visualise=False
        )
        env = env_sampler.sample_environment(task_name=effective_task_name)
        env.reset()
        return env
    except Exception as e:
        print(f"Warning: Could not create test environment for task {task_name or f'make[{item_name}]'}: {e}")
        return None

