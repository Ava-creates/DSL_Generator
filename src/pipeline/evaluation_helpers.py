#!/usr/bin/env python3
"""
Helper functions for generating evaluation function arguments that are actually present on the grid.
"""

import re
import yaml
from typing import Dict, List, Optional, Set, Tuple

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
    
    For item-related arguments (item, primitive, tool, obstacle), uses grid methods.
    For other arguments (direction, workshop, etc.), uses CFG.
    
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
    def _quote_if_string(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        is_quoted = (isinstance(value, str) and 
                    len(value) >= 2 and 
                    value.startswith('"') and value.endswith('"'))
        if is_quoted:
            return value
        # If the type is explicitly string, always quote.
        if arg_type == "str":
            return f'"{value}"'
        # If type inference was off, still quote enum-like identifiers.
        if arg_type not in ["int", "float"] and re.match(r'^[A-Z_][A-Z0-9_]*$', value):
            return f'"{value}"'
        return value

    # Determine if this is an item-related argument that should use grid methods
    arg_name_lower = arg_name.lower()
    is_item_related = arg_name_lower in ['item', 'primitive', 'tool', 'obstacle']
    
    # First, try to get value from CFG
    arg_value = None
    if cfg:
        try:
            # Lazy import to avoid dependency issues
            from src.pipeline.cfg_to_funsearch_pipeline import resolve_to_terminal_value
            arg_value = resolve_to_terminal_value(arg_name.upper(), cfg)
        except (ImportError, ModuleNotFoundError):
            # If we can't import, skip CFG-based resolution
            pass
    
    # For non-item-related arguments (like direction, workshop), use CFG directly
    if not is_item_related:
        if arg_value:
            arg_value = _quote_if_string(arg_value)
            return arg_value, "# Argument value from CFG"
        else:
            # Last resort: type-based defaults
            if arg_type == "str":
                return '"test_input"', "# Default test value"
            elif arg_type == "int":
                return "0", "# Default integer value"
            elif arg_type == "float":
                return "0.0", "# Default float value"
            else:
                return '"test_input"', "# Default test value"
    
    # For item-related arguments, use grid methods
    # If we have recipes and task_name, try to get related items
    candidate_items = set()
    
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
    items_on_grid = set()
    if env:
        items_on_grid = get_items_on_grid(env)
        # Filter candidates to only those on the grid
        candidate_items = candidate_items.intersection(items_on_grid) if candidate_items else items_on_grid
    
    # If we found items on grid, prefer those
    if items_on_grid:
        # Prefer items that match the argument name or are related to the task
        preferred = None
        if candidate_items:
            preferred = list(candidate_items)[0]  # Use first candidate
        else:
            # Use any item on grid
            preferred = list(items_on_grid)[0]
        
        if preferred:
            # Quote if string type
            if arg_type == "str":
                return f'"{preferred}"', f"# Item '{preferred}' is present on the grid"
            else:
                return preferred, f"# Item '{preferred}' is present on the grid"
    
    # Fallback to CFG value if found
    if arg_value:
        arg_value = _quote_if_string(arg_value)
        return arg_value, "# Argument value from CFG"
    
    # Last resort: type-based defaults
    if arg_type == "str":
        return '"test_input"', "# Default test value (item may not be on grid)"
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
    env_factory_module
) -> Optional[object]:
    """Create a test environment that contains the given item or its primitives on the grid.
    
    Args:
        item_name: Name of item to test (e.g., "goldarrow")
        recipes_path: Path to recipes YAML
        hints_path: Path to hints YAML
        env_factory_module: Environment factory module (e.g., env_factory)
        
    Returns:
        Environment instance or None if creation failed
    """
    try:
        # Create environment with a task related to this item
        task_name = f"make[{item_name}]"
        env_sampler = env_factory_module.EnvironmentFactory(
            recipes_path, hints_path, 7, max_steps=400,
            reuse_environments=False, visualise=False
        )
        env = env_sampler.sample_environment(task_name=task_name)
        env.reset()
        return env
    except Exception as e:
        print(f"Warning: Could not create test environment for {item_name}: {e}")
        return None

