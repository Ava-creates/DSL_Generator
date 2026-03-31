"""
Domain-Agnostic DSL Evaluator

This evaluator can parse and execute programs written in any CFG-based DSL.
It dynamically loads function implementations and executes them based on the CFG.
"""

from typing import Dict, List, Any, Optional, Callable
import copy
import os
import sys
import re
try:
    import numpy as np
except ImportError:
    np = None  # numpy is optional

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.pipeline.cfg_parser import CFGParser


class DSLEvaluator:
    """Domain-agnostic evaluator for programs written in a CFG-based DSL."""
    
    def __init__(self, cfg: str, function_implementations: Dict[str, Callable],
                 env_factory: Optional[Callable] = None,
                 env_reset: Optional[Callable] = None,
                 env_step: Optional[Callable] = None):
        """Initialize the DSL evaluator.
        
        Args:
            cfg: The CFG string defining the DSL grammar
            function_implementations: Dictionary mapping function names (sanitized) to callable implementations
            env_factory: Optional function to create environment instances (env_factory(...) -> env)
            env_reset: Optional function to reset environment (env.reset() -> None)
            env_step: Optional function to step environment (env.step(action) -> (reward, done, obs))
        """
        self.cfg_parser = CFGParser(cfg)
        self.function_implementations = function_implementations
        self.env_factory = env_factory
        self.env_reset = env_reset
        self.env_step = env_step
    
    def parse_program(self, program: str) -> bool:
        """Parse a program to check if it's valid according to the CFG.
        
        Args:
            program: Program string in DSL format
            
        Returns:
            True if program is valid, False otherwise
        """
        try:
            self.cfg_parser.parse(program)
            return True
        except Exception as e:
            print(f"Parse error: {e}")
            return False
    
    def tokenize_program(self, program: str) -> List[str]:
        """Tokenize a DSL program into function calls.
        
        Args:
            program: Program string in DSL format
            
        Returns:
            List of tokens (function calls)
        """
        # Split on semicolons and whitespace, keep function calls together
        tokens = []
        current_token = ""
        
        for char in program:
            if char == ';':
                if current_token.strip():
                    tokens.append(current_token.strip())
                    current_token = ""
            elif char.isspace() and current_token and current_token[-1] == ')':
                # End of function call
                tokens.append(current_token.strip())
                current_token = ""
            else:
                current_token += char
        
        if current_token.strip():
            tokens.append(current_token.strip())
        
        return [t for t in tokens if t]
    
    def extract_function_call(self, token: str) -> Optional[tuple]:
        """Extract function name and arguments from a token.
        
        Args:
            token: Token string (e.g., "MOVE(UP)" or "COLLECT(WOOD)")
            
        Returns:
            Tuple of (function_name, args_list) or None if not a valid function call
        """
        # First, try to match the standard pattern: FUNC_NAME(ARG1, ARG2, ...)
        match = re.match(r'^(\w+)\s*\(([^)]*)\)', token)
        if match:
            func_name = match.group(1)
            args_str = match.group(2).strip()
            if args_str:
                args = [arg.strip() for arg in args_str.split(',')]
            else:
                args = []
            return func_name, args
        
        # Fallback: allow bare function names with no parentheses, e.g. "PICKUP".
        # This supports CFGs where a terminal can appear without arguments, while
        # still letting CFGParser enforce whether such tokens are syntactically valid.
        bare_match = re.match(r'^\w+$', token.strip())
        if bare_match:
            func_name = bare_match.group(0)
            return func_name, []
        
        return None

    @staticmethod
    def _format_inventory(env) -> List[str]:
        """Format current inventory as list of 'name=count' strings (nonzero only)."""
        state = getattr(env, "_current_state", None)
        if state is None:
            return []
        inventory = getattr(state, "inventory", None)
        if inventory is None:
            return []
        world = getattr(env, "world", None)
        cookbook = getattr(world, "cookbook", None) if world else None
        index = getattr(cookbook, "index", None) if cookbook else None
        items = []
        for idx, count in enumerate(inventory):
            if not count:
                continue
            name = str(idx)
            if index is not None:
                try:
                    resolved = index.get(idx)
                    if resolved is not None:
                        name = str(resolved)
                except Exception:
                    name = str(idx)
            items.append(f"{name}={int(count)}")
        return items
    
    def evaluate_program(self, program: str, env=None, 
                        max_steps: int = 400,
                        **env_kwargs) -> Dict[str, Any]:
        """Evaluate a DSL program.
        
        Args:
            program: Program string in DSL format
            env: Optional environment instance (if None, will use env_factory)
            max_steps: Maximum number of steps to execute
            **env_kwargs: Additional keyword arguments for environment creation
            
        Returns:
            Dictionary with evaluation results:
            {
                "success": bool,
                "total_reward": float,
                "actions_taken": List[Any],
                "steps": int,
                "error": Optional[str]
            }
        """
        # Parse the program
        if not self.parse_program(program):
            return {
                "success": False,
                "error": "Invalid program syntax",
                "total_reward": 0.0,
                "actions_taken": [],
                "steps": 0,
                "inventory_trace": [],
            }
        
        # Create or use provided environment
        if env is None and self.env_factory:
            env = self.env_factory(**env_kwargs)
        
        # Reset environment if reset function provided
        if env and self.env_reset:
            self.env_reset(env)
        elif env and hasattr(env, 'reset'):
            env.reset()
        
        # Tokenize program
        tokens = self.tokenize_program(program)
        
        total_reward = 0.0
        actions_taken = []
        done = False
        step_count = 0
        inventory_trace = []  # Only when inventory changes: {"token": str, "inventory": list of "name=count"}
        last_inventory = self._format_inventory(env) if env else []  # Initial state, to detect changes

        i = 0
        while i < len(tokens) and not done and step_count < max_steps:
            token = tokens[i]
            
            # Extract function call
            func_call = self.extract_function_call(token)
            if not func_call:
                i += 1
                continue
            
            func_name, args = func_call

            # Normalize string arguments before calling the implementation
            normalized_args = [
                arg.lower() if isinstance(arg, str) else arg
                for arg in args
            ]
                        
            # Sanitize function name to match implementation keys
            safe_name = self._sanitize_function_name(func_name)
            
            # Debug: print available implementations if function not found
            if safe_name not in self.function_implementations:
                # Try alternative lookups
                func_name_lower = func_name.lower()
                if func_name_lower in self.function_implementations:
                    safe_name = func_name_lower
                elif func_name in self.function_implementations:
                    safe_name = func_name
            
            # Get function implementation
            if safe_name in self.function_implementations:
                func = self.function_implementations[safe_name]
                try:
                    env_for_func = env
                    try:
                        env_for_func = copy.deepcopy(env)
                    except Exception:
                        pass
                    
                    # Call function with environment and arguments
                    if args:
                        actions = func(env_for_func, *normalized_args)
                    else:
                        actions = func(env_for_func)

                    # Validate actions: terminal functions must return integer env action codes,
                    # NOT DSL strings like "MOVE(NORTH)" or "PICKUP(IRON)".
                    # Returning DSL strings is a design violation — terminal functions are the
                    # leaves of the DSL and must interact with the environment directly.
                    _action_list = actions if isinstance(actions, list) else [actions]
                    _bad = [a for a in _action_list if isinstance(a, str)]
                    if _bad:
                        raise ValueError(
                            f"Terminal function '{func_name}' returned DSL string tokens "
                            f"{_bad!r} instead of integer environment action codes. "
                            f"Terminal functions are the leaves of the DSL; they must return "
                            f"raw integer action codes accepted by env.step(), not DSL program "
                            f"tokens. Fix the implementation of '{func_name}' to return integers."
                        )

                    # Execute actions
                    if isinstance(actions, list):
                        for action in actions:
                            if self.env_step:
                                reward, done, observations = self.env_step(env, action)
                            elif hasattr(env, 'step'):
                                reward, done, observations = env.step(action)
                            else:
                                raise ValueError("No env_step function provided and env has no step method")
                            
                            total_reward += float(reward) if reward is not None else 0.0
                            actions_taken.append(action)
                            step_count += 1
                            if done:
                                break
                    else:
                        # Single action
                        if self.env_step:
                            reward, done, observations = self.env_step(env, actions)
                        elif hasattr(env, 'step'):
                            reward, done, observations = env.step(actions)
                        else:
                            raise ValueError("No env_step function provided and env has no step method")
                        
                        total_reward += float(reward) if reward is not None else 0.0
                        actions_taken.append(actions)
                        step_count += 1

                    # Record inventory only when it changed
                    inv_items = self._format_inventory(env)
                    if inv_items != last_inventory:
                        inventory_trace.append({"token": token, "inventory": inv_items})
                        last_inventory = inv_items
                    
                except Exception as e:
                    print(f"Error executing {func_name}: {e}")
                    return {
                        "success": False,
                        "error": f"Execution error in {func_name}: {str(e)}",
                        "total_reward": total_reward,
                        "actions_taken": actions_taken,
                        "steps": step_count,
                        "inventory_trace": inventory_trace,
                    }
            else:
                # Debug output: show what functions are available
                available_funcs = list(self.function_implementations.keys())
                print(f"Warning: No implementation found for function {func_name} (sanitized: {safe_name})")
                print(f"  Available functions: {available_funcs}")
                print(f"  Looking for: {safe_name} (from {func_name})")
            
            i += 1

        if total_reward >= 10:
            success = True
        else:
            success = False
        
        return {
            "success": success,
            "total_reward": total_reward,
            "actions_taken": actions_taken,
            "steps": step_count,
            "inventory_trace": inventory_trace,
        }
    
    @staticmethod
    def _sanitize_function_name(func_name: str) -> str:
        """Convert function name to valid Python identifier."""
        func_name = func_name.lower()
        func_name = re.sub(r'\W|^(?=\d)', '_', func_name)
        return func_name


def load_function_implementations(function_dir: str) -> Dict[str, Callable]:
    """Load function implementations from Python files in a directory.
    
    Args:
        function_dir: Directory containing Python files with function implementations
        
    Returns:
        Dictionary mapping sanitized function names to callable implementations
    """
    import importlib.util
    implementations = {}
    
    if not os.path.exists(function_dir):
        return implementations
    
    # Look for Python files
    for filename in os.listdir(function_dir):
        if filename.endswith('.py') and not filename.startswith('__'):
            module_name = filename[:-3]  # Remove .py extension
            file_path = os.path.join(function_dir, filename)
            
            try:
                # Use importlib to load the module from file path directly
                # This avoids conflicts with package names (e.g., craft.py vs craft package)
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if spec is None or spec.loader is None:
                    print(f"Warning: Could not create spec for {module_name} from {file_path}")
                    continue
                
                module = importlib.util.module_from_spec(spec)
                # Execute the module to load it
                spec.loader.exec_module(module)
                
                # Read the source file to check which functions are actually defined in it
                # (not imported from elsewhere)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        source_code = f.read()
                    # Find all function definitions in the source
                    import ast
                    source_tree = ast.parse(source_code, filename=file_path)
                    defined_functions = set()
                    for node in ast.walk(source_tree):
                        if isinstance(node, ast.FunctionDef):
                            defined_functions.add(node.name)
                except Exception:
                    # If we can't parse the source, fall back to checking all attributes
                    defined_functions = None
                
                # Look for functions in the module
                import inspect
                for attr_name in dir(module):
                    # Skip private attributes
                    if attr_name.startswith('_'):
                        continue
                    
                    # If we parsed the source, only include functions that are actually defined in this file
                    if defined_functions is not None and attr_name not in defined_functions:
                        continue
                    
                    attr = getattr(module, attr_name)
                    # Check if it's a callable function (not a class, module, etc.)
                    is_function = inspect.isfunction(attr) or inspect.ismethod(attr)
                    is_builtin = inspect.isbuiltin(attr)
                    
                    if (is_function or is_builtin) and not isinstance(attr, type):
                        # Sanitize function name
                        safe_name = DSLEvaluator._sanitize_function_name(attr_name)
                        implementations[safe_name] = attr
                        # Also map the original name if different
                        if safe_name != attr_name.lower():
                            implementations[attr_name.lower()] = attr
                        print(f"   Loaded function {attr_name} from {filename} (sanitized: {safe_name})")
            except Exception as e:
                print(f"Warning: Could not load module {module_name} from {file_path}: {e}")
                import traceback
                traceback.print_exc()
    
    return implementations


if __name__ == "__main__":
    # Example usage
    # Note: CFG format must use explicit LPAR/RPAR/SEMICOLON terminals
    cfg = """
    program ::= action_seq
    action_seq ::= action
                | action SEMICOLON action_seq
    action ::= MOVE LPAR DIR RPAR
            | COLLECT LPAR ITEM RPAR
            | CRAFT LPAR ITEM RPAR
    DIR ::= UP | DOWN | LEFT | RIGHT
    ITEM ::= WOOD | IRON | STICK
    SEMICOLON ::= ';'
    LPAR ::= '('
    RPAR ::= ')'
    """
    
    # Example function implementations
    def move(env, direction):
        # Placeholder implementation
        return [0]  # Return action code
    
    def collect(env, item):
        # Placeholder implementation
        return [1]
    
    def craft(env, item):
        # Placeholder implementation
        return [2]
    
    implementations = {
        'move': move,
        'collect': collect,
        'craft': craft
    }
    
    evaluator = DSLEvaluator(cfg, implementations)
    
    # Program format uses actual parentheses and semicolons
    program = "MOVE(UP); COLLECT(WOOD); CRAFT(STICK)"
    result = evaluator.evaluate_program(program)
    print("Result:", result)

