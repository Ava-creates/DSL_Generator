#!/usr/bin/env python3
"""
CFG Evaluator for DSL Programs

This evaluator can be used directly in program synthesis.
It takes a CFG, terminal functions, program, and environment, and evaluates the program.
"""

import os
import sys
import time
from typing import Dict, List, Any, Optional, Callable

# Add paths for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.pipeline.cfg_parser import CFGParser
from src.pipeline.dsl_evaluator import DSLEvaluator, load_function_implementations


class CFGEvaluator:
    """Evaluator for programs written in a CFG-based DSL.
    
    This evaluator can be instantiated with a CFG and terminal functions,
    then used to evaluate programs with a given environment.
    Similar to ProgramEvaluator but uses CFG-based parsing.
    """
    
    def __init__(self, cfg: str, final_functions_dir: str = None, 
                 function_implementations: Dict[str, Callable] = None):
        """Initialize the CFG evaluator.
        
        Args:
            cfg: The CFG string defining the DSL grammar
            final_functions_dir: Directory containing final function implementations (Python files)
            function_implementations: Optional dict mapping function names (sanitized) to callable implementations.
                                     If None, will load from final_functions_dir.
        """
        self.cfg = cfg
        
        # Load function implementations
        if function_implementations is not None:
            self.function_implementations = function_implementations
        elif final_functions_dir is not None:
            self.function_implementations = load_function_implementations(final_functions_dir)
        else:
            self.function_implementations = {}
        
        # Create helper functions for environment interaction
        def reset_env(env):
            """Reset the environment."""
            if hasattr(env, 'reset'):
                env.reset()
        
        def step_env(env, action):
            """Step the environment with an action."""
            if hasattr(env, 'step'):
                return env.step(action)
            else:
                raise ValueError("Environment does not have a step method")
        
        self.env_reset = reset_env
        self.env_step = step_env
        
        # Initialize DSL evaluator (without env_factory since we'll pass env directly)
        self.dsl_evaluator = DSLEvaluator(
            cfg=cfg,
            function_implementations=self.function_implementations,
            env_factory=None,  # We'll pass env directly
            env_reset=reset_env,
            env_step=step_env
        )
    
    def parse_program(self, program: str) -> bool:
        """Parse a program to check if it's valid according to the CFG.
        
        Args:
            program: Program string in DSL format
            
        Returns:
            True if program is valid, False otherwise
        """
        return self.dsl_evaluator.parse_program(program)
    
    def evaluate_program(self, program: str, env, max_steps: int = 300, 
                        timeout: Optional[float] = None) -> Dict[str, Any]:
        """Evaluate a DSL program in the given environment.
        
        Args:
            program: DSL program string to evaluate
            env: Environment instance to evaluate the program in
            max_steps: Maximum number of steps to execute
            timeout: Optional timeout in seconds (not currently enforced, for compatibility)
            
        Returns:
            Dictionary with evaluation results (similar to ProgramEvaluator):
            {
                "success": bool,
                "total_reward": float,
                "actions_taken": List[Any],
                "steps": int,
                "evaluation_time": float,
                "error": Optional[str]
            }
        """
        start_time = time.time()
        
        if env is None:
            evaluation_time = time.time() - start_time
            return {
                "success": False,
                "error": "Environment is None",
                "total_reward": 0.0,
                "actions_taken": [],
                "steps": 0,
                "evaluation_time": evaluation_time
            }
        
        # Reset environment
        self.env_reset(env)
        
        # Evaluate using DSL evaluator
        result = self.dsl_evaluator.evaluate_program(program, env=env, max_steps=max_steps)
        
        evaluation_time = time.time() - start_time
        
        # Format result similar to ProgramEvaluator
        return {
            "success": result.get("success", False),
            "total_reward": result.get("total_reward", 0.0),
            "actions_taken": result.get("actions_taken", []),
            "steps": result.get("steps", 0),
            "evaluation_time": evaluation_time,
            "error": result.get("error")
        }


def create_evaluator(cfg: str, final_functions_dir: str = None,
                    function_implementations: Dict[str, Callable] = None) -> CFGEvaluator:
    """Convenience function to create a CFG evaluator.
    
    Args:
        cfg: The CFG string defining the DSL grammar
        final_functions_dir: Directory containing final function implementations
        function_implementations: Optional dict of function implementations
        
    Returns:
        CFGEvaluator instance
    """
    return CFGEvaluator(
        cfg=cfg,
        final_functions_dir=final_functions_dir,
        function_implementations=function_implementations
    )


if __name__ == "__main__":
    # Example usage
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Evaluate DSL programs using the CFG evaluator")
    parser.add_argument(
        '--cfg',
        type=str,
        required=True,
        help='CFG string or path to CFG file (JSON with "cfg" key)'
    )
    parser.add_argument(
        '--final_functions_dir',
        type=str,
        required=True,
        help='Directory containing final function implementations'
    )
    parser.add_argument(
        '--program',
        type=str,
        default=None,
        help='DSL program to evaluate'
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=300,
        help='Maximum number of steps (default: 300)'
    )
    
    args = parser.parse_args()
    
    # Load CFG
    if os.path.exists(args.cfg):
        if args.cfg.endswith('.json'):
            with open(args.cfg, 'r') as f:
                cfg_data = json.load(f)
                cfg = cfg_data.get("cfg", "")
        else:
            with open(args.cfg, 'r') as f:
                cfg = f.read()
    else:
        cfg = args.cfg
    
    # Create evaluator
    evaluator = create_evaluator(cfg=cfg, final_functions_dir=args.final_functions_dir)
    
    if args.program:
        print(f"Evaluating program: {args.program}")
        print("Note: This example requires an environment to be passed.")
        print("In program synthesis, use: evaluator.evaluate_program(program, env, max_steps)")
    else:
        print("CFG Evaluator created successfully!")
        print(f"  CFG loaded: {len(cfg)} characters")
        print(f"  Functions loaded: {len(evaluator.function_implementations)}")
        print("\nUsage in program synthesis:")
        print("  from cfg_evaluator import CFGEvaluator")
        print("  evaluator = CFGEvaluator(cfg=cfg, final_functions_dir='path/to/functions')")
        print("  result = evaluator.evaluate_program(program, env, max_steps=300)")
