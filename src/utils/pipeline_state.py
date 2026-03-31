#!/usr/bin/env python3
"""
Pipeline State Management
Manages pipeline state using a simple text file to track terminal function counts.
"""

import os
import json
import subprocess
from typing import Dict, List


def get_state_file_path(experiment_dir: str) -> str:
    """Get the path to the pipeline state file."""
    return os.path.join(experiment_dir, "pipeline_state.txt")


def read_state(experiment_dir: str) -> Dict:
    """Read pipeline state from file.
    
    Returns:
        Dictionary with state values:
        - function_implementation_remaining: Number of function implementations still being processed
        - function_implementation_total: Total number of function implementations
          This replaces terminal_functions_remaining/total and explicit_feedback_remaining/total
          since implement_cfg packages FunSearch + Explicit Feedback together
        - test_tasks_total: Total number of test tasks (informational only, test_tasks runs in single job)
        - test_tasks_submitted: Whether test task jobs have been submitted (0 or 1)
        - phase: Current phase (initial, function_evolution, dsl_evolution)
        - dsl_round: Current DSL evolution round
        - func_evolution_round: Current function evolution round
        - tasks: JSON string of tasks list
    """
    state_file = get_state_file_path(experiment_dir)
    
    if not os.path.exists(state_file):
        return {
            "function_implementation_remaining": 0,
            "function_implementation_total": 0,
            "test_tasks_total": 0,  # Informational only (test_tasks runs in single job)
            "test_tasks_submitted": 0,
            "file_generation_submitted": 0,
            "dsl_evolution_submitted": 0,
            "phase": "initial",
            "dsl_round": 0,
            "max_dsl_evolutions": 2,
            "dsl_evolutions_remaining": 3,
            "func_evolution_round": 0,
            "max_function_evolutions": 1,
            "tasks": "[]"
        }
    
    state = {}
    try:
        with open(state_file, 'r') as f:
            for line in f:
                line = line.strip()
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    # Try to parse as int first, then keep as string
                    try:
                        state[key] = int(value)
                    except ValueError:
                        state[key] = value
    except Exception:
        pass
    
    # Ensure required keys exist with defaults
    defaults = {
        "function_implementation_remaining": 0,
        "function_implementation_total": 0,
        "test_tasks_total": 0,  # Informational only (test_tasks runs in single job)
        "test_tasks_submitted": 0,
        "file_generation_submitted": 0,
        "dsl_evolution_submitted": 0,
        "phase": "initial",
        "dsl_round": 0,
        "max_dsl_evolutions": 3,
        "dsl_evolutions_remaining": 3,
        "func_evolution_round": 0,
        "max_function_evolutions": 1,
        "tasks": "[]"
    }
    
    for key, default_value in defaults.items():
        if key not in state:
            state[key] = default_value
    
    return state


def write_state(experiment_dir: str, state: Dict) -> None:
    """Write pipeline state to file.
    
    Args:
        experiment_dir: Experiment directory
        state: Dictionary with state values to write
    """
    state_file = get_state_file_path(experiment_dir)
    
    # Ensure experiment directory exists
    os.makedirs(experiment_dir, exist_ok=True)
    
    with open(state_file, 'w') as f:
        for key, value in state.items():
            # Convert lists/dicts to JSON strings
            if isinstance(value, (list, dict)):
                value = json.dumps(value)
            f.write(f"{key}={value}\n")


def mark_explicit_feedback_submitted(experiment_dir: str) -> bool:
    """Mark explicit feedback as submitted atomically.
    
    Returns:
        True if this was the first to mark it (should submit jobs), False if already marked
    """
    import fcntl
    import time
    
    state_file = get_state_file_path(experiment_dir)
    lock_file = state_file + ".lock"
    
    # Ensure experiment directory exists
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Try to acquire lock
    max_retries = 10
    retry_delay = 0.1
    
    for attempt in range(max_retries):
        try:
            with open(lock_file, 'w') as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                
                # Read current state
                state = read_state(experiment_dir)
                
                # Check if already submitted
                if state.get("explicit_feedback_submitted", 0) == 1:
                    return False
                
                # Mark as submitted
                state["explicit_feedback_submitted"] = 1
                write_state(experiment_dir, state)
                
                return True
        except (IOError, OSError):
            # Lock is held, wait and retry
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                # Fallback: check without lock
                state = read_state(experiment_dir)
                if state.get("explicit_feedback_submitted", 0) == 1:
                    return False
                state["explicit_feedback_submitted"] = 1
                write_state(experiment_dir, state)
                return True
    
    # Fallback
    state = read_state(experiment_dir)
    if state.get("explicit_feedback_submitted", 0) == 1:
        return False
    state["explicit_feedback_submitted"] = 1
    write_state(experiment_dir, state)
    return True


def update_state(experiment_dir: str, **updates: int) -> Dict[str, int]:
    """Update pipeline state with new values.
    
    Args:
        experiment_dir: Experiment directory
        **updates: Key-value pairs to update in state
    
    Returns:
        Updated state dictionary
    """
    state = read_state(experiment_dir)
    state.update(updates)
    write_state(experiment_dir, state)
    return state


def decrement_function_implementation(experiment_dir: str) -> int:
    """Decrement function implementation remaining count atomically.
    This replaces decrement_terminal_functions and decrement_explicit_feedback
    since implement_cfg packages FunSearch + Explicit Feedback together.
    
    Uses file locking to prevent race conditions when multiple jobs complete simultaneously.
    
    Returns:
        New count after decrementing
    """
    import fcntl
    import time
    
    state_file = get_state_file_path(experiment_dir)
    lock_file = state_file + ".lock"
    
    # Ensure experiment directory exists
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Try to acquire lock (with retries for concurrent access)
    max_retries = 10
    retry_delay = 0.1
    
    for attempt in range(max_retries):
        try:
            # Open lock file and acquire exclusive lock
            with open(lock_file, 'w') as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                
                # Read current state
                state = read_state(experiment_dir)
                # Support both old and new naming for backward compatibility
                remaining = state.get("function_implementation_remaining",
                                     state.get("terminal_functions_remaining", 0))
                
                # Decrement if > 0
                if remaining > 0:
                    remaining -= 1
                
                # Write updated state
                state["function_implementation_remaining"] = remaining
                # Also update old names for backward compatibility
                if "terminal_functions_remaining" in state:
                    state["terminal_functions_remaining"] = remaining
                write_state(experiment_dir, state)
                
                # Lock is automatically released when file is closed
                return remaining
        except (IOError, OSError):
            # Lock is held by another process, wait and retry
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                # Fallback: try without lock (not ideal but better than failing)
                state = read_state(experiment_dir)
                remaining = state.get("function_implementation_remaining",
                                     state.get("terminal_functions_remaining", 0))
                if remaining > 0:
                    remaining -= 1
                state["function_implementation_remaining"] = remaining
                if "terminal_functions_remaining" in state:
                    state["terminal_functions_remaining"] = remaining
                write_state(experiment_dir, state)
                return remaining
    
    # Should not reach here, but fallback
    state = read_state(experiment_dir)
    remaining = state.get("function_implementation_remaining",
                         state.get("terminal_functions_remaining", 0))
    if remaining > 0:
        remaining -= 1
    state["function_implementation_remaining"] = remaining
    if "terminal_functions_remaining" in state:
        state["terminal_functions_remaining"] = remaining
    write_state(experiment_dir, state)
    return remaining


# decrement_explicit_feedback removed - use decrement_function_implementation instead
# Since implement_cfg packages FunSearch + Explicit Feedback together, we only need one counter
def decrement_explicit_feedback(experiment_dir: str) -> int:
    """DEPRECATED: Use decrement_function_implementation instead.
    This function now calls decrement_function_implementation for backward compatibility.
    """
    return decrement_function_implementation(experiment_dir)

# Keep decrement_terminal_functions for backward compatibility
def decrement_terminal_functions(experiment_dir: str) -> int:
    """DEPRECATED: Use decrement_function_implementation instead.
    This function now calls decrement_function_implementation for backward compatibility.
    """
    return decrement_function_implementation(experiment_dir)


# decrement_test_tasks removed - test_tasks now runs in a single job, so no counter needed



def mark_test_tasks_submitted(experiment_dir: str) -> bool:
    """Mark test tasks as submitted atomically."""
    import fcntl
    import time
    
    state_file = get_state_file_path(experiment_dir)
    lock_file = state_file + ".lock"
    os.makedirs(experiment_dir, exist_ok=True)
    
    max_retries = 10
    retry_delay = 0.1
    
    for attempt in range(max_retries):
        try:
            with open(lock_file, 'w') as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                state = read_state(experiment_dir)
                if state.get("test_tasks_submitted", 0) == 1:
                    return False
                state["test_tasks_submitted"] = 1
                write_state(experiment_dir, state)
                return True
        except (IOError, OSError):
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                state = read_state(experiment_dir)
                if state.get("test_tasks_submitted", 0) == 1:
                    return False
                state["test_tasks_submitted"] = 1
                write_state(experiment_dir, state)
                return True
    
    state = read_state(experiment_dir)
    if state.get("test_tasks_submitted", 0) == 1:
        return False
    state["test_tasks_submitted"] = 1
    write_state(experiment_dir, state)
    return True


def check_and_mark_test_tasks_ready(experiment_dir: str) -> bool:
    """Atomically check if test_tasks should be triggered and mark it as submitted.
    
    This function checks:
    1. function_implementation_remaining == 0
    2. test_tasks_submitted == 0
    
    If both conditions are true, it sets test_tasks_submitted=1 and returns True.
    Otherwise returns False.
    
    This prevents race conditions when multiple jobs finish simultaneously.
    
    Returns:
        True if this was the first to mark it (should submit test_tasks), False otherwise
    """
    import fcntl
    import time
    
    state_file = get_state_file_path(experiment_dir)
    lock_file = state_file + ".lock"
    os.makedirs(experiment_dir, exist_ok=True)
    
    max_retries = 10
    retry_delay = 0.1
    
    for attempt in range(max_retries):
        try:
            with open(lock_file, 'w') as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                
                # Read current state atomically
                state = read_state(experiment_dir)
                
                # Check conditions
                function_impl_remaining = state.get("function_implementation_remaining",
                                                  state.get("terminal_functions_remaining", 0))
                test_tasks_submitted = state.get("test_tasks_submitted", 0)
                
                # Only proceed if both conditions are met
                if function_impl_remaining == 0 and test_tasks_submitted == 0:
                    # Mark as submitted
                    state["test_tasks_submitted"] = 1
                    write_state(experiment_dir, state)
                    return True
                else:
                    return False
        except (IOError, OSError):
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                # Fallback: try without lock
                state = read_state(experiment_dir)
                function_impl_remaining = state.get("function_implementation_remaining",
                                                  state.get("terminal_functions_remaining", 0))
                test_tasks_submitted = state.get("test_tasks_submitted", 0)
                if function_impl_remaining == 0 and test_tasks_submitted == 0:
                    state["test_tasks_submitted"] = 1
                    write_state(experiment_dir, state)
                    return True
                return False
    
    # Final fallback
    state = read_state(experiment_dir)
    function_impl_remaining = state.get("function_implementation_remaining",
                                      state.get("terminal_functions_remaining", 0))
    test_tasks_submitted = state.get("test_tasks_submitted", 0)
    if function_impl_remaining == 0 and test_tasks_submitted == 0:
        state["test_tasks_submitted"] = 1
        write_state(experiment_dir, state)
        return True
    return False



def mark_file_generation_submitted(experiment_dir: str) -> bool:
    """Mark file generation as submitted atomically.
    
    Returns:
        True if this was the first to mark it (should submit job), False if already marked
    """
    import fcntl
    import time
    
    state_file = get_state_file_path(experiment_dir)
    lock_file = state_file + ".lock"
    
    # Ensure experiment directory exists
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Try to acquire lock
    max_retries = 10
    retry_delay = 0.1
    
    for attempt in range(max_retries):
        try:
            with open(lock_file, 'w') as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                
                # Read current state
                state = read_state(experiment_dir)
                
                # Check if already submitted
                if state.get("file_generation_submitted", 0) == 1:
                    return False
                
                # Mark as submitted
                state["file_generation_submitted"] = 1
                write_state(experiment_dir, state)
                
                return True
        except (IOError, OSError):
            # Lock is held, wait and retry
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                # Fallback: check without lock
                state = read_state(experiment_dir)
                if state.get("file_generation_submitted", 0) == 1:
                    return False
                state["file_generation_submitted"] = 1
                write_state(experiment_dir, state)
                return True
    
    # Fallback
    state = read_state(experiment_dir)
    if state.get("file_generation_submitted", 0) == 1:
        return False
    state["file_generation_submitted"] = 1
    write_state(experiment_dir, state)
    return True


# mark_implement_cfg_submitted removed - use status files as source of truth


def mark_dsl_evolution_submitted(experiment_dir: str) -> bool:
    """Mark DSL evolution as submitted atomically.
    
    Returns:
        True if this was the first to mark it (should submit job), False if already marked
    """
    import fcntl
    import time
    
    state_file = get_state_file_path(experiment_dir)
    lock_file = state_file + ".lock"
    
    # Ensure experiment directory exists
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Try to acquire lock
    max_retries = 10
    retry_delay = 0.1
    
    for attempt in range(max_retries):
        try:
            with open(lock_file, 'w') as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                
                # Read current state
                state = read_state(experiment_dir)
                
                # Check if already submitted
                if state.get("dsl_evolution_submitted", 0) == 1:
                    return False
                
                # Mark as submitted
                state["dsl_evolution_submitted"] = 1
                write_state(experiment_dir, state)
                
                return True
        except (IOError, OSError):
            # Lock is held, wait and retry
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                # Fallback: check without lock
                state = read_state(experiment_dir)
                if state.get("dsl_evolution_submitted", 0) == 1:
                    return False
                state["dsl_evolution_submitted"] = 1
                write_state(experiment_dir, state)
                return True
    
    # Fallback
    state = read_state(experiment_dir)
    if state.get("dsl_evolution_submitted", 0) == 1:
        return False
    state["dsl_evolution_submitted"] = 1
    write_state(experiment_dir, state)
    return True


def submit_job(slurm_script: str, dependencies: List[int] = None, env_vars: Dict[str, str] = None) -> int:
    """Submit a SLURM job and return the job ID.
    
    Args:
        slurm_script: Path to SLURM script
        dependencies: List of job IDs this job depends on
        env_vars: Environment variables to export
    
    Returns:
        Job ID as integer
    """
    cmd = ["sbatch", "--parsable"]
    
    # Add dependencies
    if dependencies:
        # Format: afterok:job1:job2:job3 (all jobs must complete successfully)
        dep_str = "afterok:" + ":".join([str(dep) for dep in dependencies])
        cmd.extend(["--dependency", dep_str])
    
    # Add environment variables
    if env_vars:
        export_str = ",".join([f"{k}={v}" for k, v in env_vars.items()])
        cmd.extend(["--export", f"ALL,{export_str}"])
    else:
        cmd.extend(["--export", "ALL"])
    
    cmd.append(slurm_script)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        job_id = int(result.stdout.strip())
        return job_id
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to submit job: {e.stderr}")
    except ValueError:
        raise RuntimeError(f"Invalid job ID returned: {result.stdout}")

