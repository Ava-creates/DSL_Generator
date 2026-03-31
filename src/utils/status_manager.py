#!/usr/bin/env python3
"""
Status File Manager
Manages organization of status files by DSL level.
Organizes status files into dsl0, dsl1, dsl2, etc. folders.
"""

import os
import json
from typing import List, Optional


def get_dsl_status_dir(experiment_dir: str, dsl_round: int) -> str:
    """Get the path to the DSL round-specific status directory.
    
    Args:
        experiment_dir: Path to experiment directory
        dsl_round: DSL evolution round number (0, 1, 2, etc.)
    
    Returns:
        Path to dsl{N}/ directory
    """
    return os.path.join(experiment_dir, "status", f"dsl{dsl_round}")


def get_stage_status_path(experiment_dir: str, dsl_round: int, stage_name: str,
                         filename: Optional[str] = None) -> str:
    """Get the path to a stage status file organized by DSL round.
    
    Args:
        experiment_dir: Path to experiment directory
        dsl_round: DSL evolution round number
        stage_name: Name of the stage (e.g., 'evolve_dsl', 'file_generation')
        filename: Optional filename (defaults to 'status')
    
    Returns:
        Path to dsl{N}/{stage_name}/filename
    """
    if filename is None:
        filename = "status"
    
    status_dir = os.path.join(experiment_dir, "status", f"dsl{dsl_round}", stage_name)
    return os.path.join(status_dir, filename)


def get_function_status_path(experiment_dir: str, dsl_round: int, stage_name: str,
                             function_name: str, ext: str = "json") -> str:
    """Get the path to a function-specific status file organized by DSL round.
    
    Args:
        experiment_dir: Path to experiment directory
        dsl_round: DSL evolution round number
        stage_name: Name of the stage (e.g., 'funsearch', 'explicit_feedback')
        function_name: Name of the function
        ext: File extension (default 'json')
    
    Returns:
        Path to dsl{N}/{stage_name}/{function_name}.{ext}
    """
    status_dir = os.path.join(experiment_dir, "status", f"dsl{dsl_round}", stage_name)
    return os.path.join(status_dir, f"{function_name}.{ext}")


def write_status(experiment_dir: str, dsl_round: int, stage_name: str,
                status_data: dict, filename: str = "status") -> None:
    """Write a status file to the DSL-organized directory structure.
    
    Args:
        experiment_dir: Path to experiment directory
        dsl_round: DSL evolution round number
        stage_name: Name of the stage
        status_data: Dictionary to write as JSON
        filename: Filename (default 'status')
    """
    # Write to versioned location only
    versioned_path = get_stage_status_path(experiment_dir, dsl_round, stage_name, filename)
    os.makedirs(os.path.dirname(versioned_path), exist_ok=True)
    
    with open(versioned_path, 'w') as f:
        json.dump(status_data, f, indent=2)


def write_function_status(experiment_dir: str, dsl_round: int, stage_name: str,
                         function_name: str, status_data: dict) -> None:
    """Write a function-specific status file to the DSL-organized directory structure.
    
    Args:
        experiment_dir: Path to experiment directory
        dsl_round: DSL evolution round number
        stage_name: Name of the stage (e.g., 'funsearch', 'explicit_feedback')
        function_name: Name of the function
        status_data: Dictionary to write as JSON
    """
    # Write to versioned location only
    versioned_path = get_function_status_path(experiment_dir, dsl_round, stage_name, function_name)
    os.makedirs(os.path.dirname(versioned_path), exist_ok=True)
    
    with open(versioned_path, 'w') as f:
        json.dump(status_data, f, indent=2)


def read_status(experiment_dir: str, dsl_round: int, stage_name: str,
               filename: str = "status") -> Optional[dict]:
    """Read a status file from the DSL-organized directory structure.
    
    Args:
        experiment_dir: Path to experiment directory
        dsl_round: DSL evolution round number
        stage_name: Name of the stage
        filename: Filename (default 'status')
    
    Returns:
        Dictionary loaded from JSON, or None if file doesn't exist
    """
    versioned_path = get_stage_status_path(experiment_dir, dsl_round, stage_name, filename)
    
    if os.path.exists(versioned_path):
        with open(versioned_path, 'r') as f:
            return json.load(f)

    if filename == "status":
        fallback_path = get_stage_status_path(experiment_dir, dsl_round, stage_name, "status.json")
        if os.path.exists(fallback_path):
            with open(fallback_path, 'r') as f:
                return json.load(f)
    
    return None


def read_function_status(experiment_dir: str, dsl_round: int, stage_name: str,
                        function_name: str) -> Optional[dict]:
    """Read a function-specific status file from the DSL-organized directory structure.
    
    Args:
        experiment_dir: Path to experiment directory
        dsl_round: DSL evolution round number
        stage_name: Name of the stage
        function_name: Name of the function
    
    Returns:
        Dictionary loaded from JSON, or None if file doesn't exist
    """
    versioned_path = get_function_status_path(experiment_dir, dsl_round, stage_name, function_name)
    
    if os.path.exists(versioned_path):
        with open(versioned_path, 'r') as f:
            return json.load(f)
    
    return None


def list_dsl_rounds(experiment_dir: str) -> List[int]:
    """List all DSL rounds that have status files.
    
    Args:
        experiment_dir: Path to experiment directory
    
    Returns:
        Sorted list of DSL round numbers that exist
    """
    status_dir = experiment_dir
    if not os.path.exists(status_dir):
        return []
    
    rounds = []
    for item in os.listdir(status_dir):
        item_path = os.path.join(status_dir, item)
        if os.path.isdir(item_path) and item.startswith("dsl"):
            try:
                round_num = int(item[3:])
                rounds.append(round_num)
            except ValueError:
                continue
    
    return sorted(rounds)


def list_function_statuses(experiment_dir: str, dsl_round: int, stage_name: str) -> List[str]:
    """List all function-specific status files for a stage and DSL round.
    
    Args:
        experiment_dir: Path to experiment directory
        dsl_round: DSL evolution round number
        stage_name: Name of the stage
    
    Returns:
        List of function names that have status files
    """
    status_dir = os.path.join(experiment_dir, "status", f"dsl{dsl_round}", stage_name)
    
    if not os.path.exists(status_dir):
        return []
    
    functions = []
    for filename in os.listdir(status_dir):
        if filename.endswith(".json"):
            functions.append(filename[:-5])  # Remove .json extension
    
    return sorted(functions)
