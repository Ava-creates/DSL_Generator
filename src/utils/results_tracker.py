"""
Results tracker for storing and plotting program synthesis results.

Tracks:
- Environment interactions during funsearch, explicit feedback, and program synthesis
- Rewards for each synthesized program
- Best reward achieved so far for each task
- CFG version (DSL round) for each result
"""

import os
import json
import sys
import tempfile
import shutil
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime


class ResultsTracker:
    """Tracks results from program synthesis pipeline."""
    
    def __init__(self, experiment_dir: str):
        """Initialize results tracker.
        
        Args:
            experiment_dir: Path to experiment directory
        """
        self.experiment_dir = experiment_dir
        self.results_dir = os.path.join(experiment_dir, "results_tracking")
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.results_file = os.path.join(self.results_dir, "synthesis_results.json")
        self.interactions_file = os.path.join(self.results_dir, "interactions.json")
        self.evolution_metrics_file = os.path.join(self.results_dir, "evolution_metrics.json")
        
        # Load existing results if available
        self.results = self._load_results()
        self.interactions = self._load_interactions()
        self.evolution_metrics = self._load_evolution_metrics()
        
        # Track cumulative interactions per phase
        # Structure: {phase: total_steps}
        # Phases: "funsearch", "explicit_feedback", "program_synthesis"
        if "funsearch" not in self.interactions:
            self.interactions["funsearch"] = 0
        if "explicit_feedback" not in self.interactions:
            self.interactions["explicit_feedback"] = 0
        if "program_synthesis" not in self.interactions:
            self.interactions["program_synthesis"] = 0
        
        # Track per-evolution interactions (reset for each evolution round)
        # Structure: {phase: steps_in_current_evolution}
        self.current_evolution_interactions = {
            "funsearch": 0,
            "explicit_feedback": 0,
            "program_synthesis": 0
        }
    
    def _load_results(self) -> List[Dict]:
        """Load existing results from file."""
        if os.path.exists(self.results_file):
            with open(self.results_file, 'r') as f:
                return json.load(f)
        return []
    
    def _load_interactions(self) -> Dict[str, int]:
        """Load existing interactions from file."""
        if os.path.exists(self.interactions_file):
            with open(self.interactions_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _load_evolution_metrics(self) -> List[Dict]:
        """Load existing evolution metrics from file."""
        if os.path.exists(self.evolution_metrics_file):
            with open(self.evolution_metrics_file, 'r') as f:
                return json.load(f)
        return []
    
    def _save_evolution_metrics(self):
        """Save evolution metrics to file using atomic write."""
        # Write to temporary file first, then atomically rename
        # This prevents corruption if process is killed during write
        try:
            file_dir = os.path.dirname(self.evolution_metrics_file) or '.'
            with tempfile.NamedTemporaryFile(mode='w', dir=file_dir, 
                                             delete=False, suffix='.tmp') as f:
                json.dump(self.evolution_metrics, f, indent=2)
                temp_path = f.name
            shutil.move(temp_path, self.evolution_metrics_file)
        except Exception as e:
            print(f" Error saving evolution metrics: {e}", file=sys.stderr)
            raise
    
    def _save_results(self):
        """Save results to file using atomic write."""
        # Write to temporary file first, then atomically rename
        # This prevents corruption if process is killed during write
        try:
            file_dir = os.path.dirname(self.results_file) or '.'
            with tempfile.NamedTemporaryFile(mode='w', dir=file_dir, 
                                             delete=False, suffix='.tmp') as f:
                json.dump(self.results, f, indent=2)
                f.flush()  # Ensure all data is written before closing
                os.fsync(f.fileno())  # Force write to disk
                temp_path = f.name
            shutil.move(temp_path, self.results_file)
        except Exception as e:
            print(f" Error saving results: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            raise
    
    def _save_interactions(self):
        """Save interactions to file using atomic write."""
        # Write to temporary file first, then atomically rename
        # This prevents corruption if process is killed during write
        try:
            file_dir = os.path.dirname(self.interactions_file) or '.'
            with tempfile.NamedTemporaryFile(mode='w', dir=file_dir, 
                                             delete=False, suffix='.tmp') as f:
                json.dump(self.interactions, f, indent=2)
                temp_path = f.name
            shutil.move(temp_path, self.interactions_file)
        except Exception as e:
            print(f" Error saving interactions: {e}", file=sys.stderr)
            raise
    
    def add_funsearch_interactions(self, steps: int):
        """Add environment steps from funsearch.
        
        Args:
            steps: Number of environment steps taken during funsearch
        """
        self.interactions["funsearch"] += steps
        self.current_evolution_interactions["funsearch"] += steps
        self._save_interactions()
    
    def add_explicit_feedback_interactions(self, steps: int):
        """Add environment steps from explicit feedback generation.
        
        Args:
            steps: Number of environment steps taken during explicit feedback
        """
        self.interactions["explicit_feedback"] += steps
        self.current_evolution_interactions["explicit_feedback"] += steps
        self._save_interactions()
    
    def add_program_synthesis_result(
        self,
        task: str,
        cfg_version: int,
        program: str,
        reward: float,
        steps: int,
        func_evolution_round: Optional[int] = None,
        success: bool = False,
        raw_llm_response: Optional[str] = None,
        prompt: Optional[str] = None,
        inventory_trace: Optional[List] = None
    ):
        """Add a program synthesis result.
        
        Args:
            task: Task name
            cfg_version: CFG version (DSL round number, 0-indexed)
            program: Synthesized program code
            reward: Reward received for this program
            steps: Number of environment steps taken during program synthesis
            func_evolution_round: Function evolution round number (None if initial testing)
            success: Whether the program successfully solved the task
            raw_llm_response: Raw LLM output for this synthesis attempt (optional)
            prompt: Prompt sent to LLM for this synthesis attempt (optional)
            inventory_trace: Intermediate inventory changes during program execution (optional)
        """
        # Calculate offset (interactions before program synthesis)
        offset = self.interactions["funsearch"] + self.interactions["explicit_feedback"]
        
        # Add program synthesis steps (both cumulative and per-evolution)
        self.interactions["program_synthesis"] += steps
        self.current_evolution_interactions["program_synthesis"] += steps
        
        # Calculate total interactions (cumulative)
        total_interactions = (
            self.interactions["funsearch"] +
            self.interactions["explicit_feedback"] +
            self.interactions["program_synthesis"]
        )
        
        # Find best reward so far for this task
        best_reward = self._get_best_reward(task)
        if reward > best_reward:
            best_reward = reward
        
        # Create result entry - ensure success is always a boolean
        # Validate all values are JSON-serializable
        result = {
            "task": str(task),
            "cfg_version": int(cfg_version),
            "func_evolution_round": func_evolution_round if func_evolution_round is not None else None,
            "program": str(program),
            "reward": float(reward),
            "best_reward_so_far": float(best_reward),
            "steps": int(steps),
            "offset": int(offset),
            "total_interactions": int(total_interactions),
            "timestamp": str(datetime.now().isoformat()),
            "success": bool(success) if success is not None else False,
            "raw_llm_response": raw_llm_response,
            "prompt": prompt,
            "inventory_trace": inventory_trace
        }
        
        # Validate the result can be serialized before adding
        try:
            json.dumps(result)
        except (TypeError, ValueError) as e:
            print(f" Error: Result contains non-serializable data: {e}", file=sys.stderr)
            print(f"  Result: {result}", file=sys.stderr)
            raise ValueError(f"Cannot serialize result: {e}") from e
        
        self.results.append(result)
        self._save_results()
        self._save_interactions()
    
    def save_evolution_metrics(
        self,
        dsl_round: int,
        func_evolution_round: Optional[int],
        steps_in_evolution: Dict[str, int],
        rewards_per_task: Dict[str, float]
    ):
        """Save metrics for a specific evolution round.
        
        Args:
            dsl_round: DSL evolution round (0-indexed)
            func_evolution_round: Function evolution round (None for initial testing, 0-indexed otherwise)
            steps_in_evolution: Dict with steps for each phase in this evolution
            rewards_per_task: Dict mapping task -> best reward achieved in this evolution
        """
        metric = {
            "dsl_round": dsl_round,
            "func_evolution_round": func_evolution_round,
            "steps": {
                "funsearch": steps_in_evolution.get("funsearch", 0),
                "explicit_feedback": steps_in_evolution.get("explicit_feedback", 0),
                "program_synthesis": steps_in_evolution.get("program_synthesis", 0),
                "total": sum(steps_in_evolution.values())
            },
            "rewards_per_task": rewards_per_task,
            "timestamp": datetime.now().isoformat()
        }
        
        self.evolution_metrics.append(metric)
        self._save_evolution_metrics()
        
        # Reset current evolution interactions for next round
        self.current_evolution_interactions = {
            "funsearch": 0,
            "explicit_feedback": 0,
            "program_synthesis": 0
        }
    
    def _get_best_reward(self, task: str) -> float:
        """Get the best reward achieved so far for a task.
        
        Args:
            task: Task name
            
        Returns:
            Best reward so far (or -inf if no results)
        """
        task_results = [r for r in self.results if r["task"] == task]
        if not task_results:
            return float('-inf')
        return max(r["best_reward_so_far"] for r in task_results)
    
    def get_task_results(self, task: str) -> List[Dict]:
        """Get all results for a specific task.
        
        Args:
            task: Task name
            
        Returns:
            List of result dictionaries for this task
        """
        return [r for r in self.results if r["task"] == task]
    
    def get_all_results(self) -> List[Dict]:
        """Get all results.
        
        Returns:
            List of all result dictionaries
        """
        return self.results.copy()
    
    def plot_reward_vs_interactions(
        self,
        output_file: Optional[str] = None,
        tasks: Optional[List[str]] = None,
        dsl_round: Optional[int] = None,
        func_evolution_round: Optional[int] = None
    ):
        """Plot best reward vs total interactions.
        
        Args:
            output_file: Optional path to save plot (auto-generated if None)
            tasks: Optional list of tasks to plot (if None, plots all tasks)
            dsl_round: Optional DSL round number for filename
            func_evolution_round: Optional function evolution round number for filename
        """
        if not self.results:
            print("No results to plot")
            return
        
        # Filter by tasks if specified
        if tasks:
            filtered_results = [r for r in self.results if r["task"] in tasks]
        else:
            filtered_results = self.results
        
        # Filter by dsl_round if specified
        if dsl_round is not None:
            filtered_results = [r for r in filtered_results if r.get("cfg_version") == dsl_round]
        
        # Filter by func_evolution_round if specified (when plotting in a specific func round folder)
        # Show all rounds up to and including the specified round (e.g., func1 shows func0 and func1)
        if func_evolution_round is not None:
            filtered_results = [
                r for r in filtered_results 
                if r.get("func_evolution_round") is not None and r.get("func_evolution_round") <= func_evolution_round
            ]
        
        if not filtered_results:
            print("No results to plot after filtering")
            return
        
        # Group by task, CFG version, and func_evolution_round
        task_cfg_func_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for result in filtered_results:
            task = result["task"]
            cfg_version = result["cfg_version"]
            func_round = result.get("func_evolution_round")
            task_cfg_func_data[task][cfg_version][func_round].append(result)
        
        # If plotting within a specific func_evolution_round, normalize interactions
        # so all tasks start from the same baseline (min_offset), but each task's steps
        # accumulate independently (not across tasks)
        # Normalize each round independently (func0, func1, etc. each have their own min_offset)
        result_to_normalized_interaction = {}  # Map result to normalized interaction value
        if func_evolution_round is not None:
            # Process each round up to and including func_evolution_round
            for round_num in range(func_evolution_round + 1):
                # Collect all results in this round (across all tasks)
                all_results_in_round = [
                    r for r in filtered_results 
                    if r.get("func_evolution_round") == round_num
                ]
                if all_results_in_round:
                    # Find minimum offset (shared starting point for all tasks in this round)
                    min_offset_for_round = min(r.get("offset", 0) for r in all_results_in_round)
                    
                    # Group results by task, then normalize each task independently
                    task_results = defaultdict(list)
                    for result in all_results_in_round:
                        task = result.get("task", "")
                        task_results[task].append(result)
                    
                    # For each task, accumulate steps independently from min_offset
                    for task, task_result_list in task_results.items():
                        # Sort this task's results chronologically
                        task_results_sorted = sorted(task_result_list, key=lambda x: x["total_interactions"])
                        
                        # Accumulate steps for this task only
                        current_accum = 0
                        for result in task_results_sorted:
                            steps = result.get("steps", 0)
                            current_accum += steps
                            # Normalized interaction = min_offset + accumulated steps for this task
                            normalized = min_offset_for_round + current_accum
                            # Use a stable key: (task, program, timestamp)
                            result_key = (
                                result.get("task", ""),
                                result.get("program", ""),
                                result.get("timestamp", "")
                            )
                            result_to_normalized_interaction[result_key] = normalized
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Color palette for different func_evolution_rounds
        # Use distinct colors: blue for func0, red for func1, green for func2, etc.
        func_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # Plot each task
        for task_idx, (task, cfg_data) in enumerate(task_cfg_func_data.items()):
            for cfg_version, func_data in cfg_data.items():
                for func_round, results in func_data.items():
                    # Sort by total interactions
                    results_sorted = sorted(results, key=lambda x: x["total_interactions"])
                    
                    # Extract data
                    if func_evolution_round is not None and func_round <= func_evolution_round and result_to_normalized_interaction:
                        # Use normalized interactions for this round (if available)
                        interactions = []
                        for r in results_sorted:
                            result_key = (r.get("task", ""), r.get("program", ""), r.get("timestamp", ""))
                            normalized = result_to_normalized_interaction.get(result_key, r["total_interactions"])
                            interactions.append(normalized)
                    else:
                        # Use original total_interactions
                        interactions = [r["total_interactions"] for r in results_sorted]
                    best_rewards = [r["best_reward_so_far"] for r in results_sorted]
                    
                    # Get color for this func_evolution_round
                    # If plotting within a specific func_evolution_round, use different colors for different rounds
                    # to show progression (func0, func1, etc.)
                    if func_evolution_round is not None:
                        # Use different colors for different func rounds to show progression
                        if func_round is None:
                            color = '#808080'  # Gray for initial testing
                        else:
                            color = func_colors[func_round % len(func_colors)]
                        # Use different linestyles for different rounds
                        linestyle = '-' if func_round == 0 else '--' if func_round == 1 else '-.' if func_round == 2 else ':'
                        func_label = f"func{func_round}" if func_round is not None else "func_init"
                        label = f"{task} ({func_label})"  # Show task name and func round
                    else:
                        # Use gray for None (initial testing), otherwise use color palette
                        if func_round is None:
                            color = '#808080'  # Gray for initial testing
                        else:
                            color = func_colors[func_round % len(func_colors)]
                        # Plot line with different linestyle for different func rounds
                        linestyle = '-' if func_round == 0 else '--' if func_round == 1 else '-.' if func_round == 2 else ':'
                        func_label = f"func{func_round}" if func_round is not None else "func_init"
                        label = f"{task} (CFG v{cfg_version + 1}, {func_label})"
                    ax.plot(interactions, best_rewards, marker='o', label=label, 
                           color=color, linewidth=2, markersize=6, linestyle=linestyle)
        
        ax.set_xlabel("Total Interactions (Cumulative)", fontsize=12)
        ax.set_ylabel("Best Reward So Far", fontsize=12)
        
        # Create title with evolution info
        title = "Best Reward vs Total Interactions"
        if dsl_round is not None:
            title += f" - DSL Round {dsl_round + 1}"
        if func_evolution_round is not None:
            # Show 0-indexed number (func_evolution_round=0 shows "Func Evolution 0")
            title += f", Func Evolution {func_evolution_round}"
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        
        plt.tight_layout()
        
        # Create subdirectory for this DSL round and function evolution round
        if dsl_round is not None:
            func_round_str = f"func{func_evolution_round}" if func_evolution_round is not None else "func_init"
            plot_dir = os.path.join(self.results_dir, f"dsl{dsl_round}", func_round_str)
            os.makedirs(plot_dir, exist_ok=True)
        else:
            plot_dir = self.results_dir
        
        # Generate filename with version numbers
        if output_file is None:
            filename = "reward_vs_interactions.png"
            output_file = os.path.join(plot_dir, filename)
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f" Saved plot to {output_file}")
        plt.close()
        
        # Also generate separate plots for each task
        print(f"\nGenerating separate plots for each task in {plot_dir}...")
        for task, cfg_data in task_cfg_func_data.items():
            # Create figure for this task
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Color palette for different func_evolution_rounds
            func_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            
            # Plot each CFG version and func_evolution_round for this task
            for cfg_version, func_data in cfg_data.items():
                for func_round, results in func_data.items():
                    # Sort by total interactions
                    results_sorted = sorted(results, key=lambda x: x["total_interactions"])
                    
                    # Extract data
                    if func_evolution_round is not None and func_round <= func_evolution_round and result_to_normalized_interaction:
                        # Use normalized interactions for this round (if available)
                        interactions = []
                        for r in results_sorted:
                            result_key = (r.get("task", ""), r.get("program", ""), r.get("timestamp", ""))
                            normalized = result_to_normalized_interaction.get(result_key, r["total_interactions"])
                            interactions.append(normalized)
                    else:
                        # Use original total_interactions
                        interactions = [r["total_interactions"] for r in results_sorted]
                    best_rewards = [r["best_reward_so_far"] for r in results_sorted]
                    
                    # Get color for this func_evolution_round
                    # If plotting within a specific func_evolution_round, use different colors for different rounds
                    # to show progression (func0, func1, etc.)
                    if func_evolution_round is not None:
                        # Use different colors for different func rounds to show progression
                        if func_round is None:
                            color = '#808080'  # Gray for initial testing
                        else:
                            color = func_colors[func_round % len(func_colors)]
                        # Use different linestyles for different rounds
                        linestyle = '-' if func_round == 0 else '--' if func_round == 1 else '-.' if func_round == 2 else ':'
                        func_label = f"func{func_round}" if func_round is not None else "func_init"
                        label = f"{func_label}"  # Show func round label
                    else:
                        # Use gray for None (initial testing), otherwise use color palette
                        if func_round is None:
                            color = '#808080'  # Gray for initial testing
                        else:
                            color = func_colors[func_round % len(func_colors)]
                        # Plot line with different linestyle for different func rounds
                        linestyle = '-' if func_round == 0 else '--' if func_round == 1 else '-.' if func_round == 2 else ':'
                        func_label = f"func{func_round}" if func_round is not None else "func_init"
                        label = f"CFG v{cfg_version + 1}, {func_label}"
                    ax.plot(interactions, best_rewards, marker='o', label=label, 
                           color=color, linewidth=2, markersize=6, linestyle=linestyle)
            
            ax.set_xlabel("Total Interactions (Cumulative)", fontsize=12, fontweight='bold')
            ax.set_ylabel("Best Reward So Far", fontsize=12, fontweight='bold')
            
            # Create title with evolution info
            task_title = f"Reward vs Interactions: {task}"
            if dsl_round is not None:
                task_title += f" - DSL Round {dsl_round + 1}"
            if func_evolution_round is not None:
                # Show 0-indexed number (func_evolution_round=0 shows "Func Evolution 0")
                task_title += f", Func Evolution {func_evolution_round}"
            
            ax.set_title(task_title, fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            # Show legend if there are multiple series (multiple CFG versions or func rounds)
            total_series = sum(len(func_data) for func_data in cfg_data.values())
            if total_series > 1:
                ax.legend(fontsize=10)
            
            # Set y-axis to start at 0 if all rewards are non-negative
            all_rewards = []
            for func_data in cfg_data.values():
                for results in func_data.values():
                    all_rewards.extend([r["best_reward_so_far"] for r in results])
            if all_rewards and min(all_rewards) >= 0:
                max_reward = max(all_rewards) if all_rewards else 1.0
                ax.set_ylim(bottom=-0.05 * max_reward if max_reward > 0 else -0.1)
            
            plt.tight_layout()
            
            # Save plot - sanitize task name for filename
            safe_task_name = task.replace("[", "_").replace("]", "_").replace("/", "_")
            task_output_file = os.path.join(plot_dir, f"reward_vs_interactions_{safe_task_name}.png")
            plt.savefig(task_output_file, dpi=300, bbox_inches='tight')
            print(f"   Saved plot for {task} to {task_output_file}")
            plt.close()
    
    def plot_all_tasks_combined(
        self,
        output_file: Optional[str] = None,
        dsl_round: Optional[int] = None,
        func_evolution_round: Optional[int] = None
    ):
        """Plot all tasks on one plot, grouped by CFG version.
        
        Args:
            output_file: Optional path to save plot (auto-generated if None)
            dsl_round: Optional DSL round number for filename
            func_evolution_round: Optional function evolution round number for filename
        """
        if not self.results:
            print("No results to plot")
            return
        
        # Group by CFG version
        cfg_data = defaultdict(list)
        for result in self.results:
            cfg_version = result["cfg_version"]
            cfg_data[cfg_version].append(result)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Color palette for different CFG versions
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        # For each CFG version, compute best reward over all tasks
        for cfg_version in sorted(cfg_data.keys()):
            results = cfg_data[cfg_version]
            
            # Sort by total interactions
            results_sorted = sorted(results, key=lambda x: x["total_interactions"])
            
            # Track best reward across all tasks at each interaction point
            interactions = []
            best_rewards = []
            current_best = float('-inf')
            
            for result in results_sorted:
                interactions.append(result["total_interactions"])
                if result["best_reward_so_far"] > current_best:
                    current_best = result["best_reward_so_far"]
                best_rewards.append(current_best)
            
            # Get color for this CFG version
            color = colors[cfg_version % len(colors)]
            
            # Plot line
            label = f"CFG Version {cfg_version + 1}"
            ax.plot(interactions, best_rewards, marker='o', label=label,
                   color=color, linewidth=2, markersize=6)
        
        ax.set_xlabel("Total Interactions (Cumulative)", fontsize=12)
        ax.set_ylabel("Best Reward So Far (All Tasks)", fontsize=12)
        
        # Create title with evolution info
        title = "Best Reward vs Total Interactions (All Tasks Combined)"
        if dsl_round is not None:
            title += f" - DSL Round {dsl_round + 1}"
        if func_evolution_round is not None:
            # Show 0-indexed number (func_evolution_round=0 shows "Func Evolution 0")
            title += f", Func Evolution {func_evolution_round}"
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        
        # Create subdirectory for this DSL round and function evolution round
        if dsl_round is not None:
            func_round_str = f"func{func_evolution_round}" if func_evolution_round is not None else "func_init"
            plot_dir = os.path.join(self.results_dir, f"dsl{dsl_round}", func_round_str)
            os.makedirs(plot_dir, exist_ok=True)
        else:
            plot_dir = self.results_dir
        
        # Generate filename with version numbers
        if output_file is None:
            filename = "reward_vs_interactions_combined.png"
            output_file = os.path.join(plot_dir, filename)
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f" Saved combined plot to {output_file}")
        plt.close()
    
    def get_summary(self) -> Dict:
        """Get summary statistics.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.results:
            return {
                "total_results": 0,
                "total_interactions": 0,
                "tasks": []
            }
        
        tasks = set(r["task"] for r in self.results)
        cfg_versions = set(r["cfg_version"] for r in self.results)
        
        task_best_rewards = {}
        for task in tasks:
            task_results = self.get_task_results(task)
            if task_results:
                task_best_rewards[task] = max(r["best_reward_so_far"] for r in task_results)
        
        return {
            "total_results": len(self.results),
            "total_interactions": {
                "funsearch": self.interactions.get("funsearch", 0),
                "explicit_feedback": self.interactions.get("explicit_feedback", 0),
                "program_synthesis": self.interactions.get("program_synthesis", 0),
                "total": (
                    self.interactions.get("funsearch", 0) +
                    self.interactions.get("explicit_feedback", 0) +
                    self.interactions.get("program_synthesis", 0)
                )
            },
            "tasks": sorted(list(tasks)),
            "cfg_versions": sorted(list(cfg_versions)),
            "best_rewards_per_task": task_best_rewards
        }
    
    def plot_tasks_separately_from_metrics(
        self,
        dsl_round: Optional[int] = None,
        func_evolution_round: Optional[int] = None
    ):
        """Generate separate plots for each task from evolution_metrics.json.
        
        This method creates individual plots for each task showing reward vs cumulative interactions.
        Plots are organized into folders by DSL round and function evolution round.
        
        Args:
            dsl_round: Optional DSL round number for folder organization
            func_evolution_round: Optional function evolution round number for folder organization
        """
        if not self.evolution_metrics:
            print("No evolution metrics to plot")
            return
        
        # Create subdirectory for this DSL round and function evolution round
        if dsl_round is not None:
            func_round_str = f"func{func_evolution_round}" if func_evolution_round is not None else "func_init"
            plot_dir = os.path.join(self.results_dir, f"dsl{dsl_round}", func_round_str)
        else:
            # If no dsl_round specified, use "all" folder
            plot_dir = os.path.join(self.results_dir, "all_rounds")
        
        os.makedirs(plot_dir, exist_ok=True)
        
        # Process metrics to get data for each task
        cumulative_interactions = 0
        task_best_rewards = defaultdict(float)  # task -> best reward so far
        task_data = defaultdict(list)  # task -> list of (interactions, reward) tuples
        
        # Get all unique tasks from the first metric entry
        all_tasks = set()
        if self.evolution_metrics:
            all_tasks = set(self.evolution_metrics[0].get("rewards_per_task", {}).keys())
        
        if not all_tasks:
            print("No tasks found in evolution metrics")
            return
        
        # Process each evolution round
        for metric in self.evolution_metrics:
            # Add steps from this evolution round
            steps = metric.get("steps", {})
            total_steps = steps.get("total", 0)
            cumulative_interactions += total_steps
            
            # Update best rewards for each task
            rewards_per_task = metric.get("rewards_per_task", {})
            for task in all_tasks:
                reward = rewards_per_task.get(task, 0.0)
                # Update best reward if this is better
                if reward > task_best_rewards[task]:
                    task_best_rewards[task] = reward
                
                # Record this point (cumulative interactions, best reward so far)
                task_data[task].append((cumulative_interactions, task_best_rewards[task]))
        
        # Generate separate plot for each task
        print(f"Generating separate plots for {len(task_data)} tasks in {plot_dir}...")
        for task in sorted(task_data.keys()):
            data = task_data[task]
            if not data:
                print(f"  No data for task {task}, skipping plot")
                continue
            
            # Extract x and y values
            interactions = [x for x, y in data]
            rewards = [y for x, y in data]
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot the data
            ax.plot(interactions, rewards, marker='o', linestyle='-', linewidth=2, 
                    markersize=8, color='#2E86AB', alpha=0.8)
            
            # Formatting
            ax.set_xlabel("Cumulative Interactions", fontsize=12, fontweight='bold')
            ax.set_ylabel("Best Reward", fontsize=12, fontweight='bold')
            ax.set_title(f"Reward vs Interactions: {task}", fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Set y-axis to start at 0 if all rewards are non-negative
            if min(rewards) >= 0:
                ax.set_ylim(bottom=-0.05 * max(rewards) if max(rewards) > 0 else -0.1)
            
            plt.tight_layout()
            
            # Save plot - sanitize task name for filename
            safe_task_name = task.replace("[", "_").replace("]", "_").replace("/", "_")
            output_file = os.path.join(plot_dir, f"plot_{safe_task_name}.png")
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"   Saved plot for {task} to {output_file}")
            plt.close()
        
        # Also generate a combined plot
        print("\nGenerating combined plot...")
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Color palette
        colors = plt.cm.tab20(np.linspace(0, 1, len(task_data)))
        
        # Plot each task
        for idx, (task, data) in enumerate(sorted(task_data.items())):
            if not data:
                continue
            
            interactions = [x for x, y in data]
            rewards = [y for x, y in data]
            
            color = colors[idx % len(colors)]
            ax.plot(interactions, rewards, marker='o', linestyle='-', linewidth=2,
                   markersize=6, label=task, color=color, alpha=0.8)
        
        ax.set_xlabel("Cumulative Interactions", fontsize=12, fontweight='bold')
        ax.set_ylabel("Best Reward", fontsize=12, fontweight='bold')
        ax.set_title("Reward vs Interactions: All Tasks", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, ncol=2)
        
        plt.tight_layout()
        
        output_file = os.path.join(plot_dir, "plot_all_tasks_combined.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"   Saved combined plot to {output_file}")
        plt.close()
        
        print(f"\n All plots generated in {plot_dir}")


def parse_funsearch_log(log_file: str) -> List[Dict]:
    """Parse a FunSearch log file into a list of metrics entries."""
    entries: List[Dict] = []
    if not os.path.exists(log_file):
        return entries
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            env_interactions = record.get("env_interactions", 0)
            scores = record.get("scores", {})
            reward = None
            if isinstance(scores, dict) and scores:
                try:
                    reward = max(float(v) for v in scores.values() if v is not None)
                except (TypeError, ValueError):
                    reward = None
            entries.append({
                "timestamp": record.get("timestamp"),
                "env_interactions": int(env_interactions) if env_interactions is not None else 0,
                "reward": reward,
            })
    return entries


def parse_explicit_feedback_file(feedback_file: str) -> List[Dict]:
    """Parse an explicit feedback JSON file into a list of metrics entries."""
    entries: List[Dict] = []
    if not os.path.exists(feedback_file):
        return entries
    try:
        with open(feedback_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return entries
    if not isinstance(data, list):
        return entries
    for entry in data:
        if not isinstance(entry, dict):
            continue
        env_interactions = entry.get("env_interactions")
        if env_interactions is None:
            env_interactions = entry.get("actions_count", 0)
        try:
            env_interactions = int(env_interactions)
        except (TypeError, ValueError):
            env_interactions = 0
        reward = entry.get("score")
        try:
            reward = float(reward) if reward is not None else None
        except (TypeError, ValueError):
            reward = None
        entries.append({
            "iteration": entry.get("iteration"),
            "env_interactions": env_interactions,
            "reward": reward,
            "runs_ok": entry.get("runs_ok", True),
        })
    return entries


def plot_funsearch_reward_vs_interactions(
    log_file: str,
    output_dir: str,
    function_name: Optional[str] = None,
) -> Optional[str]:
    """Plot best reward vs cumulative env interactions from a FunSearch log."""
    entries = parse_funsearch_log(log_file)
    if not entries:
        print(f"No FunSearch log entries found in {log_file}")
        return None

    plt.switch_backend("Agg")

    cumulative_interactions: List[int] = []
    best_rewards: List[float] = []
    raw_rewards: List[Optional[float]] = []
    total = 0
    best = float("-inf")

    for entry in entries:
        total += int(entry.get("env_interactions", 0))
        reward = entry.get("reward")
        raw_rewards.append(reward)
        if reward is not None and reward > best:
            best = reward
        cumulative_interactions.append(total)
        best_rewards.append(best if best != float("-inf") else 0.0)

    os.makedirs(output_dir, exist_ok=True)
    safe_name = function_name.replace(" ", "_") if function_name else "funsearch"
    plot_path = os.path.join(output_dir, f"{safe_name}_reward_vs_interactions.png")
    metrics_path = os.path.join(output_dir, f"{safe_name}_reward_vs_interactions.json")

    # Save metrics
    metrics = []
    for idx, entry in enumerate(entries):
        metrics.append({
            "timestamp": entry.get("timestamp"),
            "env_interactions": int(entry.get("env_interactions", 0)),
            "cumulative_interactions": cumulative_interactions[idx],
            "reward": raw_rewards[idx],
            "best_reward_so_far": best_rewards[idx],
        })
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Plot best reward vs interactions
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(cumulative_interactions, best_rewards, marker="o", linewidth=2, markersize=4)
    ax.set_title("Best Reward vs Env Interactions")
    ax.set_xlabel("Cumulative Env Interactions")
    ax.set_ylabel("Best Reward So Far")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" Saved FunSearch plot to {plot_path}")
    return plot_path


def plot_explicit_feedback_reward_vs_interactions(
    feedback_file: str,
    output_dir: str,
    function_name: Optional[str] = None,
) -> Optional[str]:
    """Plot best reward vs cumulative env interactions from explicit feedback."""
    entries = parse_explicit_feedback_file(feedback_file)
    if not entries:
        print(f"No explicit feedback entries found in {feedback_file}")
        return None

    plt.switch_backend("Agg")

    cumulative_interactions: List[int] = []
    best_rewards: List[float] = []
    raw_rewards: List[Optional[float]] = []
    total = 0
    best = float("-inf")

    for entry in entries:
        total += int(entry.get("env_interactions", 0))
        reward = entry.get("reward")
        raw_rewards.append(reward)
        if reward is not None and reward > best:
            best = reward
        cumulative_interactions.append(total)
        best_rewards.append(best if best != float("-inf") else 0.0)

    os.makedirs(output_dir, exist_ok=True)
    safe_name = function_name.replace(" ", "_") if function_name else "explicit_feedback"
    plot_path = os.path.join(output_dir, f"{safe_name}_explicit_feedback_reward_vs_interactions.png")
    metrics_path = os.path.join(output_dir, f"{safe_name}_explicit_feedback_reward_vs_interactions.json")

    metrics = []
    for idx, entry in enumerate(entries):
        metrics.append({
            "iteration": entry.get("iteration"),
            "env_interactions": int(entry.get("env_interactions", 0)),
            "cumulative_interactions": cumulative_interactions[idx],
            "reward": raw_rewards[idx],
            "best_reward_so_far": best_rewards[idx],
        })
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(cumulative_interactions, best_rewards, marker="o", linewidth=2, markersize=4)
    ax.set_title("Best Reward vs Env Interactions (Explicit Feedback)")
    ax.set_xlabel("Cumulative Env Interactions")
    ax.set_ylabel("Best Reward So Far")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" Saved explicit feedback plot to {plot_path}")
    return plot_path


def plot_baseline_reward_vs_interactions(
    funsearch_log_file: str,
    explicit_feedback_file: str,
    output_dir: str,
    function_name: Optional[str] = None,
) -> Optional[str]:
    """Plot baseline (FunSearch + explicit feedback) reward vs interactions."""
    funsearch_entries = parse_funsearch_log(funsearch_log_file)
    explicit_entries = parse_explicit_feedback_file(explicit_feedback_file)

    if not funsearch_entries and not explicit_entries:
        print("No FunSearch or explicit feedback entries found for baseline plotting")
        return None

    plt.switch_backend("Agg")

    # FunSearch series
    funsearch_cumulative: List[int] = []
    funsearch_best: List[float] = []
    total_funsearch = 0
    best = float("-inf")
    for entry in funsearch_entries:
        total_funsearch += int(entry.get("env_interactions", 0))
        reward = entry.get("reward")
        if reward is not None and reward > best:
            best = reward
        funsearch_cumulative.append(total_funsearch)
        funsearch_best.append(best if best != float("-inf") else 0.0)

    # Explicit feedback series (offset by FunSearch total)
    explicit_cumulative: List[int] = []
    explicit_best: List[float] = []
    total_explicit = total_funsearch
    best_explicit = best
    for entry in explicit_entries:
        total_explicit += int(entry.get("env_interactions", 0))
        reward = entry.get("reward")
        if reward is not None and reward > best_explicit:
            best_explicit = reward
        explicit_cumulative.append(total_explicit)
        explicit_best.append(best_explicit if best_explicit != float("-inf") else 0.0)

    os.makedirs(output_dir, exist_ok=True)
    safe_name = function_name.replace(" ", "_") if function_name else "baseline"
    plot_path = os.path.join(output_dir, f"{safe_name}_baseline_reward_vs_interactions.png")
    metrics_path = os.path.join(output_dir, f"{safe_name}_baseline_reward_vs_interactions.json")

    metrics = {
        "funsearch": {
            "cumulative_interactions": funsearch_cumulative,
            "best_rewards": funsearch_best,
        },
        "explicit_feedback": {
            "cumulative_interactions": explicit_cumulative,
            "best_rewards": explicit_best,
        },
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    fig, ax = plt.subplots(figsize=(10, 6))
    # Plot a single continuous line, switching colors between phases.
    if funsearch_entries:
        ax.plot(
            funsearch_cumulative,
            funsearch_best,
            marker="o",
            linewidth=2,
            markersize=4,
            color="#1f77b4",
            label="FunSearch",
        )
    # if explicit_entries:
    #     # Ensure continuity by connecting the last FunSearch point to the first explicit point.
    #     explicit_x = explicit_cumulative
    #     explicit_y = explicit_best
    #     if funsearch_entries and explicit_x:
    #         explicit_x = [funsearch_cumulative[-1]] + explicit_x
    #         explicit_y = [funsearch_best[-1]] + explicit_y
    #     ax.plot(
    #         explicit_x,
    #         explicit_y,
    #         marker="o",
    #         linewidth=2,
    #         markersize=4,
    #         color="#ff7f0e",
    #         label="Explicit Feedback",
    #     )

    ax.set_title("Baseline Best Reward vs Env Interactions")
    ax.set_xlabel("Cumulative Env Interactions")
    ax.set_ylabel("Best Reward So Far")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" Saved baseline plot to {plot_path}")
    return plot_path

