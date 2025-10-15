"""Plotting utilities for program synthesis results.

This module provides small helpers to plot reward vs interactions and
interaction/reward traces. It mirrors the functions that were previously
embedded in `program_synthesis.py` so other modules can reuse them.
"""
from typing import Iterable, List, Tuple
import matplotlib
# Use a non-interactive backend for headless test environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_watermark(data: List[Tuple[int, float]], task: str, out_dir: str = "results/plots") -> None:
    """Plot cumulative interactions vs best-so-far reward and save to file.

    Args:
        data: sequence of (interactions, reward) points.
        task: task name used for the filename.
        out_dir: directory to save the plot into.
    """
    if len(data) < 2:
        return

    x_values = [sum(point[0] for point in data[: i + 1]) for i in range(len(data))]
    y_values = [max(point[1] for point in data[: i + 1]) for i in range(len(data))]

    plt.plot(x_values, y_values, marker="o")
    plt.title("Reward vs Interactions")
    plt.xlabel("Number of Interactions")
    plt.ylabel("Reward")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/plot_{task}.png")
    plt.close()


def plot_interactions_rewards(interactions: Iterable[int], rewards: Iterable[float], task: str, out_path: str = "plot_{task}.png") -> None:
    """Plot rewards (or reward traces) and save to file.

    Args:
        interactions: sequence of interaction counts (x-axis optional)
        rewards: sequence of rewards to plot
        task: task name used in title
        out_path: output filename pattern (can include {task})
    """
    plt.figure(figsize=(10, 5))
    plt.plot(list(rewards), label="Rewards", marker="x")
    plt.title(f"Interactions and Rewards for Task: {task}")
    plt.xlabel("Interactions")
    plt.ylabel("Cummulative Reward")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_path.format(task=task))
    plt.close()
