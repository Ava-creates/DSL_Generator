import tempfile
import os
from DSL_Generator.plotting import plot_watermark, plot_interactions_rewards


def test_plot_watermark_creates_file():
    data = [(0, 0.0), (1, 1.0), (2, 2.0)]
    with tempfile.TemporaryDirectory() as td:
        out_dir = td
        plot_watermark(data, "unit_task", out_dir=out_dir)
        expected = os.path.join(out_dir, "plot_unit_task.png")
        assert os.path.exists(expected)


def test_plot_interactions_rewards_creates_file():
    interactions = [1, 2, 3]
    rewards = [0.0, 1.0, 2.0]
    with tempfile.TemporaryDirectory() as td:
        out_path = os.path.join(td, "plot_{task}.png")
        plot_interactions_rewards(interactions, rewards, "unit_task", out_path=out_path)
        expected = os.path.join(td, "plot_unit_task.png")
        assert os.path.exists(expected)
