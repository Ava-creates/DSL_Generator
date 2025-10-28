"""Aggregate env interactions and rewards from a JSONL log and plot them.

Usage:
    python -m prog_synth_pipeline.aggregate_and_plot <log_path> [--out-dir results/plots] [--task make[arrow]]

The script reads each JSON object per line, extracts `env_interactions` and
`scores` (prefers key "3" when present), writes a CSV summary and calls
`plot_watermark` to produce a PNG.
"""
import argparse
import json
import os
from typing import List, Tuple

from plotting import plot_watermark, plot_interactions_rewards
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_programs_results(path: str, offset: int = 26000):
    """Read a programs_results JSONL and return a list of (interaction, reward) points.

    The file contains records with 'interactions' (list) and 'rewards' (list).
    We treat each pair as an (x,y) point. An offset is added to x values so the
    program-results series.
    """
    data = []

    with open(path, 'r', encoding='utf-8') as f:
        i =0 
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rec = json.loads(ln)
            except Exception:
                continue
            x= rec.get('interactions')[-1]
            y = rec.get('total_reward', 0.0)
            x += offset
            data.append((x, y))
    print(data)
    return data

def plot_two_series(data_a, label_a: str, data_b, label_b: str, task: str, out_dir: str = 'results/plots') -> None:
    """Plot two (interaction, reward) series on the same axes and save a PNG.

    Both series are converted to best-so-far reward traces before plotting.
    """
    if not data_a and not data_b:
        print('No data to plot')
        return

    plt.figure(figsize=(10, 6))

    def best_so_far(data):
        xs = [p[0] for p in data]
        ys = [p[1] for p in data]
        best = []
        m = float('-inf')
        for v in ys:
            if v is None:
                v = 0.0
            m = max(m, v)
            best.append(m)

        # Convert xs to cumulative sum so x-axis is total interactions over time
        cumsum = []
        total = 0
        for v in xs:
            try:
                total += int(v)
            except Exception:
                total += 0
            cumsum.append(total)
        return cumsum, best

    print(data_a, data_b)

    if data_a:
        xa, ya = best_so_far(data_a)
        plt.plot(xa, ya, label=label_a, marker='o')

    if data_b:
        xb, yb = best_so_far(data_b)
        plt.plot(xb, yb, label=label_b, marker='x')

    plt.title(f'Reward vs Interactions ({task})')
    plt.xlabel('Number of Interactions')
    plt.ylabel('Reward (best-so-far)')
    plt.legend()
    plt.grid(True)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'plot_{task}_compare.png')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"✅ Saved plot to {out_path}")


def extract_reward(record: dict) -> float:
    # Prefer scores["3"] if present, else take any score value, else 0.0
    scores = record.get("scores")
    if isinstance(scores, dict):
        if "3" in scores:
            return float(scores["3"]) if scores["3"] is not None else 0.0
        # take first numeric value
        for v in scores.values():
            try:
                return float(v)
            except Exception:
                continue
    # fallback: look for top-level reward-like keys
    for key in ("reward", "score", "total_reward"):
        if key in record:
            try:
                return float(record[key])
            except Exception:
                pass
    return 0.0


def read_log(path: str) -> List[Tuple[int, float]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                # skip malformed lines
                continue
            env_inter = rec.get("env_interactions")
            if env_inter is None:
                # some logs may use snake_case or ints under other keys
                env_inter = rec.get("env_interaction") or rec.get("interactions") or 0
            try:
                env_inter = int(env_inter)
            except Exception:
                env_inter = 0
            reward = extract_reward(rec)
            data.append((env_inter, reward))
    return data




def main():
    p = argparse.ArgumentParser()
    p.add_argument("log", help="Path to JSONL log file")
    p.add_argument("--out-dir", default="results/plots", help="Directory to write CSV and plot")
    p.add_argument("--task", default="make[arrow]baseline", help="Task name used for plot filename")
    p.add_argument("--programs-file", default=None, help="Optional programs_results JSONL to overlay")
    p.add_argument("--program-offset", type=int, default=26000, help="Integer offset to add to program interactions")
    args = p.parse_args()

    data = read_log(args.log)

    # Call plot_watermark (expects list of (interactions, reward))
    # os.makedirs(args.out_dir, exist_ok=True)
    # plot_watermark(data, args.task, out_dir=args.out_dir)
    # print(f"Saved plot to {args.out_dir}")

    # If a programs file is provided, read it, offset interactions, and plot both series
    if args.programs_file:
        try:
            prog_data = read_programs_results(args.programs_file, offset=args.program_offset)
            # prod label
            label_a = "baseline"
            label_b = "out method"
            plot_two_series(data, label_a, prog_data, label_b, args.task, out_dir=args.out_dir)
        except Exception as e:
            print(f"Failed to read/plot programs file: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
