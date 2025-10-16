"""Aggregate env interactions and rewards from a JSONL log and plot them.

Usage:
    python -m prog_synth_pipeline.aggregate_and_plot <log_path> [--out-dir results/plots] [--task make[stick]]

The script reads each JSON object per line, extracts `env_interactions` and
`scores` (prefers key "3" when present), writes a CSV summary and calls
`plot_watermark` to produce a PNG.
"""
import argparse
import json
import os
from typing import List, Tuple

from plotting import plot_watermark, plot_interactions_rewards


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


def write_csv(data: List[Tuple[int, float]], out_path: str) -> None:
    import csv

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["env_interactions", "reward"])
        for e, r in data:
            w.writerow([e, r])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("log", help="Path to JSONL log file")
    p.add_argument("--out-dir", default="results/plots", help="Directory to write CSV and plot")
    p.add_argument("--task", default="make[stick]baseline", help="Task name used for plot filename")
    args = p.parse_args()

    data = read_log(args.log)
    if not data:
        print("No data parsed from", args.log)
        return 1

    csv_path = os.path.join(args.out_dir, "aggregated_log.csv")
    write_csv(data, csv_path)
    print(f"Wrote CSV to {csv_path}")

    # Call plot_watermark (expects list of (interactions, reward))
    os.makedirs(args.out_dir, exist_ok=True)
    plot_watermark(data, args.task, out_dir=args.out_dir)
    print(f"Saved plot to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
