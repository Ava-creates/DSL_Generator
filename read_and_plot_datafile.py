"""Read a simple CSV-like data file and plot env_interactions vs reward.

The expected file format is one record per line, comma-separated. Each
line should contain at least two numeric fields: env_interactions, reward,
followed by any additional text (e.g., code string). Example:

  0.0,0.0,SomeProgramText
  1.0,0.5,OtherProgram

Usage (from repo root):
  python read_and_plot_datafile.py <path-to-data-file> [--out-dir results/plots] [--task make[stick]]

The script will parse floats robustly and skip malformed lines. Use
--max-points N --tail to select the last N points (bottom N points).
"""
import os
import re
from typing import List, Tuple
import argparse
from plotting import plot_watermark


def read_data_file(path: str) -> List[Tuple[float, float]]:
    """Parse the simple CSV-like file and return list of (env_interactions, reward).

    The parser is permissive: it skips blank/malformed lines and attempts to
    extract numbers by filtering non-numeric characters if needed.
    """
    data: List[Tuple[float, float]] = []

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    # regex for lines starting with two numbers separated by a comma
    num_pattern = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?),\s*([+-]?\d+(?:\.\d+)?),")

    with open(path, 'r', encoding='utf-8') as f:
        for ln in f:
            match = num_pattern.match(ln)
            if not match:
                continue

            try:
                env_inter = float(match.group(1))
                reward = float(match.group(2))
            except ValueError:
                continue

            data.append((env_inter, reward))
    print(len(data))
    return data

import json

def extract_env_and_reward(path):
    """
    Reads a JSONL file and extracts (env_interactions, reward) pairs.

    Args:
        path (str): Path to the input file containing one JSON object per line.

    Returns:
        List[Tuple[int | None, float | None]]: A list of (env_interactions, reward) tuples.
    """
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                env_interactions = obj.get("env_interactions", None)

                # extract reward (first value in scores dict)
                scores = obj.get("scores", {})
                reward = next(iter(scores.values()), None)

                data.append((env_interactions, reward))
            except json.JSONDecodeError as e:
                print(f"Skipping bad line: {e}")
    return data




def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('datafile', help='Path to data file (CSV-like)')
    p.add_argument('--out-dir', default='results/plots', help='Directory to write the plot')
    p.add_argument('--task', default='plot_make[stick]baselinerandomsampling', help='Task name used for filename')
    p.add_argument('--max-points', type=int, default=50, help='If >0, limit to this many points')
    p.add_argument('--tail', action='store_true', help='When used with --max-points, take the last N points (tail)')
    args = p.parse_args()

        # data = read_data_file(args.datafile)
        # if not data:
        #     print('No numeric rows parsed from', args.datafile)
        #     return 2

    #

    data = extract_env_and_reward(args.datafile)

    os.makedirs(args.out_dir, exist_ok=True)
    print(data)
    plot_watermark(data, args.task, out_dir=args.out_dir)
    print(f'Wrote plot for {len(data)} points to {args.out_dir}')
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
