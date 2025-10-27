from __future__ import annotations

import os
import json
import argparse
import subprocess
import sys
import ast
import re
import tempfile
import textwrap
from datetime import datetime
from typing import List, Tuple, Optional
import time


def eval(candidate: str, support_file: str, timeout: int = 300):
    """Run `candidate` appended to `support_file` and log the result.

    The candidate should define or call evaluate() which returns either:
      - (value, actions_count)
      - {'value':..., 'actions_count': ...}
      - a single numeric value

    Returns (numeric_value_or_-1.0, success_bool, actions_count_int, stderr_or_none)
    """
    temp_dir = os.getcwd()
    unique_id = f"{os.getpid()}_{int(time.time() * 1000000)}"
    script_path = f'explicit_generated_code_{unique_id}.py'
    script_path = os.path.join(temp_dir, script_path)

    with open(support_file, 'r', encoding='utf-8') as f:
        support = f.read()

    wrapper = f'''
{support}

{candidate}

print(evaluate())
'''
    # print(wrapper)
    with open(script_path, 'w') as f:
        f.write(wrapper.strip())

    try:
        result = subprocess.run(
                    ['python', script_path],
                    capture_output=True,
                    text=True,
                    timeout=300, #this is in seconds
                    check=True,
                    encoding='utf-8',
                    errors='replace'
                )
                # Try to parse numerical output\
        output = result.stdout.strip()

        output = output.replace("np.float64", "")
        output = output.replace("np.float32", "")
        output = output.replace("(", "").replace(")", "")
        result = ast.literal_eval(output)
        print(output)
        # Access the values
        output, actions_count = result[0], result[1]
        print("output ", output)
        try:
            print(output)
            return float(output), True, actions_count, None
        except ValueError:
            return -1, True, 0 , None
    except subprocess.TimeoutExpired:
        return -1, False, 0, None
    except subprocess.CalledProcessError as e:
        print(f"Process Error: Command failed with exit code {e.returncode}")
        print(f"Command: {e.cmd}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return -1, False, 0 , e.stderr
    except Exception as e:
        print(f"Some other error occurred: {e}")
        return -1, False, 0 , "Some error"
    finally:
        # Clean up the temporary file
        if os.path.exists(script_path):
            os.remove(script_path)


def process_function_bodies_from_log(source_log: str, support_file: str, task: str = 'task', out_dir: str = 'results/plots', max_to_process: Optional[int] = None):
    """Read JSONL entries from `source_log`, extract 'function_body', wrap it as
    `def craft(env, item): <body>` (with proper indentation), and call eval() for each.

    Writes results to `eval_log.jsonl` via eval().
    """
    if not os.path.exists(source_log):
        raise FileNotFoundError(source_log)

    count = 0
    a_actions =0
    with open(source_log, 'r', encoding='utf-8') as f:
        for ln in f:
            if max_to_process and count >= max_to_process:
                break
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
            except Exception:
                continue
            try:
                if obj.get('scores')["3"]==-1:
                    continue
            except Exception:
                print(obj.get('scores'))
                continue
            func_body = obj.get('function_body') 
            if not func_body:
                continue

            wrapper = f"def craft(env, item):\n{func_body}\n"

            # Run eval on the wrapped function; eval() will append to eval_log.jsonl
            print(f'Processing function #{count+1} from {source_log}...')
            res = eval(wrapper, support_file)
            # res is a tuple: (reward, success_bool, actions_count, stderr_or_none)
            reward = res[0]
            actions_count = res[2]
            a_actions+=actions_count
            # Log the result to eval_log.jsonl
            with open('eval_log_craft.jsonl', 'a', encoding='utf-8') as log_f:
                json.dump({'value': reward, 'actions_count': actions_count}, log_f)
                log_f.write('\n')
            print(' ->', res)
            count += 1

    print(f'Processed {count} functions from {source_log}. Total actions: {a_actions}')


 

def main() -> int:
    p = argparse.ArgumentParser(description='Evaluate candidates and/or plot eval_log.jsonl')
    p.add_argument('--log', default='eval_log.jsonl', help='Path to eval JSONL log')
    p.add_argument('--task', default='craft', help='Task name used for plot filename')
    p.add_argument('--out-dir', default='results/plots', help='Directory to write plot')
    p.add_argument('--source-log', default=None, help='JSONL file that contains function bodies (e.g., funsearch logs)')
    p.add_argument('--support-file', default=None, help='Path to the python file that provides evaluate() and env supporting code')
    p.add_argument('--max-points', type=int, default=0, help='If >0, limit to this many points')
    p.add_argument('--tail', action='store_true', help='When used with --max-points, take the last N points (tail)')
    args = p.parse_args()

    # If requested, run eval() on function bodies contained in a separate log

    if not args.source_log or not args.support_file:
            print('--run-eval requires --source-log and --support-file')
            return 2
    process_function_bodies_from_log(args.source_log, args.support_file, task=args.task, out_dir=args.out_dir)

    # Always attempt to plot from the eval log specified
    # plot_from_log(log_path=args.log, task=args.task, out_dir=args.out_dir, max_points=args.max_points, tail=args.tail)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


