import os
import json
import argparse
import random
import subprocess
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional
import time
from google import genai

def get_end_score(scores: Dict[str, Any]) -> Optional[float]:
    if not isinstance(scores, dict) or not scores:
        return None
    try:
        step_keys = [int(k) for k in scores.keys()]
    except (ValueError, TypeError):
        # Fallback: if keys are not numeric, just take any deterministic "last" by insertion order
        try:
            # Python 3.7+ preserves insertion order
            last_key = next(reversed(scores))
            return float(scores[last_key])
        except Exception:
            return None
    last_step = max(step_keys)
    value = scores.get(str(last_step))
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def parse_log_file(path: str, k: int = 1) -> List[Tuple[float, str]]:
    scored_funcs: List[Tuple[float, str]] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            scores = record.get("scores")
            function_body = record.get("function_body")

            if function_body is None or scores is None:
                continue

            end_score = get_end_score(scores)
            if end_score is None:
                continue

            scored_funcs.append((end_score, function_body))

    if not scored_funcs:
        return []

    scored_funcs.sort(key=lambda x: x[0], reverse=True)

    # Find cutoff score if ties go beyond k
    cutoff = scored_funcs[k-1][0] if len(scored_funcs) >= k else scored_funcs[-1][0]

    # Keep all functions with score >= cutoff
    top_candidates = [(s, f) for (s, f) in scored_funcs if s >= cutoff]

    if len(top_candidates) > k:
        # Too many due to ties → sample exactly k at random
        return random.sample(top_candidates, k)
    else:
        return top_candidates

def eval(res, file):
    # with tempfile.TemporaryDirectory() as temp_dir:
    # Create unique filename using process ID and timestamp
    temp_dir = os.getcwd()
    unique_id = f"{os.getpid()}_{int(time.time() * 1000000)}"
    script_path = f'explicit_generated_code_{unique_id}.py'
    script_path = os.path.join(temp_dir, script_path)

    with open(file,"r") as f:
        full_program = f.read()

    full_program = f"""
{full_program}
{res}
print(evaluate())
"""
    # print(full_program)
    with open(script_path, 'w') as f:
        f.write(full_program.strip())

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
                # Try to parse numerical output
        output = result.stdout.strip()
        print("output ", output)
        try:
            print(output)
            return float(output), True
        except ValueError:
            return -1, True
    except subprocess.TimeoutExpired:
        return -1, False
    except subprocess.CalledProcessError as e:
        print(f"Process Error: Command failed with exit code {e.returncode}")
        print(f"Command: {e.cmd}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return -1, False
    finally:
        # Clean up the temporary file
        if os.path.exists(script_path):
            os.remove(script_path)

def response_gen(funcs, k, file):
    with open("prompt_specifications/specification_with_updated_nld_baseline.txt", "r") as f:
        prompt1 = f.read()

    funcs_text = "\n\n".join(
        [f"### Score: {score}\n```python\n{body}\n```" for score, body in funcs]
    )

    prompt = (
        prompt1
        + "\n\nHere are different implementations of `def make_def make_arrow(env) -> list[int]:`\n"
        + funcs_text
        + "\n\nAnalyse the functions and give natural language feedback in bullet points."
    )

    # print(prompt)
    client = genai.Client()

    response = client.models.generate_content(
                          model="gemini-2.5-pro", contents = prompt
                      )
    feedback = response.text

    correction_prompt = (
      prompt1
      + "\n\nFeedback:\n"
      + feedback
      + "\n\nHere are the failed candidate functions for `def make_arrow(env) -> list[int]:`\n"
      + funcs_text
      + "\n\nReturn a corrected and improved version of the function and just that."
  )
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

  # build filename
    log_filename = f"results/explicit_feedback/feedback_sampling_{timestamp}_{k}.json"
    i = 0
    while i< 20:
        response = client.models.generate_content(
                              model="gemini-2.5-pro", contents = correction_prompt
                          )
        b = response.text
      # print("second generation\n", b)
        try:
            # # print(feedback)
      
            b = b[b.index("```python")+len("```python")+1:]
            b = b[:b.index("```")]
          
        except:
            continue
        i+=1
        # print(feedback)
        log_entry = {
          "extracted_function_code": b,
          "evaluation_result": eval(b, file),
        }
        # print(eval_result)
        # Write to log file in JSON format
        with open(log_filename, "a") as log_file:
            log_file.write(json.dumps(log_entry, indent=2) + ", \n")
    

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--logfile", type=str, required=True,
    #                     help="Path to the feedback_sampling.json log file")
    # parser.add_argument("--k", type=int, default=5,
    #                     help="Number of top functions to extract (ties included)")
    # parser.add_argument("--file", type=str, default="evaluation_scripts/eval_collect.py",
    #                     help="Number of top functions to extract (ties included)")
    # args = parser.parse_args()

    with open("config_explicit_feedback.json", "r") as f:
        config = json.load(f)
    print(config.get("logfile"))
    funcs = parse_log_file(config.get("logfile"), k=config.get("k", 5))

    if not funcs:
        print("No functions found.")
        return

    response_gen(funcs, config.get("k"), config.get("file"))

if __name__ == "__main__":
    main()
