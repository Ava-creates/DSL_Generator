import json
import sys
from typing import Any, Dict, Optional, Tuple


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


def parse_log_file(path: str) -> Tuple[Optional[float], Optional[str]]:
    best_score: Optional[float] = None
    best_function_body: Optional[str] = None

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Expect JSON per line; skip non-JSON lines
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

            if best_score is None or end_score > best_score:
                best_score = end_score
                best_function_body = function_body

    return best_score, best_function_body


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python extract_best.py /absolute/path/to/results.log", file=sys.stderr)
        sys.exit(2)

    log_path = sys.argv[1]
    best_score, best_function_body = parse_log_file(log_path)

    if best_score is None:
        print("No valid entries with end scores and function_body were found.")
        sys.exit(1)

    print("BEST_END_SCORE:\n", best_score)
    print("\nFUNCTION_BODY:\n")
    print(best_function_body)


if __name__ == "__main__":
    main() 
    