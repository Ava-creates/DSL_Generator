import json
from datetime import datetime

def programs_per_hour(file_path):
    timestamps = []

    # Read timestamps from JSON lines (ignore the first line)
    with open(file_path, "r") as f:
        lines = f.readlines()[1:]  # skip the first line
        for line in lines:
            data = json.loads(line)
            timestamps.append(datetime.fromisoformat(data["timestamp"]))

    # Sort just in case
    timestamps.sort()

    # Take differences between pairs (0-1, 2-3, ...)
    diffs = []
    for i in range(0, len(timestamps) - 1, 2):
        diff = (timestamps[i+1] - timestamps[i]).total_seconds()
        if diff > 100:
            continue
        diffs.append(diff)

    if not diffs:
        return 0

    avg_gap = sum(diffs) / len(diffs)
    print(avg_gap)

    # Each pair = 2 programs, so rate is (2 / avg_gap) programs per second
    programs_per_hour = (2 / avg_gap) * 3600

    return programs_per_hour

# Example usage
file_path = "/Users/avanitiwari/Desktop/craft-env/results/ollama_q2.5_craft_with_updated_craft_decription_craft_func_init.py_prompt_specificationsspecification_with_updated_nld.txt_2025-09-05_22-22-00.log"
file_qwen3_30 ="/Users/avanitiwari/Desktop/craft-env/results/ollama_q2.5_craft_with_updated_craft_description_craft_func_init.py_prompt_specificationsspecification_with_updated_nld.txt_2025-09-12_15-51-25_qwen-3:30B.log"
file_qwen3_32="/Users/avanitiwari/Desktop/craft-env/results/ollama_q2.5_craft_with_updated_craft_description_craft_func_init.py_prompt_specificationsspecification_with_updated_nld.txt_2025-09-15_12-35-16.log"
print("og:", programs_per_hour(file_path))
print("qwen3 30b", programs_per_hour(file_qwen3_30))
print("qwen3 32b", programs_per_hour(file_qwen3_32))
