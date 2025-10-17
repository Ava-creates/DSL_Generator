from openai_harmony import _PyHarmonyEncoding
import os
# Replace these paths with where you stored the downloaded files
vocab_path = "/scratch/avani/openai/vocab.json"
merges_path = "/scratch/avani/openai/merges.txt"


# Use your personal scratch folder
scratch_dir = "/scratch/avani/openai"

# Optional: ensure path exists
os.makedirs(scratch_dir, exist_ok=True)


# Load the encoding offline
enc = _PyHarmonyEncoding(vocab_file=vocab_path, merges_file=merges_path)

