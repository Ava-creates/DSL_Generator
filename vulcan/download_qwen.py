import os
from huggingface_hub import snapshot_download

# Use your personal scratch folder
scratch_dir = "/scratch/avani/qwen"

# Optional: ensure path exists
os.makedirs(scratch_dir, exist_ok=True)

snapshot_download(
    repo_id="Qwen/Qwen3-Coder-30B-A3B-Instruct",
    local_dir=scratch_dir,
    local_dir_use_symlinks=False  # avoids symlinks, useful for portability
)
