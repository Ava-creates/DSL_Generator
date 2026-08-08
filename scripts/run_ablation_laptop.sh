#!/usr/bin/env bash
# Run baseline ablation locally (no SLURM). Uses Aleph OpenAI-compatible API.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="${1:-config/ablation_laptop_smoke.yaml}"
TASKS=()
if [[ "${2:-}" == "--tasks" ]]; then
  shift 2
  TASKS=("$@")
fi

if [[ -z "${OPENAI_COMPAT_API_KEY:-}" && ! -f "${OPENAI_COMPAT_KEY_FILE:-key.txt}" && ! -f key.txt ]]; then
  echo "Set OPENAI_COMPAT_API_KEY or create key.txt (see key.txt.example)" >&2
  exit 1
fi

export EXPERIMENT_CONFIG="$CONFIG"
export MODEL_TYPE="${MODEL_TYPE:-openai_compat}"
export OPENAI_COMPAT_BASE_URL="${OPENAI_COMPAT_BASE_URL:-https://inference.vulcan.alliancecan.ca}"
export OPENAI_COMPAT_CHAT_PATH="${OPENAI_COMPAT_CHAT_PATH:-/v1/chat/completions}"
export OPENAI_COMPAT_MODEL="${OPENAI_COMPAT_MODEL:-gpt-oss-120b}"
export OPENAI_COMPAT_MAX_PARALLEL="${OPENAI_COMPAT_MAX_PARALLEL:-4}"

PYTHON="${PYTHON:-python3}"
if [[ -d .venv ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
elif [[ -d new_dsl_env ]]; then
  # shellcheck disable=SC1091
  source new_dsl_env/bin/activate
fi

CMD=(
  "$PYTHON" src/baseline.py
  --local
  --model_type openai_compat
  --baseline_variant task_env
)
if ((${#TASKS[@]})); then
  CMD+=(--tasks "${TASKS[@]}")
fi

echo "Config: $CONFIG"
echo "Aleph:  $OPENAI_COMPAT_BASE_URL ($OPENAI_COMPAT_MODEL)"
exec "${CMD[@]}"
